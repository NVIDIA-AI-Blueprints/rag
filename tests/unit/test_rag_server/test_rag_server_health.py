# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

import pytest

from nvidia_rag.rag_server.health import (
    check_service_health,
    check_minio_health,
    is_nvidia_api_catalog_url,
    check_all_services_health,
    print_health_report,
    check_and_print_services_health,
    check_services_health,
)


class MockAsyncContextManager:
    """Helper class to properly mock async context managers for aiohttp"""
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestCheckServiceHealth:
    """Test cases for check_service_health function"""

    @pytest.mark.asyncio
    async def test_check_service_health_success(self):
        """Test successful service health check"""
        mock_response = MagicMock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_get = MockAsyncContextManager(mock_response)
            mock_session.get.return_value = mock_get
            mock_session_class.return_value.__aenter__.return_value = mock_session

            with patch("time.time", side_effect=[0.0, 0.1]):
                result = await check_service_health("http://localhost:8080/health", "Test Service")

            assert result["status"] == "healthy"
            assert result["service"] == "Test Service"
            assert result["http_status"] == 200
            assert result["latency_ms"] == 100.0

    @pytest.mark.asyncio
    async def test_check_service_health_unhealthy_status(self):
        """Test service health check with unhealthy HTTP status"""
        mock_response = MagicMock()
        mock_response.status = 500

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_get = MockAsyncContextManager(mock_response)
            mock_session.get.return_value = mock_get
            mock_session_class.return_value.__aenter__.return_value = mock_session

            with patch("time.time", side_effect=[0.0, 0.05]):
                result = await check_service_health("http://localhost:8080/health", "Test Service")

            assert result["status"] == "unhealthy"
            assert result["http_status"] == 500

    @pytest.mark.asyncio
    async def test_check_service_health_timeout(self):
        """Test service health check with timeout"""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.side_effect = asyncio.TimeoutError()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await check_service_health("http://localhost:8080/health", "Test Service", timeout=1)

            assert result["status"] == "timeout"
            assert "timed out after 1s" in result["error"]

    @pytest.mark.asyncio
    async def test_check_service_health_client_error(self):
        """Test service health check with client error"""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await check_service_health("http://localhost:8080/health", "Test Service")

            assert result["status"] == "error"
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_check_service_health_post_with_json(self):
        """Test POST request with JSON data"""
        mock_response = MagicMock()
        mock_response.status = 201

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_post = MockAsyncContextManager(mock_response)
            mock_session.post.return_value = mock_post
            mock_session_class.return_value.__aenter__.return_value = mock_session

            json_data = {"test": "data"}
            result = await check_service_health(
                "http://localhost:8080/api", 
                "API Service", 
                method="POST", 
                json_data=json_data
            )

            assert result["status"] == "healthy"
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_health_url_with_scheme_added(self):
        """Test URL without scheme gets http:// added"""
        mock_response = MagicMock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_get = MockAsyncContextManager(mock_response)
            mock_session.get.return_value = mock_get
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await check_service_health("localhost:8080/health", "Test Service")

            assert result["status"] == "healthy"
            # Verify the URL was called with http:// prefix
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            assert call_args[0][0] == "http://localhost:8080/health"

    @pytest.mark.asyncio
    async def test_check_service_health_no_url(self):
        """Test with empty URL"""
        result = await check_service_health("", "Test Service")

        assert result["status"] == "skipped"
        assert result["error"] == "No URL provided"

    @pytest.mark.asyncio
    async def test_check_service_health_with_headers(self):
        """Test service health check with custom headers"""
        mock_response = MagicMock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_get = MockAsyncContextManager(mock_response)
            mock_session.get.return_value = mock_get
            mock_session_class.return_value.__aenter__.return_value = mock_session

            headers = {"Authorization": "Bearer token"}
            result = await check_service_health(
                "http://localhost:8080/health", 
                "Test Service", 
                headers=headers
            )

            assert result["status"] == "healthy"


class TestCheckMinioHealth:
    """Test cases for check_minio_health function"""

    @pytest.mark.asyncio
    async def test_check_minio_health_success(self):
        """Test successful MinIO health check"""
        mock_buckets = [MagicMock(), MagicMock(), MagicMock()]

        with patch("nvidia_rag.rag_server.health.MinioOperator") as mock_minio_class:
            mock_minio = MagicMock()
            mock_minio.client.list_buckets.return_value = mock_buckets
            mock_minio_class.return_value = mock_minio

            with patch("time.time", side_effect=[0.0, 0.15]):
                result = await check_minio_health("localhost:9000", "access_key", "secret_key")

            assert result["status"] == "healthy"
            assert result["service"] == "MinIO"
            assert result["url"] == "localhost:9000"
            assert result["buckets"] == 3
            assert result["latency_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_check_minio_health_connection_error(self):
        """Test MinIO health check with connection error"""
        with patch("nvidia_rag.rag_server.health.MinioOperator") as mock_minio_class:
            mock_minio_class.side_effect = Exception("Connection refused")

            result = await check_minio_health("localhost:9000", "access_key", "secret_key")

            assert result["status"] == "error"
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_check_minio_health_list_buckets_error(self):
        """Test MinIO health check with list_buckets error"""
        with patch("nvidia_rag.rag_server.health.MinioOperator") as mock_minio_class:
            mock_minio = MagicMock()
            mock_minio.client.list_buckets.side_effect = Exception("Access denied")
            mock_minio_class.return_value = mock_minio

            result = await check_minio_health("localhost:9000", "wrong_key", "wrong_secret")

            assert result["status"] == "error"
            assert "Access denied" in result["error"]

    @pytest.mark.asyncio
    async def test_check_minio_health_no_endpoint(self):
        """Test MinIO health check with no endpoint"""
        result = await check_minio_health("", "access_key", "secret_key")

        assert result["status"] == "skipped"
        assert result["error"] == "No endpoint provided"


class TestIsNvidiaApiCatalogUrl:
    """Test cases for is_nvidia_api_catalog_url function"""

    def test_is_nvidia_api_catalog_url_empty(self):
        """Test with empty URL"""
        assert is_nvidia_api_catalog_url("") is True

    def test_is_nvidia_api_catalog_url_none(self):
        """Test with None URL"""
        assert is_nvidia_api_catalog_url(None) is True

    def test_is_nvidia_api_catalog_url_nvidia_integrate_prefix(self):
        """Test with NVIDIA integrate API catalog URL"""
        assert is_nvidia_api_catalog_url("https://integrate.api.nvidia.com/v1/models") is True

    def test_is_nvidia_api_catalog_url_nvidia_ai_prefix(self):
        """Test with NVIDIA AI API catalog URL"""
        assert is_nvidia_api_catalog_url("https://ai.api.nvidia.com/v1/chat/completions") is True

    def test_is_nvidia_api_catalog_url_nvidia_nvcf_prefix(self):
        """Test with NVIDIA NVCF API catalog URL"""
        assert is_nvidia_api_catalog_url("https://api.nvcf.nvidia.com/v2/functions") is True

    def test_is_nvidia_api_catalog_url_local_url(self):
        """Test with local URL"""
        assert is_nvidia_api_catalog_url("http://localhost:8080/v1/models") is False

    def test_is_nvidia_api_catalog_url_other_domain(self):
        """Test with other domain URL"""
        assert is_nvidia_api_catalog_url("https://example.com/api/v1/models") is False

    def test_is_nvidia_api_catalog_url_partial_match(self):
        """Test with URL that contains but doesn't start with NVIDIA prefix"""
        assert is_nvidia_api_catalog_url("https://example.com/https://integrate.api.nvidia.com") is False


class TestCheckAllServicesHealth:
    """Test cases for check_all_services_health function"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = MagicMock()
        
        # MinIO config
        config.minio.endpoint = "localhost:9000"
        config.minio.access_key = "test_access_key"
        config.minio.secret_key = "test_secret_key"
        
        # Vector store config
        config.vector_store.name = "milvus"
        config.vector_store.url = "http://localhost:19530"
        
        # LLM config
        config.llm.server_url = "http://localhost:8001"
        config.llm.model_name = "llama-2-7b-chat"
        
        # Query rewriter config
        config.query_rewriter.enable_query_rewriter = True
        config.query_rewriter.server_url = "http://localhost:8002"
        config.query_rewriter.model_name = "nv-rewrite-qr-mistral-7b"
        
        # Embeddings config
        config.embeddings.server_url = "http://localhost:8003"
        config.embeddings.model_name = "nv-embedqa-e5-v5"
        
        # Ranking config
        config.ranking.enable_reranker = True
        config.ranking.server_url = "http://localhost:8004"
        config.ranking.model_name = "nv-rerankqa-mistral-4b-v3"
        
        # Guardrails config
        config.enable_guardrails = True
        
        return config

    @pytest.mark.asyncio
    async def test_check_all_services_health_success(self, mock_config):
        """Test successful check of all services"""
        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(return_value={
            "service": "Vector Store (milvus)",
            "status": "healthy",
            "latency_ms": 50
        })

        # Create side effect function to return proper service health results
        def mock_service_health_side_effect(*args, **kwargs):
            service_name = kwargs.get('service_name', args[1] if len(args) > 1 else 'Unknown')
            return {
                "service": service_name,
                "status": "healthy",
                "latency_ms": 100,
                "url": args[0] if args else "http://localhost:8000"
            }

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            with patch.dict(os.environ, {
                "NEMO_GUARDRAILS_URL": "http://localhost:8005",
                "ENABLE_REFLECTION": "true",
                "REFLECTION_LLM": "llama-2-13b-chat",
                "REFLECTION_LLM_SERVERURL": "http://localhost:8006"
            }):
                with patch("nvidia_rag.rag_server.health.check_minio_health") as mock_minio:
                    with patch("nvidia_rag.rag_server.health.check_service_health", side_effect=mock_service_health_side_effect) as mock_service:
                        
                        # Setup mock returns
                        mock_minio.return_value = {"service": "MinIO", "status": "healthy"}

                        result = await check_all_services_health(mock_vdb_op)

                        assert "databases" in result
                        assert "object_storage" in result
                        assert "nim" in result
                        assert len(result["databases"]) == 1
                        assert len(result["object_storage"]) == 1
                        assert len(result["nim"]) >= 5  # LLM, Query Rewriter, Embeddings, Ranking, Guardrails

    @pytest.mark.asyncio
    async def test_check_all_services_health_nvidia_api_catalog(self, mock_config):
        """Test with NVIDIA API Catalog services"""
        # Configure services to use API catalog
        mock_config.llm.server_url = "https://integrate.api.nvidia.com/v1"
        mock_config.query_rewriter.server_url = "https://ai.api.nvidia.com/v1"
        mock_config.embeddings.server_url = "https://api.nvcf.nvidia.com/v2"
        mock_config.ranking.server_url = ""  # Empty URL should be treated as API catalog

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(return_value={
            "service": "Vector Store",
            "status": "healthy"
        })

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            with patch("nvidia_rag.rag_server.health.check_minio_health"):
                
                result = await check_all_services_health(mock_vdb_op)

                nim_services = result["nim"]
                
                # Find API catalog services
                api_catalog_services = [s for s in nim_services if s.get("message") == "Using NVIDIA API Catalog"]
                assert len(api_catalog_services) >= 4  # LLM, Query Rewriter, Embeddings, Ranking
                
                # Check that all API catalog services are marked as healthy
                for service in api_catalog_services:
                    assert service["status"] == "healthy"
                    assert service["latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_check_all_services_health_disabled_features(self, mock_config):
        """Test with disabled features"""
        # Disable optional features
        mock_config.query_rewriter.enable_query_rewriter = False
        mock_config.ranking.enable_reranker = False
        mock_config.enable_guardrails = False

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(return_value={
            "service": "Vector Store",
            "status": "healthy"
        })

        # Create side effect function to return proper service health results
        def mock_service_health_side_effect(*args, **kwargs):
            service_name = kwargs.get('service_name', args[1] if len(args) > 1 else 'Unknown')
            return {
                "service": service_name,
                "status": "healthy",
                "latency_ms": 100,
                "url": args[0] if args else "http://localhost:8000"
            }

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"ENABLE_REFLECTION": "false"}):
                with patch("nvidia_rag.rag_server.health.check_minio_health") as mock_minio:
                    with patch("nvidia_rag.rag_server.health.check_service_health", side_effect=mock_service_health_side_effect) as mock_service:
                        
                        mock_minio.return_value = {"service": "MinIO", "status": "healthy"}
                        
                        result = await check_all_services_health(mock_vdb_op)

                        nim_services = result["nim"]
                        service_names = [s["service"] for s in nim_services]
                        
                        # Should only have LLM and Embeddings (required services)
                        assert "LLM" in service_names
                        assert "Embeddings" in service_names
                        assert "Query Rewriter" not in service_names
                        assert "Ranking" not in service_names
                        assert "NemoGuardrails" not in service_names
                        assert "Reflection LLM" not in service_names

    @pytest.mark.asyncio
    async def test_check_all_services_health_vector_store_error(self, mock_config):
        """Test with vector store health check error"""
        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health.side_effect = Exception("Vector store connection failed")

        # Create side effect function to return proper service health results
        def mock_service_health_side_effect(*args, **kwargs):
            service_name = kwargs.get('service_name', args[1] if len(args) > 1 else 'Unknown')
            return {
                "service": service_name,
                "status": "healthy",
                "latency_ms": 100,
                "url": args[0] if args else "http://localhost:8000"
            }

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            with patch("nvidia_rag.rag_server.health.check_minio_health"):
                with patch("nvidia_rag.rag_server.health.check_service_health", side_effect=mock_service_health_side_effect):
                    
                    result = await check_all_services_health(mock_vdb_op)

                    db_services = result["databases"]
                    assert len(db_services) == 1
                    assert db_services[0]["status"] == "unknown"
                    assert "Error checking vector store health: Vector store connection failed" in db_services[0]["error"]

    @pytest.mark.asyncio
    async def test_check_all_services_health_no_minio_endpoint(self, mock_config):
        """Test with no MinIO endpoint configured"""
        mock_config.minio.endpoint = ""

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(return_value={
            "service": "Vector Store",
            "status": "healthy"
        })

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            
            result = await check_all_services_health(mock_vdb_op)

            # Should have no object storage entries when endpoint is empty
            assert len(result["object_storage"]) == 0

    @pytest.mark.asyncio
    async def test_check_all_services_health_guardrails_no_url(self, mock_config):
        """Test guardrails health check with no URL configured"""
        mock_config.enable_guardrails = True

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(return_value={
            "service": "Vector Store",
            "status": "healthy"
        })

        # Create side effect function to return proper service health results
        def mock_service_health_side_effect(*args, **kwargs):
            service_name = kwargs.get('service_name', args[1] if len(args) > 1 else 'Unknown')
            return {
                "service": service_name,
                "status": "healthy",
                "latency_ms": 100,
                "url": args[0] if args else "http://localhost:8000"
            }

        with patch("nvidia_rag.rag_server.health.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"NEMO_GUARDRAILS_URL": ""}):
                with patch("nvidia_rag.rag_server.health.check_minio_health"):
                    with patch("nvidia_rag.rag_server.health.check_service_health", side_effect=mock_service_health_side_effect):
                        
                        result = await check_all_services_health(mock_vdb_op)

                        nim_services = result["nim"]
                        guardrails_services = [s for s in nim_services if s["service"] == "NemoGuardrails"]
                        
                        assert len(guardrails_services) == 1
                        assert guardrails_services[0]["status"] == "skipped"
                        assert guardrails_services[0]["message"] == "URL not provided"


class TestPrintHealthReport:
    """Test cases for print_health_report function"""

    def test_print_health_report_all_healthy(self, caplog):
        """Test printing report with all healthy services"""
        health_results = {
            "databases": [
                {"service": "Milvus", "status": "healthy", "latency_ms": 45}
            ],
            "object_storage": [
                {"service": "MinIO", "status": "healthy", "latency_ms": 120, "buckets": 3}
            ],
            "nim": [
                {"service": "LLM", "status": "healthy", "latency_ms": 250, "model": "llama-2-7b-chat"},
                {"service": "Embeddings", "status": "healthy", "latency_ms": 180, "model": "nv-embedqa-e5-v5"}
            ]
        }

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "SERVICE HEALTH STATUS" in caplog.text
        assert "Service 'Milvus' is healthy - Response time: 45ms" in caplog.text
        assert "Service 'MinIO' is healthy - Response time: 120ms" in caplog.text
        assert "Service 'LLM' is healthy - Response time: 250ms" in caplog.text

    def test_print_health_report_mixed_status(self, caplog):
        """Test printing report with mixed service status"""
        health_results = {
            "databases": [
                {"service": "Milvus", "status": "error", "error": "Connection timeout"}
            ],
            "nim": [
                {"service": "LLM", "status": "skipped", "error": "No URL provided"},
                {"service": "Embeddings", "status": "unhealthy", "error": "Service unavailable"}
            ]
        }

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "Service 'Milvus' is not healthy - Issue: Connection timeout" in caplog.text
        assert "Service 'LLM' check skipped - Reason: No URL provided" in caplog.text
        assert "Service 'Embeddings' is not healthy - Issue: Service unavailable" in caplog.text

    def test_print_health_report_empty_results(self, caplog):
        """Test printing report with empty results"""
        health_results = {}

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "SERVICE HEALTH STATUS" in caplog.text

    def test_print_health_report_none_services(self, caplog):
        """Test printing report with None services"""
        health_results = {
            "databases": None,
            "nim": []
        }

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "SERVICE HEALTH STATUS" in caplog.text

    def test_print_health_report_latency_na(self, caplog):
        """Test printing report with N/A latency"""
        health_results = {
            "nim": [
                {"service": "LLM", "status": "healthy", "model": "test-model"}  # No latency_ms field
            ]
        }

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "Service 'LLM' is healthy - Response time: N/Ams" in caplog.text


class TestCheckAndPrintServicesHealth:
    """Test cases for check_and_print_services_health function"""

    @pytest.mark.asyncio
    async def test_check_and_print_services_health(self):
        """Test check and print services health function"""
        mock_results = {
            "databases": [{"service": "Milvus", "status": "healthy"}],
            "object_storage": [{"service": "MinIO", "status": "healthy"}],
            "nim": [{"service": "LLM", "status": "healthy"}]
        }
        mock_vdb_op = MagicMock()

        with patch("nvidia_rag.rag_server.health.check_all_services_health") as mock_check:
            with patch("nvidia_rag.rag_server.health.print_health_report") as mock_print:
                mock_check.return_value = mock_results

                result = await check_and_print_services_health(mock_vdb_op)

                assert result == mock_results
                mock_check.assert_called_once_with(mock_vdb_op)
                mock_print.assert_called_once_with(mock_results)


class TestCheckServicesHealth:
    """Test cases for check_services_health function"""

    def test_check_services_health(self):
        """Test synchronous wrapper for checking service health"""
        mock_results = {
            "databases": [{"service": "Milvus", "status": "healthy"}],
            "object_storage": [{"service": "MinIO", "status": "healthy"}],
            "nim": [{"service": "LLM", "status": "healthy"}]
        }
        mock_vdb_op = MagicMock()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = mock_results

            result = check_services_health(mock_vdb_op)

            assert result == mock_results
            mock_run.assert_called_once()

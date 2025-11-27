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
from pydantic import SecretStr

from nvidia_rag.ingestor_server.health import (
    check_all_services_health,
    check_and_print_services_health,
    check_minio_health,
    check_nv_ingest_health,
    check_redis_health,
    check_service_health,
    check_services_health,
    is_nvidia_api_catalog_url,
    print_health_report,
)
from nvidia_rag.utils.health_models import (
    DatabaseHealthInfo,
    IngestorHealthResponse,
    NIMServiceHealthInfo,
    ProcessingHealthInfo,
    ServiceStatus,
    StorageHealthInfo,
    TaskManagementHealthInfo,
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
    async def test_check_service_health_no_url(self):
        """Test with empty URL"""
        result = await check_service_health("", "Test Service")

        assert result.status == ServiceStatus.SKIPPED
        assert result.error == "No URL provided"


class TestCheckMinioHealth:
    """Test cases for check_minio_health function"""

    @pytest.mark.asyncio
    async def test_check_minio_health_success(self):
        """Test successful MinIO health check"""
        mock_buckets = [MagicMock(), MagicMock()]

        with patch(
            "nvidia_rag.ingestor_server.health.MinioOperator"
        ) as mock_minio_class:
            mock_minio = MagicMock()
            mock_minio.client.list_buckets.return_value = mock_buckets
            mock_minio_class.return_value = mock_minio

            with patch("time.time", side_effect=[0.0, 0.2]):
                result = await check_minio_health("localhost:9000", "key", "secret")

            assert result.status == ServiceStatus.HEALTHY
            assert result.buckets == 2

    @pytest.mark.asyncio
    async def test_check_minio_health_error(self):
        """Test MinIO health check with error"""
        with patch(
            "nvidia_rag.ingestor_server.health.MinioOperator"
        ) as mock_minio_class:
            mock_minio_class.side_effect = Exception("Connection failed")

            result = await check_minio_health("localhost:9000", "key", "secret")

            assert result.status == ServiceStatus.ERROR
            assert "Connection failed" in result.error

    @pytest.mark.asyncio
    async def test_check_minio_health_no_endpoint(self):
        """Test MinIO health check with no endpoint"""
        result = await check_minio_health("", "key", "secret")

        assert result.status == ServiceStatus.SKIPPED
        assert result.error == "No endpoint provided"


class TestCheckNvIngestHealth:
    """Test cases for check_nv_ingest_health function"""

    @pytest.mark.asyncio
    async def test_check_nv_ingest_health_no_hostname_or_port(self):
        """Test NV-Ingest health check with missing hostname or port"""
        result = await check_nv_ingest_health("", 8080)
        assert result.status == ServiceStatus.SKIPPED


class TestCheckRedisHealth:
    """Test cases for check_redis_health function"""

    @pytest.mark.asyncio
    async def test_check_redis_health_success(self):
        """Test successful Redis health check"""
        with patch("builtins.__import__") as mock_import:
            mock_redis_module = MagicMock()
            mock_redis_class = MagicMock()
            mock_redis = MagicMock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            mock_redis_module.Redis = mock_redis_class
            mock_import.return_value = mock_redis_module

            with patch("time.time", side_effect=[0.0, 0.05]):
                result = await check_redis_health("localhost", 6379, 0)

            assert result.status == ServiceStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_redis_health_ping_failed(self):
        """Test Redis health check with ping failure"""
        with patch("builtins.__import__") as mock_import:
            mock_redis_module = MagicMock()
            mock_redis_class = MagicMock()
            mock_redis = MagicMock()
            mock_redis.ping.return_value = False
            mock_redis_class.return_value = mock_redis
            mock_redis_module.Redis = mock_redis_class
            mock_import.return_value = mock_redis_module

            result = await check_redis_health("localhost", 6379, 0)

            assert result.status == ServiceStatus.UNHEALTHY
            assert result.error == "Redis ping failed"

    @pytest.mark.asyncio
    async def test_check_redis_health_import_error(self):
        """Test Redis health check with ImportError"""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'redis'")

            result = await check_redis_health("localhost", 6379, 0)

            assert result.status == ServiceStatus.SKIPPED
            assert result.error == "Redis not available (library not installed)"

    @pytest.mark.asyncio
    async def test_check_redis_health_no_host_or_port(self):
        """Test Redis health check with missing host or port"""
        result = await check_redis_health("", 6379, 0)
        assert result.status == ServiceStatus.SKIPPED


class TestIsNvidiaApiCatalogUrl:
    """Test cases for is_nvidia_api_catalog_url function"""

    def test_is_nvidia_api_catalog_url_empty(self):
        """Test with empty URL"""
        assert is_nvidia_api_catalog_url("") is True

    def test_is_nvidia_api_catalog_url_none(self):
        """Test with None URL"""
        assert is_nvidia_api_catalog_url(None) is True

    def test_is_nvidia_api_catalog_url_nvidia_prefix(self):
        """Test with NVIDIA API catalog URLs"""
        assert (
            is_nvidia_api_catalog_url("https://integrate.api.nvidia.com/v1/models")
            is True
        )
        assert is_nvidia_api_catalog_url("https://ai.api.nvidia.com/v1/models") is True
        assert (
            is_nvidia_api_catalog_url("https://api.nvcf.nvidia.com/v2/functions")
            is True
        )

    def test_is_nvidia_api_catalog_url_other_url(self):
        """Test with non-NVIDIA API Catalog URL"""
        assert is_nvidia_api_catalog_url("https://example.com/api") is False
        assert is_nvidia_api_catalog_url("http://localhost:8080") is False


class TestCheckAllServicesHealth:
    """Test cases for check_all_services_health function"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = MagicMock()
        config.minio.endpoint = "localhost:9000"
        config.minio.access_key = SecretStr("test_key")
        config.minio.secret_key = SecretStr("test_secret")
        config.vector_store.url = "http://localhost:19530"
        config.vector_store.name = "milvus"
        config.nv_ingest.message_client_hostname = "localhost"
        config.nv_ingest.message_client_port = 8080
        config.embeddings.server_url = "http://localhost:8081"
        config.embeddings.model_name = "test-embedding-model"
        config.summarizer.server_url = "http://localhost:8082"
        config.summarizer.model_name = "test-llm-model"
        config.nv_ingest.extract_images = True
        config.nv_ingest.caption_endpoint_url = (
            "http://localhost:8083/v1/chat/completions"
        )
        config.nv_ingest.caption_model_name = "test-caption-model"
        return config

    @pytest.mark.asyncio
    async def test_check_all_services_health_success(self, mock_config):
        """Test successful check of all services"""
        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(
            return_value={
                "service": "Vector DB",
                "url": "http://vectordb:19530",
                "status": "healthy",
            }
        )

        with patch.dict(
            os.environ,
            {"REDIS_HOST": "localhost", "REDIS_PORT": "6379", "REDIS_DB": "0"},
        ):
            with patch(
                "nvidia_rag.ingestor_server.health.check_minio_health"
            ) as mock_minio:
                with patch(
                    "nvidia_rag.ingestor_server.health.check_nv_ingest_health"
                ) as mock_nv_ingest:
                    with patch(
                        "nvidia_rag.ingestor_server.health.check_service_health"
                    ) as mock_service:
                        with patch(
                            "nvidia_rag.ingestor_server.health.check_redis_health"
                        ) as mock_redis:
                            # Setup mock returns - health check functions return Pydantic models
                            mock_minio.return_value = StorageHealthInfo(
                                service="MinIO",
                                url="http://minio:9000",
                                status=ServiceStatus.HEALTHY,
                            )
                            mock_nv_ingest.return_value = ProcessingHealthInfo(
                                service="NV-Ingest",
                                url="http://nv-ingest:8000",
                                status=ServiceStatus.HEALTHY,
                            )
                            mock_service.return_value = NIMServiceHealthInfo(
                                service="Test",
                                url="http://test:8000",
                                status=ServiceStatus.HEALTHY,
                            )
                            mock_redis.return_value = TaskManagementHealthInfo(
                                service="Redis",
                                url="redis://redis:6379",
                                status=ServiceStatus.HEALTHY,
                            )

                            result = await check_all_services_health(mock_vdb_op, mock_config)

                            assert isinstance(result, IngestorHealthResponse)
                            assert hasattr(result, "databases")
                            assert hasattr(result, "object_storage")
                            assert hasattr(result, "nim")
                            assert hasattr(result, "processing")
                            assert hasattr(result, "task_management")

    @pytest.mark.asyncio
    async def test_check_all_services_health_nvidia_api_catalog(self, mock_config):
        """Test with NVIDIA API Catalog services"""
        mock_config.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
        mock_config.summarizer.server_url = "https://ai.api.nvidia.com/v1"
        mock_config.nv_ingest.caption_endpoint_url = "https://api.nvcf.nvidia.com/v2"

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health = AsyncMock(
            return_value={
                "service": "Vector DB",
                "url": "http://vectordb:19530",
                "status": "healthy",
            }
        )

        with patch.dict(
            os.environ,
            {"REDIS_HOST": "localhost", "REDIS_PORT": "6379", "REDIS_DB": "0"},
        ):
            with patch("nvidia_rag.ingestor_server.health.check_minio_health", return_value=StorageHealthInfo(
                service="MinIO", url="http://minio:9000", status=ServiceStatus.HEALTHY
            )):
                with patch(
                    "nvidia_rag.ingestor_server.health.check_nv_ingest_health",
                    return_value=ProcessingHealthInfo(
                        service="NV-Ingest", url="http://nv-ingest:8000", status=ServiceStatus.HEALTHY
                    )
                ):
                    with patch(
                        "nvidia_rag.ingestor_server.health.check_redis_health",
                        return_value=TaskManagementHealthInfo(
                            service="Redis", url="redis://redis:6379", status=ServiceStatus.HEALTHY
                        )
                    ):
                        result = await check_all_services_health(mock_vdb_op, mock_config)

                        nim_services = result.nim
                        assert len(nim_services) == 3

                        # Check that each service has the expected URL and model from the mock config
                        expected_services = [
                            {
                                "url": "https://integrate.api.nvidia.com/v1",  # embeddings
                                "model": "test-embedding-model",
                            },
                            {
                                "url": "https://ai.api.nvidia.com/v1",  # summarizer
                                "model": "test-llm-model",
                            },
                            {
                                "url": "https://api.nvcf.nvidia.com/v2",  # caption endpoint
                                "model": "test-caption-model",
                            },
                        ]

                        for i, service in enumerate(nim_services):
                            assert service.url == expected_services[i]["url"]
                            assert service.status == ServiceStatus.HEALTHY
                            assert service.model == expected_services[i]["model"]

    @pytest.mark.asyncio
    async def test_check_all_services_health_unknown_vector_store(self, mock_config):
        """Test with unknown vector store type"""
        mock_config.vector_store.name = "unknown_db"

        mock_vdb_op = MagicMock()
        mock_vdb_op.check_health.side_effect = Exception(
            "Unsupported vector store type"
        )

        with patch.dict(
            os.environ,
            {"REDIS_HOST": "localhost", "REDIS_PORT": "6379", "REDIS_DB": "0"},
        ):
            with patch("nvidia_rag.ingestor_server.health.check_minio_health", return_value=StorageHealthInfo(
                service="MinIO", url="http://minio:9000", status=ServiceStatus.HEALTHY
            )):
                with patch(
                    "nvidia_rag.ingestor_server.health.check_nv_ingest_health",
                    return_value=ProcessingHealthInfo(
                        service="NV-Ingest", url="http://nv-ingest:8000", status=ServiceStatus.HEALTHY
                    )
                ):
                    with patch(
                        "nvidia_rag.ingestor_server.health.check_service_health",
                        return_value=NIMServiceHealthInfo(
                            service="Test", url="http://test:8000", status=ServiceStatus.HEALTHY
                        )
                    ):
                        with patch(
                            "nvidia_rag.ingestor_server.health.check_redis_health",
                            return_value=TaskManagementHealthInfo(
                                service="Redis", url="redis://redis:6379", status=ServiceStatus.HEALTHY
                            )
                        ):
                            result = await check_all_services_health(mock_vdb_op, mock_config)

                            db_services = result.databases
                            assert len(db_services) == 1
                            assert db_services[0].status == ServiceStatus.UNKNOWN
                            assert (
                                "Unsupported vector store type"
                                in db_services[0].error
                            )


class TestPrintHealthReport:
    """Test cases for print_health_report function"""

    def test_print_health_report_all_healthy(self, caplog):
        """Test printing report with all healthy services"""
        from nvidia_rag.utils.health_models import (
            DatabaseHealthInfo,
            NIMServiceHealthInfo,
        )

        health_results = IngestorHealthResponse(
            databases=[
                DatabaseHealthInfo(service="Milvus", url="", status=ServiceStatus.HEALTHY, latency_ms=100)
            ],
            nim=[
                NIMServiceHealthInfo(
                    service="Embeddings",
                    url="",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=50,
                    model="test-embedding-model",
                )
            ],
        )

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "INGESTOR SERVICE HEALTH STATUS" in caplog.text
        assert "✓ Milvus is healthy" in caplog.text

    def test_print_health_report_mixed_status(self, caplog):
        """Test printing report with mixed service status"""
        from nvidia_rag.utils.health_models import (
            DatabaseHealthInfo,
            NIMServiceHealthInfo,
        )

        health_results = IngestorHealthResponse(
            databases=[
                DatabaseHealthInfo(service="Redis", url="", status=ServiceStatus.ERROR, error="Connection failed")
            ],
            nim=[
                NIMServiceHealthInfo(
                    service="Embeddings",
                    url="",
                    status=ServiceStatus.SKIPPED,
                    error="No URL provided",
                    model="test-embedding-model",
                )
            ],
        )

        with caplog.at_level(logging.INFO):
            print_health_report(health_results)

        assert "✗ Redis is not healthy" in caplog.text
        assert "- Embeddings check skipped" in caplog.text


class TestCheckAndPrintServicesHealth:
    """Test cases for check_and_print_services_health function"""

    @pytest.mark.asyncio
    async def test_check_and_print_services_health(self):
        """Test check and print services health function"""
        mock_results = IngestorHealthResponse()
        mock_vdb_op = MagicMock()

        with patch(
            "nvidia_rag.ingestor_server.health.check_all_services_health"
        ) as mock_check:
            with patch(
                "nvidia_rag.ingestor_server.health.print_health_report"
            ) as mock_print:
                mock_check.return_value = mock_results

                result = await check_and_print_services_health(mock_vdb_op)

                assert result == mock_results
                mock_check.assert_called_once_with(mock_vdb_op, None)
                mock_print.assert_called_once_with(mock_results)


class TestCheckServicesHealth:
    """Test cases for check_services_health function"""

    def test_check_services_health(self):
        """Test synchronous wrapper for checking service health"""
        mock_results = IngestorHealthResponse()
        mock_vdb_op = MagicMock()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = mock_results

            result = check_services_health(mock_vdb_op)

            assert result == mock_results
            mock_run.assert_called_once()

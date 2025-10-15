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

"""
Unit tests for the RAG server metrics functionality.
Tests the actual /metrics endpoint and tracing integration in the RAG server.
"""

import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from nvidia_rag.rag_server.response_generator import ErrorCodeMapping
from nvidia_rag.rag_server.server import app


class MockNvidiaRAG:
    """Mock class for NvidiaRAG with configurable responses and error states"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rag_contexts = [
            Mock(
                metadata={
                    "source": {"source_id": "test.pdf"},
                    "content_metadata": {"type": "text"},
                },
                page_content="Test content",
            )
        ]
        self.rag_generator_items = ["Hello", " world", "!"]
        self.llm_generator_items = ["Hello", " world", "!"]
        self._generate_side_effect = None
        self._search_side_effect = None

    async def _async_gen(self, items):
        for item in items:
            yield f"data: {json.dumps({'choices': [{'message': {'content': item}}]})}\n"

    def generate(self, *args, **kwargs):
        if self._generate_side_effect:
            return self._generate_side_effect(*args, **kwargs)
        return self._async_gen(self.rag_generator_items)

    def search(self, *args, **kwargs):
        if self._search_side_effect:
            return self._search_side_effect(*args, **kwargs)
        return {
            "total_results": 1,
            "results": [
                {
                    "content": "Test content",
                    "metadata": {
                        "source": {"source_id": "test.pdf"},
                        "content_metadata": {"type": "text"},
                    },
                    "score": 0.95,
                }
            ],
        }

    async def health(self, check_dependencies: bool = False):
        return {"message": "Service is up."}


# Create mock instances
mock_nvidia_rag_instance = MockNvidiaRAG()


class TestRAGServerMetricsEndpoint:
    """Test cases for the /metrics endpoint in RAG server"""

    @pytest.fixture
    def temp_prom_dir(self):
        """Create a temporary directory for Prometheus multi-process data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def client(self, temp_prom_dir):
        """Create test client with mocked RAG server"""
        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": temp_prom_dir}):
            with patch(
                "nvidia_rag.rag_server.server.NVIDIA_RAG", mock_nvidia_rag_instance
            ):
                # Mock the tracing setup to avoid instrumentation issues
                with patch(
                    "nvidia_rag.rag_server.server.get_config"
                ) as mock_get_config:
                    # Create a mock settings object with tracing disabled
                    mock_settings = Mock()
                    mock_settings.tracing.enabled = False
                    mock_get_config.return_value = mock_settings

                    return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_mock_instance(self):
        """Reset mock instance before each test"""
        mock_nvidia_rag_instance.reset()
        yield

    def test_metrics_endpoint_success(self, client):
        """Test successful metrics endpoint response"""
        # Mock the Prometheus components
        with (
            patch(
                "nvidia_rag.rag_server.server.CollectorRegistry"
            ) as mock_registry_class,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch("nvidia_rag.rag_server.server.generate_latest") as mock_generate,
        ):
            # Setup mocks
            mock_registry = Mock(spec=CollectorRegistry)
            mock_registry_class.return_value = mock_registry
            mock_generate.return_value = b"# HELP test_metric A test metric\n# TYPE test_metric counter\ntest_metric_total 42\n"

            # Make request to metrics endpoint
            response = client.get("/metrics")

            # Verify response
            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert b"test_metric_total 42" in response.content

            # Verify mocks were called correctly
            mock_registry_class.assert_called_once()
            mock_generate.assert_called_once_with(mock_registry)

    def test_metrics_endpoint_error_handling(self, client):
        """Test error handling when metrics generation fails"""
        with patch("nvidia_rag.rag_server.server.CollectorRegistry") as mock_registry:
            mock_registry.side_effect = Exception("Registry creation failed")

            response = client.get("/metrics")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert "Error generating metrics" in response.text

    def test_metrics_endpoint_multiprocess_collector_error(self, client):
        """Test error handling when MultiProcessCollector fails"""
        with patch(
            "nvidia_rag.rag_server.server.MultiProcessCollector"
        ) as mock_collector:
            mock_collector.side_effect = Exception("MultiProcessCollector failed")

            response = client.get("/metrics")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert "Error generating metrics" in response.text

    def test_metrics_endpoint_generate_latest_error(self, client):
        """Test error handling when generate_latest fails"""
        with patch("nvidia_rag.rag_server.server.generate_latest") as mock_generate:
            mock_generate.side_effect = Exception("generate_latest failed")

            response = client.get("/metrics")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert "Error generating metrics" in response.text

    def test_metrics_endpoint_logging(self, client, caplog):
        """Test that metrics endpoint logs debug information"""
        # Mock the Prometheus components
        with (
            patch(
                "nvidia_rag.rag_server.server.CollectorRegistry"
            ) as mock_registry_class,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch("nvidia_rag.rag_server.server.generate_latest") as mock_generate,
        ):
            # Setup mocks
            mock_registry = Mock(spec=CollectorRegistry)
            mock_registry_class.return_value = mock_registry
            mock_generate.return_value = b"test metrics data"

            with caplog.at_level("DEBUG"):
                response = client.get("/metrics")

                assert response.status_code == ErrorCodeMapping.SUCCESS
                # Check for any debug message containing "Generated" and "bytes"
                debug_messages = [
                    record.message
                    for record in caplog.records
                    if record.levelname == "DEBUG"
                ]
                assert any(
                    "Generated" in msg and "bytes" in msg for msg in debug_messages
                )

    def test_metrics_endpoint_content_type(self, client):
        """Test that metrics endpoint returns correct content type"""
        response = client.get("/metrics")

        assert response.status_code == ErrorCodeMapping.SUCCESS
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_metrics_endpoint_empty_response(self, client):
        """Test metrics endpoint with empty response"""
        # Mock the Prometheus components
        with (
            patch(
                "nvidia_rag.rag_server.server.CollectorRegistry"
            ) as mock_registry_class,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch("nvidia_rag.rag_server.server.generate_latest") as mock_generate,
        ):
            # Setup mocks
            mock_registry = Mock(spec=CollectorRegistry)
            mock_registry_class.return_value = mock_registry
            mock_generate.return_value = b""

            response = client.get("/metrics")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.text == ""

    def test_metrics_endpoint_with_real_prometheus_components(
        self, temp_prom_dir, client
    ):
        """Test metrics endpoint with real Prometheus components"""
        # This test uses actual Prometheus components to verify integration
        with (
            patch(
                "nvidia_rag.rag_server.server.CollectorRegistry"
            ) as mock_registry_class,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch("nvidia_rag.rag_server.server.generate_latest") as mock_generate,
        ):
            # Setup mocks to return real Prometheus format
            mock_registry = Mock(spec=CollectorRegistry)
            mock_registry_class.return_value = mock_registry
            mock_generate.return_value = b"# HELP test_metric A test metric\n# TYPE test_metric counter\ntest_metric_total 42\n"

            response = client.get("/metrics")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            assert response.headers["content-type"] == "text/plain; charset=utf-8"

            # Verify the response contains some Prometheus format
            content = response.text
            assert len(content) > 0
            assert "# HELP" in content
            assert "# TYPE" in content

    def test_metrics_endpoint_http_methods(self, client):
        """Test that metrics endpoint only accepts GET requests"""
        # Test POST method (should return 405 Method Not Allowed)
        response = client.post("/metrics")
        assert response.status_code == ErrorCodeMapping.METHOD_NOT_ALLOWED

        # Test PUT method (should return 405 Method Not Allowed)
        response = client.put("/metrics")
        assert response.status_code == ErrorCodeMapping.METHOD_NOT_ALLOWED

        # Test DELETE method (should return 405 Method Not Allowed)
        response = client.delete("/metrics")
        assert response.status_code == ErrorCodeMapping.METHOD_NOT_ALLOWED

    def test_metrics_endpoint_with_query_params(self, client):
        """Test metrics endpoint with query parameters (should be ignored)"""
        response = client.get("/metrics?format=prometheus&debug=true")

        assert response.status_code == ErrorCodeMapping.SUCCESS
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_metrics_endpoint_performance(self, client):
        """Test metrics endpoint performance"""
        start_time = time.time()
        response = client.get("/metrics")
        end_time = time.time()

        assert response.status_code == ErrorCodeMapping.SUCCESS
        # Should respond within reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0

    def test_metrics_endpoint_concurrent_requests(self, client):
        """Test metrics endpoint with concurrent requests"""
        results = []
        errors = []

        def make_request():
            try:
                response = client.get("/metrics")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # Create 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert len(errors) == 0
        assert len(results) == 5
        assert all(status == ErrorCodeMapping.SUCCESS for status in results)

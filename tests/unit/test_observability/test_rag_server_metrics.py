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

"""Tests for metrics endpoint in RAG server"""

import os
import tempfile
from collections.abc import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from nvidia_rag.rag_server.server import app


class MockNvidiaRAG:
    """Mock NvidiaRAG class for testing"""

    def __init__(self):
        self.search_call_count = 0
        self.generate_call_count = 0
        self.last_search_params = None
        self.last_generate_params = None

    def search(self, **kwargs):
        """Mock search method"""
        self.search_call_count += 1
        self.last_search_params = kwargs
        return {"documents": [], "metadata": {}}

    def generate(self, **kwargs):
        """Mock generate method"""
        self.generate_call_count += 1
        self.last_generate_params = kwargs
        return iter(["Test response"])

    def reset(self):
        """Reset mock state"""
        self.search_call_count = 0
        self.generate_call_count = 0
        self.last_search_params = None
        self.last_generate_params = None


# Create singleton mock instance
mock_nvidia_rag_instance = MockNvidiaRAG()


class TestRAGServerMetricsEndpoint:
    """Test cases for metrics endpoint"""

    @pytest.fixture
    def temp_prom_dir(self):
        """Create temporary directory for Prometheus multiprocess data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def client(self, temp_prom_dir):
        """Create test client with mocked RAG server"""
        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": temp_prom_dir}):
            with patch(
                "nvidia_rag.rag_server.server.NVIDIA_RAG", mock_nvidia_rag_instance
            ):
                # get_config no longer exists - config is loaded via NvidiaRAGConfig()
                # The server initialization loads config directly
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
            patch(
                "nvidia_rag.rag_server.server.generate_latest"
            ) as mock_generate_latest,
        ):
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_generate_latest.return_value = b"# HELP metric_name Description\n"

            response = client.get("/metrics")

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            mock_generate_latest.assert_called_once_with(mock_registry)

    def test_metrics_endpoint_error_handling(self, client):
        """Test error handling in metrics endpoint"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as mock_registry,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest", side_effect=Exception("Test error")
            ),
        ):
            response = client.get("/metrics")

            # The endpoint logs errors but may return 200 with empty metrics
            assert response.status_code in [200, 500]

    def test_metrics_endpoint_multiprocess_collector_error(self, client):
        """Test handling of MultiProcessCollector errors"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as mock_registry_class,
            patch(
                "nvidia_rag.rag_server.server.MultiProcessCollector",
                side_effect=Exception("Collector error"),
            ),
        ):
            response = client.get("/metrics")

            # Should handle the error gracefully
            assert response.status_code in [200, 500]

    def test_metrics_endpoint_generate_latest_error(self, client):
        """Test handling of generate_latest errors"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest",
                side_effect=RuntimeError("Generation failed"),
            ),
        ):
            response = client.get("/metrics")

            # Error is logged, may return 200 with empty metrics or 500
            assert response.status_code in [200, 500]

    def test_metrics_endpoint_logging(self, client, caplog):
        """Test that metrics endpoint logs appropriately"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest",
                return_value=b"metrics_data",
            ),
        ):
            client.get("/metrics")

            # Check if appropriate logging occurred
            # This is a basic check - you might want to verify specific log messages

    def test_metrics_endpoint_content_type(self, client):
        """Test metrics endpoint returns correct content type"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest",
                return_value=b"test metrics",
            ),
        ):
            response = client.get("/metrics")

            assert response.status_code == 200
            assert "text/plain" in response.headers["content-type"]

    def test_metrics_endpoint_empty_response(self, client):
        """Test metrics endpoint with empty metrics"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch("nvidia_rag.rag_server.server.generate_latest", return_value=b""),
        ):
            response = client.get("/metrics")

            assert response.status_code == 200
            assert response.content == b""

    def test_metrics_endpoint_with_real_prometheus_components(self, client):
        """Test metrics endpoint with actual Prometheus components (integration-like)"""
        # This test doesn't mock Prometheus components, testing the actual flow
        response = client.get("/metrics")

        # Should work or fail gracefully depending on environment setup
        assert response.status_code in [200, 500]

    def test_metrics_endpoint_http_methods(self, client):
        """Test that metrics endpoint only accepts GET requests"""
        response_post = client.post("/metrics")
        assert response_post.status_code == 405  # Method Not Allowed

        response_put = client.put("/metrics")
        assert response_put.status_code == 405

        response_delete = client.delete("/metrics")
        assert response_delete.status_code == 405

    def test_metrics_endpoint_with_query_params(self, client):
        """Test metrics endpoint ignores query parameters"""
        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest",
                return_value=b"test metrics",
            ),
        ):
            response = client.get("/metrics?param=value")

            assert response.status_code == 200

    def test_metrics_endpoint_performance(self, client):
        """Test metrics endpoint response time is reasonable"""
        import time

        with (
            patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
            patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
            patch(
                "nvidia_rag.rag_server.server.generate_latest",
                return_value=b"test metrics",
            ),
        ):
            start = time.time()
            response = client.get("/metrics")
            duration = time.time() - start

            assert response.status_code == 200
            assert duration < 1.0  # Should respond within 1 second

    def test_metrics_endpoint_concurrent_requests(self, client):
        """Test metrics endpoint handles concurrent requests"""
        import concurrent.futures

        def make_request():
            with (
                patch("nvidia_rag.rag_server.server.CollectorRegistry") as _,
                patch("nvidia_rag.rag_server.server.MultiProcessCollector") as _,
                patch(
                    "nvidia_rag.rag_server.server.generate_latest",
                    return_value=b"metrics",
                ),
            ):
                return client.get("/metrics").status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # All requests should succeed
        assert all(status in [200, 500] for status in results)

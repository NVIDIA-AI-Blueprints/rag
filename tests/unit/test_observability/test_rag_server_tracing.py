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
Unit tests for the RAG server tracing integration.
Tests the actual tracing.py functions and their integration with the RAG server.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from opentelemetry.sdk.resources import Resource

import nvidia_rag.utils.observability.tracing as tracing
from nvidia_rag.utils.observability.tracing import instrument


class TestRAGServerTracingIntegration:
    """Test cases for tracing integration in RAG server"""

    @pytest.fixture
    def temp_prom_dir(self):
        """Create a temporary directory for Prometheus multi-process data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_settings(self, temp_prom_dir):
        """Create mock settings for testing"""
        settings = Mock()
        settings.tracing.enabled = True
        settings.tracing.prometheus_multiproc_dir = temp_prom_dir
        settings.tracing.otlp_grpc_endpoint = ""
        settings.tracing.otlp_http_endpoint = ""
        return settings

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app for testing"""
        return FastAPI()

    def test_prometheus_multiproc_dir_creation(
        self, temp_prom_dir, mock_settings, mock_app
    ):
        """Test that PROMETHEUS_MULTIPROC_DIR is created when it doesn't exist"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs") as mock_makedirs:
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        # Call instrument function
                        instrument(mock_app, mock_settings)

                        # Verify directory creation was called
                        mock_makedirs.assert_called_once_with(
                            temp_prom_dir, exist_ok=True
                        )

    def test_prometheus_meter_provider_setup(
        self, temp_prom_dir, mock_settings, mock_app
    ):
        """Test that PrometheusMeterProvider is set up correctly"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch(
                "nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"
            ) as mock_provider:
                with patch(
                    "nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"
                ) as mock_set_meter:
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        # Call instrument function
                        instrument(mock_app, mock_settings)

                        # Verify PrometheusMeterProvider was called
                        mock_provider.assert_called_once()
                        mock_set_meter.assert_called_once()

    def test_otlp_provider_setup_with_endpoint(self, temp_prom_dir, mock_app):
        """Test OTLP provider setup when endpoint is provided"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        with patch(
                            "nvidia_rag.utils.observability.tracing.OTLPMetricExporter"
                        ) as mock_exporter:
                            # Mock settings with OTLP endpoint
                            settings = Mock()
                            settings.tracing.enabled = True
                            settings.tracing.prometheus_multiproc_dir = temp_prom_dir
                            settings.tracing.otlp_grpc_endpoint = (
                                "http://localhost:4317"
                            )
                            settings.tracing.otlp_http_endpoint = ""

                            # Call instrument function
                            instrument(mock_app, settings)

                            # Verify OTLP exporter was called
                            mock_exporter.assert_called_once()

    def test_otlp_provider_setup_without_endpoint(
        self, temp_prom_dir, mock_settings, mock_app
    ):
        """Test OTLP provider setup when no endpoint is provided"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        with patch(
                            "nvidia_rag.utils.observability.tracing.OTLPMetricExporter"
                        ) as mock_exporter:
                            # Call instrument function
                            instrument(mock_app, mock_settings)

                            # Verify OTLP exporter was NOT called
                            mock_exporter.assert_not_called()

    def test_fastapi_instrumentation(self, temp_prom_dir, mock_settings, mock_app):
        """Test that FastAPI is properly instrumented"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch(
                        "nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"
                    ) as mock_instrumentor_class:
                        # Mock the instance method
                        mock_instrumentor_instance = Mock()
                        mock_instrumentor_class.return_value = (
                            mock_instrumentor_instance
                        )

                        # Call instrument function
                        instrument(mock_app, mock_settings)

                        # Verify FastAPI instrumentation was called
                        mock_instrumentor_instance.instrument_app.assert_called_once()

    def test_environment_variables(self, temp_prom_dir, mock_settings, mock_app):
        """Test that environment variables are set correctly"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        # Call instrument function
                        instrument(mock_app, mock_settings)

                        # Verify environment variable is set
                        assert (
                            os.environ.get("PROMETHEUS_MULTIPROC_DIR") == temp_prom_dir
                        )

    def test_error_handling_in_instrument(self, temp_prom_dir, mock_app):
        """Test error handling in instrument function"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = Exception("Directory creation failed")

            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        # Mock settings
                        settings = Mock()
                        settings.tracing.enabled = True
                        settings.tracing.prometheus_multiproc_dir = temp_prom_dir
                        settings.tracing.otlp_grpc_endpoint = ""
                        settings.tracing.otlp_http_endpoint = ""

                        # Call instrument function - should raise the exception
                        with pytest.raises(
                            Exception, match="Directory creation failed"
                        ):
                            instrument(mock_app, settings)

    def test_instrument_function_returns_otel_metrics(
        self, temp_prom_dir, mock_settings, mock_app
    ):
        """Test that instrument function returns OTel metrics instance"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        with patch(
                            "nvidia_rag.utils.observability.tracing.OtelMetrics"
                        ) as mock_otel_metrics:
                            # Mock OTel metrics instance
                            mock_otel_instance = Mock()
                            mock_otel_metrics.return_value = mock_otel_instance

                            # Call instrument function
                            result = instrument(mock_app, mock_settings)

                            # Verify OTel metrics was created and returned
                            mock_otel_metrics.assert_called_once()
                            assert result == mock_otel_instance

    def test_instrument_function_with_otlp_endpoint(self, temp_prom_dir, mock_app):
        """Test instrument function with OTLP endpoint configured"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        with patch(
                            "nvidia_rag.utils.observability.tracing.OTLPMetricExporter"
                        ) as mock_exporter:
                            with patch(
                                "nvidia_rag.utils.observability.tracing.OtelMetrics"
                            ) as mock_otel_metrics:
                                # Mock settings with OTLP endpoint
                                settings = Mock()
                                settings.tracing.enabled = True
                                settings.tracing.prometheus_multiproc_dir = (
                                    temp_prom_dir
                                )
                                settings.tracing.otlp_grpc_endpoint = (
                                    "http://localhost:4317"
                                )
                                settings.tracing.otlp_http_endpoint = ""

                                # Mock OTel metrics instance
                                mock_otel_instance = Mock()
                                mock_otel_metrics.return_value = mock_otel_instance

                                # Call instrument function
                                result = instrument(mock_app, settings)

                                # Verify OTLP exporter was called
                                mock_exporter.assert_called_once()
                                # Verify OTel metrics was created and returned
                                mock_otel_metrics.assert_called_once()
                                assert result == mock_otel_instance

    def test_instrument_function_without_otlp_endpoint(
        self, temp_prom_dir, mock_settings, mock_app
    ):
        """Test instrument function without OTLP endpoint configured"""
        with patch("nvidia_rag.utils.observability.tracing.os.makedirs"):
            with patch("nvidia_rag.utils.observability.tracing.PrometheusMeterProvider"):
                with patch("nvidia_rag.utils.observability.tracing.metrics.set_meter_provider"):
                    with patch("nvidia_rag.utils.observability.tracing.FastAPIInstrumentor"):
                        with patch(
                            "nvidia_rag.utils.observability.tracing.OTLPMetricExporter"
                        ) as mock_exporter:
                            with patch(
                                "nvidia_rag.utils.observability.tracing.OtelMetrics"
                            ) as mock_otel_metrics:
                                # Mock OTel metrics instance
                                mock_otel_instance = Mock()
                                mock_otel_metrics.return_value = mock_otel_instance

                                # Call instrument function
                                result = instrument(mock_app, mock_settings)

                                # Verify OTLP exporter was NOT called
                                mock_exporter.assert_not_called()
                                # Verify OTel metrics was created and returned
                                mock_otel_metrics.assert_called_once()
                                assert result == mock_otel_instance

    def test_instrument_function_with_disabled_tracing(self, temp_prom_dir, mock_app):
        """Test instrument function when tracing is disabled"""
        # Mock settings with tracing disabled
        settings = Mock()
        settings.tracing.enabled = False

        # Call instrument function
        result = instrument(mock_app, settings)

        # Verify function returns None when tracing is disabled
        assert result is None

    def test_build_span_filters_only_for_ingestor(self):
        """Ensure span filters are returned only for the ingestor service."""
        ingestor_filters = tracing._build_span_filters("ingestor")
        assert len(ingestor_filters) == 1

        other_filters = tracing._build_span_filters("rag")
        assert other_filters == []

    def test_filtering_span_exporter_drops_noisy_http_receive_spans(self):
        """Verify noisy FastAPI http receive spans are filtered before export."""

        def _mock_span(name: str, service: str = "ingestor"):
            span = MagicMock()
            span.name = name
            span.resource = Resource(attributes={"service.name": service})
            span.attributes = {"asgi.event.type": "http.request"}
            span.instrumentation_scope = SimpleNamespace(
                name="opentelemetry.instrumentation.fastapi"
            )
            return span

        downstream_exporter = MagicMock()
        filtering_exporter = tracing.FilteringSpanExporter(
            downstream_exporter, skip_predicates=tracing._build_span_filters("ingestor")
        )

        noisy_span = _mock_span("post /documents http receive")
        clean_span = _mock_span("ingestor.main.process")

        filtering_exporter.export([noisy_span, clean_span])
        downstream_exporter.export.assert_called_once_with([clean_span])

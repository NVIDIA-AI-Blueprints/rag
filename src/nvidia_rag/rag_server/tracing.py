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

"""Module to enable Observervability and Tracing instrumentation
1. instrument(): Instrument the FastAPI app with OpenTelemetry.
"""

import logging
import os
from typing import Any

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.milvus import MilvusInstrumentor
from opentelemetry.processor.baggage import ALLOW_ALL_BAGGAGE_KEYS, BaggageSpanProcessor
from opentelemetry.sdk.extension.prometheus_multiprocess import PrometheusMeterProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from observability.langchain_instrumentor import LangchainInstrumentor
from observability.otel_metrics import OtelMetrics

logger = logging.getLogger(__name__)


def _fastapi_server_request_hook(span: Span, scope: dict[str, Any]):
    """Utility function"""

    if span and span.is_recording():
        for k, v in scope.get("headers"):
            if k == b"x-benchmark-id":
                span.set_attribute("x-benchmark-id", v)


def instrument(app: FastAPI, settings):
    """Function to enable OTLP export and instrumentation for traces and metrics"""

    otel_metrics = None
    if settings.tracing.enabled:
        resource = Resource(attributes={"service.name": "rag"})
        # Always set up Prometheus multi-process directory
        prom_dir = settings.tracing.prometheus_multiproc_dir
        os.makedirs(prom_dir, exist_ok=True)

        # Use PrometheusMeterProvider for multi-process support
        # This ensures /metrics endpoint works with multi-worker aggregation
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
        metrics.set_meter_provider(PrometheusMeterProvider())

        # Set up OTLP export separately if endpoint is configured
        if settings.tracing.otlp_grpc_endpoint != "":
            # Create a separate OTLP provider for metrics export
            exporter_grpc = OTLPMetricExporter(
                endpoint=settings.tracing.otlp_grpc_endpoint, insecure=True
            )
            otlp_reader = PeriodicExportingMetricReader(exporter_grpc)
            otlp_provider = MeterProvider(
                resource=resource, metric_readers=[otlp_reader]
            )
            # Store OTLP provider for later use
            logging.info(
                f"OTLP metrics export configured for: {settings.tracing.otlp_grpc_endpoint}"
            )
        else:
            otlp_provider = None
        otel_metrics = OtelMetrics(service_name="rag")

        # Set up OTLP meter if available
        if otlp_provider:
            otel_metrics.setup_otlp_meter(otlp_provider)

        # Oberservability Tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        exporter_http = None
        if settings.tracing.otlp_http_endpoint != "":
            logger.debug(
                f"configuring otlp http exporter {settings.tracing.otlp_http_endpoint}"
            )
            exporter_http = OTLPSpanExporter(
                endpoint=settings.tracing.otlp_http_endpoint
            )
        else:
            logger.debug(f"configuring console exporter {settings.tracing}")
            exporter_http = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(exporter_http)
        trace.get_tracer_provider().add_span_processor(
            BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS)
        )
        trace.get_tracer_provider().add_span_processor(span_processor)
        LangchainInstrumentor().instrument(
            tracer_provider=trace.get_tracer_provider(), metrics=otel_metrics
        )
        MilvusInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())

        FastAPIInstrumentor().instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            server_request_hook=_fastapi_server_request_hook,
            meter_provider=otlp_provider,
        )
    return otel_metrics

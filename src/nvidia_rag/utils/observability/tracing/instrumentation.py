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

"""Module to enable Observability and Tracing instrumentation
1. instrument(): Instrument the FastAPI app with OpenTelemetry.
"""

from collections.abc import Callable, Sequence
from functools import partial
import logging
import os

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.milvus import MilvusInstrumentor
from opentelemetry.processor.baggage import ALLOW_ALL_BAGGAGE_KEYS, BaggageSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)

from nvidia_rag.utils.observability.langchain_instrumentor import LangchainInstrumentor
from nvidia_rag.utils.observability.otel_metrics import OtelMetrics

logger = logging.getLogger(__name__)


class InvalidKindError(ValueError):
    """Exception raised when an invalid span kind is provided."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"Invalid kind: {kind}")

try:
    from opentelemetry.sdk.extension.prometheus_multiprocess import (
        PrometheusMeterProvider,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    PrometheusMeterProvider = None  # type: ignore[assignment]
    logger.debug(
        "Prometheus multiprocess metrics support is not available. "
        "Install 'opentelemetry-sdk-extension-prometheus' to enable it."
    )


_NOISY_HTTP_RECEIVE_SUBSTRING = "post /documents http receive"
_NOISY_HTTP_SEND_SUBSTRING = "post /generate http send"


_METRICS_ENDPOINT_PATH = "/metrics"


class FilteringSpanExporter(SpanExporter):
    """Span exporter wrapper that filters spans before delegating to the actual exporter."""

    def __init__(
        self,
        exporter: SpanExporter,
        skip_predicates: Sequence[Callable[[ReadableSpan], bool]] | None = None,
    ) -> None:
        self._exporter = exporter
        self._skip_predicates = tuple(skip_predicates or [])

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not self._skip_predicates:
            return self._exporter.export(spans)

        filtered_spans = [
            span
            for span in spans
            if not any(predicate(span) for predicate in self._skip_predicates)
        ]
        if not filtered_spans:
            return SpanExportResult.SUCCESS

        return self._exporter.export(filtered_spans)

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis=timeout_millis)


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _is_fastapi_http_noise(span: ReadableSpan, kind: str = "receive") -> bool:
    """True when the span matches the noisy FastAPI request receive span for POST /documents."""

    normalized_span_name = _normalize_text(span.name)
    if kind == "receive":
        if _NOISY_HTTP_RECEIVE_SUBSTRING not in normalized_span_name:
            return False
    elif kind == "send":
        if _NOISY_HTTP_SEND_SUBSTRING not in normalized_span_name:
            return False
    else:
        raise InvalidKindError(kind)

    # Only skip for FastAPI spans to avoid suppressing similar spans elsewhere.
    instrumentation_scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    instrumentation_name = _normalize_text(
        getattr(instrumentation_scope, "name", "") if instrumentation_scope else ""
    )
    if "fastapi" not in instrumentation_name:
        return False

    asgi_event_type = span.attributes.get("asgi.event.type")
    if isinstance(asgi_event_type, str):
        asgi_event_type = asgi_event_type.lower()
        if kind == "receive":
            # Receive-side noise should be limited to HTTP request events.
            if asgi_event_type != "http.request":
                return False
        elif kind == "send":
            # Send-side noise corresponds to HTTP response events; FastAPI commonly
            # uses "http.response.start" and "http.response.body".
            if asgi_event_type not in ("http.response.start", "http.response.body"):
                return False

    return True


def _is_metrics_endpoint_span(span: ReadableSpan) -> bool:
    """True when the span corresponds to the FastAPI /metrics endpoint.

    The metrics endpoint is typically scraped frequently and does not add much
    value in tracing, so we filter it out to reduce noise.
    """

    # Prefer explicit route/target attributes if present
    route = span.attributes.get("http.route") or span.attributes.get("http.target")
    if isinstance(route, str):
        normalized_route = _normalize_text(route)
        # Match exact "/metrics" or any path that ends with "/metrics"
        if normalized_route == _METRICS_ENDPOINT_PATH or normalized_route.endswith(
            _METRICS_ENDPOINT_PATH
        ):
            return True

    # Fallback: many instrumentations name spans like "GET /metrics"
    normalized_span_name = _normalize_text(span.name)
    if "get /metrics" in normalized_span_name or normalized_span_name.endswith(
        _METRICS_ENDPOINT_PATH
    ):
        return True

    return False


def _build_span_filters(service_name: str) -> list[Callable[[ReadableSpan], bool]]:
    """Return predicates for spans that should be filtered for the provided service."""

    filters: list[Callable[[ReadableSpan], bool]] = []
    # Filter noisy FastAPI HTTP receive spans for the ingestor and send spans for
    # the rag service. We use partials so that each predicate only requires the span.
    if _normalize_text(service_name) == "ingestor":
        filters.append(partial(_is_fastapi_http_noise, kind="receive"))
    elif _normalize_text(service_name) == "rag":
        filters.append(partial(_is_fastapi_http_noise, kind="send"))
    # Always filter noisy /metrics endpoint spans for all services to avoid
    # cluttering traces with frequent scrape requests that add little value.
    filters.append(_is_metrics_endpoint_span)
    return filters


def instrument(app: FastAPI, settings, service_name: str = "rag"):
    """Function to enable OTLP export and instrumentation for traces and metrics"""

    otel_metrics = None
    if settings.tracing.enabled:
        resource = Resource(attributes={"service.name": service_name})
        # Always set up Prometheus multi-process directory
        prom_dir = settings.tracing.prometheus_multiproc_dir
        os.makedirs(prom_dir, exist_ok=True)

        # meter_provider is used for our custom application metrics (via OtelMetrics)
        # asgi_meter_provider is used for FastAPI/ASGI HTTP server metrics.
        # We intentionally decouple these when Prometheus multiprocess is enabled
        # because the PrometheusMeterProvider currently crashes when creating
        # certain HTTP server instruments (e.g. http.server.active_requests)
        # in multi-process setups.
        meter_provider = None
        asgi_meter_provider = None
        if PrometheusMeterProvider is not None:
            # Use PrometheusMeterProvider for multi-process support
            # This ensures /metrics endpoint works with multi-worker aggregation
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir

            # Check if PrometheusMeterProvider is already set in the global metrics registry
            current_provider = metrics.get_meter_provider()
            if isinstance(PrometheusMeterProvider, type) and isinstance(
                current_provider, PrometheusMeterProvider
            ):
                meter_provider = current_provider
                logger.debug("Reusing existing PrometheusMeterProvider")
            else:
                meter_provider = PrometheusMeterProvider()
                metrics.set_meter_provider(meter_provider)
                logger.debug("Initialized new PrometheusMeterProvider")

            # IMPORTANT:
            # Use a separate, plain MeterProvider for FastAPI HTTP metrics
            # to avoid crashes in Prometheus multiprocess provider when
            # creating built-in HTTP instruments like http.server.active_requests.
            asgi_meter_provider = MeterProvider(resource=resource)
        else:
            logger.warning(
                "opentelemetry-sdk-extension-prometheus is not installed; "
                "falling back to the default MeterProvider without multi-process "
                "aggregation for the /metrics endpoint."
            )
            meter_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(meter_provider)
            # When Prometheus is not available, FastAPI can safely share the same provider
            asgi_meter_provider = meter_provider

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
            logger.info(
                "OTLP metrics export configured for: %s",
                settings.tracing.otlp_grpc_endpoint,
            )
        else:
            otlp_provider = None
        otel_metrics = OtelMetrics(service_name=service_name)

        # Set up OTLP meter if available
        if otlp_provider:
            otel_metrics.setup_otlp_meter(otlp_provider)

        # Observability Tracing
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
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

        span_filters = _build_span_filters(service_name)
        if span_filters:
            exporter_http = FilteringSpanExporter(
                exporter_http, skip_predicates=span_filters
            )

        span_processor = BatchSpanProcessor(exporter_http)
        trace.get_tracer_provider().add_span_processor(
            BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS)
        )
        trace.get_tracer_provider().add_span_processor(span_processor)

        # Instrument FastAPI:
        # - Traces use the configured tracer provider.
        # - HTTP metrics use asgi_meter_provider which is decoupled from the
        #   Prometheus multiprocess provider to prevent crashes when /metrics
        #   or other endpoints are accessed under multi-worker setups.
        FastAPIInstrumentor().instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            meter_provider=asgi_meter_provider,
        )
        LangchainInstrumentor().instrument(
            tracer_provider=trace.get_tracer_provider(), metrics=otel_metrics
        )
        MilvusInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        logger.info(
            "FastAPI automatic HTTP instrumentation enabled for %s service",
            service_name,
        )
    return otel_metrics

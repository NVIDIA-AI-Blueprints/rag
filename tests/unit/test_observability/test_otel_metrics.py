# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

import nvidia_rag.utils.observability.otel_metrics as om
from nvidia_rag.utils.observability.otel_metrics import OtelMetrics, get_otel_metrics


class FakeHistogram:
    def __init__(self, name: str):
        self.name = name
        self.records: list[float] = []

    def record(self, value: float):
        self.records.append(value)


class FakeGauge:
    def __init__(self, name: str):
        self.name = name
        self.value = None

    def set(self, value):
        self.value = value


class FakeCounter:
    def __init__(self, name: str):
        self.name = name
        self.add_calls: list[tuple[int, dict]] = []

    def add(self, amount: int, attrs: dict | None = None):
        self.add_calls.append((amount, attrs or {}))


class FakeMeter:
    def __init__(self):
        self.histograms: dict[str, FakeHistogram] = {}
        self.gauges: dict[str, FakeGauge] = {}
        self.counters: dict[str, FakeCounter] = {}

    def create_histogram(self, name: str, description: str = ""):
        hist = self.histograms.get(name)
        if hist is None:
            hist = FakeHistogram(name)
            self.histograms[name] = hist
        return hist

    def create_gauge(self, name: str, description: str = ""):
        gauge = self.gauges.get(name)
        if gauge is None:
            gauge = FakeGauge(name)
            self.gauges[name] = gauge
        return gauge

    def create_counter(self, name: str, description: str = ""):
        counter = self.counters.get(name)
        if counter is None:
            counter = FakeCounter(name)
            self.counters[name] = counter
        return counter


@pytest.fixture()
def fake_meter(monkeypatch):
    meter = FakeMeter()

    # Monkeypatch opentelemetry.metrics in the imported module to use our fake meter
    fake_metrics_module = types.SimpleNamespace(get_meter=lambda service_name: meter)
    monkeypatch.setattr(om, "metrics", fake_metrics_module, raising=True)
    return meter


def test_otel_metrics_setup_and_updates(fake_meter):
    m = OtelMetrics(service_name="rag")

    # Instruments created
    assert "api_requests_total" in fake_meter.counters
    assert "input_tokens" in fake_meter.gauges
    assert "output_tokens" in fake_meter.gauges
    assert "total_tokens" in fake_meter.gauges
    assert set(m.latency_hists.keys()) == {
        "rag_ttft_ms",
        "llm_ttft_ms",
        "context_reranker_time_ms",
        "retrieval_time_ms",
        "llm_generation_time_ms",
    }

    # API requests counter update
    m.update_api_requests(method="GET", endpoint="/v1/health")
    assert fake_meter.counters["api_requests_total"].add_calls[-1] == (
        1,
        {"method": "GET", "endpoint": "/v1/health"},
    )

    # Token gauges and histogram
    m.update_llm_tokens(input_t=3, output_t=7)
    assert fake_meter.gauges["input_tokens"].value == 3
    assert fake_meter.gauges["output_tokens"].value == 7
    assert fake_meter.gauges["total_tokens"].value == 10
    assert fake_meter.histograms["token_usage_distribution"].records[-1] == 10

    # Latency metrics
    m.update_latency_metrics({"rag_ttft_ms": 12.5, "llm_generation_time_ms": 40.0})
    assert fake_meter.histograms["rag_ttft_ms"].records[-1] == 12.5
    assert fake_meter.histograms["llm_generation_time_ms"].records[-1] == 40.0


def test_otel_metrics_reinit_guard(fake_meter):
    m1 = OtelMetrics(service_name="rag")

    # Capture instrument identities
    hist_ids_before = {k: id(v) for k, v in m1.latency_hists.items()}
    counter_id_before = id(fake_meter.counters["api_requests_total"])

    # Re-run setup to test guard (should not recreate instruments)
    m1._setup_metrics()

    hist_ids_after = {k: id(v) for k, v in m1.latency_hists.items()}
    counter_id_after = id(fake_meter.counters["api_requests_total"])

    assert hist_ids_after == hist_ids_before
    assert counter_id_after == counter_id_before


def test_setup_otlp_meter_none_provider(fake_meter):
    """Test that setup_otlp_meter handles None provider gracefully."""
    m = OtelMetrics(service_name="rag")
    m.setup_otlp_meter(None)

    assert not hasattr(m, "_otlp_api_request_counter")


def test_setup_otlp_meter_success(fake_meter):
    """Test successful OTLP meter setup."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            return fake_meter

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    assert hasattr(m, "_otlp_api_request_counter")
    assert hasattr(m, "_otlp_input_token_gauge")
    assert hasattr(m, "_otlp_latency_hists")


def test_setup_otlp_meter_exception_handling(fake_meter):
    """Test that setup_otlp_meter handles exceptions gracefully."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            raise Exception("OTLP connection failed")

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    assert m._otlp_meter is None


def test_update_api_requests_with_otlp(fake_meter):
    """Test update_api_requests updates OTLP meter when available."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            return fake_meter

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    m.update_api_requests(method="POST", endpoint="/v1/generate")

    assert len(fake_meter.counters["api_requests_total"].add_calls) == 2


def test_update_llm_tokens_with_otlp(fake_meter):
    """Test update_llm_tokens updates OTLP meter when available."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            return fake_meter

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    m.update_llm_tokens(input_t=10, output_t=20)

    assert fake_meter.gauges["input_tokens"].value == 10
    assert fake_meter.gauges["output_tokens"].value == 20
    assert fake_meter.gauges["total_tokens"].value == 30


def test_update_avg_words_per_chunk_with_otlp(fake_meter):
    """Test update_avg_words_per_chunk updates OTLP meter when available."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            return fake_meter

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    m.update_avg_words_per_chunk(avg_words_per_chunk=50)

    assert fake_meter.gauges["avg_words_per_chunk"].value == 50


def test_update_latency_metrics_with_otlp(fake_meter):
    """Test update_latency_metrics updates OTLP meter when available."""
    m = OtelMetrics(service_name="rag")

    class MockOTLPProvider:
        def get_meter(self, service_name):
            return fake_meter

    provider = MockOTLPProvider()
    m.setup_otlp_meter(provider)

    m.update_latency_metrics({"retrieval_time_ms": 15.5})

    assert fake_meter.histograms["retrieval_time_ms"].records[-1] == 15.5


def test_update_api_requests_without_method_endpoint(fake_meter):
    """Test update_api_requests does nothing when method/endpoint are None."""
    m = OtelMetrics(service_name="rag")
    initial_calls = len(fake_meter.counters["api_requests_total"].add_calls)

    m.update_api_requests(method=None, endpoint=None)
    m.update_api_requests(method="GET", endpoint=None)
    m.update_api_requests(method=None, endpoint="/v1/health")

    assert len(fake_meter.counters["api_requests_total"].add_calls) == initial_calls


def test_update_llm_tokens_partial_params(fake_meter):
    """Test update_llm_tokens does nothing when params are None."""
    m = OtelMetrics(service_name="rag")
    initial_input = fake_meter.gauges["input_tokens"].value

    m.update_llm_tokens(input_t=None, output_t=10)
    m.update_llm_tokens(input_t=10, output_t=None)
    m.update_llm_tokens(input_t=None, output_t=None)

    assert fake_meter.gauges["input_tokens"].value == initial_input


def test_update_avg_words_per_chunk_none(fake_meter):
    """Test update_avg_words_per_chunk does nothing when value is None."""
    m = OtelMetrics(service_name="rag")
    initial_value = fake_meter.gauges["avg_words_per_chunk"].value

    m.update_avg_words_per_chunk(avg_words_per_chunk=None)

    assert fake_meter.gauges["avg_words_per_chunk"].value == initial_value


def test_update_latency_metrics_missing_hist(fake_meter):
    """Test update_latency_metrics handles missing histogram gracefully."""
    m = OtelMetrics(service_name="rag")

    m.update_latency_metrics({"unknown_metric": 10.0})

    assert "unknown_metric" not in fake_meter.histograms


def test_update_latency_metrics_none_value(fake_meter):
    """Test update_latency_metrics handles None values gracefully."""
    m = OtelMetrics(service_name="rag")
    initial_records = len(fake_meter.histograms["rag_ttft_ms"].records)

    m.update_latency_metrics({"rag_ttft_ms": None})

    assert len(fake_meter.histograms["rag_ttft_ms"].records) == initial_records


def test_get_otel_metrics_singleton(fake_meter):
    """Test that get_otel_metrics returns singleton instance."""
    m1 = get_otel_metrics("rag")
    m2 = get_otel_metrics("rag")

    assert m1 is m2
    assert isinstance(m1, OtelMetrics)


def test_get_otel_metrics_different_service_names(fake_meter):
    """Test that get_otel_metrics returns same singleton regardless of service name."""
    m1 = get_otel_metrics("rag")
    m2 = get_otel_metrics("ingestor")

    assert m1 is m2

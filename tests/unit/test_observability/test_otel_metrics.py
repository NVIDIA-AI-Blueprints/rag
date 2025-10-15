# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


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
    import observability.otel_metrics as om

    fake_metrics_module = types.SimpleNamespace(get_meter=lambda service_name: meter)
    monkeypatch.setattr(om, "metrics", fake_metrics_module, raising=True)
    return meter


def test_otel_metrics_setup_and_updates(fake_meter):
    from observability.otel_metrics import OtelMetrics

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
    from observability.otel_metrics import OtelMetrics

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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from pathlib import Path

from nvidia_rag.rag_server.agentic_rag.tracing import (
    LLMCallRecord,
    NodeTiming,
    QueryTrace,
    RetrievalRecord,
)
from nvidia_rag.utils.observability.agentic_metrics import AgenticRAGMetrics


class FakeHistogram:
    def __init__(self, name: str):
        self.name = name
        self.records: list[tuple[float, dict]] = []

    def record(self, value: float, attrs: dict | None = None):
        self.records.append((value, attrs or {}))


class FakeCounter:
    def __init__(self, name: str):
        self.name = name
        self.add_calls: list[tuple[int, dict]] = []

    def add(self, amount: int, attrs: dict | None = None):
        self.add_calls.append((amount, attrs or {}))


class FakeMeter:
    def __init__(self):
        self.histograms: dict[str, FakeHistogram] = {}
        self.counters: dict[str, FakeCounter] = {}

    def create_histogram(self, name: str, description: str = ""):
        hist = self.histograms.get(name)
        if hist is None:
            hist = FakeHistogram(name)
            self.histograms[name] = hist
        return hist

    def create_counter(self, name: str, description: str = ""):
        counter = self.counters.get(name)
        if counter is None:
            counter = FakeCounter(name)
            self.counters[name] = counter
        return counter


def _build_trace() -> QueryTrace:
    trace = QueryTrace(query_text="what changed?")
    trace.llm_calls.extend(
        [
            LLMCallRecord("Planner (phase 1)", 10, 20, 30.0),
            LLMCallRecord("Task t1 answer (attempt 1)", 5, 15, 25.0),
            LLMCallRecord("Task t1 seed gen (attempt 2)", 3, 4, 5.0),
            LLMCallRecord("Verification", 7, 8, 9.0),
            LLMCallRecord("Answer synthesis", 11, 12, 13.0),
        ]
    )
    trace.node_timings.extend(
        [
            NodeTiming("initial_retrieval", 12.0),
            NodeTiming("plan (phase 1)", 34.0),
            NodeTiming("execute (answer)", 56.0),
            NodeTiming("verify_execute", 78.0),
        ]
    )
    trace.retrieval_calls.extend(
        [
            RetrievalRecord("initial_retrieval", chunks=3, duration_ms=10.0),
            RetrievalRecord("execute", chunks=2, duration_ms=20.0),
            RetrievalRecord("verify_execute", chunks=0, duration_ms=5.0, error=True),
        ]
    )
    trace.plan_summary = {"scope_only": False, "task_count": 2, "scope_rounds": 1}
    trace.task_results_summary = {
        "t1": {"status": "answered", "attempts": 1},
        "t2": {"status": "no_data", "attempts": 2},
    }
    trace.verification_outcome = {
        "passed": False,
        "issues": ["missing figure"],
        "follow_up_tasks": 1,
    }
    trace.finalize()
    return trace


def test_agentic_metrics_records_query_trace():
    meter = FakeMeter()
    metrics = AgenticRAGMetrics("rag", meter)

    metrics.record_query_trace(
        _build_trace(), status="success", verification_enabled=True
    )

    assert "agentic_requests_total" in meter.counters
    assert meter.counters["agentic_requests_total"].add_calls[-1] == (
        1,
        {"status": "success", "verification_enabled": "true"},
    )
    assert meter.histograms["agentic_request_duration_ms"].records
    assert any(
        attrs == {"stage": "plan", "status": "success"}
        for _, attrs in meter.histograms["agentic_stage_duration_ms"].records
    )
    assert (1, {"role": "planner", "status": "success"}) in meter.counters[
        "agentic_llm_calls_total"
    ].add_calls
    assert (10, {"role": "planner", "type": "input", "status": "success"}) in meter.counters[
        "agentic_llm_tokens_total"
    ].add_calls
    assert any(
        attrs == {"plan_type": "answer", "status": "success"}
        for _, attrs in meter.histograms["agentic_plan_tasks"].records
    )
    assert (1, {"stage": "verify_execute"}) in meter.counters[
        "agentic_errors_total"
    ].add_calls
    assert (1, {"result": "failed", "status": "success"}) in meter.counters[
        "agentic_verification_total"
    ].add_calls


def test_agentic_metrics_records_otlp_meter():
    meter = FakeMeter()
    otlp_meter = FakeMeter()
    metrics = AgenticRAGMetrics("rag", meter)
    metrics.setup_otlp_meter(otlp_meter)

    metrics.record_query_trace(_build_trace(), status="error", verification_enabled=False)

    assert meter.counters["agentic_requests_total"].add_calls[-1][1]["status"] == "error"
    assert otlp_meter.counters["agentic_requests_total"].add_calls[-1][1]["status"] == "error"


def test_agentic_metrics_handles_sparse_trace():
    meter = FakeMeter()
    metrics = AgenticRAGMetrics("rag", meter)
    trace = QueryTrace(query_text="empty")
    trace.finalize()

    metrics.record_query_trace(trace, status="unknown", verification_enabled=False)

    assert meter.counters["agentic_requests_total"].add_calls[-1][1] == {
        "status": "success",
        "verification_enabled": "false",
    }
    assert meter.counters["agentic_verification_total"].add_calls[-1] == (
        1,
        {"result": "disabled", "status": "success"},
    )


def test_agentic_dashboard_json_uses_agentic_metrics():
    dashboard = json.loads(
        Path("deploy/config/agentic-rag-metrics-dashboard.json").read_text()
    )
    expressions = []

    def collect_exprs(value):
        if isinstance(value, dict):
            expr = value.get("expr")
            if expr:
                expressions.append(expr)
            for child in value.values():
                collect_exprs(child)
        elif isinstance(value, list):
            for child in value:
                collect_exprs(child)

    collect_exprs(dashboard["panels"])

    assert expressions
    assert all("agentic_" in expr for expr in expressions)

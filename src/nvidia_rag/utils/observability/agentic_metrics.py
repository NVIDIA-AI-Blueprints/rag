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

"""OpenTelemetry metrics for the Agentic RAG pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvidia_rag.rag_server.agentic_rag.tracing import QueryTrace

logger = logging.getLogger(__name__)

_STAGES = (
    "initial_retrieval",
    "plan",
    "execute",
    "synthesize",
    "verify_execute",
    "verify",
)


class AgenticRAGMetrics:
    """Encapsulates Agentic RAG aggregate metrics.

    The exporter consumes the per-request ``QueryTrace`` object once the
    agentic graph finishes. This keeps metric emission centralized and avoids
    high-cardinality labels from query, task, or document content.
    """

    def __init__(self, service_name: str, meter: Any):
        self.service_name = service_name
        self._instruments = self._create_instruments(meter)
        self._otlp_instruments = None

    def _create_instruments(self, meter: Any) -> dict[str, Any]:
        return {
            "requests": meter.create_counter(
                "agentic_requests_total",
                description="Total Agentic RAG requests",
            ),
            "request_duration": meter.create_histogram(
                "agentic_request_duration_ms",
                description="Agentic RAG request duration in milliseconds",
            ),
            "stage_duration": meter.create_histogram(
                "agentic_stage_duration_ms",
                description="Agentic RAG graph stage duration in milliseconds",
            ),
            "llm_calls": meter.create_counter(
                "agentic_llm_calls_total",
                description="Agentic RAG LLM calls",
            ),
            "llm_call_duration": meter.create_histogram(
                "agentic_llm_call_duration_ms",
                description="Agentic RAG LLM call duration in milliseconds",
            ),
            "llm_tokens": meter.create_counter(
                "agentic_llm_tokens_total",
                description="Agentic RAG LLM token usage",
            ),
            "plan_tasks": meter.create_histogram(
                "agentic_plan_tasks",
                description="Agentic RAG planned task count per request",
            ),
            "scope_rounds": meter.create_histogram(
                "agentic_scope_rounds",
                description="Agentic RAG scope discovery rounds per request",
            ),
            "retrieval_calls": meter.create_counter(
                "agentic_retrieval_calls_total",
                description="Agentic RAG retrieval calls",
            ),
            "retrieved_chunks": meter.create_histogram(
                "agentic_retrieved_chunks",
                description="Agentic RAG retrieved chunks per retrieval call",
            ),
            "task_results": meter.create_counter(
                "agentic_task_results_total",
                description="Agentic RAG task outcomes",
            ),
            "task_attempts": meter.create_histogram(
                "agentic_task_attempts",
                description="Agentic RAG task attempts",
            ),
            "verification": meter.create_counter(
                "agentic_verification_total",
                description="Agentic RAG verification outcomes",
            ),
            "verification_followup_tasks": meter.create_histogram(
                "agentic_verification_followup_tasks",
                description="Agentic RAG verification follow-up task count",
            ),
            "errors": meter.create_counter(
                "agentic_errors_total",
                description="Agentic RAG errors",
            ),
        }

    def setup_otlp_meter(self, otlp_meter: Any) -> None:
        """Set up an OTLP meter for dual export."""
        try:
            self._otlp_instruments = self._create_instruments(otlp_meter)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to setup Agentic RAG OTLP metrics: %s", e)
            self._otlp_instruments = None

    def record_query_trace(
        self,
        trace: QueryTrace,
        *,
        status: str,
        verification_enabled: bool,
    ) -> None:
        """Record aggregate metrics for one completed Agentic RAG query."""
        try:
            status_label = _normalize_status(status)
            verification_label = _bool_label(verification_enabled)
            for instruments in self._instrument_sets():
                self._record_with_instruments(
                    instruments,
                    trace,
                    status=status_label,
                    verification_enabled=verification_label,
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to record Agentic RAG metrics: %s", e)

    def _instrument_sets(self) -> list[dict[str, Any]]:
        instrument_sets = [self._instruments]
        if self._otlp_instruments is not None:
            instrument_sets.append(self._otlp_instruments)
        return instrument_sets

    def _record_with_instruments(
        self,
        instruments: dict[str, Any],
        trace: QueryTrace,
        *,
        status: str,
        verification_enabled: str,
    ) -> None:
        request_attrs = {
            "status": status,
            "verification_enabled": verification_enabled,
        }
        instruments["requests"].add(1, request_attrs)
        instruments["request_duration"].record(trace.total_duration_ms, request_attrs)

        if status == "error":
            instruments["errors"].add(1, {"stage": "request"})

        for timing in getattr(trace, "node_timings", []):
            stage = _normalize_stage(getattr(timing, "node_name", "unknown"))
            instruments["stage_duration"].record(
                getattr(timing, "duration_ms", 0.0),
                {"stage": stage, "status": status},
            )

        for call in getattr(trace, "llm_calls", []):
            role = _normalize_role(getattr(call, "step_name", "unknown"))
            attrs = {"role": role, "status": status}
            instruments["llm_calls"].add(1, attrs)
            instruments["llm_call_duration"].record(
                getattr(call, "duration_ms", 0.0),
                attrs,
            )
            input_tokens = int(getattr(call, "input_tokens", 0) or 0)
            output_tokens = int(getattr(call, "output_tokens", 0) or 0)
            if input_tokens:
                instruments["llm_tokens"].add(
                    input_tokens, {"role": role, "type": "input", "status": status}
                )
            if output_tokens:
                instruments["llm_tokens"].add(
                    output_tokens, {"role": role, "type": "output", "status": status}
                )

        plan_summary = getattr(trace, "plan_summary", {}) or {}
        if plan_summary:
            plan_type = "scope" if plan_summary.get("scope_only") else "answer"
            instruments["plan_tasks"].record(
                int(plan_summary.get("task_count") or 0),
                {"plan_type": plan_type, "status": status},
            )
            instruments["scope_rounds"].record(
                int(plan_summary.get("scope_rounds") or 0),
                {"status": status},
            )

        for retrieval in getattr(trace, "retrieval_calls", []):
            stage = _normalize_stage(getattr(retrieval, "stage", "unknown"))
            attrs = {"stage": stage, "status": status}
            instruments["retrieval_calls"].add(1, attrs)
            instruments["retrieved_chunks"].record(
                int(getattr(retrieval, "chunks", 0) or 0),
                attrs,
            )
            if getattr(retrieval, "error", False):
                instruments["errors"].add(1, {"stage": stage})

        task_results = getattr(trace, "task_results_summary", {}) or {}
        for task in task_results.values():
            result = _normalize_result(task.get("status", "unknown"))
            attrs = {"result": result, "status": status}
            instruments["task_results"].add(1, attrs)
            instruments["task_attempts"].record(
                int(task.get("attempts") or 0),
                attrs,
            )

        verification = getattr(trace, "verification_outcome", {}) or {}
        if verification:
            result = "passed" if verification.get("passed") else "failed"
        elif verification_enabled == "true":
            result = "skipped"
        else:
            result = "disabled"
        instruments["verification"].add(1, {"result": result, "status": status})
        instruments["verification_followup_tasks"].record(
            int(verification.get("follow_up_tasks") or 0),
            {"status": status},
        )


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _normalize_status(status: str) -> str:
    return "error" if status == "error" else "success"


def _normalize_result(result: str) -> str:
    normalized = (result or "unknown").lower()
    if normalized in {"answered", "no_data", "error"}:
        return normalized
    return "unknown"


def _normalize_role(step_name: str) -> str:
    normalized = (step_name or "").lower()
    if "seed gen" in normalized or "seed" in normalized:
        return "seed_generator"
    if "task" in normalized and "answer" in normalized:
        return "task"
    if "planner" in normalized:
        return "planner"
    if "verification" in normalized or "verify" in normalized:
        return "verifier"
    if "synth" in normalized or "answer synthesis" in normalized:
        return "synthesis"
    return "unknown"


def _normalize_stage(stage_name: str) -> str:
    normalized = (stage_name or "").lower().strip()
    for stage in _STAGES:
        if normalized.startswith(stage):
            return stage
    if normalized.startswith("execute_scope"):
        return "execute"
    return "unknown"

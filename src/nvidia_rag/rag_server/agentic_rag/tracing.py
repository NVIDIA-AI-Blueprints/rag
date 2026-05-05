# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Per-query tracing and cross-query aggregate metrics.

Provides concurrency-safe per-query instrumentation via ``contextvars``.
Each query gets its own ``QueryTrace`` object that records LLM calls,
graph-node timings, and workflow metadata.  At the end of the query the
trace is serialized to a JSON file and fed into ``AgentMetrics`` for
rolling min/max/avg statistics.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_P = "[AGENTIC_RAG]"

# ---------------------------------------------------------------------------
# Context variable — one QueryTrace per asyncio task / query invocation
# ---------------------------------------------------------------------------

_current_trace: contextvars.ContextVar[QueryTrace | None] = contextvars.ContextVar(
    "_current_trace", default=None
)


def get_current_trace() -> QueryTrace | None:
    """Return the ``QueryTrace`` for the current asyncio task, or *None*."""
    return _current_trace.get()


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    """Single LLM invocation record."""

    step_name: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    attempt_number: int = 1


@dataclass
class NodeTiming:
    """Wall-clock duration for one graph node execution."""

    node_name: str
    duration_ms: float


# ---------------------------------------------------------------------------
# QueryTrace — per-query collector
# ---------------------------------------------------------------------------


@dataclass
class QueryTrace:
    """Collects all instrumentation data for a single query."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    start_wall: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: float | None = None

    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    node_timings: list[NodeTiming] = field(default_factory=list)

    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    plan_summary: dict[str, Any] = field(default_factory=dict)
    task_results_summary: dict[str, Any] = field(default_factory=dict)
    verification_outcome: dict[str, Any] = field(default_factory=dict)

    final_answer: str = ""
    error: str | None = None

    # ---- recording helpers ------------------------------------------------

    def record_llm_call(
        self,
        step_name: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        attempt_number: int = 1,
    ) -> None:
        self.llm_calls.append(
            LLMCallRecord(
                step_name=step_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                attempt_number=attempt_number,
            )
        )

    @contextmanager
    def trace_node(self, node_name: str):
        """Context manager that records wall-clock duration of a graph node."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.node_timings.append(NodeTiming(node_name=node_name, duration_ms=elapsed_ms))

    # ---- properties -------------------------------------------------------

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def total_duration_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    # ---- finalize / serialize --------------------------------------------

    def finalize(self) -> None:
        """Mark the trace as complete."""
        self.end_time = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "start_time_iso": self.start_wall.isoformat(),
            "total_duration_ms": round(self.total_duration_ms, 1),
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "error": self.error,
            "node_timings": [
                {"node": nt.node_name, "duration_ms": round(nt.duration_ms, 1)}
                for nt in self.node_timings
            ],
            "llm_calls": [
                {
                    "step": c.step_name,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "duration_ms": round(c.duration_ms, 1),
                    "attempt": c.attempt_number,
                }
                for c in self.llm_calls
            ],
            "retrieval_stats": self.retrieval_stats,
            "plan_summary": self.plan_summary,
            "task_results_summary": self.task_results_summary,
            "verification": self.verification_outcome,
            "final_answer": self.final_answer,
        }

    def write(self, output_dir: str) -> str | None:
        """Write the trace to ``output_dir/{query_id}.json``. Returns the path."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"{self.query_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.debug("%s Trace written: %s", _P, path)
            return path
        except Exception as ex:
            logger.warning("%s Failed to write trace file: %s", _P, ex)
            return None

    def one_line_summary(self) -> str:
        dur = self.total_duration_ms / 1000
        return (
            f"query_id={self.query_id[:8]} "
            f"llm_calls={self.total_llm_calls} "
            f"tokens_in={self.total_input_tokens} "
            f"tokens_out={self.total_output_tokens} "
            f"duration={dur:.1f}s" + (f" ERROR={self.error[:80]}" if self.error else "")
        )


# ---------------------------------------------------------------------------
# AgentMetrics — rolling aggregates across queries
# ---------------------------------------------------------------------------


class AgentMetrics:
    """Accumulates per-query summaries and computes min/max/avg statistics."""

    def __init__(self) -> None:
        self._query_summaries: list[dict[str, Any]] = []

    def update(self, trace: QueryTrace) -> None:
        self._query_summaries.append(
            {
                "query_id": trace.query_id,
                "llm_calls": trace.total_llm_calls,
                "duration_ms": trace.total_duration_ms,
                "input_tokens": trace.total_input_tokens,
                "output_tokens": trace.total_output_tokens,
                "error": trace.error is not None,
            }
        )

    def reset(self) -> None:
        self._query_summaries.clear()

    @property
    def total_queries(self) -> int:
        return len(self._query_summaries)

    def summary(self) -> dict[str, Any]:
        n = len(self._query_summaries)
        if n == 0:
            return {"total_queries": 0}

        def _stats(key: str) -> dict[str, float]:
            vals = [s[key] for s in self._query_summaries]
            return {
                "min": round(min(vals), 1),
                "max": round(max(vals), 1),
                "avg": round(sum(vals) / len(vals), 1),
            }

        errors = sum(1 for s in self._query_summaries if s["error"])
        return {
            "total_queries": n,
            "errors": errors,
            "llm_calls": _stats("llm_calls"),
            "duration_ms": _stats("duration_ms"),
            "input_tokens": _stats("input_tokens"),
            "output_tokens": _stats("output_tokens"),
        }

    def log_summary(self) -> None:
        s = self.summary()
        n = s["total_queries"]
        if n == 0:
            return
        lc = s["llm_calls"]
        dur = s["duration_ms"]
        ti = s["input_tokens"]
        to_ = s["output_tokens"]
        logger.info(
            "%s Metrics (%d queries): "
            "LLM calls min=%g max=%g avg=%.1f | "
            "Latency min=%.1fs max=%.1fs avg=%.1fs | "
            "Tokens in avg=%.0f out avg=%.0f | "
            "Errors: %d",
            _P,
            n,
            lc["min"],
            lc["max"],
            lc["avg"],
            dur["min"] / 1000,
            dur["max"] / 1000,
            dur["avg"] / 1000,
            ti["avg"],
            to_["avg"],
            s["errors"],
        )

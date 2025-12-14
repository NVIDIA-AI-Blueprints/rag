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
Reusable helpers for OpenTelemetry tracing.

This module provides:

1. get_tracer: Obtain or create a tracer with a default namespace.
2. traced_span: Context manager to wrap arbitrary blocks with a span.
3. trace_function: Decorator to automatically trace sync/async functions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

DEFAULT_TRACER_NAME = "nvidia_rag"
T = TypeVar("T")
AsyncFn = Callable[..., Awaitable[T]]
SyncFn = Callable[..., T]
Function = AsyncFn[T] | SyncFn[T]


def _normalize_text(value: str | None) -> str:
    """Normalize free-form span/label text for robust comparisons."""

    return " ".join((value or "").lower().split())


def get_tracer(name: str = DEFAULT_TRACER_NAME) -> Tracer:
    """Return an OpenTelemetry tracer."""

    return trace.get_tracer(name)


@contextmanager
def traced_span(
    name: str,
    tracer: Tracer | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Span:
    """Context manager that starts a span and records errors automatically."""

    tracer = tracer or get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - best effort
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def trace_function(name: str | None = None, tracer: Tracer | None = None):
    """Decorator that wraps sync/async functions in a traced span."""

    def decorator(func: Function[T]) -> Function[T]:
        span_name = name or func.__qualname__

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with traced_span(span_name, tracer) as _:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with traced_span(span_name, tracer):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def process_nv_ingest_traces(
    traces: Iterable[Mapping[str, Any]] | None,
    *,
    tracer: Tracer | None = None,
    span_namespace: str = "nv_ingest",
    collection_name: str | None = None,
    batch_number: int | None = None,
    reference_time_ns: int | None = None,
) -> None:
    """Convert NV-Ingest timing traces into OpenTelemetry sub-spans.

    This is a standalone version of the helper originally defined on
    ``NvidiaRAGIngestor`` so it can be reused by other services.

    Args:
        traces: Raw trace dictionaries returned by ``nv_ingest_client``.
        tracer: Optional tracer instance. When omitted, the default tracer is used.
        span_namespace: Prefix for generated span names.
        collection_name: Optional collection identifier for span attributes.
        batch_number: Optional batch identifier for span attributes.
        reference_time_ns: Absolute timestamp captured right before
            ``nv_ingest_ingestor.ingest()`` starts. Relative trace offsets (which
            are typically reported in nanoseconds since ingest start) are anchored
            to this value so the derived spans appear before downstream waits such
            as ``ingestor.nvingest.patched_wait_for_index``.
    """

    if not traces:
        return

    tracer = tracer or get_tracer()

    RELATIVE_TS_THRESHOLD_NS = 1_000_000_000_000_000  # ~16 minutes in ns

    def _normalize_timestamp(raw_value: int | float) -> int:
        """
        NV-Ingest reports per-stage timings either as absolute nanoseconds
        or as relative offsets since ingestion start. When the sampled value
        looks like a small relative offset, anchor it to the provided
        reference_time_ns (captured right before nv_ingest_ingestor.ingest()).
        """

        timestamp = int(raw_value)
        if reference_time_ns is not None and timestamp < RELATIVE_TS_THRESHOLD_NS:
            return reference_time_ns + timestamp
        return timestamp

    SKIPPED_TRACE_STAGE_NAMES = {
        "ingestor: post /documents http receive",
        "ingestor.post /documents http receive",
        "ingestor::post /documents http receive",
    }
    SKIPPED_TRACE_SUBSTRINGS = {"post /documents http receive"}

    def _compute_trace_duration(trace_record: Mapping[str, Any]) -> int:
        entry_times = [
            _normalize_timestamp(value)
            for key, value in trace_record.items()
            if key.startswith("trace::entry::") and isinstance(value, int | float)
        ]
        exit_times = [
            _normalize_timestamp(value)
            for key, value in trace_record.items()
            if key.startswith("trace::exit::") and isinstance(value, int | float)
        ]
        if not entry_times or not exit_times:
            return 0
        return max(exit_times) - min(entry_times)

    try:
        largest_trace = max(traces, key=_compute_trace_duration)
        largest_trace_duration = _compute_trace_duration(largest_trace)
        if largest_trace_duration <= 0:
            return

        entry_events = {
            key.removeprefix("trace::entry::"): _normalize_timestamp(value)
            for key, value in largest_trace.items()
            if key.startswith("trace::entry::") and isinstance(value, int | float)
        }
        exit_events = {
            key.removeprefix("trace::exit::"): _normalize_timestamp(value)
            for key, value in largest_trace.items()
            if key.startswith("trace::exit::") and isinstance(value, int | float)
        }

        if not entry_events or not exit_events:
            return

        parent_start = min(entry_events.values())
        parent_end = max(exit_events.values())
        if parent_end <= parent_start:
            return

        namespace = span_namespace.strip().replace(" ", "_") or "nv_ingest"
        parent_span_name = f"{namespace}.largest_trace"
        parent_attributes: dict[str, Any] = {
            "nv_ingest.stage_count": len(entry_events),
            "nv_ingest.duration_ns": largest_trace_duration,
        }
        if collection_name:
            parent_attributes["collection_name"] = collection_name
        if batch_number is not None:
            parent_attributes["batch_number"] = batch_number

        with tracer.start_as_current_span(
            parent_span_name,
            attributes=parent_attributes,
            start_time=parent_start,
            end_on_exit=False,
        ) as parent_span:
            try:
                for stage, start_time in sorted(
                    entry_events.items(), key=lambda item: item[1]
                ):
                    end_time = exit_events.get(stage)
                    if end_time is None or end_time <= start_time:
                        continue
                    stage_name = stage.replace("::", ".")
                    normalized_stage_name = _normalize_text(stage_name)
                    normalized_raw_stage = _normalize_text(stage)
                    if (
                        normalized_stage_name in SKIPPED_TRACE_STAGE_NAMES
                        or normalized_raw_stage in SKIPPED_TRACE_STAGE_NAMES
                        or any(
                            substring in normalized_stage_name
                            for substring in SKIPPED_TRACE_SUBSTRINGS
                        )
                        or any(
                            substring in normalized_raw_stage
                            for substring in SKIPPED_TRACE_SUBSTRINGS
                        )
                    ):
                        continue
                    child_attributes = {
                        "nv_ingest.stage": stage,
                        "nv_ingest.duration_ns": end_time - start_time,
                    }
                    if collection_name:
                        child_attributes["collection_name"] = collection_name
                    if batch_number is not None:
                        child_attributes["batch_number"] = batch_number

                    with tracer.start_as_current_span(
                        f"{namespace}.{stage_name}",
                        attributes=child_attributes,
                        start_time=start_time,
                        end_on_exit=False,
                    ) as child_span:
                        child_span.end(end_time=end_time)
            finally:
                parent_span.end(end_time=parent_end)
    except Exception as e:
        # Tracing should be best-effort and must not break ingestion flows.
        try:
            span = trace.get_tracer(__name__).get_current_span()
            if span:
                span.record_exception(e)  # pragma: no cover
        except Exception:
            # Ignore any errors during exception recording
            pass
        # We intentionally swallow errors here.
        return


def create_nv_ingest_trace_context(
    *,
    span_namespace: str,
    collection_name: str | None = None,
    batch_number: int | None = None,
) -> dict[str, Any]:
    """Convenience helper to build a standard NV-Ingest trace_context dict.

    This keeps call sites consistent and centralizes any future changes to the
    trace context schema in a single place.
    """

    context: dict[str, Any] = {"span_namespace": span_namespace}
    if collection_name is not None:
        context["collection_name"] = collection_name
    if batch_number is not None:
        context["batch_number"] = batch_number
    return context

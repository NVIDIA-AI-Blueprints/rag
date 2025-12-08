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
from contextlib import contextmanager
from functools import wraps
from typing import Any, Awaitable, Callable, Mapping, TypeVar, Union

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

DEFAULT_TRACER_NAME = "nvidia_rag"
T = TypeVar("T")
AsyncFn = Callable[..., Awaitable[T]]
SyncFn = Callable[..., T]
Function = Union[AsyncFn[T], SyncFn[T]]


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
                with traced_span(span_name, tracer) as span:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with traced_span(span_name, tracer):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


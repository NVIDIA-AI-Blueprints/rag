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

"""Agentic RAG pipeline runner.

Executes a pre-built ``AgenticRag`` graph for a single request and returns
a ``RAGResponse`` whose generator yields SSE-formatted chunks ready for
``server.py``'s ``StreamingResponse`` to forward to the HTTP client.

Two execution modes
-------------------
* ``enable_streaming=True`` (default) — graph runs via ``astream`` with combined
  ``stream_mode=["messages", "custom", "debug"]``.  ``streaming.translate_graph_stream``
  consumes the events live and emits per-chunk SSE strings: stage announcements,
  intermediate-stage reasoning + output tokens, final-stage reasoning, and the
  final answer (with citations + TTFT on the first final-answer chunk).
* ``enable_streaming=False`` — legacy path: the graph runs to completion via
  ``ainvoke`` and the full answer is wrapped in a single async-generator chunk
  for ``generate_answer_async``.  Preserves the pre-streaming contract for
  clients that don't want intermediate events.

ContextVars (``_current_trace``, ``_agentic_search_params``,
``_agentic_all_citations``) are bound for the lifetime of the request and
unbound in the ``finally`` block, making this function safe under concurrent
async invocations.

Usage (called from ``NvidiaRAG._agentic_chain``)::

    from nvidia_rag.rag_server.agentic_rag.runner import run_agentic_pipeline
    rag_response = await run_agentic_pipeline(
        agent=agent,
        graph=graph,
        query=rewritten_query,
        cfg=self.config.agentic_rag,
        search_params=AgenticSearchParams(...),
        enable_streaming=True,
        ...
    )
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nvidia_rag.rag_server.agentic_rag.agentic_rag import AgenticRAGGraphState
from nvidia_rag.rag_server.agentic_rag.builder import (
    AgenticSearchParams,
    _agentic_all_citations,
    _agentic_search_params,
)
from nvidia_rag.rag_server.agentic_rag.streaming import translate_graph_stream
from nvidia_rag.rag_server.agentic_rag.tracing import QueryTrace, _current_trace
from nvidia_rag.rag_server.response_generator import (
    Citations,
    ErrorCodeMapping,
    RAGResponse,
    SourceResult,
    generate_answer_async,
)

if TYPE_CHECKING:
    from nvidia_rag.rag_server.agentic_rag.agentic_rag import AgenticRag
    from nvidia_rag.utils.agentic_rag_config import AgenticRAGConfig
    from nvidia_rag.utils.observability.otel_metrics import OtelMetrics

logger = logging.getLogger(__name__)


async def _aiter_single(text: str):
    """Async generator that yields a single string — compatible with generate_answer_async."""
    yield text


def _collate_citations(
    acc: OrderedDict[str, list[SourceResult]],
) -> Citations | None:
    """Pick citations from the last stage that performed retrieval.

    The accumulator preserves insertion order, so the last value is always
    the most recently active stage — no stage names need to be hardcoded.
    """
    last_stage_results = next(reversed(acc.values()), [])
    if not last_stage_results:
        return None
    return Citations(
        total_results=len(last_stage_results),
        results=last_stage_results,
    )


async def run_agentic_pipeline(
    *,
    agent: AgenticRag,
    graph: Any,
    query: str,
    cfg: AgenticRAGConfig,
    search_params: AgenticSearchParams,
    enable_citations: bool = True,
    use_nrl_citations: bool = False,
    model: str = "",
    collection_names: list[str] | None = None,
    enable_streaming: bool = True,
    rag_start_time_sec: float | None = None,
    metrics: OtelMetrics | None = None,
) -> RAGResponse:
    """Run the compiled agentic RAG graph for one request and return a RAGResponse.

    Args:
        agent:              Compiled AgenticRag (used for metrics bookkeeping).
        graph:              Compiled LangGraph returned by AgenticRag.build_graph().
        query:              User query; already rewritten if query rewriting is enabled.
        cfg:                AgenticRAGConfig sub-config (recursion limit, debug stream, etc).
        search_params:      All per-request NvidiaRAG.search() parameters.
        enable_citations:   Forward to generate_answer_async (non-streaming path) /
                            controls citation collation (streaming path).
        use_nrl_citations:  Forward to generate_answer_async (NRL/LanceDB mode).
        model:              LLM model name forwarded to generate_answer_async.
        collection_names:   Used for citation metadata; first entry used as collection_name.
        enable_streaming:   True (default) → live event-stream translator.
                            False → legacy ainvoke + single-chunk path.
        rag_start_time_sec: Request-start timestamp for end-to-end latency metrics.
        metrics:            Optional OTel metrics client.
    """
    import opentelemetry.trace as otel_trace

    tracer = otel_trace.get_tracer(__name__)

    trace = QueryTrace(query_text=query)
    trace_token = _current_trace.set(trace)
    params_token = _agentic_search_params.set(search_params)
    citations_acc: OrderedDict[str, list[SourceResult]] = OrderedDict()
    citations_token = _agentic_all_citations.set(citations_acc)

    initial_state = AgenticRAGGraphState(user_query=query)
    collection_name = collection_names[0] if collection_names else ""

    if enable_streaming:
        return await _run_streaming(
            agent=agent,
            graph=graph,
            initial_state=initial_state,
            cfg=cfg,
            tracer=tracer,
            trace=trace,
            citations_acc=citations_acc,
            enable_citations=enable_citations,
            model=model,
            collection_name=collection_name,
            rag_start_time_sec=rag_start_time_sec,
            metrics=metrics,
            cleanup=lambda: (
                _current_trace.reset(trace_token),
                _agentic_search_params.reset(params_token),
                _agentic_all_citations.reset(citations_token),
            ),
        )

    # ------------------------------------------------------------------
    # Non-streaming path — preserves legacy behaviour.
    # ------------------------------------------------------------------
    try:
        with tracer.start_as_current_span("agentic_rag_query") as root_span:
            root_span.set_attribute("openinference.span.kind", "CHAIN")
            root_span.set_attribute("input.value", query)

            final_state = await graph.ainvoke(
                initial_state,
                config={"recursion_limit": cfg.recursion_limit},
            )

            answer = (
                final_state.get("final_answer", "No answer generated.")
                if isinstance(final_state, dict)
                else (final_state.final_answer or "No answer generated.")
            )

            trace.final_answer = answer
            trace.finalize()
            agent.metrics.update(trace)

            root_span.set_attribute("output.value", answer)
            logger.info("[AGENTIC_RAG] Query done: %s", trace.one_line_summary())
            agent.metrics.log_summary()

        collated_citations = _collate_citations(_agentic_all_citations.get())

        return RAGResponse(
            generate_answer_async(
                _aiter_single(answer),
                [],
                model=model,
                collection_name=collection_name,
                enable_citations=enable_citations,
                use_nrl_citations=use_nrl_citations,
                rag_start_time_sec=rag_start_time_sec,
                otel_metrics_client=metrics,
                citations=collated_citations,
            ),
            status_code=ErrorCodeMapping.SUCCESS,
        )

    except Exception as ex:
        logger.exception("Agentic RAG pipeline failed: %s", ex)
        trace.error = str(ex)[:500]
        trace.finalize()
        agent.metrics.update(trace)
        logger.info("[AGENTIC_RAG] Query failed: %s", trace.one_line_summary())
        agent.metrics.log_summary()

        error_msg = (
            "I encountered an error while processing your request via agentic RAG. "
            "Please check the server logs for details."
        )
        return RAGResponse(
            generate_answer_async(
                _aiter_single(error_msg),
                [],
                model=model,
                collection_name=collection_name,
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )

    finally:
        _current_trace.reset(trace_token)
        _agentic_search_params.reset(params_token)
        _agentic_all_citations.reset(citations_token)


async def _run_streaming(
    *,
    agent: AgenticRag,
    graph: Any,
    initial_state: AgenticRAGGraphState,
    cfg: AgenticRAGConfig,
    tracer: Any,
    trace: QueryTrace,
    citations_acc: OrderedDict[str, list[SourceResult]],
    enable_citations: bool,
    model: str,
    collection_name: str,
    rag_start_time_sec: float | None,
    metrics: OtelMetrics | None,
    cleanup: Any,
) -> RAGResponse:
    """Build a RAGResponse whose generator delegates to ``translate_graph_stream``.

    The OTel root span and ContextVar cleanup are owned by the wrapper async
    generator below so they live for the full lifetime of the stream — not
    just until this function returns.
    """
    debug_stream = bool(getattr(cfg, "enable_debug_stream", False))
    verification_enabled = bool(getattr(agent.verification_cfg, "enabled", False))

    def _build_citations_now() -> Citations | None:
        if not enable_citations:
            return None
        return _collate_citations(citations_acc)

    async def _on_complete(final_answer: str) -> None:
        trace.final_answer = final_answer
        trace.finalize()
        agent.metrics.update(trace)
        logger.info("[AGENTIC_RAG] Query done: %s", trace.one_line_summary())
        agent.metrics.log_summary()

    async def _stream() -> AsyncIterator[str]:
        try:
            with tracer.start_as_current_span("agentic_rag_query") as root_span:
                root_span.set_attribute("openinference.span.kind", "CHAIN")
                root_span.set_attribute("input.value", initial_state.user_query)
                async for sse in translate_graph_stream(
                    graph,
                    initial_state,
                    model=model,
                    recursion_limit=cfg.recursion_limit,
                    verification_enabled=verification_enabled,
                    citations_provider=_build_citations_now,
                    enable_debug_stream=debug_stream,
                    rag_start_time_sec=rag_start_time_sec,
                    on_complete=_on_complete,
                ):
                    yield sse
                root_span.set_attribute("output.value", trace.final_answer or "")
        except Exception as ex:
            logger.exception("Agentic RAG streaming pipeline failed: %s", ex)
            trace.error = str(ex)[:500]
            trace.finalize()
            agent.metrics.update(trace)
            logger.info("[AGENTIC_RAG] Query failed: %s", trace.one_line_summary())
            agent.metrics.log_summary()
            raise
        finally:
            try:
                cleanup()
            except Exception as cex:  # noqa: BLE001
                logger.warning("Streaming cleanup failed: %s", cex)
            # ``metrics`` is passed in for parity with the non-streaming path;
            # the translator already records TTFT/generation time on the
            # finishing chunk, but we still want OTel histogram updates.
            if metrics is not None:
                try:
                    payload = {
                        "rag_ttft_ms": getattr(trace, "rag_ttft_ms", None),
                        "llm_ttft_ms": getattr(trace, "llm_ttft_ms", None),
                    }
                    payload = {k: v for k, v in payload.items() if v is not None}
                    if payload:
                        metrics.update_latency_metrics(payload)
                except Exception as mex:  # noqa: BLE001
                    logger.debug("OTel latency update failed: %s", mex)

    # collection_name not currently surfaced in streaming chunks, but kept in
    # signature for parity with the non-streaming branch.
    _ = collection_name

    return RAGResponse(_stream(), status_code=ErrorCodeMapping.SUCCESS)

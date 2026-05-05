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

Executes a pre-built AgenticRag graph for a single request and wraps
the final answer in a streaming RAGResponse so downstream code (server.py
StreamingResponse) works without modification.

Usage (called from NvidiaRAG._agentic_chain):
    from nvidia_rag.rag_server.agentic_rag.runner import run_agentic_pipeline
    rag_response = await run_agentic_pipeline(
        agent=agent,
        graph=graph,
        query=rewritten_query,
        cfg=self.config.agentic_rag,
        search_params=AgenticSearchParams(...),
        ...
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nvidia_rag.rag_server.agentic_rag.agentic_rag import AgenticRAGGraphState
from nvidia_rag.rag_server.agentic_rag.builder import (
    AgenticSearchParams,
    _agentic_all_citations,
    _agentic_search_params,
)
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


async def run_agentic_pipeline(
    *,
    agent: "AgenticRag",
    graph: Any,
    query: str,
    cfg: "AgenticRAGConfig",
    search_params: AgenticSearchParams,
    enable_citations: bool = True,
    use_nrl_citations: bool = False,
    model: str = "",
    collection_names: list[str] | None = None,
    rag_start_time_sec: float | None = None,
    metrics: "OtelMetrics | None" = None,
) -> RAGResponse:
    """Run the compiled agentic RAG graph for one request and return a RAGResponse.

    The agent graph runs to completion (non-streaming) inside an OTel root span.
    The resulting ``final_answer`` string is wrapped in a single-chunk async
    generator so the rest of the rag-server pipeline (generate_answer_async →
    StreamingResponse) works unchanged.

    ``_current_trace`` and ``_agentic_search_params`` ContextVars are set for
    the duration of this call and reset in the ``finally`` block, making this
    function safe for concurrent async requests.

    Args:
        agent:              Compiled AgenticRag (used for metrics bookkeeping).
        graph:              Compiled LangGraph returned by AgenticRag.build_graph().
        query:              User query; already rewritten if query rewriting is enabled.
        cfg:                AgenticRAGConfig sub-config (recursion limit, JSON traces, …).
        search_params:      All per-request NvidiaRAG.search() parameters.
        enable_citations:   Forward to generate_answer_async.
        use_nrl_citations:  Forward to generate_answer_async (NRL/LanceDB mode).
        model:              LLM model name forwarded to generate_answer_async.
        collection_names:   Used for citation metadata; first entry used as collection_name.
        rag_start_time_sec: Request-start timestamp for end-to-end latency metrics.
        metrics:            Optional OTel metrics client.
    """
    import opentelemetry.trace as otel_trace

    tracer = otel_trace.get_tracer(__name__)

    trace = QueryTrace(query_text=query)
    trace_token = _current_trace.set(trace)
    params_token = _agentic_search_params.set(search_params)
    citations_acc: list[SourceResult] = []
    citations_token = _agentic_all_citations.set(citations_acc)

    try:
        with tracer.start_as_current_span("agentic_rag_query") as root_span:
            root_span.set_attribute("openinference.span.kind", "CHAIN")
            root_span.set_attribute("input.value", query)

            initial_state = AgenticRAGGraphState(user_query=query)
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
            if cfg.json_traces:
                trace.write(cfg.json_traces_output_dir)
            agent.metrics.update(trace)

            root_span.set_attribute("output.value", answer)
            logger.info("[AGENTIC_RAG] Query done: %s", trace.one_line_summary())
            agent.metrics.log_summary()

        all_source_results = _agentic_all_citations.get()
        collated_citations = (
            Citations(total_results=len(all_source_results), results=all_source_results)
            if all_source_results
            else None
        )

        return RAGResponse(
            generate_answer_async(
                _aiter_single(answer),
                [],
                model=model,
                collection_name=collection_names[0] if collection_names else "",
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
        if cfg.json_traces:
            trace.write(cfg.json_traces_output_dir)
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
                collection_name=collection_names[0] if collection_names else "",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )

    finally:
        _current_trace.reset(trace_token)
        _agentic_search_params.reset(params_token)
        _agentic_all_citations.reset(citations_token)

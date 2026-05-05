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

"""Agentic RAG builder: wires NvidiaRAG infrastructure into AgenticRag.

This module replaces the NAT register.py files from the standalone agentic_rag
example.  It provides two public helpers:

  make_retriever_fn       — creates an async retriever callable backed by
                            NvidiaRAG.search(), returning Citations.model_dump()
                            which is directly compatible with AgenticRag's
                            _extract_chunks() parser.

  build_agentic_rag_agent — constructs an AgenticRag and compiles its
                            LangGraph from the NvidiaRAGConfig.agentic_rag
                            sub-config.  Called lazily on the first agentic
                            request; the result is cached by NvidiaRAG.

Per-request search parameters
------------------------------
The agent and its compiled graph are built once and cached across requests.
All runtime search() parameters (collection_names, reranker_top_k, etc.) are
supplied per-request via the ``_agentic_search_params`` ContextVar.

Before each graph.ainvoke() call, the caller (NvidiaRAG._agentic_chain) sets
this ContextVar for the current asyncio task; the retriever_fn reads it at
every invocation.  The ContextVar is async-safe: concurrent requests each see
their own value.

Usage from _agentic_chain (to be wired in main.py):

    from nvidia_rag.rag_server.agentic_rag.builder import (
        AgenticSearchParams,
        _agentic_search_params,
    )

    params = AgenticSearchParams(
        collection_names=collection_names,
        reranker_top_k=reranker_top_k,
        ...
    )
    token = _agentic_search_params.set(params)
    try:
        await graph.ainvoke(...)
    finally:
        _agentic_search_params.reset(token)
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nvidia_rag.rag_server.agentic_rag.agentic_rag import AgenticRag
from nvidia_rag.rag_server.response_generator import SourceResult

if TYPE_CHECKING:
    from nvidia_rag.rag_server.main import NvidiaRAG
    from nvidia_rag.utils.agentic_rag_config import AgenticAnyLLMConfig
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PER-REQUEST SEARCH PARAMETERS
# =============================================================================


@dataclass
class AgenticSearchParams:
    """All runtime parameters forwarded to NvidiaRAG.search() per request.

    Fields map 1-to-1 to NvidiaRAG.search() keyword arguments (excluding
    ``query``, which is provided by the agent itself for each retrieval call).

    Set via ``_agentic_search_params`` ContextVar before each graph.ainvoke().
    """

    # --- Retrieval scope ----------------------------------------------------
    collection_names: list[str] | None = None
    vdb_top_k: int | None = None
    vdb_endpoint: str | None = None
    vdb_auth_token: str = ""

    # --- Reranking ----------------------------------------------------------
    reranker_top_k: int | None = None
    reranker_model: str | None = None
    reranker_endpoint: str | None = None
    enable_reranker: bool | None = None

    # --- Embedding ----------------------------------------------------------
    embedding_model: str | None = None
    embedding_endpoint: str | None = None

    # --- Query processing ---------------------------------------------------
    enable_query_rewriting: bool | None = None
    enable_filter_generator: bool | None = None
    filter_expr: str | list[dict[str, Any]] = field(default_factory=str)

    # --- Result filtering ---------------------------------------------------
    confidence_threshold: float | None = None
    enable_citations: bool | None = None


# ContextVar holding per-request search params.
# No mutable default is stored here (B039). Callers use .get(AgenticSearchParams()) so
# each fallback access creates a fresh instance rather than sharing one mutable object.
_agentic_search_params: contextvars.ContextVar[AgenticSearchParams] = contextvars.ContextVar(
    "_agentic_search_params"
)

# ContextVar holding the accumulated citations list for the current agentic request.
# Each entry is a SourceResult with the stage field set to the pipeline stage that produced it.
# Must be initialised to a fresh list before each graph.ainvoke() call (done in runner.py).
_agentic_all_citations: contextvars.ContextVar[list[SourceResult]] = contextvars.ContextVar(
    "_agentic_all_citations"
)


# =============================================================================
# RETRIEVER BRIDGE
# =============================================================================


def make_retriever_fn(
    nvidia_rag: "NvidiaRAG",
    default_reranker_top_k: int,
) -> Callable[[str], Any]:
    """Return an async retriever function backed by NvidiaRAG.search().

    The returned coroutine matches the signature expected by AgenticRag:

        async def retriever_fn(query: str) -> dict | None

    It calls NvidiaRAG.search() and returns Citations.model_dump(), whose
    ``results`` list is directly parseable by AgenticRag._extract_chunks().
    Each element carries ``document_name``, ``score``, ``document_type``, and
    ``content`` — exactly what the agent needs.

    All search parameters are resolved at call time from ``_agentic_search_params``
    so that each request can override any search argument without rebuilding the
    cached agent.  ``default_reranker_top_k`` is used only when the per-request
    params leave ``reranker_top_k`` as None.

    Args:
        nvidia_rag: Initialised NvidiaRAG instance.
        default_reranker_top_k: Fallback reranker_top_k from NvidiaRAGConfig.
    """

    async def retriever_fn(query: str, stage: str = "rag") -> dict | None:
        # Read all per-request search parameters set by _agentic_chain.
        # Passing the fallback here (not as ContextVar default) ensures each
        # fallback access gets a fresh AgenticSearchParams rather than a shared one.
        p = _agentic_search_params.get(AgenticSearchParams())
        try:
            citations = await nvidia_rag.search(
                query=query,
                collection_names=p.collection_names,
                reranker_top_k=p.reranker_top_k if p.reranker_top_k is not None else default_reranker_top_k,
                vdb_top_k=p.vdb_top_k,
                vdb_endpoint=p.vdb_endpoint,
                vdb_auth_token=p.vdb_auth_token,
                enable_reranker=p.enable_reranker,
                enable_query_rewriting=p.enable_query_rewriting,
                enable_filter_generator=p.enable_filter_generator,
                embedding_model=p.embedding_model,
                embedding_endpoint=p.embedding_endpoint,
                reranker_model=p.reranker_model,
                reranker_endpoint=p.reranker_endpoint,
                filter_expr=p.filter_expr,
                confidence_threshold=p.confidence_threshold,
                enable_citations=p.enable_citations,
                stage=stage,
            )
            # Accumulate stage-annotated citations for the final RAGResponse.
            try:
                _agentic_all_citations.get().extend(citations.results)
            except LookupError:
                pass  # Not in an agentic request context (e.g. direct search() call)
            return citations.model_dump()
        except Exception as ex:
            logger.warning(
                "Agentic retriever failed for query %r: %s", query[:80], ex
            )
            return None

    return retriever_fn


# =============================================================================
# LLM FACTORY
# =============================================================================


def _make_role_llm(
    role_cfg: "AgenticAnyLLMConfig",
    fallback_cfg: "AgenticAnyLLMConfig",
    rag_config: "NvidiaRAGConfig",
) -> Any:
    """Construct a LangChain LLM for one agentic role.

    Resolution order for model_name / server_url / api_key:
      1. role_cfg (if model_name is non-empty)
      2. fallback_cfg (planner LLM config)
      3. rag_config.llm (main RAG LLM config)
    """
    from nvidia_rag.utils.llm import get_llm

    cfg = role_cfg if role_cfg.model_name else fallback_cfg

    model_name = cfg.model_name or rag_config.llm.model_name
    server_url = cfg.server_url or rag_config.llm.server_url
    api_key = cfg.get_api_key() or rag_config.llm.get_api_key()

    logger.debug(
        "Creating agentic LLM: model=%s, url=%s",
        model_name,
        server_url or "(api-catalog)",
    )

    return get_llm(
        config=rag_config,
        model=model_name,
        llm_endpoint=server_url,
        api_key=api_key,
    )


# =============================================================================
# AGENT BUILDER
# =============================================================================


async def build_agentic_rag_agent(
    nvidia_rag: "NvidiaRAG",
) -> tuple[AgenticRag, Any]:
    """Build and compile an AgenticRag from a NvidiaRAG instance.

    Reads nvidia_rag.config.agentic_rag for all tunable parameters and wires in
    the NvidiaRAG.search()-backed retriever.  No search parameters are baked in
    at build time — they are all supplied per-request via the
    ``_agentic_search_params`` ContextVar before each graph.ainvoke() call.

    Args:
        nvidia_rag: Initialised NvidiaRAG instance (provides config + search).

    Returns:
        (agent, compiled_graph) — caller is responsible for caching these.
    """
    cfg = nvidia_rag.config.agentic_rag
    rag_config = nvidia_rag.config

    logger.info(
        "Building AgenticRag (concurrency=%d, recursion_limit=%d)",
        cfg.concurrency_limit,
        cfg.recursion_limit,
    )

    planner_llm = _make_role_llm(cfg.planner_llm, cfg.planner_llm, rag_config)
    task_llm = _make_role_llm(cfg.task_llm, cfg.planner_llm, rag_config)
    seed_gen_llm = _make_role_llm(cfg.seed_gen_llm, cfg.planner_llm, rag_config)
    synthesis_llm = _make_role_llm(cfg.synthesis_llm, cfg.planner_llm, rag_config)

    default_reranker_top_k = rag_config.retriever.top_k
    retriever_fn = make_retriever_fn(nvidia_rag, default_reranker_top_k)

    agent = AgenticRag(
        planner_llm=planner_llm,
        task_llm=task_llm,
        seed_gen_llm=seed_gen_llm,
        synthesis_llm=synthesis_llm,
        retriever_fn=retriever_fn,
        log_level=cfg.log_level,
        concurrency_limit=cfg.concurrency_limit,
        planner_config=cfg.planner,
        task_execution_config=cfg.task_execution,
        llm_config=cfg.llm,
        verification_config=cfg.verification,
        context_config=cfg.context,
    )

    graph = await agent.build_graph()
    logger.info("AgenticRag graph compiled successfully")
    return agent, graph

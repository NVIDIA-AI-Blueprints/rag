# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal MCP server exposing RAG generate and search as MCP tools over stdio.
#
# Tools:
# - generate: Run the RAG generation pipeline (with or without knowledge base)
# - search: Retrieve relevant documents/citations from the vector database
#
# Usage:
#   This module is intended to be launched by an MCP-compatible client via stdio.
#   It relies on environment variables/config files the same way FastAPI server does.
#
# Notes:
# - Streaming responses are consumed and returned as a single concatenated string.
# - Tool parameters align closely with NvidiaRAG.generate/search signatures.

from __future__ import annotations

import anyio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server

from nvidia_rag.rag_server.main import NvidiaRAG
from nvidia_rag.utils.configuration import NvidiaRAGConfig


server = Server("nvidia-rag-mcp")

# Initialize the core RAG instance once (loads config from env/defaults)
RAG = NvidiaRAG(config=NvidiaRAGConfig())


@server.tool(
    "generate",
    description="Generate an answer using NVIDIA RAG (optionally with knowledge base). "
    "Provide chat messages and optional generation parameters.",
)
async def tool_generate(
    messages: list[dict[str, Any]],
    use_knowledge_base: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    min_tokens: int | None = None,
    ignore_eos: bool | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    reranker_top_k: int | None = None,
    vdb_top_k: int | None = None,
    vdb_endpoint: str | None = None,
    collection_name: str = "",
    collection_names: list[str] | None = None,
    enable_query_rewriting: bool | None = None,
    enable_reranker: bool | None = None,
    enable_guardrails: bool | None = None,
    enable_citations: bool | None = None,
    enable_vlm_inference: bool | None = None,
    enable_filter_generator: bool | None = None,
    model: str | None = None,
    llm_endpoint: str | None = None,
    embedding_model: str | None = None,
    embedding_endpoint: str | None = None,
    reranker_model: str | None = None,
    reranker_endpoint: str | None = None,
    vlm_model: str | None = None,
    vlm_endpoint: str | None = None,
    filter_expr: str | list[dict[str, Any]] = "",
    confidence_threshold: float | None = None,
) -> str:
    """
    Run the generation pipeline and return the concatenated textual answer.
    """
    # Execute core pipeline (returns RAGResponse with a generator)
    rag_response = RAG.generate(
        messages=messages,
        use_knowledge_base=use_knowledge_base,
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_tokens,
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
        stop=stop or [],
        reranker_top_k=reranker_top_k,
        vdb_top_k=vdb_top_k,
        vdb_endpoint=vdb_endpoint,
        collection_name=collection_name,
        collection_names=collection_names,
        enable_query_rewriting=enable_query_rewriting,
        enable_reranker=enable_reranker,
        enable_guardrails=enable_guardrails,
        enable_citations=enable_citations,
        enable_vlm_inference=enable_vlm_inference,
        enable_filter_generator=enable_filter_generator,
        model=model,
        llm_endpoint=llm_endpoint,
        embedding_model=embedding_model,
        embedding_endpoint=embedding_endpoint,
        reranker_model=reranker_model,
        reranker_endpoint=reranker_endpoint,
        vlm_model=vlm_model,
        vlm_endpoint=vlm_endpoint,
        filter_expr=filter_expr,
        confidence_threshold=confidence_threshold,
        metrics=None,
    )

    # Consume the stream and aggregate textual content
    output_text: str = ""
    for chunk in rag_response.generator:
        try:
            # strip any SSE prefixes if present; generator yields plain strings in server
            output_text += str(chunk)
        except Exception:
            # Best-effort concatenation
            output_text += ""
    return output_text


@server.tool(
    "search",
    description="Search the vector database and return citations for a given query.",
)
async def tool_search(
    query: str | list[dict[str, Any]],
    messages: list[dict[str, str]] | None = None,
    reranker_top_k: int | None = None,
    vdb_top_k: int | None = None,
    collection_name: str = "",
    collection_names: list[str] | None = None,
    vdb_endpoint: str | None = None,
    enable_query_rewriting: bool | None = None,
    enable_reranker: bool | None = None,
    enable_filter_generator: bool | None = None,
    embedding_model: str | None = None,
    embedding_endpoint: str | None = None,
    reranker_model: str | None = None,
    reranker_endpoint: str | None = None,
    filter_expr: str | list[dict[str, Any]] = "",
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run the search pipeline and return citations as a plain JSON-serializable dict.
    """
    citations = RAG.search(
        query=query,
        messages=messages,
        reranker_top_k=reranker_top_k,
        vdb_top_k=vdb_top_k,
        collection_name=collection_name,
        collection_names=collection_names,
        vdb_endpoint=vdb_endpoint,
        enable_query_rewriting=enable_query_rewriting,
        enable_reranker=enable_reranker,
        enable_filter_generator=enable_filter_generator,
        embedding_model=embedding_model,
        embedding_endpoint=embedding_endpoint,
        reranker_model=reranker_model,
        reranker_endpoint=reranker_endpoint,
        filter_expr=filter_expr,
        confidence_threshold=confidence_threshold,
    )
    # Pydantic model -> dict
    return citations.model_dump()


async def _amain() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write)


def main() -> None:
    anyio.run(_amain)


if __name__ == "__main__":
    main()

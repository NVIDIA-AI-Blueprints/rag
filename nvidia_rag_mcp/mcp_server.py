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

from __future__ import annotations
"""
NVIDIA RAG MCP Server
---------------------

This server exposes NVIDIA RAG capabilities as MCP tools using FastMCP.
Transports:
  - sse: Server-Sent Events endpoint
  - streamable_http: FastMCP streamable-http (recommended for HTTP)

Implementation notes:
  - The server forwards requests to the RAG HTTP API discovered via _rag_base_url.
  - Tool implementations are thin adapters around REST endpoints to keep the
    surface predictable for MCP clients.
"""

import argparse
import aiohttp
import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP


server = FastMCP("nvidia-rag-mcp-server")

def _rag_base_url() -> str:
    """
    Resolve the base URL for the RAG HTTP API.
    Priority:
      - VITE_API_CHAT_URL env var (e.g., http://localhost:8081)
    Fallback:
      - http://localhost:8081
    """
    return os.environ.get("VITE_API_CHAT_URL", "http://localhost:8081").rstrip("/")


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
    Generate an answer using the RAG pipeline.
    Streams SSE chunks when available and concatenates them into a single string.
    Returns:
        str: Full generated text.
    """
    base_url = _rag_base_url()
    url = f"{base_url}/v1/generate"

    payload: dict[str, Any] = {
        "messages": messages,
    }
    
    if use_knowledge_base is not None:
        payload["use_knowledge_base"] = use_knowledge_base
    if stop is not None:
        payload["stop"] = stop
    if filter_expr is not None:
        payload["filter_expr"] = filter_expr
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if min_tokens is not None:
        payload["min_tokens"] = min_tokens
    if ignore_eos is not None:
        payload["ignore_eos"] = ignore_eos
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reranker_top_k is not None:
        payload["reranker_top_k"] = reranker_top_k
    if vdb_top_k is not None:
        payload["vdb_top_k"] = vdb_top_k
    if vdb_endpoint is not None:
        payload["vdb_endpoint"] = vdb_endpoint
    if collection_name:
        payload["collection_name"] = collection_name
    if collection_names is not None:
        payload["collection_names"] = collection_names
    if enable_query_rewriting is not None:
        payload["enable_query_rewriting"] = enable_query_rewriting
    if enable_reranker is not None:
        payload["enable_reranker"] = enable_reranker
    if enable_guardrails is not None:
        payload["enable_guardrails"] = enable_guardrails
    if enable_citations is not None:
        payload["enable_citations"] = enable_citations
    if enable_vlm_inference is not None:
        payload["enable_vlm_inference"] = enable_vlm_inference
    if enable_filter_generator is not None:
        payload["enable_filter_generator"] = enable_filter_generator
    if model is not None:
        payload["model"] = model
    if llm_endpoint is not None:
        payload["llm_endpoint"] = llm_endpoint
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
    if embedding_endpoint is not None:
        payload["embedding_endpoint"] = embedding_endpoint
    if reranker_model is not None:
        payload["reranker_model"] = reranker_model
    if reranker_endpoint is not None:
        payload["reranker_endpoint"] = reranker_endpoint
    if vlm_model is not None:
        payload["vlm_model"] = vlm_model
    if vlm_endpoint is not None:
        payload["vlm_endpoint"] = vlm_endpoint
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold

    timeout = aiohttp.ClientTimeout(total=300)
    concatenated_text: list[str] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type or resp.status == 200:
                buffer = ""
                async for chunk in resp.content.iter_chunked(8192):
                    if not chunk:
                        continue
                    try:
                        decoded = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    buffer += decoded
                    lines = buffer.split("\n")
                    buffer = lines[-1]
                    for line in lines[:-1]:
                        line = line.strip()
                        if not line.startswith("data: "):
                            if not line:
                                continue
                            try:
                                data_obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                        else:
                            json_str = line[6:].strip()
                            if not json_str:
                                continue
                            try:
                                data_obj = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue

                        message_part = (
                            data_obj.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if message_part:
                            concatenated_text.append(str(message_part))

                        finish_reason = (
                            data_obj.get("choices", [{}])[0].get("finish_reason")
                        )
                        if finish_reason == "stop":
                            return "".join(concatenated_text)
                if concatenated_text:
                    return "".join(concatenated_text)


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
    Search the vector database for relevant documents.
    Returns:
        dict[str, Any]: JSON body returned by the RAG search endpoint.
    """
    base_url = _rag_base_url()
    url = f"{base_url}/v1/search"

    payload: dict[str, Any] = {
        "query": query,
    }

    if messages is not None:
        payload["messages"] = messages
    if filter_expr is not None:
        payload["filter_expr"] = filter_expr
    if reranker_top_k is not None:
        payload["reranker_top_k"] = reranker_top_k
    if vdb_top_k is not None:
        payload["vdb_top_k"] = vdb_top_k
    if collection_name:
        payload["collection_name"] = collection_name
    if collection_names is not None:
        payload["collection_names"] = collection_names
    if vdb_endpoint is not None:
        payload["vdb_endpoint"] = vdb_endpoint
    if enable_query_rewriting is not None:
        payload["enable_query_rewriting"] = enable_query_rewriting
    if enable_reranker is not None:
        payload["enable_reranker"] = enable_reranker
    if enable_filter_generator is not None:
        payload["enable_filter_generator"] = enable_filter_generator
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
    if embedding_endpoint is not None:
        payload["embedding_endpoint"] = embedding_endpoint
    if reranker_model is not None:
        payload["reranker_model"] = reranker_model
    if reranker_endpoint is not None:
        payload["reranker_endpoint"] = reranker_endpoint
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold


    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {"error": "Non-JSON response", "status": resp.status, "body": text}


@server.tool(
    "get_summary",
    description="Retrieve the pre-generated summary for a document from a collection. "
    "Set blocking=true to wait up to timeout seconds for summary generation.",
)
async def tool_get_summary(
    collection_name: str,
    file_name: str,
    blocking: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Retrieve pre-generated summary for a document.
    Returns:
        dict[str, Any]: Summary or status (pending/timeout/error).
    """
    base_url = _rag_base_url()
    url = f"{base_url}/v1/summary"

    params = {
        "collection_name": collection_name,
        "file_name": file_name,
        "blocking": str(bool(blocking)).lower(),
        "timeout": timeout,
    }
    timeout_cfg = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url, params=params) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {"error": "Non-JSON response", "status": resp.status, "body": text}


def main() -> None:
    """
    Main entry point for the MCP server.
    Examples:
      SSE:
        python nvidia_rag_mcp/mcp_server.py --transport sse --host 127.0.0.1 --port 8000
      streamable_http:
        python nvidia_rag_mcp/mcp_server.py --transport streamable_http
    """
    parser = argparse.ArgumentParser(description="NVIDIA RAG MCP server")
    parser.add_argument("--transport", choices=["sse", "streamable_http"], help="Transport mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports")
    ns = parser.parse_args()

    if ns.transport == "streamable_http":
        try:
            server.run(
                transport="streamable-http",
                host=ns.host,
                port=ns.port,
            )
        except TypeError:
            server.run(transport="streamable-http")
    elif ns.transport == "sse":
        try:
            server.run(
                transport="sse",
                host=ns.host,
                port=ns.port,
            )
        except TypeError:
            server.run(transport="sse")


if __name__ == "__main__":
    main()

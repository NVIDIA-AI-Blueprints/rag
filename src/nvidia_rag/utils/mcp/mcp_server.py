# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
NVIDIA RAG MCP Server

Exposes RAG functionality as MCP tools over stdio, SSE, or streamable HTTP transports.

Available Tools:
- generate: Run RAG generation pipeline (with or without knowledge base)
- search: Retrieve relevant documents/citations from vector database
- get_summary: Retrieve pre-generated document summaries

Supported Transports:
- stdio: Local process-to-process communication
- sse: Server-Sent Events over HTTP
- streamable_http: Streamable HTTP transport (uses /messages endpoint)

Usage:
  python -m nvidia_rag.utils.mcp.mcp_server --transport stdio --api-key YOUR_KEY
  python -m nvidia_rag.utils.mcp.mcp_server --transport sse --host 127.0.0.1 --port 8000
  python -m nvidia_rag.utils.mcp.mcp_server --transport streamable_http --host 127.0.0.1 --port 8000

Notes:
- Streaming responses are consumed and returned as concatenated text
- Tool parameters align with NvidiaRAG.generate/search/get_summary signatures
"""

from __future__ import annotations

import argparse
import anyio
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from nvidia_rag.rag_server.main import NvidiaRAG
from nvidia_rag.utils.configuration import NvidiaRAGConfig
try:
    from mcp.server.http import create_sse_app  # type: ignore
except Exception:
    try:
        from fastmcp.server.http import create_sse_app  # type: ignore
    except Exception:
        create_sse_app = None  # type: ignore


server = FastMCP("nvidia-rag-mcp-server")

# Lazy initialization of RAG instance (initialized after environment is configured)
RAG: NvidiaRAG | None = None


def _ensure_rag_initialized() -> None:
    """Initialize the RAG instance if not already initialized."""
    global RAG
    if RAG is None:
        RAG = NvidiaRAG(config=NvidiaRAGConfig())

class SSEAuthEnvMiddleware:
    """
    ASGI middleware to extract API key from HTTP headers and set NVIDIA_API_KEY env var.
    
    Supports:
    - Authorization: Bearer <token>
    - X-API-Key: <token>
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = {}
            for k, v in (scope.get("headers") or []):
                try:
                    headers[k.decode("latin1").lower()] = v.decode("latin1")
                except Exception:
                    continue
            
            # Extract API key from Authorization or X-API-Key headers
            token: str | None = None
            auth = headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
            if not token:
                api_key = headers.get("x-api-key", "").strip()
                if api_key:
                    token = api_key
            
            if token:
                os.environ["NVIDIA_API_KEY"] = token
        
        await self.app(scope, receive, send)


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
    
    Returns the complete generated text as a concatenated string.
    """
    _ensure_rag_initialized()
    assert RAG is not None
    
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

    # Consume streaming response and concatenate text
    output_text: str = ""
    for chunk in rag_response.generator:
        try:
            output_text += str(chunk)
        except Exception:
            pass
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
    Search the vector database for relevant documents.
    
    Returns citations as a JSON-serializable dictionary.
    """
    _ensure_rag_initialized()
    assert RAG is not None
    
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
    
    return citations.model_dump()


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
        JSON-serializable dict with summary or status (pending/timeout/error).
    """
    _ensure_rag_initialized()
    assert RAG is not None
    
    summary = await NvidiaRAG.get_summary(
        collection_name=collection_name,
        file_name=file_name,
        blocking=blocking,
        timeout=timeout,
    )
    return summary


async def _amain(ns: argparse.Namespace) -> None:
    """Main async entry point for the MCP server (stdio and SSE transports)."""
    # stdio: run FastMCP over stdio
    if ns.transport == "stdio":
        if ns.api_key:
            os.environ["NVIDIA_API_KEY"] = ns.api_key
        await server.run_stdio_async()
        return

    # HTTP transport: SSE uses explicit ASGI app via create_sse_app
    if ns.transport == "sse":
        if ns.api_key:
            os.environ["NVIDIA_API_KEY"] = ns.api_key

        try:
            from uvicorn import Config, Server  # type: ignore
        except Exception:
            raise SystemExit("uvicorn is required for HTTP transports. Install with: pip install uvicorn")

        # Create HTTP app using FastMCP helpers
        if create_sse_app is not None:
            message_path = "/messages/"
            sse_path = "/sse/"

            # Ensure compatibility with FastMCP
            if not hasattr(server, "_additional_http_routes"):
                setattr(server, "_additional_http_routes", [])
            if not hasattr(server, "_additional_middleware"):
                setattr(server, "_additional_middleware", [])

            try:
                app = create_sse_app(  # type: ignore[arg-type]
                    server=server,
                    message_path=message_path,
                    sse_path=sse_path,
                    debug=False,
                    routes=[],
                    middleware=[],
                )
            except Exception:
                # Fallback for older SDK versions
                try:
                    app = create_sse_app(
                        server=server,
                        message_path=message_path,
                        sse_path=sse_path,
                        debug=False,
                    )  # type: ignore[arg-type]
                except Exception as e:
                    raise SystemExit(f"Failed to create HTTP app: {e}")
        else:
            try:
                app = server.sse_app("/")  # type: ignore[attr-defined]
            except Exception as e:
                raise SystemExit(f"Failed to create HTTP app: {e}")

        # Add authentication middleware
        try:
            app.add_middleware(SSEAuthEnvMiddleware)  # type: ignore[attr-defined]
        except Exception:
            app = SSEAuthEnvMiddleware(app)

        config = Config(app=app, host=ns.host, port=ns.port, log_level="info")
        uv_server = Server(config)
        await uv_server.serve()
        return


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="NVIDIA RAG MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable_http"], default="stdio", help="Transport mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=9901, help="Port for HTTP transports")
    parser.add_argument("--api-key", help="API key to set as NVIDIA_API_KEY environment variable")
    ns = parser.parse_args()

    # streamable_http: let FastMCP manage the HTTP app and event loop directly.
    # We avoid wrapping this in anyio.run to prevent nested event loops.
    if ns.transport == "streamable_http":
        if ns.api_key:
            os.environ["NVIDIA_API_KEY"] = ns.api_key
        # Use FastMCP.run to start a streamable-http server; mount_path controls URL prefix.
        # With mount_path="/mcp", streamable_http endpoints will be under /mcp (e.g., /mcp/messages).
        # FastMCP.run in the current SDK manages its own anyio.run for streamable-http.
        server.run(transport="streamable-http", mount_path="/mcp")  # type: ignore[attr-defined]
        return

    # stdio and SSE go through the async helper
    anyio.run(_amain, ns)


if __name__ == "__main__":
    main()

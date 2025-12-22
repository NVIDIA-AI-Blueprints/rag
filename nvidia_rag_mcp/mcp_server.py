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

This server exposes NVIDIA RAG and Ingestor capabilities as MCP tools using FastMCP.
Transports:
  - sse: Server-Sent Events endpoint (HTTP)
  - streamable_http: FastMCP streamable-http (recommended for HTTP)
  - stdio: Standard IO transport (good for local processes)

Implementation notes:
  - The server forwards requests to the RAG HTTP API discovered via _rag_base_url
    and to the Ingestor HTTP API discovered via _ingestor_base_url.
  - Tool implementations are thin adapters around REST endpoints to keep the
    surface predictable for MCP clients.

Environment variables:
  - VITE_API_CHAT_URL: Base URL for RAG HTTP API (default http://localhost:8081)
  - INGESTOR_URL: Base URL for Ingestor API (default http://127.0.0.1:8082)
"""

import argparse
import aiohttp
import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP


server = FastMCP("nvidia-rag-mcp-server")

def _ingestor_base_url() -> str:
    """
    Resolve the base URL for the Ingestor HTTP API.
    Priority:
      - INGESTOR_URL env var (e.g., http://127.0.0.1:8082)
    Fallback:
      - http://127.0.0.1:8082
    """
    return os.environ.get("INGESTOR_URL", "http://127.0.0.1:8082").rstrip("/")

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
    description="""Generate an answer using NVIDIA RAG (optionally with knowledge base).
Args JSON:
{
  "messages": [{"role": "user", "content": "..."}],
  "use_knowledge_base": true,
  "collection_names": ["c1", "c2"],
  "temperature": 0.2,
  "top_p": 0.9,
  "min_tokens": 0,
  "ignore_eos": false,
  "max_tokens": 256,
  "stop": ["\\n"],
  "reranker_top_k": 2,
  "vdb_top_k": 5,
  "vdb_endpoint": "",
  "enable_query_rewriting": false,
  "enable_reranker": true,
  "enable_guardrails": false,
  "enable_citations": false,
  "enable_vlm_inference": false,
  "enable_filter_generator": false,
  "model": "",
  "llm_endpoint": "",
  "embedding_model": "",
  "embedding_endpoint": "",
  "reranker_model": "",
  "reranker_endpoint": "",
  "vlm_model": "",
  "vlm_endpoint": "",
  "filter_expr": "",
  "confidence_threshold": 0.5
}
""",
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
    description="""Search the vector database and return citations for a given query.
Args JSON:
{
  "query": "text or structured query",
  "messages": [{"role": "user", "content": "..."}],
  "collection_names": ["c1", "c2"],
  "vdb_endpoint": "",
  "reranker_top_k": 2,
  "vdb_top_k": 5,
  "enable_query_rewriting": false,
  "enable_reranker": true,
  "enable_filter_generator": false,
  "embedding_model": "",
  "embedding_endpoint": "",
  "reranker_model": "",
  "reranker_endpoint": "",
  "filter_expr": "",
  "confidence_threshold": 0.5
}
""",
)
async def tool_search(
    query: str | list[dict[str, Any]],
    messages: list[dict[str, str]] | None = None,
    reranker_top_k: int | None = None,
    vdb_top_k: int | None = None,
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
    description="""Retrieve the pre-generated summary for a document from a collection.
Set blocking=true to wait up to timeout seconds for summary generation.
Args JSON:
{
  "collection_name": "test",
  "file_name": "woods_frost.pdf",
  "blocking": false,
  "timeout": 60
}
""",
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


@server.tool(
    "get_documents",
    description="""List documents in a collection via the Ingestor service.
Args JSON:
{
  "collection_name": "my_collection",
  "vdb_endpoint": "http://milvus:19530"
}
""",
)
async def tool_get_documents(
    collection_name: str,
    vdb_endpoint: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve documents ingested into a given collection.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/documents"
    params: dict[str, Any] = {"collection_name": collection_name}
    if vdb_endpoint is not None:
        params["vdb_endpoint"] = vdb_endpoint

    timeout_cfg = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url, params=params) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "delete_documents",
    description="""Delete one or more documents from a collection via the Ingestor service.
Args JSON:
{
  "collection_name": "my_collection",
  "document_names": ["file1.pdf", "file2.pdf"]
}
""",
)
async def tool_delete_documents(
    collection_name: str,
    document_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Delete one or more documents from the specified collection.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/documents"
    names = document_names or []
    params: list[tuple[str, str]] = [("collection_name", collection_name)]
    for name in names:
        params.append(("document_names", name))

    timeout_cfg = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.delete(url, params=params) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "update_documents",
    description="""Update (re-upload) one or more local files for a collection (Ingestor).
Same semantics as upload_documents but uses PATCH /v1/documents.
Args JSON:
{
  "collection_name": "test",
  "file_paths": ["/abs/path/a.pdf"],
  "blocking": true,
  "generate_summary": false,
  "custom_metadata": [{}],
  "split_options": {
    "chunk_size": 512,
    "chunk_overlap": 150
  }
}
""",
)
async def tool_update_documents(
    collection_name: str,
    file_paths: list[str],
    blocking: bool = True,
    generate_summary: bool = False,
    custom_metadata: list[dict[str, Any]] | None = None,
    split_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update documents in the Ingestor for a given collection (PATCH /v1/documents).
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/documents"
    form_data = aiohttp.FormData()

    for path in file_paths or []:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    form_data.add_field(
                        "documents",
                        f.read(),
                        filename=os.path.basename(path),
                        content_type="application/octet-stream",
                    )
            else:
                logging.getLogger(__name__).warning(f"File not found, skipping: {path}")
        except OSError as e:
            logging.getLogger(__name__).warning(f"Failed to read file {path}: {e}")
            continue

    data: dict[str, Any] = {
        "collection_name": collection_name,
        "blocking": bool(blocking),
        "custom_metadata": custom_metadata or [],
        "generate_summary": bool(generate_summary),
    }
    if split_options:
        data["split_options"] = split_options
    form_data.add_field("data", json.dumps(data), content_type="application/json")

    timeout_cfg = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.patch(url, data=form_data) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "list_collections",
    description="""List collections from the Ingestor service.
Args JSON:
{
  "vdb_endpoint": "http://milvus:19530"
}
""",
)
async def tool_list_collections(
    vdb_endpoint: str | None = None,
) -> dict[str, Any]:
    """
    List collections known to the underlying vector database via the Ingestor.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/collections"
    params: dict[str, Any] = {}
    if vdb_endpoint is not None:
        params["vdb_endpoint"] = vdb_endpoint

    timeout_cfg = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url, params=params or None) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "update_collection_metadata",
    description="""Update catalog metadata for an existing collection via the Ingestor service.
Args JSON:
{
  "collection_name": "my_collection",
  "description": "Updated description",
  "tags": ["tag1", "tag2"],
  "owner": "owner@example.com",
  "business_domain": "demo",
  "status": "Active"
}
""",
)
async def tool_update_collection_metadata(
    collection_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
    owner: str | None = None,
    business_domain: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Update catalog metadata for the specified collection.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/collections/{collection_name}/metadata"

    body: dict[str, Any] = {}
    if description is not None:
        body["description"] = description
    if tags is not None:
        body["tags"] = tags
    if owner is not None:
        body["owner"] = owner
    if business_domain is not None:
        body["business_domain"] = business_domain
    if status is not None:
        body["status"] = status

    timeout_cfg = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.patch(url, json=body) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "update_document_metadata",
    description="""Update catalog metadata for a specific document in a collection via the Ingestor service.
Args JSON:
{
  "collection_name": "my_collection",
  "document_name": "file1.pdf",
  "description": "Updated description",
  "tags": ["tag1", "tag2"]
}
""",
)
async def tool_update_document_metadata(
    collection_name: str,
    document_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Update catalog metadata for a specific document in a collection.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/collections/{collection_name}/documents/{document_name}/metadata"

    body: dict[str, Any] = {}
    if description is not None:
        body["description"] = description
    if tags is not None:
        body["tags"] = tags

    timeout_cfg = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.patch(url, json=body) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {
                    "error": "Non-JSON response",
                    "status": resp.status,
                    "body": text,
                }


@server.tool(
    "create_collections",
    description="""Create one or more collections in the Ingestor service.
Args JSON:
{
  "collection_names": ["c1", "c2"]
}
""",
)
async def tool_create_collections(
    collection_names: list[str],
) -> dict[str, Any]:
    """
    Create one or more collections.
    Returns:
        dict[str, Any]: Response JSON from the Ingestor service or error info.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/collections"
    timeout_cfg = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.post(url, json=collection_names) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {"error": "Non-JSON response", "status": resp.status, "body": text}


@server.tool(
    "delete_collections",
    description="""Delete one or more collections in the Ingestor service.
Args JSON:
{
  "collection_names": ["c1", "c2"]
}
""",
)
async def tool_delete_collections(
    collection_names: list[str],
) -> dict[str, Any]:
    """
    Delete one or more collections.
    Returns:
        dict[str, Any]: Response JSON from the Ingestor service or error info.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/collections"
    timeout_cfg = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.delete(url, json=collection_names) as resp:
            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                text = await resp.text()
                return {"error": "Non-JSON response", "status": resp.status, "body": text}


@server.tool(
    "upload_documents",
    description="""Upload one or more local files to a collection (Ingestor).
Supports generate_summary and basic split options.
Args JSON:
{
  "collection_name": "test",
  "file_paths": ["/abs/path/a.pdf"],
  "blocking": true,
  "generate_summary": true,
  "custom_metadata": [{}],
  "split_options": {
    "chunk_size": 512,
    "chunk_overlap": 150
  }
}
""",
)
async def tool_upload_documents(
    collection_name: str,
    file_paths: list[str],
    blocking: bool = True,
    generate_summary: bool = True,
    custom_metadata: list[dict[str, Any]] | None = None,
    split_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Upload local files to the Ingestor for a given collection.
    Args:
        collection_name: Target collection.
        file_paths: Absolute/relative paths on the MCP server host.
        blocking: If true, wait for ingestion to complete.
        generate_summary: If true, request summary generation.
        custom_metadata: Optional per-document metadata.
        split_options: Optional chunking config, e.g., {'chunk_size':512,'chunk_overlap':150}
    Returns:
        dict[str, Any]: Server response JSON or error info.
    """
    base_url = _ingestor_base_url()
    url = f"{base_url}/v1/documents"
    form_data = aiohttp.FormData()
    # Add files
    for path in file_paths or []:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    form_data.add_field(
                        "documents",
                        f.read(),
                        filename=os.path.basename(path),
                        content_type="application/octet-stream",
                    )
            else:
                print(f"File not found, skipping: {path}")
        except OSError as e:
            print(f"Failed to read file {path}: {e}")
            continue
    data = {
        "collection_name": collection_name,
        "blocking": bool(blocking),
        "custom_metadata": custom_metadata or [],
        "generate_summary": bool(generate_summary),
    }
    if split_options:
        data["split_options"] = split_options
    form_data.add_field("data", json.dumps(data), content_type="application/json")

    timeout_cfg = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.post(url, data=form_data) as resp:
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
        python nvidia_rag_mcp/mcp_server.py --transport sse
      streamable_http:
        python nvidia_rag_mcp/mcp_server.py --transport streamable_http
    """
    parser = argparse.ArgumentParser(description="NVIDIA RAG MCP server")
    parser.add_argument("--transport", choices=["sse", "streamable_http", "stdio"], help="Transport mode")
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
    elif ns.transport == "stdio":
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

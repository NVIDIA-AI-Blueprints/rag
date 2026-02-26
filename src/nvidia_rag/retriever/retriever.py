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

"""LangChain retriever for the NVIDIA NVIDIA RAG Blueprint.

``NvidiaRAGRetriever`` is a thin, thread-safe HTTP client that wraps
the NVIDIA RAG server's ``/v1/search`` endpoint as a LangChain
``BaseRetriever``.  Ingestion is assumed to be complete before the
retriever is used.

Typical usage::

    from nvidia_rag.retriever import NvidiaRAGRetriever

    retriever = NvidiaRAGRetriever(
        base_url="http://localhost:8081",
        collection_name="my_collection",
        top_k=5,
    )

    # Synchronous
    docs = retriever.invoke("What is CUDA?")

    # Asynchronous
    docs = await retriever.ainvoke("What is CUDA?")

    # Inside a LangChain chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr, SecretStr, model_validator

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 10
_DEFAULT_VDB_TOP_K = 100
_DEFAULT_TIMEOUT = 60.0


class NvidiaRAGRetriever(BaseRetriever):
    """LangChain retriever backed by a NVIDIA RAG Blueprint server.

    This retriever sends natural-language queries to a running NVIDIA RAG
    server and returns ranked documents as LangChain ``Document`` objects.

    The server handles embedding, vector search, and optional reranking –
    the retriever itself is a stateless HTTP client.

    Args:
        base_url: Root URL of the NVIDIA RAG server (e.g. ``http://localhost:8081``).
        api_key: Optional bearer token forwarded in the ``Authorization`` header.
        collection_name: Vector-store collection to search. When *None* the
            server's default collection is used.
        top_k: Maximum number of documents to return after reranking.
        vdb_top_k: Number of candidates to retrieve from the vector DB
            before reranking.  Increase for higher recall at the cost of
            latency.
        filters: Optional filter expression passed to the vector store.
            Accepts a Milvus filter string (e.g.
            ``'content_metadata["author"] == "Jane"'``) or an Elasticsearch
            query DSL list.
        enable_reranker: Whether the server should rerank results.
        enable_query_rewriting: Whether the server should rewrite the query
            for better retrieval.
        embedding_model: Override the server-default embedding model name.
        reranker_model: Override the server-default reranker model name.
        timeout: HTTP request timeout in seconds.

    Example::

        retriever = NvidiaRAGRetriever(
            base_url="http://localhost:8081",
            api_key="nvapi-...",
            collection_name="product_docs",
            top_k=5,
        )
        docs = retriever.invoke("How do I install TensorRT?")
    """

    # ── Public configuration (Pydantic fields) ────────────────────────────
    base_url: str = Field(
        ...,
        description="Root URL of the NVIDIA RAG server.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Bearer token for the RAG server (forwarded as Authorization header).",
    )
    collection_name: str | None = Field(
        default=None,
        description="Collection to search. None uses the server default.",
    )
    top_k: int = Field(
        default=_DEFAULT_TOP_K,
        ge=1,
        le=25,
        description="Max documents returned after reranking.",
    )
    vdb_top_k: int = Field(
        default=_DEFAULT_VDB_TOP_K,
        ge=1,
        le=400,
        description="Candidates retrieved from vector DB before reranking.",
    )
    filters: str | list[dict[str, Any]] = Field(
        default="",
        description="Filter expression for the vector store.",
    )
    enable_reranker: bool | None = Field(
        default=None,
        description="Enable server-side reranking.  None defers to server config.",
    )
    enable_query_rewriting: bool | None = Field(
        default=None,
        description="Enable server-side query rewriting.  None defers to server config.",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Override the embedding model name.",
    )
    reranker_model: str | None = Field(
        default=None,
        description="Override the reranker model name.",
    )
    timeout: float = Field(
        default=_DEFAULT_TIMEOUT,
        gt=0,
        description="HTTP timeout in seconds.",
    )

    # ── Private state ─────────────────────────────────────────────────────
    _sync_client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    # ── Validators ────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _strip_trailing_slash(self) -> "NvidiaRAGRetriever":
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        return self

    def model_post_init(self, __context: Any) -> None:
        """Create httpx clients after Pydantic validation."""
        super().model_post_init(__context)
        headers = self._build_headers()
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

    # ── LangChain interface ───────────────────────────────────────────────
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Synchronously retrieve documents from the NVIDIA RAG server."""
        payload = self._build_payload(query)
        logger.debug("NvidiaRAGRetriever POST /v1/search payload=%s", payload)

        response = self._sync_client.post("/v1/search", json=payload)
        response.raise_for_status()

        return self._parse_response(response.json())

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
    ) -> list[Document]:
        """Asynchronously retrieve documents from the NVIDIA RAG server."""
        payload = self._build_payload(query)
        logger.debug("NvidiaRAGRetriever async POST /v1/search payload=%s", payload)

        response = await self._async_client.post("/v1/search", json=payload)
        response.raise_for_status()

        return self._parse_response(response.json())

    # ── Payload construction ──────────────────────────────────────────────
    def _build_payload(self, query: str) -> dict[str, Any]:
        """Build the JSON payload for ``POST /v1/search``."""
        payload: dict[str, Any] = {
            "query": query,
            "reranker_top_k": self.top_k,
            "vdb_top_k": self.vdb_top_k,
        }

        if self.collection_name is not None:
            payload["collection_names"] = [self.collection_name]

        if self.filters:
            payload["filter_expr"] = self.filters

        if self.enable_reranker is not None:
            payload["enable_reranker"] = self.enable_reranker

        if self.enable_query_rewriting is not None:
            payload["enable_query_rewriting"] = self.enable_query_rewriting

        if self.embedding_model is not None:
            payload["embedding_model"] = self.embedding_model

        if self.reranker_model is not None:
            payload["reranker_model"] = self.reranker_model

        return payload

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers, including optional bearer auth."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    # ── Response parsing ──────────────────────────────────────────────────
    @staticmethod
    def _parse_response(data: dict[str, Any]) -> list[Document]:
        """Convert the ``/v1/search`` JSON response into LangChain Documents.

        The server returns a ``Citations`` object::

            {
                "total_results": N,
                "results": [
                    {
                        "content": "...",
                        "document_name": "...",
                        "document_type": "text",
                        "score": 0.89,
                        "metadata": { ... }
                    },
                    ...
                ]
            }

        Each result is mapped to a ``Document`` with ``page_content`` set to
        the chunk text and ``metadata`` carrying all server-returned fields.
        """
        documents: list[Document] = []

        for result in data.get("results", []):
            metadata: dict[str, Any] = {
                "source": result.get("document_name", ""),
                "document_id": result.get("document_id", ""),
                "document_type": result.get("document_type", "text"),
                "score": result.get("score", 0.0),
            }

            server_meta = result.get("metadata", {})
            if server_meta:
                metadata["page_number"] = server_meta.get("page_number", 0)
                metadata["language"] = server_meta.get("language", "")
                metadata["date_created"] = server_meta.get("date_created", "")
                metadata["last_modified"] = server_meta.get("last_modified", "")
                metadata["description"] = server_meta.get("description", "")

                if server_meta.get("content_metadata"):
                    metadata["content_metadata"] = server_meta["content_metadata"]

                if server_meta.get("location"):
                    metadata["location"] = server_meta["location"]
                if server_meta.get("location_max_dimensions"):
                    metadata["location_max_dimensions"] = server_meta[
                        "location_max_dimensions"
                    ]

            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata=metadata,
                )
            )

        return documents

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Close underlying HTTP clients.  Safe to call multiple times."""
        self._sync_client.close()

    async def aclose(self) -> None:
        """Async close for the underlying HTTP clients."""
        await self._async_client.aclose()

    def __del__(self) -> None:
        try:
            self._sync_client.close()
        except Exception:
            pass

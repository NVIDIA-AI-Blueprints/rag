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

"""This defines the main modules for RAG server which manages the core functionality.

Public async methods:
1. generate(): Generate a response using the RAG chain (async).
2. search(): Search for the most relevant documents for the given search parameters (async).
3. get_summary(): Get the summary of a document (async).
4. health(): Check the health of dependent services (async).

Private async methods:
1. _llm_chain(): Execute a simple LLM chain using the components defined above (async).
2. _rag_chain(): Execute a RAG chain using the components defined above (async).

Private helper methods:
1. _eager_prefetch_astream(): Eagerly prefetch the first chunk from an async stream.
2. _print_conversation_history(): Print the conversation history.
3. _normalize_relevance_scores(): Normalize the relevance scores of the documents.
4. _format_document_with_source(): Format the document with the source.

"""

import json
import logging
import math
import os
import time
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from traceback import print_exc
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from opentelemetry import context as otel_context
from requests import ConnectTimeout

from nvidia_rag.rag_server.health import check_all_services_health
from nvidia_rag.rag_server.query_decomposition import iterative_query_decomposition
from nvidia_rag.rag_server.reflection import (
    ReflectionCounter,
    check_context_relevance,
    check_response_groundedness,
)
from nvidia_rag.rag_server.response_generator import (
    APIError,
    Citations,
    ErrorCodeMapping,
    RAGResponse,
    generate_answer_async,
    prepare_citations,
    prepare_llm_request,
    retrieve_summary,
)
from nvidia_rag.rag_server.validation import (
    validate_model_info,
    validate_reranker_k,
    validate_temperature,
    validate_top_p,
    validate_use_knowledge_base,
    validate_vdb_top_k,
)
from nvidia_rag.rag_server.vlm import VLM
from nvidia_rag.utils.common import (
    filter_documents_by_confidence,
    process_filter_expr,
    validate_filter_expr,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.filter_expression_generator import (
    generate_filter_from_natural_language,
)
from nvidia_rag.utils.health_models import RAGHealthResponse
from nvidia_rag.utils.llm import (
    get_llm,
    get_prompts,
    get_streaming_filter_think_parser_async,
)
from nvidia_rag.utils.observability.otel_metrics import OtelMetrics
from nvidia_rag.utils.reranker import get_ranking_model
from nvidia_rag.utils.vdb import _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag


async def _async_iter(items) -> AsyncGenerator[Any, None]:
    """Helper to convert a list to an async generator."""
    for item in items:
        yield item


logger = logging.getLogger(__name__)

MAX_COLLECTION_NAMES = 5


class NvidiaRAG:
    def __init__(
        self,
        config: NvidiaRAGConfig = None,
        vdb_op: VDBRag = None,
        prompts: str | dict | None = None,
    ):
        """Initialize NvidiaRAG with configuration.

        Attempts to initialize all NIM services during startup. If any NIM is unavailable,
        logs a warning and continues initialization. Unavailable services will return
        proper error responses at request time.

        Args:
            config: Configuration object. If None, loads from environment.
            vdb_op: Optional vector database operator. If None, will be created as needed.
            prompts: Optional path to a YAML/JSON file or a dictionary of prompts.
        """
        # Store config
        self.config = config or NvidiaRAGConfig()
        self.vdb_op = vdb_op

        if self.vdb_op is not None:
            if not isinstance(self.vdb_op, VDBRag):
                raise ValueError(
                    "vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag. "
                    "Please make sure all the required methods are implemented."
                )

        # Initialize models and utilities from config
        logger.info("Initializing NvidiaRAG models...")

        # Track initialization errors for runtime reporting
        self._init_errors = {}

        # Default embedding model
        try:
            self.document_embedder = get_embedding_model(
                model=self.config.embeddings.model_name,
                url=self.config.embeddings.server_url,
                config=self.config,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            self.document_embedder = None
            self._init_errors["embeddings"] = str(e)
            logger.warning(
                "Embedding NIM unavailable at %s - will fail at request time: %s",
                self.config.embeddings.server_url,
                e,
            )

        # Default ranker
        try:
            self.ranker = get_ranking_model(
                model=self.config.ranking.model_name,
                url=self.config.ranking.server_url,
                top_n=self.config.retriever.top_k,
                config=self.config,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            self.ranker = None
            self._init_errors["ranking"] = str(e)
            logger.warning(
                "Ranking NIM unavailable at %s - will fail at request time: %s",
                self.config.ranking.server_url,
                e,
            )

        # Query rewriter LLM
        query_rewriter_llm_config = {
            "temperature": 0,
            "top_p": 0.1,
            "api_key": self.config.query_rewriter.get_api_key(),
        }
        # Log config without sensitive api_key
        safe_config = {k: v for k, v in query_rewriter_llm_config.items() if k != "api_key"}
        logger.info(
            "Query rewriter llm config: model name %s, url %s, config %s",
            self.config.query_rewriter.model_name,
            self.config.query_rewriter.server_url,
            safe_config,
        )
        try:
            self.query_rewriter_llm = get_llm(
                config=self.config,
                model=self.config.query_rewriter.model_name,
                llm_endpoint=self.config.query_rewriter.server_url,
                **query_rewriter_llm_config,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            self.query_rewriter_llm = None
            self._init_errors["query_rewriter"] = str(e)
            logger.warning(
                "Query rewriter NIM unavailable at %s - will fail at request time: %s",
                self.config.query_rewriter.server_url,
                e,
            )

        # Filter expression generator LLM
        filter_generator_llm_config = {
            "temperature": self.config.filter_expression_generator.temperature,
            "top_p": self.config.filter_expression_generator.top_p,
            "max_tokens": self.config.filter_expression_generator.max_tokens,
            "api_key": self.config.filter_expression_generator.get_api_key(),
        }
        try:
            self.filter_generator_llm = get_llm(
                config=self.config,
                model=self.config.filter_expression_generator.model_name,
                llm_endpoint=self.config.filter_expression_generator.server_url,
                **filter_generator_llm_config,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            self.filter_generator_llm = None
            self._init_errors["filter_generator"] = str(e)
            logger.warning(
                "Filter generator NIM unavailable at %s - will fail at request time: %s",
                self.config.filter_expression_generator.server_url,
                e,
            )

        # Load prompts and other utilities
        self.prompts = get_prompts(prompts)
        self.vdb_top_k = int(self.config.retriever.vdb_top_k)
        self.StreamingFilterThinkParser = get_streaming_filter_think_parser_async()

        if self._init_errors:
            logger.warning(
                "NvidiaRAG initialization completed with %d unavailable service(s): %s. "
                "Server will start but requests using these services will fail.",
                len(self._init_errors),
                list(self._init_errors.keys()),
            )
        else:
            logger.info("NvidiaRAG initialization complete - all services available")

    @staticmethod
    async def _eager_prefetch_astream(stream_gen):
        """
        Eagerly fetch the first chunk from an async stream to trigger any errors early.

        Args:
            stream_gen: Async generator to prefetch from

        Returns:
            Async generator that yields the prefetched chunk followed by the rest

        Raises:
            StopAsyncIteration: If the stream is empty (converted to empty generator)
        """
        try:
            first_chunk = await stream_gen.__anext__()

            async def complete_stream():
                yield first_chunk
                async for chunk in stream_gen:
                    yield chunk

            return complete_stream()
        except StopAsyncIteration:
            logger.warning("LLM produced no output.")

            async def empty_gen():
                return
                yield  # Make it an async generator

            return empty_gen()

    async def health(self, check_dependencies: bool = False) -> RAGHealthResponse:
        """Check the health of the RAG server."""
        if check_dependencies:
            vdb_op = self._prepare_vdb_op()
            return await check_all_services_health(vdb_op, self.config)

        return RAGHealthResponse(message="Service is up.")

    def _prepare_vdb_op(
        self,
        vdb_endpoint: str | None = None,
        embedding_model: str | None = None,
        embedding_endpoint: str | None = None,
        vdb_auth_token: str = "",
    ) -> VDBRag:
        """
        Prepare the VDBRag object for generation.
        """
        if self.vdb_op is not None:
            if vdb_endpoint is not None:
                raise ValueError(
                    "vdb_endpoint is not supported when vdb_op is provided during initialization."
                )
            if embedding_model is not None:
                raise ValueError(
                    "embedding_model is not supported when vdb_op is provided during initialization."
                )
            if embedding_endpoint is not None:
                raise ValueError(
                    "embedding_endpoint is not supported when vdb_op is provided during initialization."
                )

            return self.vdb_op

        document_embedder = get_embedding_model(
            model=embedding_model or self.config.embeddings.model_name,
            url=embedding_endpoint or self.config.embeddings.server_url,
            config=self.config,
        )

        return _get_vdb_op(
            vdb_endpoint=vdb_endpoint or self.config.vector_store.url,
            embedding_model=document_embedder,
            config=self.config,
            vdb_auth_token=vdb_auth_token,
        )

    def _validate_collections_exist(
        self, collection_names: list[str], vdb_op: VDBRag
    ) -> None:
        """Validate that all specified collections exist in the vector database.

        Args:
            collection_names: List of collection names to validate
            vdb_op: Vector database operation instance
        Raises:
            APIError: If any collection does not exist
        """
        for collection_name in collection_names:
            if not vdb_op.check_collection_exists(collection_name):
                raise APIError(
                    f"Collection {collection_name} does not exist. Ensure a collection is created using POST /collection endpoint first "
                    f"and documents are uploaded using POST /document endpoint",
                    ErrorCodeMapping.BAD_REQUEST,
                )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        use_knowledge_base: bool = True,
        vdb_auth_token: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
        ignore_eos: bool | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        reranker_top_k: int | None = None,
        vdb_top_k: int | None = None,
        vdb_endpoint: str | None = None,
        min_thinking_tokens: int | None = None,
        max_thinking_tokens: int | None = None,
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
        vlm_temperature: float | None = None,
        vlm_top_p: float | None = None,
        vlm_max_tokens: int | None = None,
        vlm_max_total_images: int | None = None,
        filter_expr: str | list[dict[str, Any]] = "",
        enable_query_decomposition: bool | None = None,
        confidence_threshold: float | None = None,
        rag_start_time_sec: float | None = None,
        metrics: OtelMetrics | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True` or `False`.

        Args:
            messages: List of conversation messages
            use_knowledge_base: Whether to use knowledge base for generation
            temperature: Sampling temperature for generation
            top_p: Top-p sampling mass
            min_tokens: Minimum tokens to generate
            ignore_eos: Whether to generate tokens after the EOS token is generated
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            reranker_top_k: Number of documents to return after reranking
            vdb_top_k: Number of documents to retrieve from vector DB
            collection_names: List of collection names to use
            enable_query_rewriting: Whether to enable query rewriting
            enable_reranker: Whether to enable reranking
            enable_guardrails: Whether to enable guardrails
            enable_citations: Whether to enable citations
            model: Name of the LLM model
            llm_endpoint: LLM server endpoint URL
            reranker_model: Name of the reranker model
            reranker_endpoint: Reranker server endpoint URL
            filter_expr: Filter expression to filter document from vector DB
        """
        # === STAGE 1: PIPELINE ENTRY ===
        logger.info("=" * 80)
        logger.info("RAG PIPELINE START - generate() method invoked")
        logger.info("=" * 80)
        logger.info("Input Parameters:")
        logger.info("  - use_knowledge_base: %s", use_knowledge_base)
        logger.info("  - messages count: %d", len(messages) if messages else 0)
        logger.info("  - collection_names: %s", collection_names)
        logger.info("  - enable_query_rewriting: %s", enable_query_rewriting)
        logger.info("  - enable_reranker: %s", enable_reranker)
        logger.info("  - enable_filter_generator: %s", enable_filter_generator)
        logger.info("  - enable_query_decomposition: %s", enable_query_decomposition)
        logger.info("  - enable_vlm_inference: %s", enable_vlm_inference)
        logger.info("  - enable_reflection: %s", self.config.reflection.enable_reflection)
        logger.info("  - vdb_top_k: %s, reranker_top_k: %s", vdb_top_k, reranker_top_k)
        logger.info("  - temperature: %s, top_p: %s, max_tokens: %s", temperature, top_p, max_tokens)
        logger.info("  - model: %s", model)
        logger.info("-" * 80)
        
        # Apply defaults from config for None values
        model_params = self.config.llm.get_model_parameters()
        temperature = (
            temperature if temperature is not None else model_params["temperature"]
        )
        top_p = top_p if top_p is not None else model_params["top_p"]
        min_tokens = (
            min_tokens if min_tokens is not None else model_params["min_tokens"]
        )
        ignore_eos = (
            ignore_eos if ignore_eos is not None else model_params["ignore_eos"]
        )
        max_tokens = (
            max_tokens if max_tokens is not None else model_params["max_tokens"]
        )
        reranker_top_k = (
            reranker_top_k
            if reranker_top_k is not None
            else self.config.retriever.top_k
        )
        vdb_top_k = (
            vdb_top_k if vdb_top_k is not None else self.config.retriever.vdb_top_k
        )
        enable_query_rewriting = (
            enable_query_rewriting
            if enable_query_rewriting is not None
            else self.config.query_rewriter.enable_query_rewriter
        )
        enable_reranker = (
            enable_reranker
            if enable_reranker is not None
            else self.config.ranking.enable_reranker
        )
        enable_guardrails = (
            enable_guardrails
            if enable_guardrails is not None
            else self.config.enable_guardrails
        )
        enable_citations = (
            enable_citations
            if enable_citations is not None
            else self.config.enable_citations
        )
        enable_vlm_inference = (
            enable_vlm_inference
            if enable_vlm_inference is not None
            else self.config.enable_vlm_inference
        )
        enable_filter_generator = (
            enable_filter_generator
            if enable_filter_generator is not None
            else self.config.filter_expression_generator.enable_filter_generator
        )
        model = model if model is not None else self.config.llm.model_name
        llm_endpoint = (
            llm_endpoint if llm_endpoint is not None else self.config.llm.server_url
        )
        reranker_model = (
            reranker_model
            if reranker_model is not None
            else self.config.ranking.model_name
        )
        reranker_endpoint = (
            reranker_endpoint
            if reranker_endpoint is not None
            else self.config.ranking.server_url
        )
        vlm_model = vlm_model if vlm_model is not None else self.config.vlm.model_name
        vlm_endpoint = (
            vlm_endpoint if vlm_endpoint is not None else self.config.vlm.server_url
        )
        enable_query_decomposition = (
            enable_query_decomposition
            if enable_query_decomposition is not None
            else self.config.query_decomposition.enable_query_decomposition
        )
        confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.default_confidence_threshold
        )

        vdb_op = self._prepare_vdb_op(
            vdb_endpoint=vdb_endpoint,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            vdb_auth_token=vdb_auth_token,
        )

        # Validate boolean and float parameters
        use_knowledge_base = validate_use_knowledge_base(use_knowledge_base)
        temperature = validate_temperature(temperature)
        top_p = validate_top_p(top_p)

        # Validate top_k parameters
        vdb_top_k = validate_vdb_top_k(vdb_top_k)
        reranker_top_k = validate_reranker_k(reranker_top_k, vdb_top_k)

        # Normalize all model and endpoint values using validation functions
        (
            model,
            llm_endpoint,
            reranker_model,
            reranker_endpoint,
            vlm_model,
            vlm_endpoint,
        ) = (
            validate_model_info(model, "model"),
            validate_model_info(llm_endpoint, "llm_endpoint"),
            validate_model_info(reranker_model, "reranker_model"),
            validate_model_info(reranker_endpoint, "reranker_endpoint"),
            validate_model_info(vlm_model, "vlm_model"),
            validate_model_info(vlm_endpoint, "vlm_endpoint"),
        )

        if stop is None:
            stop = []
        if collection_names is None:
            collection_names = [self.config.vector_store.default_collection_name]

        query, chat_history = prepare_llm_request(messages)
        
        # Log extracted query
        query_text = self._extract_text_from_content(query)
        logger.info("Extracted Query: '%s'", query_text[:200] if query_text else "")
        logger.info("Chat History: %d message(s)", len(chat_history) if chat_history else 0)
        
        llm_settings = {
            "model": model,
            "llm_endpoint": llm_endpoint,
            "temperature": temperature,
            "top_p": top_p,
            "min_tokens": min_tokens,
            "ignore_eos": ignore_eos,
            "max_tokens": max_tokens,
            "min_thinking_tokens": min_thinking_tokens,
            "max_thinking_tokens": max_thinking_tokens,
            "enable_guardrails": enable_guardrails,
            "stop": stop,
        }

        # Resolve VLM overrides to concrete values (fall back to config when None)
        vlm_temperature = (
            vlm_temperature
            if vlm_temperature is not None
            else self.config.vlm.temperature
        )
        vlm_top_p = vlm_top_p if vlm_top_p is not None else self.config.vlm.top_p
        vlm_max_tokens = (
            vlm_max_tokens if vlm_max_tokens is not None else self.config.vlm.max_tokens
        )
        vlm_max_total_images = (
            vlm_max_total_images
            if vlm_max_total_images is not None
            else self.config.vlm.max_total_images
        )

        vlm_settings = {
            "vlm_model": vlm_model,
            "vlm_endpoint": vlm_endpoint,
            "vlm_temperature": vlm_temperature,
            "vlm_top_p": vlm_top_p,
            "vlm_max_tokens": vlm_max_tokens,
            "vlm_max_total_images": vlm_max_total_images,
        }

        if use_knowledge_base:
            logger.info("=" * 80)
            logger.info("PIPELINE MODE: RAG Chain (with knowledge base)")
            logger.info("=" * 80)
            return await self._rag_chain(
                llm_settings=llm_settings,
                query=query,
                chat_history=chat_history,
                reranker_top_k=reranker_top_k,
                vdb_top_k=vdb_top_k,
                collection_names=collection_names,
                enable_reranker=enable_reranker,
                reranker_model=reranker_model,
                reranker_endpoint=reranker_endpoint,
                enable_vlm_inference=enable_vlm_inference,
                vlm_settings=vlm_settings,
                model=model,
                enable_query_rewriting=enable_query_rewriting,
                enable_citations=enable_citations,
                filter_expr=filter_expr,
                enable_filter_generator=enable_filter_generator,
                vdb_op=vdb_op,
                enable_query_decomposition=enable_query_decomposition,
                confidence_threshold=confidence_threshold,
                rag_start_time_sec=rag_start_time_sec,
                metrics=metrics,
            )
        else:
            logger.info("=" * 80)
            logger.info("PIPELINE MODE: Direct LLM Chain (without knowledge base)")
            logger.info("=" * 80)
            return await self._llm_chain(
                llm_settings=llm_settings,
                query=query,
                chat_history=chat_history,
                model=model,
                enable_citations=enable_citations,
                metrics=metrics,
            )

    async def search(
        self,
        query: str | list[dict[str, Any]],
        messages: list[dict[str, str]] | None = None,
        reranker_top_k: int | None = None,
        vdb_top_k: int | None = None,
        collection_names: list[str] | None = None,
        vdb_endpoint: str | None = None,
        vdb_auth_token: str = "",
        enable_query_rewriting: bool | None = None,
        enable_reranker: bool | None = None,
        enable_filter_generator: bool | None = None,
        embedding_model: str | None = None,
        embedding_endpoint: str | None = None,
        reranker_model: str | None = None,
        reranker_endpoint: str | None = None,
        filter_expr: str | list[dict[str, Any]] = "",
        confidence_threshold: float | None = None,
        enable_citations: bool | None = None,
    ) -> Citations:
        """Search for the most relevant documents for the given search parameters.
        It's called when the `/search` API is invoked.

        Args:
            query (str | list[dict[str, Any]]): Query to be searched from vectorstore. Can be a string or multimodal content with text and images.
            messages (List[Dict[str, str]]): List of chat messages for context.
            reranker_top_k (int): Number of document chunks to retrieve after reranking.
            vdb_top_k (int): Number of top results to retrieve from vector database.
            collection_names (List[str]): List of collection names to be searched from vectorstore.
            vdb_endpoint (str): Endpoint URL of the vector database server.
            enable_query_rewriting (bool): Whether to enable query rewriting.
            enable_reranker (bool): Whether to enable reranking by the ranker model.
            embedding_model (str): Name of the embedding model used for vectorization.
            embedding_endpoint (str): Endpoint URL for the embedding model server.
            reranker_model (str): Name of the reranker model used for ranking results.
            reranker_endpoint (Optional[str]): Endpoint URL for the reranker model server.
            filter_expr (Union[str, List[Dict[str, Any]]]): Filter expression to filter document from vector DB
        Returns:
            Citations: Retrieved documents.
        """

        query_text = self._extract_text_from_content(query)
        logger.info("=" * 80)
        logger.info("RAG PIPELINE START - search() method invoked")
        logger.info("=" * 80)
        logger.info("Search Parameters:")
        logger.info("  - Query: '%s'", query_text[:200] if query_text else "")
        logger.info("  - Collection Names: %s", collection_names)
        logger.info("  - VDB Top-K: %s, Reranker Top-K: %s", vdb_top_k, reranker_top_k)
        logger.info("  - Enable Query Rewriting: %s", enable_query_rewriting)
        logger.info("  - Enable Reranker: %s", enable_reranker)
        logger.info("  - Enable Filter Generator: %s", enable_filter_generator)
        logger.info("-" * 80)

        # Apply defaults from config for None values
        reranker_top_k = (
            reranker_top_k
            if reranker_top_k is not None
            else self.config.retriever.top_k
        )
        vdb_top_k = (
            vdb_top_k if vdb_top_k is not None else self.config.retriever.vdb_top_k
        )
        enable_query_rewriting = (
            enable_query_rewriting
            if enable_query_rewriting is not None
            else self.config.query_rewriter.enable_query_rewriter
        )
        enable_reranker = (
            enable_reranker
            if enable_reranker is not None
            else self.config.ranking.enable_reranker
        )
        enable_filter_generator = (
            enable_filter_generator
            if enable_filter_generator is not None
            else self.config.filter_expression_generator.enable_filter_generator
        )
        reranker_model = (
            reranker_model
            if reranker_model is not None
            else self.config.ranking.model_name
        )
        reranker_endpoint = (
            reranker_endpoint
            if reranker_endpoint is not None
            else self.config.ranking.server_url
        )
        confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.default_confidence_threshold
        )
        enable_citations = (
            enable_citations
            if enable_citations is not None
            else self.config.enable_citations
        )

        vdb_op = self._prepare_vdb_op(
            vdb_endpoint=vdb_endpoint,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            vdb_auth_token=vdb_auth_token,
        )

        if messages is None:
            messages = []
        if collection_names is None:
            collection_names = [self.config.vector_store.default_collection_name]

        # Validate top_k parameters
        vdb_top_k = validate_vdb_top_k(vdb_top_k)
        reranker_top_k = validate_reranker_k(reranker_top_k, vdb_top_k)

        # Normalize all model and endpoint values using validation functions
        reranker_model, reranker_endpoint = (
            validate_model_info(reranker_model, "reranker_model"),
            validate_model_info(reranker_endpoint, "reranker_endpoint"),
        )

        try:
            if not collection_names:
                raise APIError(
                    "Collection names are not provided.", ErrorCodeMapping.BAD_REQUEST
                )

            if len(collection_names) > 1 and not enable_reranker:
                raise APIError(
                    "Reranking is not enabled but multiple collection names are provided.",
                    ErrorCodeMapping.BAD_REQUEST,
                )

            if len(collection_names) > MAX_COLLECTION_NAMES:
                raise APIError(
                    f"Only {MAX_COLLECTION_NAMES} collections are supported at a time.",
                    ErrorCodeMapping.BAD_REQUEST,
                )

            self._validate_collections_exist(collection_names, vdb_op)

            metadata_schemas = {}

            if (
                filter_expr
                and (not isinstance(filter_expr, str) or filter_expr.strip() != "")
                or enable_filter_generator
            ):
                for collection_name in collection_names:
                    metadata_schemas[collection_name] = vdb_op.get_metadata_schema(
                        collection_name
                    )

            if not filter_expr or (
                isinstance(filter_expr, str) and filter_expr.strip() == ""
            ):
                validation_result = {
                    "status": True,
                    "validated_collections": collection_names,
                }
            else:
                validation_result = validate_filter_expr(
                    filter_expr, collection_names, metadata_schemas, config=self.config
                )

            if not validation_result["status"]:
                error_message = validation_result.get(
                    "error_message", "Invalid filter expression"
                )
                error_details = validation_result.get("details", "")
                full_error = f"Invalid filter expression: {error_message}"
                if error_details:
                    full_error += f"\n Details: {error_details}"
                raise APIError(full_error, ErrorCodeMapping.BAD_REQUEST)

            validated_collections = validation_result.get(
                "validated_collections", collection_names
            )

            if len(validated_collections) < len(collection_names):
                skipped_collections = [
                    name
                    for name in collection_names
                    if name not in validated_collections
                ]
                logger.info(
                    f"Collections {skipped_collections} do not support the filter expression and will be skipped"
                )

            if not filter_expr or (
                isinstance(filter_expr, str) and filter_expr.strip() == ""
            ):
                collection_filter_mapping = dict.fromkeys(validated_collections, "")
                logger.debug(
                    "Filter expression is empty, skipping processing for all collections"
                )
            else:

                def process_filter_for_collection(collection_name):
                    metadata_schema_data = metadata_schemas.get(collection_name)
                    processed_filter_expr = process_filter_expr(
                        filter_expr,
                        collection_name,
                        metadata_schema_data,
                        config=self.config,
                    )
                    logger.debug(
                        f"Filter expression processed for collection '{collection_name}': '{filter_expr}' -> '{processed_filter_expr}'"
                    )
                    return collection_name, processed_filter_expr

                collection_filter_mapping = {}
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_filter_for_collection, collection_name)
                        for collection_name in validated_collections
                    ]
                    for future in futures:
                        collection_name, processed_filter_expr = future.result()
                        collection_filter_mapping[collection_name] = (
                            processed_filter_expr
                        )

            docs = []
            local_ranker = None
            if enable_reranker:
                try:
                    local_ranker = get_ranking_model(
                        model=reranker_model,
                        url=reranker_endpoint,
                        top_n=reranker_top_k,
                        config=self.config,
                    )
                except APIError:
                    # Re-raise APIError as-is
                    raise
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException,
                    ConnectionError,
                    OSError,
                ) as e:
                    # Wrap connection errors from reranker service
                    reranker_url = reranker_endpoint or self.config.ranking.server_url
                    error_msg = f"Reranker NIM unavailable at {reranker_url}. Please verify the service is running and accessible."
                    logger.error("Connection error in reranker initialization: %s", e)
                    raise APIError(
                        error_msg,
                        ErrorCodeMapping.SERVICE_UNAVAILABLE,
                    ) from e

            top_k = vdb_top_k if local_ranker and enable_reranker else reranker_top_k
            logger.info("Setting top k as: %s.", top_k)

            # Build retriever query from multimodal content (similar to generate method)
            retriever_query, is_image_query = self._build_retriever_query_from_content(
                query
            )
            # Query used for specific tasks (filter generation, reflection) - stays clean without history concatenation
            processed_query = retriever_query

            # Handle multi-turn conversations with two different strategies:
            # 1. Query rewriting: Creates a standalone, context-aware query (good for both retrieval and tasks)
            # 2. Query combination: Concatenates history for retrieval, keeps original for specific tasks
            if messages and not is_image_query:
                # Check CONVERSATION_HISTORY setting
                conversation_history_count = int(
                    os.environ.get("CONVERSATION_HISTORY", 0)
                )

                if enable_query_rewriting:
                    # Skip query rewriting if conversation history is disabled
                    if conversation_history_count == 0:
                        logger.warning(
                            "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0. "
                            "Query rewriting requires conversation history to work effectively. "
                            "Skipping query rewriting. Set CONVERSATION_HISTORY > 0 to enable query rewriting."
                        )
                    # Check if query rewriter is available (fails early if not)
                    elif self.query_rewriter_llm is None:
                        raise APIError(
                            "Query rewriting is enabled but the query rewriter NIM is unavailable. "
                            f"Please verify the service is running at {self.config.query_rewriter.server_url}.",
                            ErrorCodeMapping.SERVICE_UNAVAILABLE,
                        )
                    else:
                        # conversation is tuple so it should be multiple of two
                        # -1 is to keep last k conversation
                        history_count = conversation_history_count * 2 * -1
                        messages = messages[history_count:]
                        conversation_history = []

                        for message in messages:
                            if message.get("role") != "system":
                                conversation_history.append(
                                    (message.get("role"), message.get("content"))
                                )

                        # Based on conversation history recreate query for better
                        # document retrieval
                        contextualize_q_system_prompt = (
                            "Given a chat history and the latest user question "
                            "which might reference context in the chat history, "
                            "formulate a standalone question which can be understood "
                            "without the chat history. Do NOT answer the question, "
                            "just reformulate it if needed and otherwise return it as is."
                        )
                        query_rewriter_prompt_config = self.prompts.get(
                            "query_rewriter_prompt", {}
                        )
                        system_prompt = query_rewriter_prompt_config.get(
                            "system", contextualize_q_system_prompt
                        )
                        human_prompt = query_rewriter_prompt_config.get(
                            "human", "{input}"
                        )

                        # Format conversation history as a string
                        formatted_history = ""
                        if conversation_history:
                            formatted_history = "\n".join(
                                [
                                    f"{role.capitalize()}: {content}"
                                    for role, content in conversation_history
                                ]
                            )

                        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", system_prompt),
                                ("human", human_prompt),
                            ]
                        )
                        q_prompt = (
                            contextualize_q_prompt
                            | self.query_rewriter_llm
                            | self.StreamingFilterThinkParser
                            | StrOutputParser()
                        )

                        # Log the complete prompt that will be sent to LLM
                        try:
                            formatted_prompt = contextualize_q_prompt.format_messages(
                                input=query, chat_history=formatted_history
                            )
                            logger.info("Complete query rewriter prompt sent to LLM:")
                            for i, message in enumerate(formatted_prompt):
                                logger.info(
                                    "  Message %d [%s]: %s",
                                    i,
                                    message.type,
                                    message.content,
                                )
                        except Exception as e:
                            logger.warning("Could not format prompt for logging: %s", e)

                        try:
                            retriever_query = await q_prompt.ainvoke(
                                {"input": query, "chat_history": formatted_history}
                            )
                        except (ConnectionError, OSError, Exception) as e:
                            # Wrap connection errors from query rewriter LLM
                            if isinstance(e, APIError):
                                raise
                            query_rewriter_url = self.config.query_rewriter.server_url
                            endpoint_msg = (
                                f" at {query_rewriter_url}"
                                if query_rewriter_url
                                else ""
                            )
                            raise APIError(
                                f"Query rewriter LLM NIM unavailable{endpoint_msg}. Please verify the service is running and accessible or disable query rewriting.",
                                ErrorCodeMapping.SERVICE_UNAVAILABLE,
                            ) from e

                        logger.info("Rewritten Query: %s", retriever_query)

                        # When query rewriting is enabled, we can use it as processed_query for other modules
                        processed_query = retriever_query
                else:
                    # Query combination strategy: Concatenate history for better retrieval context
                    # Note: processed_query remains unchanged (original query) for clean task processing
                    if self.config.query_rewriter.multiturn_retrieval_simple:
                        user_queries = [
                            msg.get("content")
                            for msg in messages
                            if msg.get("role") == "user"
                        ]
                        retriever_query = ". ".join(
                            [*user_queries, self._extract_text_from_content(query)]
                        )
                        logger.info("Combined retriever query: %s", retriever_query)
                    else:
                        # Use only the current query, ignore conversation history
                        logger.info(
                            "Using only current query: %s for retrieval (conversation history disabled)",
                            retriever_query,
                        )

            if enable_filter_generator and not is_image_query:
                if self.config.vector_store.name != "milvus":
                    logger.warning(
                        f"Filter expression generator is currently only supported for Milvus. "
                        f"Current vector store: {self.config.vector_store.name}. Skipping filter generation."
                    )
                else:
                    # Check if filter generator LLM is available (fails early if not)
                    if self.filter_generator_llm is None:
                        raise APIError(
                            "Filter expression generator is enabled but the filter generator NIM is unavailable. "
                            f"Please verify the service is running at {self.config.filter_expression_generator.server_url}.",
                            ErrorCodeMapping.SERVICE_UNAVAILABLE,
                        )

                    logger.debug(
                        "Filter expression generator enabled, attempting to generate filter from query"
                    )
                    try:

                        def generate_filter_for_collection(collection_name):
                            try:
                                metadata_schema_data = metadata_schemas.get(
                                    collection_name
                                )

                                generated_filter = (
                                    generate_filter_from_natural_language(
                                        user_request=processed_query,
                                        collection_name=collection_name,
                                        metadata_schema=metadata_schema_data,
                                        prompt_template=self.prompts.get(
                                            "filter_expression_generator_prompt"
                                        ),
                                        llm=self.filter_generator_llm,
                                        existing_filter_expr=filter_expr,
                                    )
                                )

                                if generated_filter:
                                    logger.debug(
                                        f"Generated filter expression for collection '{collection_name}': {generated_filter}"
                                    )

                                    processed_filter_expr = process_filter_expr(
                                        generated_filter,
                                        collection_name,
                                        metadata_schema_data,
                                        is_generated_filter=True,
                                        config=self.config,
                                    )
                                    return collection_name, processed_filter_expr
                                else:
                                    logger.debug(
                                        f"No filter expression generated for collection '{collection_name}'"
                                    )
                                    return collection_name, ""
                            except Exception as e:
                                logger.warning(
                                    f"Error generating filter for collection '{collection_name}': {str(e)}"
                                )
                                return collection_name, ""

                        with ThreadPoolExecutor() as executor:
                            futures = [
                                executor.submit(
                                    generate_filter_for_collection, collection_name
                                )
                                for collection_name in validated_collections
                            ]

                            for future in futures:
                                collection_name, processed_filter_expr = future.result()
                                collection_filter_mapping[collection_name] = (
                                    processed_filter_expr
                                )

                        generated_count = len(
                            [f for f in collection_filter_mapping.values() if f]
                        )
                        if generated_count > 0:
                            logger.info(
                                f"Generated filter expressions for {generated_count}/{len(validated_collections)} collections"
                            )
                        else:
                            logger.info(
                                "No filter expressions generated for any collection"
                            )

                    except Exception as e:
                        logger.error(f"Error generating filter expression: {str(e)}")

            if confidence_threshold > 0.0 and not enable_reranker:
                logger.warning(
                    f"confidence_threshold is set to {confidence_threshold} but enable_reranker is explicitly set to False. "
                    f"Confidence threshold filtering requires reranker to be enabled to generate relevance scores. "
                    f"Consider setting enable_reranker=True for effective filtering."
                )

            # Get relevant documents with optional reflection
            otel_ctx = otel_context.get_current()
            if self.config.reflection.enable_reflection:
                context_reflection_counter = ReflectionCounter(self.config.reflection.max_loops)
                docs, is_relevant = await check_context_relevance(
                    vdb_op=vdb_op,
                    retriever_query=processed_query,
                    collection_names=validated_collections,
                    ranker=local_ranker,
                    reflection_counter=context_reflection_counter,
                    top_k=top_k,
                    enable_reranker=enable_reranker,
                    collection_filter_mapping=collection_filter_mapping,
                    config=self.config,
                )
                if local_ranker and enable_reranker:
                    docs = self._normalize_relevance_scores(docs)
                    if confidence_threshold > 0.0:
                        docs = filter_documents_by_confidence(
                            documents=docs,
                            confidence_threshold=confidence_threshold,
                        )
                if not is_relevant:
                    logger.warning(
                        "Could not find sufficiently relevant context after maximum attempts"
                    )
                return prepare_citations(retrieved_documents=docs, force_citations=True, enable_citations=enable_citations)
            else:
                if local_ranker and enable_reranker and not is_image_query:
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to %s with the reranker for search",
                        top_k,
                        reranker_top_k,
                    )
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)
                    # Update number of document to be retriever by ranker
                    local_ranker.top_n = reranker_top_k

                    context_reranker = RunnableAssign(
                        {
                            "context": lambda input: local_ranker.compress_documents(
                                query=input["question"], documents=input["context"]
                            )
                        }
                    )

                    # Perform parallel retrieval from all vector stores with their specific filter expressions
                    docs = []
                    vectorstores = []
                    for collection_name in validated_collections:
                        vectorstores.append(
                            vdb_op.get_langchain_vectorstore(collection_name)
                        )

                    with ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                vdb_op.retrieval_langchain,
                                query=retriever_query,
                                collection_name=collection_name,
                                vectorstore=vectorstore,
                                top_k=top_k,
                                filter_expr=collection_filter_mapping.get(
                                    collection_name, ""
                                ),
                                otel_ctx=otel_ctx,
                            )
                            for collection_name, vectorstore in zip(
                                validated_collections, vectorstores, strict=False
                            )
                        ]
                        for future in futures:
                            docs.extend(future.result())

                    context_reranker_start_time = time.time()
                    try:
                        docs = await context_reranker.ainvoke(
                            {"context": docs, "question": processed_query},
                            config={"run_name": "context_reranker"},
                        )
                    except (
                        requests.exceptions.ConnectionError,
                        ConnectionError,
                        OSError,
                    ) as e:
                        reranker_url = (
                            reranker_endpoint or self.config.ranking.server_url
                        )
                        error_msg = f"Reranker NIM unavailable at {reranker_url}. Please verify the service is running and accessible."
                        logger.error("Connection error in reranker: %s", e)
                        raise APIError(
                            error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE
                        ) from e

                    logger.info(
                        "    == Context reranker time: %.2f ms ==",
                        (time.time() - context_reranker_start_time) * 1000,
                    )

                    # Normalize scores to 0-1 range"
                    docs = self._normalize_relevance_scores(docs.get("context", []))
                    if confidence_threshold > 0.0:
                        docs = filter_documents_by_confidence(
                            documents=docs,
                            confidence_threshold=confidence_threshold,
                        )

                    return prepare_citations(
                        retrieved_documents=docs, force_citations=True, enable_citations=enable_citations
                    )
                else:
                    # Handle case where reranker is disabled or image query
                    if is_image_query:
                        docs = vdb_op.retrieval_image_langchain(
                            query=retriever_query,
                            collection_name=validated_collections[0],
                            vectorstore=vdb_op.get_langchain_vectorstore(
                                validated_collections[0]
                            ),
                            top_k=top_k,
                            reranker_top_k=reranker_top_k,
                            # Note: Filter expressions may not be supported for image queries
                            # filter_expr=collection_filter_mapping.get(validated_collections[0], ""),
                            # otel_ctx=otel_ctx,
                        )
                    else:
                        docs = vdb_op.retrieval_langchain(
                            query=retriever_query,
                            collection_name=validated_collections[0],
                            vectorstore=vdb_op.get_langchain_vectorstore(
                                validated_collections[0]
                            ),
                            top_k=top_k,
                            filter_expr=collection_filter_mapping.get(
                                validated_collections[0], ""
                            ),
                            otel_ctx=otel_ctx,
                        )
                    return prepare_citations(
                        retrieved_documents=docs, force_citations=True, enable_citations=enable_citations
                    )

        except APIError:
            # Re-raise APIError as-is to preserve status_code
            raise
        except Exception as e:
            # Only wrap non-APIError exceptions - default to 500 for unexpected errors
            raise APIError(
                f"Failed to search documents. {str(e)}",
                ErrorCodeMapping.INTERNAL_SERVER_ERROR,
            ) from e

    @staticmethod
    async def get_summary(
        collection_name: str,
        file_name: str,
        blocking: bool = False,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Get the summary of a document."""

        summary_response = await retrieve_summary(
            collection_name=collection_name,
            file_name=file_name,
            wait=blocking,
            timeout=timeout,
        )
        return summary_response

    def _handle_prompt_processing(
        self,
        chat_history: list[dict[str, Any]],
        model: str,
        template_key: str = "chat_template",
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[tuple[str, str]],
    ]:
        """Handle common prompt processing logic for both LLM and RAG chains.

        Args:
            chat_history: List of conversation messages
            model: Name of the model used for generation
            template_key: Key to get the appropriate template from prompts

        Returns:
            Tuple containing:
            - system_message: List of system message tuples
            - conversation_history: List of conversation history tuples
            - user_message: List of user message tuples from prompt template
        """

        # Get the base template
        system_prompt = self.prompts.get(template_key, {}).get("system", "")
        # Support both "human" and "user" keys with fallback
        template_dict = self.prompts.get(template_key, {})
        user_prompt = template_dict.get("human", template_dict.get("user", ""))
        conversation_history = []
        user_message = []

        is_nemotron_v1 = str(model).endswith("llama-3.3-nemotron-super-49b-v1")

        # Nemotron controls thinking using system prompt, if nemotron v1 model is used update system prompt to enable/disable think
        if is_nemotron_v1:
            logger.info("Nemotron v1 model detected, updating system prompt")
            if os.environ.get("ENABLE_NEMOTRON_THINKING", "false").lower() == "true":
                logger.info("Setting system prompt as detailed thinking on")
                system_prompt = "detailed thinking on"
            else:
                logger.info("Setting system prompt as detailed thinking off")
                system_prompt = "detailed thinking off"

        # Process chat history
        for message in chat_history:
            # Overwrite system message if provided in conversation history
            if message.get("role") == "system":
                content_text = self._extract_text_from_content(message.get("content"))
                system_prompt = system_prompt + " " + content_text
            else:
                content_text = self._extract_text_from_content(message.get("content"))
                conversation_history.append((message.get("role"), content_text))

        system_message = [("system", system_prompt)]
        if user_prompt:
            user_message = [("user", user_prompt)]

        return (
            system_message,
            conversation_history,
            user_message,
        )

    async def _llm_chain(
        self,
        llm_settings: dict[str, Any],
        query: str | list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        model: str = "",
        enable_citations: bool = True,
        metrics: OtelMetrics | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute a simple LLM chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `False`.

        Args:
            llm_settings: Dictionary containing LLM settings
            query: The user's query
            chat_history: List of conversation messages
            model: Name of the model used for generation
            enable_citations: Whether to enable citations in the response
        """
        try:
            # Limit conversation history to prevent overwhelming the model
            # conversation is tuple so it should be multiple of two
            # -1 is to keep last k conversation
            conversation_history_count = int(os.environ.get("CONVERSATION_HISTORY", 0))
            if conversation_history_count == 0:
                chat_history = []
            else:
                history_count = conversation_history_count * 2 * -1
                chat_history = chat_history[history_count:]

            # Use the new prompt processing method
            (
                system_message,
                conversation_history,
                user_message,
            ) = self._handle_prompt_processing(chat_history, model, "chat_template")

            logger.debug("System message: %s", system_message)
            logger.debug("User message: %s", user_message)
            logger.debug("Conversation history: %s", conversation_history)
            # Prompt template with system message, user message from prompt template
            message = system_message + user_message

            # If conversation history exists, add it as formatted message
            if conversation_history:
                # Format conversation history
                formatted_history = "\n".join(
                    [
                        f"{role.title()}: {content}"
                        for role, content in conversation_history
                    ]
                )
                message += [("user", f"Conversation history:\n{formatted_history}")]

            # Add user query to prompt
            user_query = []
            # Extract text from query for processing
            query_text = self._extract_text_from_content(query)
            logger.info("Query is: %s", query_text)
            if query_text is not None and query_text != "":
                user_query += [("user", "Query: {question}")]

            # Add user query
            message += user_query

            self._print_conversation_history(message, query_text)

            prompt_template = ChatPromptTemplate.from_messages(message)
            llm = get_llm(config=self.config, **llm_settings)

            logger.info("=" * 80)
            logger.info("STAGE: LLM Generation (Direct)")
            logger.info("=" * 80)
            logger.info("LLM Configuration:")
            logger.info("  - Model: %s", model)
            llm_endpoint_display = llm_settings.get("llm_endpoint") or "api catalog"
            logger.info("  - Endpoint: %s", llm_endpoint_display)
            logger.info("  - Temperature: %s, Top-P: %s, Max Tokens: %s", 
                       llm_settings.get("temperature"), 
                       llm_settings.get("top_p"), 
                       llm_settings.get("max_tokens"))
            logger.info("Input:")
            logger.info("  - Query: '%s'", query_text[:200] if query_text else "")
            logger.info("Starting LLM stream generation...")
            logger.info("-" * 80)

            chain = (
                prompt_template
                | llm
                | self.StreamingFilterThinkParser
                | StrOutputParser()
            )
            # Create async stream generator
            stream_gen = chain.astream(
                {"question": query_text}, config={"run_name": "llm-stream"}
            )
            # Eagerly fetch first chunk to trigger any errors before returning response
            prefetched_stream = await self._eager_prefetch_astream(stream_gen)
            
            logger.info("LLM stream initiated successfully (first chunk received)")
            logger.info("-" * 80)

            return RAGResponse(
                generate_answer_async(
                    prefetched_stream,
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.SUCCESS,
            )
        except ConnectTimeout as e:
            logger.warning(
                "Connection timed out while making a request to the LLM endpoint: %s", e
            )
            return RAGResponse(
                generate_answer_async(
                    _async_iter(
                        [
                            "Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."
                        ]
                    ),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.REQUEST_TIMEOUT,
            )

        except (requests.exceptions.ConnectionError, ConnectionError, OSError):
            # Fallback for uncaught LLM connection errors
            llm_url = llm_settings.get("llm_endpoint") or self.config.llm.server_url
            error_msg = f"LLM NIM unavailable at {llm_url}. Please verify the service is running and accessible."
            logger.exception("Connection error (LLM)")
            return RAGResponse(
                generate_answer_async(
                    _async_iter([error_msg]),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
            )

        except Exception as e:
            # Extract just the error type and message for cleaner logs
            error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
            logger.warning(
                "Failed to generate response due to exception: %s", error_msg
            )

            # Only show full traceback at DEBUG level
            if logger.getEffectiveLevel() <= logging.DEBUG:
                print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning(
                    "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                )
                return RAGResponse(
                    generate_answer_async(
                        _async_iter(
                            [
                                "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                            ]
                        ),
                        [],
                        model=model,
                        collection_name="",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.FORBIDDEN,
                )
            elif "[404] Not Found" in str(e):
                # Check if this is a VLM-related error
                error_msg = "Model or endpoint not found. Please verify the API endpoint and your payload. Ensure that the model name is valid."
                logger.warning(f"Model not found: {error_msg}")

                return RAGResponse(
                    generate_answer_async(
                        _async_iter([error_msg]),
                        [],
                        model=model,
                        collection_name="",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.NOT_FOUND,
                )
            else:
                return RAGResponse(
                    generate_answer_async(
                        _async_iter([str(e)]),
                        [],
                        model=model,
                        collection_name="",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.BAD_REQUEST,
                )

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text content from either string or multimodal content.

        Args:
            content: Either a string or a list of content objects (multimodal)

        Returns:
            str: Extracted text content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text from multimodal content
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Note: We ignore image_url content for text extraction
            return " ".join(text_parts)
        else:
            # Fallback for any other content type
            return str(content) if content is not None else ""

    def _contains_images(self, content: Any) -> bool:
        """Check if content contains any images.

        Args:
            content: Either a string or a list of content objects (multimodal)

        Returns:
            bool: True if content contains images, False otherwise
        """
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
        return False

    def _build_retriever_query_from_content(self, content: Any) -> tuple[str, bool]:
        """Build retriever query from either string or multimodal content.
        For multimodal content, includes both text and base64 images for VLM embedding support.

        Args:
            content: Either a string or a list of content objects (multimodal)

        Returns:
            tuple[str, bool]: Query string that may include base64 image data for VLM embeddings
            bool: True if image URL is provided, False otherwise
        """
        if isinstance(content, str):
            return content, False
        elif isinstance(content, list):
            # Build multimodal query with both text and base64 images
            query_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content = item.get("text", "").strip()
                        if text_content:
                            query_parts.append(text_content)
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            # If image URL is provided, return it as is
                            return image_url, True
            # If no image URL is provided, return the text content
            return "\n\n".join(query_parts), False
        else:
            # Fallback for any other content type
            return (str(content) if content is not None else ""), False

    async def _rag_chain(
        self,
        llm_settings: dict[str, Any],
        query: str | list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        reranker_top_k: int = 10,
        vdb_top_k: int = 40,
        collection_names: list[str] | None = None,
        enable_reranker: bool = True,
        reranker_model: str = "",
        reranker_endpoint: str | None = None,
        enable_vlm_inference: bool = False,
        vlm_settings: dict[str, Any] | None = None,
        model: str = "",
        enable_query_rewriting: bool = False,
        enable_citations: bool = True,
        filter_expr: str | list[dict[str, Any]] | None = "",
        enable_filter_generator: bool = False,
        vdb_op: VDBRag | None = None,
        enable_query_decomposition: bool = False,
        confidence_threshold: float | None = None,
        rag_start_time_sec: float | None = None,
        metrics: OtelMetrics | None = None,
    ) -> tuple[AsyncGenerator[str, None], list[dict[str, Any]]]:
        """Execute a RAG chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True`.

        Args:
            llm_settings: Dictionary containing LLM settings
            query: The user's query
            chat_history: List of conversation messages
            reranker_top_k: Number of documents to return after reranking
            vdb_top_k: Number of documents to retrieve from vector DB
            collection_names: List of collection names to use
            embedding_model: Name of the embedding model
            embedding_endpoint: Embedding server endpoint URL
            vdb_endpoint: Vector database endpoint URL
            enable_reranker: Whether to enable reranking
            reranker_model: Name of the reranker model
            reranker_endpoint: Reranker server endpoint URL
            model: Name of the LLM model
            enable_query_rewriting: Whether to enable query rewriting
            enable_citations: Whether to enable citations
            filter_expr: Filter expression to filter document from vector DB
            enable_filter_generator: Whether to enable automatic filter generation
            enable_query_decomposition: Whether to use iterative query decomposition for complex queries
        """
        # TODO: Remove image whille printing logs and add image as place holder to not pollute logs
        logger.info(
            "Using multiturn rag to generate response from document for the query: %s",
            self._extract_text_from_content(query),
        )

        try:
            # Apply default from config for None value
            confidence_threshold = (
                confidence_threshold
                if confidence_threshold is not None
                else self.config.default_confidence_threshold
            )

            # Check if collection names are provided
            if not collection_names:
                raise APIError(
                    "Collection names are not provided.", ErrorCodeMapping.BAD_REQUEST
                )

            if len(collection_names) > 1 and not enable_reranker:
                raise APIError(
                    "Reranking is not enabled but multiple collection names are provided.",
                    ErrorCodeMapping.BAD_REQUEST,
                )
            if len(collection_names) > MAX_COLLECTION_NAMES:
                raise APIError(
                    f"Only {MAX_COLLECTION_NAMES} collections are supported at a time.",
                    ErrorCodeMapping.BAD_REQUEST,
                )

            self._validate_collections_exist(collection_names, vdb_op)

            metadata_schemas = {}
            if (
                filter_expr
                and (not isinstance(filter_expr, str) or filter_expr.strip() != "")
            ) or enable_filter_generator:
                for collection_name in collection_names:
                    metadata_schemas[collection_name] = vdb_op.get_metadata_schema(
                        collection_name
                    )

            if not filter_expr or (
                isinstance(filter_expr, str) and filter_expr.strip() == ""
            ):
                validation_result = {
                    "status": True,
                    "validated_collections": collection_names,
                }
            else:
                validation_result = validate_filter_expr(
                    filter_expr, collection_names, metadata_schemas, config=self.config
                )

            if not validation_result["status"]:
                error_message = validation_result.get(
                    "error_message", "Invalid filter expression"
                )
                error_details = validation_result.get("details", "")
                full_error = f"Invalid filter expression: {error_message}"
                if error_details:
                    full_error += f"\n Details: {error_details}"
                raise APIError(full_error, ErrorCodeMapping.BAD_REQUEST)

            validated_collections = validation_result.get(
                "validated_collections", collection_names
            )

            if len(validated_collections) < len(collection_names):
                skipped_collections = [
                    name
                    for name in collection_names
                    if name not in validated_collections
                ]
                logger.info(
                    f"Collections {skipped_collections} do not support the filter expression and will be skipped"
                )

            if not filter_expr or (
                isinstance(filter_expr, str) and filter_expr.strip() == ""
            ):
                collection_filter_mapping = dict.fromkeys(validated_collections, "")
                logger.debug(
                    "Filter expression is empty, skipping processing for all collections"
                )
            else:

                def process_filter_for_collection(collection_name):
                    # Use cached metadata schema to avoid duplicate API call
                    metadata_schema_data = metadata_schemas.get(collection_name)
                    processed_filter_expr = process_filter_expr(
                        filter_expr,
                        collection_name,
                        metadata_schema_data,
                        config=self.config,
                    )
                    logger.debug(
                        f"Filter expression processed for collection '{collection_name}': '{filter_expr}' -> '{processed_filter_expr}'"
                    )
                    return collection_name, processed_filter_expr

                collection_filter_mapping = {}
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_filter_for_collection, collection_name)
                        for collection_name in validated_collections
                    ]
                    for future in futures:
                        collection_name, processed_filter_expr = future.result()
                        collection_filter_mapping[collection_name] = (
                            processed_filter_expr
                        )

            # LLM and ranker creation - let the existing exception handler at the bottom catch runtime errors
            llm = get_llm(config=self.config, **llm_settings)
            logger.info("Ranker enabled: %s", enable_reranker)

            # Try to get ranking model if reranker is enabled
            ranker = None
            if enable_reranker:
                try:
                    ranker = get_ranking_model(
                        model=reranker_model,
                        url=reranker_endpoint,
                        top_n=reranker_top_k,
                        config=self.config,
                    )
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException,
                ) as e:
                    service_url = (
                        reranker_endpoint
                        or reranker_model
                        or self.config.ranking.server_url
                    )
                    raise APIError(
                        f"Ranking NIM unavailable at {service_url}. "
                        f"Please verify the service is running and accessible.",
                        ErrorCodeMapping.SERVICE_UNAVAILABLE,
                    ) from e

            top_k = vdb_top_k if ranker and enable_reranker else reranker_top_k
            logger.info("Setting retriever top k as: %s.", top_k)

            # conversation is tuple so it should be multiple of two
            # -1 is to keep last k conversation
            conversation_history_count = int(os.environ.get("CONVERSATION_HISTORY", 0))
            if conversation_history_count == 0:
                chat_history = []
                # Warn if query rewriting is enabled but conversation history is disabled
                if enable_query_rewriting:
                    logger.warning(
                        "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0. "
                        "Query rewriting requires conversation history to work effectively. "
                        "Skipping query rewriting. Set CONVERSATION_HISTORY > 0 to enable query rewriting."
                    )
            else:
                history_count = conversation_history_count * 2 * -1
                chat_history = chat_history[history_count:]
            retrieval_time_ms = None
            context_reranker_time_ms = None

            # Use the new prompt processing method
            (
                system_message,
                conversation_history,
                user_message,
            ) = self._handle_prompt_processing(chat_history, model, "rag_template")
            logger.debug("System message: %s", system_message)
            logger.debug("User message: %s", user_message)
            logger.debug("Conversation history: %s", conversation_history)
            # for multimoda query only image is used for retrieval
            retriever_query, is_image_query = self._build_retriever_query_from_content(
                query
            )
            # Query used for specific tasks (filter generation, reflection) - stays clean without history concatenation
            processed_query = retriever_query

            # Handle multi-turn conversations with two different strategies:
            # 1. Query rewriting: Creates a standalone, context-aware query (good for both retrieval and tasks)
            # 2. Query combination: Concatenates history for retrieval, keeps original for specific tasks
            if chat_history and not is_image_query:
                if enable_query_rewriting:
                    logger.info("=" * 80)
                    logger.info("STAGE: Query Rewriting")
                    logger.info("=" * 80)
                    logger.info("Configuration:")
                    logger.info("  - Model: %s", self.config.query_rewriter.model_name)
                    logger.info("  - Endpoint: %s", self.config.query_rewriter.server_url)
                    logger.info("Input:")
                    logger.info("  - Query: '%s'", retriever_query[:200] if retriever_query else "")
                    logger.info("  - Chat History Messages: %d", len(chat_history) if chat_history else 0)
                    logger.info("-" * 80)
                    
                    # Skip query rewriting if conversation history is disabled
                    if conversation_history_count == 0:
                        logger.warning(
                            "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0. "
                            "Query rewriting requires conversation history to work effectively. "
                            "Skipping query rewriting. Set CONVERSATION_HISTORY > 0 to enable query rewriting."
                        )
                    # Check if query rewriter is available (fails early if not)
                    # Only check if we actually need it (CONVERSATION_HISTORY > 0)
                    elif self.query_rewriter_llm is None:
                        raise APIError(
                            "Query rewriting is enabled but the query rewriter NIM is unavailable. "
                            f"Please verify the service is running at {self.config.query_rewriter.server_url}.",
                            ErrorCodeMapping.SERVICE_UNAVAILABLE,
                        )
                    else:
                        # Based on conversation history recreate query for better
                        # document retrieval
                        contextualize_q_system_prompt = (
                            "Given a chat history and the latest user question "
                            "which might reference context in the chat history, "
                            "formulate a standalone question which can be understood "
                            "without the chat history. Do NOT answer the question, "
                            "just reformulate it if needed and otherwise return it as is."
                        )
                        query_rewriter_prompt_config = self.prompts.get(
                            "query_rewriter_prompt", {}
                        )
                        system_prompt = query_rewriter_prompt_config.get(
                            "system", contextualize_q_system_prompt
                        )
                        human_prompt = query_rewriter_prompt_config.get(
                            "human", "{input}"
                        )

                        # Format conversation history as a string
                        formatted_history = ""
                        if conversation_history:
                            formatted_history = "\n".join(
                                [
                                    f"{role.capitalize()}: {content}"
                                    for role, content in conversation_history
                                ]
                            )

                        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", system_prompt),
                                ("human", human_prompt),
                            ]
                        )
                        q_prompt = (
                            contextualize_q_prompt
                            | self.query_rewriter_llm
                            | self.StreamingFilterThinkParser
                            | StrOutputParser()
                        )
                        # query to be used for document retrieval
                        # logger.info("Query rewriter prompt: %s", contextualize_q_prompt)

                        # Log the complete prompt that will be sent to LLM
                        try:
                            formatted_prompt = contextualize_q_prompt.format_messages(
                                input=retriever_query, chat_history=formatted_history
                            )
                            logger.info("Complete query rewriter prompt sent to LLM:")
                            for i, message in enumerate(formatted_prompt):
                                logger.info(
                                    "  Message %d [%s]: %s",
                                    i,
                                    message.type,
                                    message.content,
                                )
                        except Exception as e:
                            logger.warning("Could not format prompt for logging: %s", e)

                        try:
                            retriever_query = await q_prompt.ainvoke(
                                {
                                    "input": retriever_query,
                                    "chat_history": formatted_history,
                                },
                                config={"run_name": "query-rewriter"},
                            )
                        except (ConnectionError, OSError, Exception) as e:
                            # Wrap connection errors from query rewriter LLM
                            if isinstance(e, APIError):
                                raise
                            query_rewriter_url = self.config.query_rewriter.server_url
                            endpoint_msg = (
                                f" at {query_rewriter_url}"
                                if query_rewriter_url
                                else ""
                            )
                            raise APIError(
                                f"Query rewriter LLM NIM unavailable{endpoint_msg}. Please verify the service is running and accessible or disable query rewriting.",
                                ErrorCodeMapping.SERVICE_UNAVAILABLE,
                            ) from e

                        logger.info("Query Rewriting Output:")
                        logger.info("  - Original Query: '%s'", processed_query[:200] if processed_query else "")
                        logger.info("  - Rewritten Query: '%s'", retriever_query[:200] if retriever_query else "")
                        logger.info("  - Rewritten Query Length: %d characters", len(retriever_query))
                        logger.info("Query rewriting completed successfully")
                        logger.info("-" * 80)

                        # When query rewriting is enabled, we can use it as processed_query for other modules
                        processed_query = retriever_query
                else:
                    # Query combination strategy: Concatenate history for better retrieval context
                    # Note: processed_query remains unchanged (original query) for clean task processing
                    if self.config.query_rewriter.multiturn_retrieval_simple:
                        user_query_results = [
                            self._build_retriever_query_from_content(msg.get("content"))
                            for msg in chat_history
                            if msg.get("role") == "user"
                        ][-1:]
                        # Extract just the query strings from the tuples
                        user_queries = [query for query, _ in user_query_results]
                        retriever_query = ". ".join([*user_queries, retriever_query])
                        logger.info("Combined retriever query: %s", retriever_query)
                    else:
                        # Use only the current query, ignore conversation history
                        logger.info(
                            "Using only current query %s for retrieval (conversation history disabled)",
                            retriever_query,
                        )

            if enable_filter_generator and not is_image_query:
                if self.config.vector_store.name != "milvus":
                    logger.warning(
                        f"Filter expression generator is currently only supported for Milvus. "
                        f"Current vector store: {self.config.vector_store.name}. Skipping filter generation."
                    )
                else:
                    # Check if filter generator LLM is available (fails early if not)
                    if self.filter_generator_llm is None:
                        raise APIError(
                            "Filter expression generator is enabled but the filter generator NIM is unavailable. "
                            f"Please verify the service is running at {self.config.filter_expression_generator.server_url}.",
                            ErrorCodeMapping.SERVICE_UNAVAILABLE,
                        )

                    logger.info("=" * 80)
                    logger.info("STAGE: Filter Expression Generation")
                    logger.info("=" * 80)
                    logger.info("Configuration:")
                    logger.info("  - Model: %s", self.config.filter_expression_generator.model_name)
                    logger.info("  - Endpoint: %s", self.config.filter_expression_generator.server_url)
                    logger.info("Input:")
                    logger.info("  - Query: '%s'", processed_query[:200] if processed_query else "")
                    logger.info("  - Collections: %s", validated_collections)
                    logger.info("Generating filter expressions for collections...")
                    logger.info("-" * 80)
                    
                    try:

                        def generate_filter_for_collection(collection_name):
                            try:
                                metadata_schema_data = metadata_schemas.get(
                                    collection_name
                                )

                                generated_filter = (
                                    generate_filter_from_natural_language(
                                        user_request=processed_query,
                                        collection_name=collection_name,
                                        metadata_schema=metadata_schema_data,
                                        prompt_template=self.prompts.get(
                                            "filter_expression_generator_prompt"
                                        ),
                                        llm=self.filter_generator_llm,
                                        existing_filter_expr=filter_expr,
                                    )
                                )

                                if generated_filter:
                                    logger.info(
                                        f"Generated filter expression for collection '{collection_name}': {generated_filter}"
                                    )
                                    processed_filter_expr = process_filter_expr(
                                        generated_filter,
                                        collection_name,
                                        metadata_schema_data,
                                        is_generated_filter=True,
                                        config=self.config,
                                    )
                                    return collection_name, processed_filter_expr
                                else:
                                    logger.info(
                                        f"No filter expression generated for collection '{collection_name}'"
                                    )
                                    return collection_name, ""
                            except Exception as e:
                                logger.warning(
                                    f"Error generating filter for collection '{collection_name}': {str(e)}"
                                )
                                return collection_name, ""

                        with ThreadPoolExecutor() as executor:
                            futures = [
                                executor.submit(
                                    generate_filter_for_collection, collection_name
                                )
                                for collection_name in validated_collections
                            ]

                            for future in futures:
                                collection_name, processed_filter_expr = future.result()
                                collection_filter_mapping[collection_name] = (
                                    processed_filter_expr
                                )

                        generated_count = len(
                            [f for f in collection_filter_mapping.values() if f]
                        )
                        if generated_count > 0:
                            logger.info("Filter Generation Output:")
                            logger.info("  - Successfully generated filters for %d/%d collections", 
                                       generated_count, len(validated_collections))
                            for coll_name, filter_val in collection_filter_mapping.items():
                                if filter_val:
                                    logger.info("  - Collection '%s': %s", coll_name, filter_val[:100])
                        else:
                            logger.info("Filter Generation Output: No filter expressions generated")
                        logger.info("-" * 80)

                    except Exception as e:
                        logger.warning(f"Error generating filter expression: {str(e)}")

            if enable_query_decomposition and not is_image_query:
                logger.info("=" * 80)
                logger.info("STAGE: Query Decomposition (Iterative)")
                logger.info("=" * 80)
                logger.info("Configuration:")
                logger.info("  - LLM Model: %s", model)
                llm_endpoint_display = llm_settings.get("llm_endpoint") or "api catalog"
                logger.info("  - LLM Endpoint: %s", llm_endpoint_display)
                logger.info("  - Recursion Depth: %d", self.config.query_decomposition.recursion_depth)
                logger.info("  - Enable Reranker: %s", enable_reranker)
                if enable_reranker:
                    logger.info("  - Reranker Model: %s", reranker_model)
                    logger.info("  - Reranker Top-K: %d", reranker_top_k)
                logger.info("Input:")
                logger.info("  - Query: '%s'", self._extract_text_from_content(query)[:200])
                logger.info("  - Collection: %s", validated_collections[0] if validated_collections else "")
                logger.info("  - Retrieval Top-K: %d", top_k)
                logger.info("  - Confidence Threshold: %.2f", confidence_threshold)
                logger.info("Starting iterative query decomposition...")
                logger.info("-" * 80)
                
                # TODO: Pass processed_query instead of query and check accuracy
                return await iterative_query_decomposition(
                    query=query,
                    history=conversation_history,
                    llm=llm,
                    vdb_op=vdb_op,
                    ranker=ranker if enable_reranker else None,
                    recursion_depth=self.config.query_decomposition.recursion_depth,
                    enable_citations=enable_citations,
                    collection_name=validated_collections[0]
                    if validated_collections
                    else "",
                    top_k=top_k,
                    ranker_top_k=reranker_top_k,
                    confidence_threshold=confidence_threshold,
                    llm_settings=llm_settings,
                    prompts=self.prompts,
                )

            if confidence_threshold > 0.0 and not enable_reranker:
                logger.warning(
                    f"confidence_threshold is set to {confidence_threshold} but enable_reranker is explicitly set to False. "
                    f"Confidence threshold filtering requires reranker to be enabled to generate relevance scores. "
                    f"Consider setting enable_reranker=True for effective filtering."
                )

            # Get relevant documents with optional reflection
            if self.config.reflection.enable_reflection:
                logger.info("=" * 80)
                logger.info("STAGE: Context Relevance Reflection")
                logger.info("=" * 80)
                logger.info("Reflection Configuration:")
                logger.info("  - Model: %s", self.config.reflection.model_name)
                logger.info("  - Endpoint: %s", self.config.reflection.server_url)
                logger.info("  - Max Loops: %d", self.config.reflection.max_loops)
                logger.info("  - Relevance Threshold: %d", self.config.reflection.context_relevance_threshold)
                logger.info("Starting context relevance check...")
                logger.info("-" * 80)
                
                context_reflection_counter = ReflectionCounter(self.config.reflection.max_loops)

                try:
                    context_to_show, is_relevant = await check_context_relevance(
                        vdb_op=vdb_op,
                        retriever_query=processed_query,
                        collection_names=validated_collections,
                        ranker=ranker,
                        reflection_counter=context_reflection_counter,
                        top_k=top_k,
                        enable_reranker=enable_reranker,
                        collection_filter_mapping=collection_filter_mapping,
                        config=self.config,
                        prompts=self.prompts,
                    )
                except (
                    ConnectionError,
                    OSError,
                    requests.exceptions.ConnectionError,
                ) as e:
                    # Wrap connection errors from reflection LLM with proper message
                    reflection_llm_endpoint = self.config.reflection.server_url
                    endpoint_msg = (
                        f" at {reflection_llm_endpoint}"
                        if reflection_llm_endpoint
                        else ""
                    )
                    raise APIError(
                        f"Reflection LLM NIM unavailable{endpoint_msg}. Please verify the service is running and accessible or disable reflection.",
                        ErrorCodeMapping.SERVICE_UNAVAILABLE,
                    ) from e
                except APIError:
                    # Re-raise APIError as-is
                    raise

                # Normalize scores to 0-1 range
                if ranker and enable_reranker:
                    context_to_show = self._normalize_relevance_scores(context_to_show)

                logger.info("Reflection Output:")
                logger.info("  - Context Relevant: %s", is_relevant)
                logger.info("  - Reflection Iterations: %d", context_reflection_counter.current_count)
                logger.info("  - Final Documents: %d", len(context_to_show))
                if not is_relevant:
                    logger.warning("  - Could not find sufficiently relevant context after %d attempts",
                                 context_reflection_counter.current_count)
                logger.info("-" * 80)
            else:
                otel_ctx = otel_context.get_current()
                # Current reranker is not supported for image query
                if ranker and enable_reranker and not is_image_query:
                    logger.info("=" * 80)
                    logger.info("STAGE: VDB Retrieval + Reranking")
                    logger.info("=" * 80)
                    logger.info("Embedding Configuration:")
                    if hasattr(vdb_op, 'embedding_model') and vdb_op.embedding_model:
                        logger.info("  - Model: %s", vdb_op.embedding_model._model if hasattr(vdb_op.embedding_model, '_model') else 'N/A')
                        logger.info("  - Endpoint: %s", getattr(vdb_op.embedding_model, '_client', {}).base_url if hasattr(vdb_op.embedding_model, '_client') else 'N/A')
                    else:
                        logger.info("  - Model: N/A")
                        logger.info("  - Endpoint: N/A")
                    logger.info("Reranker Configuration:")
                    logger.info("  - Model: %s", reranker_model)
                    logger.info("  - Endpoint: %s", reranker_endpoint or self.config.ranking.server_url)
                    logger.info("Retrieval Configuration:")
                    logger.info("  - Query: '%s'", retriever_query[:200] if retriever_query else "")
                    logger.info("  - Collections: %s", validated_collections)
                    logger.info("  - VDB Top-K: %d", top_k)
                    logger.info("  - Reranker Top-K: %d", reranker_top_k)
                    logger.info("  - Filter Expressions: %s", 
                               {k: v[:50] + "..." if len(v) > 50 else v 
                                for k, v in collection_filter_mapping.items()})
                    logger.info("Starting parallel retrieval from collections...")
                    logger.info("-" * 80)
                    
                    context_reranker = RunnableAssign(
                        {
                            "context": lambda input: ranker.compress_documents(
                                query=input["question"], documents=input["context"]
                            )
                        }
                    )

                    # Perform parallel retrieval from all vector stores
                    docs = []
                    # Start measuring retrieval latency across collections
                    retrieval_start_time = time.time()
                    vectorstores = []
                    for collection_name in validated_collections:
                        vectorstores.append(
                            vdb_op.get_langchain_vectorstore(collection_name)
                        )
                    with ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                vdb_op.retrieval_langchain,
                                query=retriever_query,
                                collection_name=collection_name,
                                vectorstore=vectorstore,
                                top_k=top_k,
                                filter_expr=collection_filter_mapping.get(
                                    collection_name, ""
                                ),
                                otel_ctx=otel_ctx,
                            )
                            for collection_name, vectorstore in zip(
                                validated_collections, vectorstores, strict=False
                            )
                        ]
                        for future in futures:
                            docs.extend(future.result())

                    retrieval_time_ms = (time.time() - retrieval_start_time) * 1000
                    logger.info("Retrieval Output:")
                    logger.info("  - Retrieved Documents: %d", len(docs))
                    logger.info("  - Retrieval Time: %.2f ms", retrieval_time_ms)
                    logger.info("-" * 80)

                    context_reranker_start_time = time.time()
                    logger.info("Starting reranking with query: '%s'", processed_query[:200] if processed_query else "")
                    try:
                        docs = await context_reranker.ainvoke(
                            {"context": docs, "question": processed_query},
                            config={"run_name": "context_reranker"},
                        )
                    except (
                        requests.exceptions.ConnectionError,
                        ConnectionError,
                        OSError,
                    ) as e:
                        reranker_url = (
                            reranker_endpoint or self.config.ranking.server_url
                        )
                        error_msg = f"Reranker NIM unavailable at {reranker_url}. Please verify the service is running and accessible."
                        logger.error("Connection error in reranker: %s", e)
                        raise APIError(
                            error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE
                        ) from e

                    context_reranker_time_ms = (
                        time.time() - context_reranker_start_time
                    ) * 1000
                    context_to_show = docs.get("context", [])
                    
                    # Normalize scores to 0-1 range
                    context_to_show = self._normalize_relevance_scores(context_to_show)
                    
                    logger.info("Reranking Output:")
                    logger.info("  - Reranked Documents: %d", len(context_to_show))
                    logger.info("  - Reranking Time: %.2f ms", context_reranker_time_ms)
                    if context_to_show:
                        scores = [doc.metadata.get("relevance_score", "N/A") for doc in context_to_show[:3]]
                        logger.info("  - Top Document Scores (normalized): %s", 
                                   [f"{s:.4f}" if isinstance(s, (int, float)) else s for s in scores])
                    logger.info("-" * 80)
                else:
                    # Multiple retrievers are not supported when reranking is disabled
                    logger.info("=" * 80)
                    logger.info("STAGE: VDB Retrieval (without reranking)")
                    logger.info("=" * 80)
                    logger.info("Embedding Configuration:")
                    if hasattr(vdb_op, 'embedding_model') and vdb_op.embedding_model:
                        logger.info("  - Model: %s", vdb_op.embedding_model._model if hasattr(vdb_op.embedding_model, '_model') else 'N/A')
                        logger.info("  - Endpoint: %s", getattr(vdb_op.embedding_model, '_client', {}).base_url if hasattr(vdb_op.embedding_model, '_client') else 'N/A')
                    else:
                        logger.info("  - Model: N/A")
                        logger.info("  - Endpoint: N/A")
                    logger.info("Retrieval Configuration:")
                    logger.info("  - Query: '%s'", retriever_query[:200] if retriever_query else "")
                    logger.info("  - Collection: %s", validated_collections[0] if validated_collections else "")
                    logger.info("  - Top-K: %d", top_k)
                    logger.info("  - Is Image Query: %s", is_image_query)
                    logger.info("Starting retrieval...")
                    logger.info("-" * 80)
                    
                    retrieval_start_time = time.time()
                    if is_image_query:
                        docs = vdb_op.retrieval_image_langchain(
                            query=retriever_query,
                            collection_name=validated_collections[0],
                            vectorstore=vdb_op.get_langchain_vectorstore(
                                validated_collections[0]
                            ),
                            top_k=top_k,
                            reranker_top_k=reranker_top_k,
                            # filter_expr=collection_filter_mapping.get(
                            #     validated_collections[0], ""
                            # ),
                            # otel_ctx=otel_ctx,
                        )
                        context_to_show = docs
                    else:
                        docs = vdb_op.retrieval_langchain(
                            query=retriever_query,
                            collection_name=validated_collections[0],
                            vectorstore=vdb_op.get_langchain_vectorstore(
                                validated_collections[0]
                            ),
                            top_k=top_k,
                            filter_expr=collection_filter_mapping.get(
                                validated_collections[0], ""
                            ),
                            otel_ctx=otel_ctx,
                        )
                        context_to_show = docs
                    retrieval_time_ms = (time.time() - retrieval_start_time) * 1000
                    
                    logger.info("Retrieval Output:")
                    logger.info("  - Retrieved Documents: %d", len(context_to_show))
                    logger.info("  - Retrieval Time: %.2f ms", retrieval_time_ms)
                    logger.info("-" * 80)

            if ranker and enable_reranker and confidence_threshold > 0.0:
                logger.info("=" * 80)
                logger.info("STAGE: Confidence Threshold Filtering")
                logger.info("=" * 80)
                logger.info("Configuration:")
                logger.info("  - Input Documents: %d", len(context_to_show))
                logger.info("  - Confidence Threshold: %.2f", confidence_threshold)
                logger.info("-" * 80)
                context_to_show = filter_documents_by_confidence(
                    documents=context_to_show,
                    confidence_threshold=confidence_threshold,
                )
                
                logger.info("Filtering Output:")
                logger.info("  - Filtered Documents: %d", len(context_to_show))
                logger.info("-" * 80)

            if enable_vlm_inference or is_image_query:
                # Initialize vlm_settings if not provided
                vlm_settings = vlm_settings or {}
                # Fast pre-check: determine where images are present
                has_images_in_query = self._contains_images(query)
                has_images_in_history = any(
                    self._contains_images(m.get("content")) for m in chat_history or []
                )
                has_images_in_messages = has_images_in_query or has_images_in_history
                has_images_in_context = False
                try:
                    for d in context_to_show:
                        meta = getattr(d, "metadata", {}) or {}
                        content_md = meta.get("content_metadata", {}) or {}
                        if content_md.get("type") in ["image", "structured"]:
                            has_images_in_context = True
                            break
                except Exception:
                    # If metadata inspection fails, be conservative and proceed
                    has_images_in_context = False

                # Control whether we are allowed to silently fall back to LLM when no images are present
                vlm_to_llm_fallback = getattr(self.config, "vlm_to_llm_fallback", True)

                # Decide if we should call VLM:
                # - Always when any images are present (messages, context, or explicit image query)
                # - Additionally, when VLM_TO_LLM_FALLBACK is disabled, even if no images are present
                should_call_vlm = (
                    has_images_in_messages
                    or has_images_in_context
                    or is_image_query
                    or not vlm_to_llm_fallback
                )

                if should_call_vlm:
                    try:
                        # Resolve all VLM settings to concrete values (no None)
                        vlm_model_cfg = (
                            vlm_settings.get("vlm_model") or self.config.vlm.model_name
                        )
                        vlm_endpoint_cfg = (
                            vlm_settings.get("vlm_endpoint")
                            or self.config.vlm.server_url
                        )
                        vlm_temperature_cfg = (
                            vlm_settings.get("vlm_temperature")
                            or self.config.vlm.temperature
                        )
                        vlm_top_p_cfg = (
                            vlm_settings.get("vlm_top_p") or self.config.vlm.top_p
                        )
                        vlm_max_tokens_cfg = (
                            vlm_settings.get("vlm_max_tokens")
                            or self.config.vlm.max_tokens
                        )
                        vlm_max_total_images_cfg = (
                            vlm_settings.get("vlm_max_total_images")
                            or self.config.vlm.max_total_images
                        )
                        
                        logger.info("=" * 80)
                        logger.info("STAGE: VLM Generation (Vision Language Model)")
                        logger.info("=" * 80)
                        logger.info("VLM Configuration:")
                        logger.info("  - Model: %s", vlm_model_cfg)
                        logger.info("  - Endpoint: %s", vlm_endpoint_cfg)
                        logger.info("  - Temperature: %s, Top-P: %s, Max Tokens: %s", 
                                   vlm_temperature_cfg, vlm_top_p_cfg, vlm_max_tokens_cfg)
                        logger.info("  - Max Total Images: %d", vlm_max_total_images_cfg)
                        logger.info("Input:")
                        logger.info("  - Has Images in Messages: %s", has_images_in_messages)
                        logger.info("  - Has Images in Context: %s", has_images_in_context)
                        logger.info("  - Is Image Query: %s", is_image_query)
                        logger.info("  - Context Documents: %d", len(context_to_show))
                        logger.info("Starting VLM stream generation...")
                        logger.info("-" * 80)

                        vlm = VLM(
                            vlm_model=vlm_model_cfg,
                            vlm_endpoint=vlm_endpoint_cfg,
                            config=self.config,
                            prompts=self.prompts,
                        )
                        # Build full messages: prior history + current query as a final user turn
                        vlm_messages = [
                            *(chat_history or []),
                            {"role": "user", "content": query},
                        ]
                        # Build textual context identical to LLM "context" (before mutation below)
                        vlm_text_context = "\n\n".join(
                            [
                                self._format_document_with_source(d)
                                for d in context_to_show
                            ]
                        )
                        # Always stream VLM response directly using async streaming (reasoning gate deprecated)
                        logger.info("Streaming VLM response directly (async).")
                        vlm_generator = vlm.stream_with_messages(
                            docs=context_to_show,
                            messages=vlm_messages,
                            context_text=vlm_text_context,
                            question_text=self._extract_text_from_content(query),
                            temperature=vlm_temperature_cfg,
                            top_p=vlm_top_p_cfg,
                            max_tokens=vlm_max_tokens_cfg,
                            max_total_images=vlm_max_total_images_cfg,
                        )
                        # Eagerly prefetch first chunk to trigger any errors before creating RAGResponse
                        # ensures connection errors are caught early
                        prefetched_vlm_stream = await self._eager_prefetch_astream(
                            vlm_generator
                        )
                        
                        logger.info("VLM stream initiated successfully (first chunk received)")
                        logger.info("-" * 80)

                        return RAGResponse(
                            generate_answer_async(
                                prefetched_vlm_stream,
                                context_to_show,
                                model=model,
                                collection_name=validated_collections[0] if validated_collections else "",
                                enable_citations=enable_citations,
                            ),
                            status_code=ErrorCodeMapping.SUCCESS,
                        )
                    except APIError as e:
                        # Catch APIError from VLM (raised during eager prefetch) and return with correct status code
                        logger.warning("APIError from VLM in _rag_chain: %s", e.message)
                        return RAGResponse(
                            generate_answer_async(
                                _async_iter([e.message]),
                                [],
                                model=model,
                                collection_name=validated_collections[0] if validated_collections else "",
                                enable_citations=enable_citations,
                                otel_metrics_client=metrics,
                            ),
                            status_code=e.status_code,
                        )
                    except (OSError, ValueError, ConnectionError) as e:
                        logger.warning(
                            "VLM processing failed for query='%s', collection='%s': %s",
                            query,
                            validated_collections[0] if validated_collections else "",
                            e,
                            exc_info=True,
                        )
                        # Provide specific error message for VLM issues
                        vlm_error_msg = f"VLM processing failed: {str(e)}. Please check your VLM configuration and ensure the VLM service is running."
                        # Don't yield here, let the exception propagate to be caught by the server
                        raise APIError(
                            vlm_error_msg, ErrorCodeMapping.BAD_REQUEST
                        ) from e

                    except Exception as e:
                        logger.error(
                            "Unexpected error during VLM processing for query='%s', collection='%s': %s",
                            query,
                            validated_collections[0] if validated_collections else "",
                            e,
                            exc_info=True,
                        )
                        # Provide specific error message for unexpected VLM issues
                        vlm_error_msg = f"Unexpected VLM error: {str(e)}. Please check your VLM configuration and try again."
                        # Don't yield here, let the exception propagate to be caught by the server
                        raise APIError(
                            vlm_error_msg, ErrorCodeMapping.BAD_REQUEST
                        ) from e
                else:
                    # No images found and VLM_TO_LLM_FALLBACK is enabled: skip VLM and continue with standard LLM RAG flow.
                    logger.info(
                        "Skipping VLM because no images are present and VLM_TO_LLM_FALLBACK is enabled; "
                        "falling back to regular LLM flow."
                    )

            docs = [self._format_document_with_source(d) for d in context_to_show]

            logger.info("=" * 80)
            logger.info("STAGE: Context Preparation for LLM")
            logger.info("=" * 80)
            logger.info("Context Configuration:")
            logger.info("  - Final Context Documents: %d", len(context_to_show))
            if context_to_show:
                total_context_length = sum(len(d.page_content) for d in context_to_show)
                logger.info("  - Total Context Length: %d characters", total_context_length)
                logger.info("  - First Document Preview: %s...", 
                           context_to_show[0].page_content[:100] if context_to_show else "")
            logger.info("-" * 80)

            # Prompt for response generation based on context
            message = system_message + user_message

            if conversation_history:
                # Format conversation history
                formatted_history = "\n".join(
                    [
                        f"{role.title()}: {content}"
                        for role, content in conversation_history
                    ]
                )
                message += [("user", f"Conversation history:\n{formatted_history}")]

            # Add user query to prompt
            user_query = [("user", "Query: {question}\n\nAnswer: ")]
            message += user_query

            self._print_conversation_history(message)
            prompt = ChatPromptTemplate.from_messages(message)

            logger.info("=" * 80)
            logger.info("STAGE: LLM Generation (RAG)")
            logger.info("=" * 80)
            logger.info("LLM Configuration:")
            logger.info("  - Model: %s", model)
            llm_endpoint_display = llm_settings.get("llm_endpoint") or "api catalog"
            logger.info("  - Endpoint: %s", llm_endpoint_display)
            logger.info("  - Temperature: %s, Top-P: %s, Max Tokens: %s",
                       llm_settings.get("temperature"),
                       llm_settings.get("top_p"),
                       llm_settings.get("max_tokens"))
            logger.info("Input:")
            logger.info("  - Query: '%s'", self._extract_text_from_content(query)[:200])
            logger.info("  - Context Documents: %d", len(docs))
            logger.info("-" * 80)

            chain = prompt | llm | self.StreamingFilterThinkParser | StrOutputParser()

            # Check response groundedness when reflection is enabled.
            # Use a separate ReflectionCounter for response groundedness so that
            # it can perform up to the configured number of attempts
            # independently of the context relevance reflection loop.
            if self.config.reflection.enable_reflection:
                logger.info("=" * 80)
                logger.info("STAGE: Response Groundedness Reflection")
                logger.info("=" * 80)
                logger.info("Reflection Configuration:")
                logger.info("  - Model: %s", self.config.reflection.model_name)
                logger.info("  - Endpoint: %s", self.config.reflection.server_url)
                logger.info("  - Max Loops: %d", self.config.reflection.max_loops)
                logger.info("  - Groundedness Threshold: %d", self.config.reflection.response_groundedness_threshold)
                logger.info("Starting LLM generation with reflection enabled...")
                logger.info("-" * 80)
                
                response_reflection_counter = ReflectionCounter(
                    self.config.reflection.max_loops
                )
                initial_response = await chain.ainvoke(
                    {"question": query, "context": docs}
                )
                logger.info("Initial LLM response generated, checking groundedness...")
                try:
                    final_response, is_grounded = await check_response_groundedness(
                        query,
                        initial_response,
                        docs,
                        response_reflection_counter,
                        config=self.config,
                        prompts=self.prompts,
                    )
                except (
                    ConnectionError,
                    OSError,
                    requests.exceptions.ConnectionError,
                ) as e:
                    # Wrap connection errors from reflection LLM with proper message
                    reflection_llm_endpoint = self.config.reflection.server_url
                    endpoint_msg = (
                        f" at {reflection_llm_endpoint}"
                        if reflection_llm_endpoint
                        else ""
                    )
                    raise APIError(
                        f"Reflection LLM NIM unavailable{endpoint_msg}. Please verify the service is running and accessible or disable reflection.",
                        ErrorCodeMapping.SERVICE_UNAVAILABLE,
                    ) from e
                except APIError:
                    # Re-raise APIError as-is
                    raise
                logger.info("Reflection Output:")
                logger.info("  - Response Grounded: %s", is_grounded)
                logger.info("  - Reflection Iterations: %d", response_reflection_counter.current_count)
                logger.info("  - Response Length: %d characters", len(final_response) if final_response else 0)
                logger.info("  - Response Preview: %s...", final_response[:100] if final_response else "")
                if not is_grounded:
                    logger.warning("  - Could not generate sufficiently grounded response after %d attempts",
                                 response_reflection_counter.current_count)
                logger.info("-" * 80)
                logger.info("=" * 80)
                logger.info("RAG PIPELINE COMPLETE")
                logger.info("=" * 80)
                
                return RAGResponse(
                    generate_answer_async(
                        _async_iter([final_response]),
                        context_to_show,
                        model=model,
                        collection_name=validated_collections[0] if validated_collections else "",
                        enable_citations=enable_citations,
                        context_reranker_time_ms=context_reranker_time_ms,
                        retrieval_time_ms=retrieval_time_ms,
                        rag_start_time_sec=rag_start_time_sec,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.SUCCESS,
                )
            else:
                logger.info("Starting LLM stream generation...")
                # Create async stream generator
                stream_gen = chain.astream(
                    {"question": query, "context": docs},
                    config={"run_name": "llm-stream"},
                )
                # Eagerly fetch first chunk to trigger any errors before returning response
                prefetched_stream = await self._eager_prefetch_astream(stream_gen)
                
                logger.info("LLM stream initiated successfully (first chunk received)")
                logger.info("-" * 80)
                logger.info("=" * 80)
                logger.info("RAG PIPELINE COMPLETE - Starting response streaming")
                logger.info("=" * 80)

                return RAGResponse(
                    generate_answer_async(
                        prefetched_stream,
                        context_to_show,
                        model=model,
                        collection_name=validated_collections[0] if validated_collections else "",
                        enable_citations=enable_citations,
                        context_reranker_time_ms=context_reranker_time_ms,
                        retrieval_time_ms=retrieval_time_ms,
                        rag_start_time_sec=rag_start_time_sec,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.SUCCESS,
                )

        except ConnectTimeout as e:
            logger.warning(
                "Connection timed out while making a request to the LLM endpoint: %s", e
            )
            return RAGResponse(
                generate_answer_async(
                    _async_iter(
                        [
                            "Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."
                        ]
                    ),
                    [],
                    model=model,
                    collection_name=collection_names[0] if collection_names else "",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.REQUEST_TIMEOUT,
            )

        except APIError as e:
            # APIError from any service (embedding, reranker, etc.) - convert to RAGResponse
            logger.warning("APIError in _rag_chain: %s", e.message)
            return RAGResponse(
                generate_answer_async(
                    _async_iter([e.message]),
                    [],
                    model=model,
                    collection_name=collection_names[0] if collection_names else "",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=e.status_code,
            )

        except (requests.exceptions.ConnectionError, ConnectionError, OSError):
            # Fallback for uncaught LLM connection errors
            llm_url = llm_settings.get("llm_endpoint") or self.config.llm.server_url
            error_msg = f"LLM NIM unavailable at {llm_url}. Please verify the service is running and accessible."
            logger.exception("Connection error (LLM)")
            return RAGResponse(
                generate_answer_async(
                    _async_iter([error_msg]),
                    [],
                    model=model,
                    collection_name=collection_names[0] if collection_names else "",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
            )

        except Exception as e:
            # Extract just the error type and message for cleaner logs
            error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
            logger.warning(
                "Failed to generate response due to exception: %s", error_msg
            )

            # Only show full traceback at DEBUG level
            if logger.getEffectiveLevel() <= logging.DEBUG:
                print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning(
                    "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                )
                return RAGResponse(
                    generate_answer_async(
                        _async_iter(
                            [
                                "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                            ]
                        ),
                        [],
                        model=model,
                        collection_name=collection_names[0] if collection_names else "",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.FORBIDDEN,
                )
            elif "[404] Not Found" in str(e):
                # Check if this is a VLM-related error
                requested_vlm_model = None
                if isinstance(vlm_settings, dict):
                    requested_vlm_model = vlm_settings.get("vlm_model")
                effective_vlm_model = requested_vlm_model or self.config.vlm.model_name

                if enable_vlm_inference and effective_vlm_model:
                    error_msg = (
                        f"VLM model '{effective_vlm_model}' not found. "
                        "Please verify the VLM model name and ensure it's available in your NVIDIA API account."
                    )
                    logger.warning(f"VLM model not found: {error_msg}")
                else:
                    error_msg = "Model or endpoint not found. Please verify the API endpoint and your payload. Ensure that the model name is valid."
                    logger.warning(f"Model not found: {error_msg}")

                return RAGResponse(
                    generate_answer_async(
                        _async_iter([error_msg]),
                        [],
                        model=model,
                        collection_name=collection_names[0] if collection_names else "",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.NOT_FOUND,
                )
            else:
                return RAGResponse(
                    generate_answer_async(
                        _async_iter([str(e)]),
                        [],
                        model=model,
                        collection_name=collection_names[0] if collection_names else "",
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.BAD_REQUEST,
                )

    def _print_conversation_history(
        self, conversation_history: list[str] = None, query: str | None = None
    ) -> None:
        if conversation_history is not None:
            for role, content in conversation_history:
                logger.debug("Role: %s", role)
                logger.debug("Content: %s\n", content)

    def _normalize_relevance_scores(
        self, documents: list["Document"]
    ) -> list["Document"]:
        """
        Normalize relevance scores in a list of documents to be between 0 and 1 using sigmoid function.

        Args:
            documents: List of Document objects with relevance_score in metadata

        Returns:
            The same list of documents with normalized scores
        """
        if not documents:
            return documents

        # Apply sigmoid normalization (1 / (1 + e^-x))
        for doc in documents:
            if "relevance_score" in doc.metadata:
                original_score = doc.metadata["relevance_score"]
                scaled_score = original_score * 0.1
                normalized_score = 1 / (1 + math.exp(-scaled_score))
                doc.metadata["relevance_score"] = normalized_score

        return documents

    def _format_document_with_source(self, doc: "Document") -> str:
        """Format document content with its source filename.

        Args:
            doc: Document object with metadata and page_content

        Returns:
            str: Formatted string with filename and content if ENABLE_SOURCE_METADATA is True,
                otherwise returns just the content
        """
        # Debug log before formatting
        logger.debug(f"Before format_document_with_source - Document: {doc}")

        # Check if source metadata is enabled via environment variable
        enable_metadata = os.getenv("ENABLE_SOURCE_METADATA", "True").lower() == "true"

        # Return just content if metadata is disabled or doc has no metadata
        if not enable_metadata or not hasattr(doc, "metadata"):
            result = doc.page_content
            logger.debug(
                f"After format_document_with_source (metadata disabled) - Result: {result}"
            )
            return result

        # Handle nested metadata structure
        source = doc.metadata.get("source", {})
        source_path = (
            source.get("source_name", "") if isinstance(source, dict) else source
        )

        # If no source path is found, return just the content
        if not source_path:
            result = doc.page_content
            logger.debug(
                f"After format_document_with_source (no source path) - Result: {result}"
            )
            return result

        filename = os.path.splitext(os.path.basename(source_path))[0]
        logger.debug(f"Before format_document_with_source - Filename: {filename}")
        result = f"File: {filename}\nContent: {doc.page_content}"

        # Debug log after formatting
        logger.debug(f"After format_document_with_source - Result: {result}")

        return result

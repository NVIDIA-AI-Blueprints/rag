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
1. generate(): Generate a response using the RAG chain.
2. search(): Search for the most relevant documents for the given search parameters.
3. get_summary(): Get the summary of a document.

Private methods:
1. __llm_chain: Execute a simple LLM chain using the components defined above.
2. __rag_chain: Execute a RAG chain using the components defined above.
3. __print_conversation_history: Print the conversation history.
4. __normalize_relevance_scores: Normalize the relevance scores of the documents.
5. __format_document_with_source: Format the document with the source.

"""

import json
import logging
import math
import os
import time
from collections.abc import Generator
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
    Citations,
    ErrorCodeMapping,
    RAGResponse,
    generate_answer,
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
    get_config,
    process_filter_expr,
    validate_filter_expr,
)
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.filter_expression_generator import (
    generate_filter_from_natural_language,
)
from nvidia_rag.utils.llm import get_llm, get_prompts, get_streaming_filter_think_parser
from nvidia_rag.utils.reranker import get_ranking_model
from nvidia_rag.utils.vdb import _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from observability.otel_metrics import OtelMetrics

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Get the model parameters from the config
model_params = CONFIG.llm.get_model_parameters()
default_max_tokens = model_params["max_tokens"]
default_temperature = model_params["temperature"]
default_top_p = model_params["top_p"]

document_embedder = get_embedding_model(
    model=CONFIG.embeddings.model_name, url=CONFIG.embeddings.server_url
)
ranker = get_ranking_model(
    model=CONFIG.ranking.model_name,
    url=CONFIG.ranking.server_url,
    top_n=CONFIG.retriever.top_k,
)
query_rewriter_llm_config = {"temperature": 0, "top_p": 0.1}
logger.info(
    "Query rewriter llm config: model name %s, url %s, config %s",
    CONFIG.query_rewriter.model_name,
    CONFIG.query_rewriter.server_url,
    query_rewriter_llm_config,
)
query_rewriter_llm = get_llm(
    model=CONFIG.query_rewriter.model_name,
    llm_endpoint=CONFIG.query_rewriter.server_url,
    **query_rewriter_llm_config,
)

# Initialize filter expression generator LLM
filter_generator_llm_config = {
    "temperature": CONFIG.filter_expression_generator.temperature,
    "top_p": CONFIG.filter_expression_generator.top_p,
    "max_tokens": CONFIG.filter_expression_generator.max_tokens,
}

filter_generator_llm = get_llm(
    model=CONFIG.filter_expression_generator.model_name,
    llm_endpoint=CONFIG.filter_expression_generator.server_url,
    **filter_generator_llm_config,
)

prompts = get_prompts()
vdb_top_k = int(CONFIG.retriever.vdb_top_k)

MAX_COLLECTION_NAMES = 5

# Get a StreamingFilterThinkParser based on configuration
StreamingFilterThinkParser = get_streaming_filter_think_parser()


class APIError(Exception):
    """Custom exception class for API errors."""

    def __init__(self, message: str, code: int = ErrorCodeMapping.BAD_REQUEST):
        logger.error("APIError occurred: %s with HTTP status: %d", message, code)
        print_exc()
        self.message = message
        self.code = code
        super().__init__(message)


class NvidiaRAG:
    def __init__(
        self,
        vdb_op: VDBRag = None,
    ):
        self.vdb_op = vdb_op

        if self.vdb_op is not None:
            if not isinstance(self.vdb_op, VDBRag):
                raise ValueError(
                    "vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag. "
                    "Please make sure all the required methods are implemented."
                )

    async def health(self, check_dependencies: bool = False) -> dict[str, Any]:
        """Check the health of the RAG server."""
        response_message = "Service is up."
        health_results = {}
        health_results["message"] = response_message

        vdb_op = self.__prepare_vdb_op()

        if check_dependencies:
            dependencies_results = await check_all_services_health(vdb_op)
            health_results.update(dependencies_results)
        return health_results

    def __prepare_vdb_op(
        self,
        vdb_endpoint: str = None,
        embedding_model: str = None,
        embedding_endpoint: str = None,
    ):
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
            model=embedding_model or CONFIG.embeddings.model_name,
            url=embedding_endpoint or CONFIG.embeddings.server_url,
        )

        return _get_vdb_op(
            vdb_endpoint=vdb_endpoint or CONFIG.vector_store.url,
            embedding_model=document_embedder,
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

    def generate(
        self,
        messages: list[dict[str, Any]],
        use_knowledge_base: bool = True,
        temperature: float = default_temperature,
        top_p: float = default_top_p,
        max_tokens: int = default_max_tokens,
        stop: list[str] = None,
        reranker_top_k: int = int(CONFIG.retriever.top_k),
        vdb_top_k: int = int(CONFIG.retriever.vdb_top_k),
        vdb_endpoint: str = None,
        collection_name: str = "",
        collection_names: list[str] = None,
        enable_query_rewriting: bool = CONFIG.query_rewriter.enable_query_rewriter,
        enable_reranker: bool = CONFIG.ranking.enable_reranker,
        enable_guardrails: bool = CONFIG.enable_guardrails,
        enable_citations: bool = CONFIG.enable_citations,
        enable_vlm_inference: bool = CONFIG.enable_vlm_inference,
        enable_filter_generator: bool = CONFIG.filter_expression_generator.enable_filter_generator,
        model: str = CONFIG.llm.model_name,
        llm_endpoint: str = CONFIG.llm.server_url,
        embedding_model: str = None,
        embedding_endpoint: str = None,
        reranker_model: str = CONFIG.ranking.model_name,
        reranker_endpoint: str = CONFIG.ranking.server_url,
        vlm_model: str = CONFIG.vlm.model_name,
        vlm_endpoint: str = CONFIG.vlm.server_url,
        filter_expr: str | list[dict[str, Any]] = "",
        enable_query_decomposition: bool = CONFIG.query_decomposition.enable_query_decomposition,
        confidence_threshold: float = CONFIG.default_confidence_threshold,
        rag_start_time_sec: float | None = None,
        metrics: OtelMetrics | None = None,
    ) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True` or `False`.

        Args:
            messages: List of conversation messages
            use_knowledge_base: Whether to use knowledge base for generation
            temperature: Sampling temperature for generation
            top_p: Top-p sampling mass
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            reranker_top_k: Number of documents to return after reranking
            vdb_top_k: Number of documents to retrieve from vector DB
            collection_name: Name of the collection to use
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

        vdb_op = self.__prepare_vdb_op(
            vdb_endpoint=vdb_endpoint,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
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
            collection_names = [CONFIG.vector_store.default_collection_name]

        query, chat_history = prepare_llm_request(messages)
        llm_settings = {
            "model": model,
            "llm_endpoint": llm_endpoint,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "enable_guardrails": enable_guardrails,
            "stop": stop,
        }

        if use_knowledge_base:
            logger.info("Using knowledge base to generate response.")
            return self.__rag_chain(
                llm_settings=llm_settings,
                query=query,
                chat_history=chat_history,
                reranker_top_k=reranker_top_k,
                vdb_top_k=vdb_top_k,
                collection_name=collection_name,
                collection_names=collection_names,
                enable_reranker=enable_reranker,
                reranker_model=reranker_model,
                reranker_endpoint=reranker_endpoint,
                enable_vlm_inference=enable_vlm_inference,
                vlm_model=vlm_model,
                vlm_endpoint=vlm_endpoint,
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
            logger.info(
                "Using LLM to generate response directly without knowledge base."
            )
            return self.__llm_chain(
                llm_settings=llm_settings,
                query=query,
                chat_history=chat_history,
                model=model,
                collection_name=collection_name,
                enable_citations=enable_citations,
                metrics=metrics,
            )

    def search(
        self,
        query: str,
        messages: list[dict[str, str]] = None,
        reranker_top_k: int = int(CONFIG.retriever.top_k),
        vdb_top_k: int = int(CONFIG.retriever.vdb_top_k),
        collection_name: str = "",
        collection_names: list[str] = None,
        vdb_endpoint: str = None,
        enable_query_rewriting: bool = CONFIG.query_rewriter.enable_query_rewriter,
        enable_reranker: bool = CONFIG.ranking.enable_reranker,
        enable_filter_generator: bool = CONFIG.filter_expression_generator.enable_filter_generator,
        embedding_model: str = None,
        embedding_endpoint: str = None,
        reranker_model: str = CONFIG.ranking.model_name,
        reranker_endpoint: str | None = CONFIG.ranking.server_url,
        filter_expr: str | list[dict[str, Any]] = "",
        confidence_threshold: float = CONFIG.default_confidence_threshold,
    ) -> Citations:
        """Search for the most relevant documents for the given search parameters.
        It's called when the `/search` API is invoked.

        Args:
            query (str): Query to be searched from vectorstore.
            messages (List[Dict[str, str]]): List of chat messages for context.
            reranker_top_k (int): Number of document chunks to retrieve after reranking.
            vdb_top_k (int): Number of top results to retrieve from vector database.
            collection_name (str): Name of the collection to be searched from vectorstore.
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

        logger.info("Searching relevant document for the query: %s", query)

        vdb_op = self.__prepare_vdb_op(
            vdb_endpoint=vdb_endpoint,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
        )

        if messages is None:
            messages = []
        if collection_names is None:
            collection_names = [CONFIG.vector_store.default_collection_name]

        # Validate top_k parameters
        vdb_top_k = validate_vdb_top_k(vdb_top_k)
        reranker_top_k = validate_reranker_k(reranker_top_k, vdb_top_k)

        # Normalize all model and endpoint values using validation functions
        reranker_model, reranker_endpoint = (
            validate_model_info(reranker_model, "reranker_model"),
            validate_model_info(reranker_endpoint, "reranker_endpoint"),
        )

        try:
            if collection_name:  # Would be deprecated in the future
                logger.warning(
                    "'collection_name' parameter is provided. This will be deprecated in the future. Use 'collection_names' instead."
                )
                collection_names = [collection_name]

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
                    filter_expr, collection_names, metadata_schemas
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
                        filter_expr, collection_name, metadata_schema_data
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
            local_ranker = get_ranking_model(
                model=reranker_model, url=reranker_endpoint, top_n=reranker_top_k
            )
            top_k = vdb_top_k if local_ranker and enable_reranker else reranker_top_k
            logger.info("Setting top k as: %s.", top_k)

            retriever_query = query
            # Query used for specific tasks (filter generation, reflection) - stays clean without history concatenation
            processed_query = retriever_query

            # Handle multi-turn conversations with two different strategies:
            # 1. Query rewriting: Creates a standalone, context-aware query (good for both retrieval and tasks)
            # 2. Query combination: Concatenates history for retrieval, keeps original for specific tasks
            if messages:
                if enable_query_rewriting:
                    # conversation is tuple so it should be multiple of two
                    # -1 is to keep last k conversation
                    history_count = (
                        int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
                    )
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
                    query_rewriter_prompt_config = prompts.get(
                        "query_rewriter_prompt", {}
                    )
                    system_prompt = query_rewriter_prompt_config.get(
                        "system", contextualize_q_system_prompt
                    )
                    human_prompt = query_rewriter_prompt_config.get("human", "{input}")

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
                        | query_rewriter_llm
                        | StreamingFilterThinkParser
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

                    retriever_query = q_prompt.invoke(
                        {"input": query, "chat_history": formatted_history}
                    )
                    logger.info("Rewritten Query: %s", retriever_query)

                    # When query rewriting is enabled, we can use it as processed_query for other modules
                    processed_query = retriever_query
                else:
                    # Query combination strategy: Concatenate history for better retrieval context
                    # Note: processed_query remains unchanged (original query) for clean task processing
                    user_queries = [
                        msg.get("content")
                        for msg in messages
                        if msg.get("role") == "user"
                    ]
                    retriever_query = ". ".join([*user_queries, query])
                    logger.info("Combined retriever query: %s", retriever_query)

            if enable_filter_generator:
                if CONFIG.vector_store.name != "milvus":
                    logger.warning(
                        f"Filter expression generator is currently only supported for Milvus. "
                        f"Current vector store: {CONFIG.vector_store.name}. Skipping filter generation."
                    )
                else:
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
                                        prompt_template=prompts.get(
                                            "filter_expression_generator_prompt"
                                        ),
                                        llm=filter_generator_llm,
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
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)
                docs, is_relevant = check_context_relevance(
                    vdb_op=vdb_op,
                    retriever_query=processed_query,
                    collection_names=validated_collections,
                    ranker=local_ranker,
                    reflection_counter=reflection_counter,
                    top_k=top_k,
                    enable_reranker=enable_reranker,
                    collection_filter_mapping=collection_filter_mapping,
                )
                if local_ranker and enable_reranker:
                    docs = self.__normalize_relevance_scores(docs)
                    if confidence_threshold > 0.0:
                        docs = filter_documents_by_confidence(
                            documents=docs,
                            confidence_threshold=confidence_threshold,
                        )
                if not is_relevant:
                    logger.warning(
                        "Could not find sufficiently relevant context after maximum attempts"
                    )
                return prepare_citations(retrieved_documents=docs, force_citations=True)
            else:
                if local_ranker and enable_reranker:
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to %s with the reranker for rag"
                        " chain.",
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
                    docs = context_reranker.invoke(
                        {"context": docs, "question": processed_query},
                        config={"run_name": "context_reranker"},
                    )
                    logger.info(
                        "    == Context reranker time: %.2f ms ==",
                        (time.time() - context_reranker_start_time) * 1000,
                    )

                    # Normalize scores to 0-1 range"
                    docs = self.__normalize_relevance_scores(docs.get("context", []))
                    if confidence_threshold > 0.0:
                        docs = filter_documents_by_confidence(
                            documents=docs,
                            confidence_threshold=confidence_threshold,
                        )

                    return prepare_citations(
                        retrieved_documents=docs, force_citations=True
                    )

            docs = vdb_op.retrieval_langchain(
                query=retriever_query,
                collection_name=validated_collections[0],
                vectorstore=vdb_op.get_langchain_vectorstore(validated_collections[0]),
                top_k=top_k,
                filter_expr=collection_filter_mapping.get(validated_collections[0], ""),
                otel_ctx=otel_ctx,
            )
            # TODO: Check how to get the relevance score from milvus
            return prepare_citations(retrieved_documents=docs, force_citations=True)

        except Exception as e:
            raise APIError(f"Failed to search documents. {str(e)}") from e

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
        system_prompt = prompts.get(template_key, {}).get("system", "")
        # Support both "human" and "user" keys with fallback
        template_dict = prompts.get(template_key, {})
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

    def __llm_chain(
        self,
        llm_settings: dict[str, Any],
        query: str | list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        model: str = "",
        collection_name: str = "",
        enable_citations: bool = True,
        metrics: OtelMetrics | None = None,
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `False`.

        Args:
            llm_settings: Dictionary containing LLM settings
            query: The user's query
            chat_history: List of conversation messages
            model: Name of the model used for generation
            collection_name: Name of the collection used for retrieval
            enable_citations: Whether to enable citations in the response
        """
        try:
            # Limit conversation history to prevent overwhelming the model
            # conversation is tuple so it should be multiple of two
            # -1 is to keep last k conversation
            history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
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

            self.__print_conversation_history(message, query_text)

            prompt_template = ChatPromptTemplate.from_messages(message)
            llm = get_llm(**llm_settings)

            chain = (
                prompt_template | llm | StreamingFilterThinkParser | StrOutputParser()
            )
            return RAGResponse(
                generate_answer(
                    chain.stream(
                        {"question": query_text}, config={"run_name": "llm-stream"}
                    ),
                    [],
                    model=model,
                    collection_name=collection_name,
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
                generate_answer(
                    iter(
                        [
                            "Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."
                        ]
                    ),
                    [],
                    model=model,
                    collection_name=collection_name,
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.REQUEST_TIMEOUT,
            )

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning(
                    "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                )
                return RAGResponse(
                    generate_answer(
                        iter(
                            [
                                "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                            ]
                        ),
                        [],
                        model=model,
                        collection_name=collection_name,
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
                    generate_answer(
                        iter([error_msg]),
                        [],
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.NOT_FOUND,
                )
            else:
                return RAGResponse(
                    generate_answer(
                        iter([f"{str(e)}"]),
                        [],
                        model=model,
                        collection_name=collection_name,
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

    def _build_retriever_query_from_content(self, content: Any) -> str:
        """Build retriever query from either string or multimodal content.
        For multimodal content, includes both text and base64 images for VLM embedding support.

        Args:
            content: Either a string or a list of content objects (multimodal)

        Returns:
            str: Query string that may include base64 image data for VLM embeddings
        """
        if isinstance(content, str):
            return content
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
                            # Include the base64 image data for VLM embedding
                            query_parts.append(image_url)
            return "\n\n".join(query_parts)
        else:
            # Fallback for any other content type
            return str(content) if content is not None else ""

    def __rag_chain(
        self,
        llm_settings: dict[str, Any],
        query: str | list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        reranker_top_k: int = 10,
        vdb_top_k: int = 40,
        collection_name: str = "",
        collection_names: list[str] = None,
        enable_reranker: bool = True,
        reranker_model: str = "",
        reranker_endpoint: str | None = None,
        enable_vlm_inference: bool = False,
        vlm_model: str = "",
        vlm_endpoint: str = "",
        model: str = "",
        enable_query_rewriting: bool = False,
        enable_citations: bool = True,
        filter_expr: str | list[dict[str, Any]] | None = "",
        enable_filter_generator: bool = False,
        vdb_op: VDBRag = None,
        enable_query_decomposition: bool = False,
        confidence_threshold: float = CONFIG.default_confidence_threshold,
        rag_start_time_sec: float | None = None,
        metrics: OtelMetrics | None = None,
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        """Execute a RAG chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True`.

        Args:
            llm_settings: Dictionary containing LLM settings
            query: The user's query
            chat_history: List of conversation messages
            reranker_top_k: Number of documents to return after reranking
            vdb_top_k: Number of documents to retrieve from vector DB
            collection_name: Name of the collection to use
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
            # If collection_name is provided, use it as the collection name,
            # Otherwise, use the collection names from the kwargs
            if collection_name:  # Would be deprecated in the future
                logger.warning(
                    "'collection_name' parameter is provided. This will be deprecated in the future. Use 'collection_names' instead."
                )
                collection_names = [collection_name]
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
                    filter_expr, collection_names, metadata_schemas
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
                        filter_expr, collection_name, metadata_schema_data
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

            llm = get_llm(**llm_settings)
            logger.info("Ranker enabled: %s", enable_reranker)
            ranker = get_ranking_model(
                model=reranker_model, url=reranker_endpoint, top_n=reranker_top_k
            )
            top_k = vdb_top_k if ranker and enable_reranker else reranker_top_k
            logger.info("Setting retriever top k as: %s.", top_k)

            # conversation is tuple so it should be multiple of two
            # -1 is to keep last k conversation
            history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
            chat_history = chat_history[history_count:]
            retrieval_time_ms = None
            context_reranker_time_ms = None

            # Use the new prompt processing method
            (
                system_message,
                conversation_history,
                user_message,
            ) = self._handle_prompt_processing(chat_history, model, "rag_template")
            vlm_message = []
            logger.debug("System message: %s", system_message)
            logger.debug("User message: %s", user_message)
            logger.debug("Conversation history: %s", conversation_history)
            # Build retriever query from multimodal content (includes text and base64 images for VLM embedding)
            retriever_query = self._build_retriever_query_from_content(query)
            # Query used for specific tasks (filter generation, reflection) - stays clean without history concatenation
            processed_query = retriever_query

            # Handle multi-turn conversations with two different strategies:
            # 1. Query rewriting: Creates a standalone, context-aware query (good for both retrieval and tasks)
            # 2. Query combination: Concatenates history for retrieval, keeps original for specific tasks
            if chat_history:
                if enable_query_rewriting:
                    # Based on conversation history recreate query for better
                    # document retrieval
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                    query_rewriter_prompt_config = prompts.get(
                        "query_rewriter_prompt", {}
                    )
                    system_prompt = query_rewriter_prompt_config.get(
                        "system", contextualize_q_system_prompt
                    )
                    human_prompt = query_rewriter_prompt_config.get("human", "{input}")

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
                        | query_rewriter_llm
                        | StreamingFilterThinkParser
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

                    retriever_query = q_prompt.invoke(
                        {
                            "input": retriever_query,
                            "chat_history": formatted_history,
                        },
                        config={"run_name": "query-rewriter"},
                    )
                    logger.info(
                        "Rewritten Query: %s %s", retriever_query, len(retriever_query)
                    )

                    # When query rewriting is enabled, we can use it as processed_query for other modules
                    processed_query = retriever_query
                else:
                    # Query combination strategy: Concatenate history for better retrieval context
                    # Note: processed_query remains unchanged (original query) for clean task processing
                    user_queries = [
                        self._build_retriever_query_from_content(msg.get("content"))
                        for msg in chat_history
                        if msg.get("role") == "user"
                    ][-1:]
                    # TODO: Find a better way to join this when queries already
                    # have punctuation
                    retriever_query = ". ".join([*user_queries, retriever_query])
                    logger.info("Combined retriever query: %s", retriever_query)

            if enable_filter_generator:
                if CONFIG.vector_store.name != "milvus":
                    logger.warning(
                        f"Filter expression generator is currently only supported for Milvus. "
                        f"Current vector store: {CONFIG.vector_store.name}. Skipping filter generation."
                    )
                else:
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
                                        prompt_template=prompts.get(
                                            "filter_expression_generator_prompt"
                                        ),
                                        llm=filter_generator_llm,
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
                            logger.debug(
                                f"Generated filter expressions for {generated_count}/{len(validated_collections)} collections"
                            )
                        else:
                            logger.debug(
                                "No filter expressions generated for any collection"
                            )

                    except Exception as e:
                        logger.warning(f"Error generating filter expression: {str(e)}")

            if enable_query_decomposition:
                logger.info("Using query decomposition for complex query processing")
                # TODO: Pass processed_query instead of query and check accuracy
                return iterative_query_decomposition(
                    query=query,
                    history=conversation_history,
                    llm=llm,
                    vdb_op=vdb_op,
                    ranker=ranker if enable_reranker else None,
                    recursion_depth=CONFIG.query_decomposition.recursion_depth,
                    enable_citations=enable_citations,
                    collection_name=validated_collections[0]
                    if validated_collections
                    else "",
                    top_k=top_k,
                    ranker_top_k=reranker_top_k,
                    confidence_threshold=confidence_threshold,
                    llm_settings=llm_settings,
                )

            if confidence_threshold > 0.0 and not enable_reranker:
                logger.warning(
                    f"confidence_threshold is set to {confidence_threshold} but enable_reranker is explicitly set to False. "
                    f"Confidence threshold filtering requires reranker to be enabled to generate relevance scores. "
                    f"Consider setting enable_reranker=True for effective filtering."
                )

            # Get relevant documents with optional reflection
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)

                context_to_show, is_relevant = check_context_relevance(
                    vdb_op=vdb_op,
                    retriever_query=processed_query,
                    collection_names=validated_collections,
                    ranker=ranker,
                    reflection_counter=reflection_counter,
                    top_k=top_k,
                    enable_reranker=enable_reranker,
                    collection_filter_mapping=collection_filter_mapping,
                )

                # Normalize scores to 0-1 range
                if ranker and enable_reranker:
                    context_to_show = self.__normalize_relevance_scores(context_to_show)

                if not is_relevant:
                    logger.warning(
                        "Could not find sufficiently relevant context after %d attempts",
                        reflection_counter.current_count,
                    )
            else:
                otel_ctx = otel_context.get_current()
                if ranker and enable_reranker:
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to "
                        "%s with the reranker for rag chain.",
                        top_k,
                        reranker_top_k,
                    )
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)
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
                    logger.debug(
                        "Using retriever query for retrieval %s", retriever_query
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
                    logger.info(
                        "== Total retrieval time: %.2f ms ==", retrieval_time_ms
                    )

                    context_reranker_start_time = time.time()
                    logger.debug(
                        "Using processed query for reranker %s", processed_query
                    )
                    docs = context_reranker.invoke(
                        {"context": docs, "question": processed_query},
                        config={"run_name": "context_reranker"},
                    )
                    context_reranker_time_ms = (
                        time.time() - context_reranker_start_time
                    ) * 1000
                    logger.info(
                        "    == Context reranker time: %.2f ms ==",
                        context_reranker_time_ms,
                    )
                    context_to_show = docs.get("context", [])
                    # Normalize scores to 0-1 range
                    context_to_show = self.__normalize_relevance_scores(context_to_show)
                else:
                    # Multiple retrievers are not supported when reranking is disabled
                    retrieval_start_time = time.time()
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

            if ranker and enable_reranker and confidence_threshold > 0.0:
                context_to_show = filter_documents_by_confidence(
                    documents=context_to_show,
                    confidence_threshold=confidence_threshold,
                )

            if enable_vlm_inference:
                # Fast pre-check: skip VLM entirely if no images in query or context
                has_images_in_query = self._contains_images(query)
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

                if not (has_images_in_query or has_images_in_context):
                    logger.warning(
                        "Skipping VLM: no images found in query or retrieved context."
                    )
                    # fall through without VLM
                else:
                    logger.info("Calling VLM to analyze images cited in the context")
                    vlm_response: str = ""
                    try:
                        vlm = VLM(vlm_model, vlm_endpoint)
                        vlm_response = vlm.analyze_images_from_context(
                            context_to_show, query
                        )

                        vlm_response_stripped = (
                            vlm_response.strip()
                            if isinstance(vlm_response, str)
                            else ""
                        )
                        if vlm_response_stripped:
                            if CONFIG.vlm.enable_vlm_response_reasoning:
                                should_use_vlm_response = vlm.reason_on_vlm_response(
                                    query,
                                    vlm_response_stripped,
                                    context_to_show,
                                    llm_settings,
                                )
                            else:
                                # Reasoning gate disabled: always include VLM output
                                should_use_vlm_response = True

                            # If query contains images, or vlm_response_as_final_answer is enabled, return as final answer
                            if should_use_vlm_response and (
                                CONFIG.vlm.vlm_response_as_final_answer
                                or self._contains_images(query)
                            ):
                                logger.info(
                                    "VLM response as final answer: %s",
                                    vlm_response_stripped,
                                )
                                return RAGResponse(
                                    generate_answer(
                                        iter([vlm_response_stripped]),
                                        context_to_show,
                                        model=model,
                                        collection_name=collection_name,
                                        enable_citations=enable_citations,
                                    ),
                                    status_code=ErrorCodeMapping.SUCCESS,
                                )
                            else:
                                logger.info("Query type: %s", type(query))
                                logger.info(
                                    "VLM response not as final answer: %s",
                                    vlm_response_stripped,
                                )

                            if should_use_vlm_response:
                                logger.info(
                                    "VLM response validated and added to prompt: %s",
                                    vlm_response_stripped,
                                )
                                injection_tmpl = prompts.get(
                                    "vlm_response_injection_template",
                                    "The following is an answer generated by a Vision-Language Model (VLM) based solely on images cited in the context:\n---\n{vlm_response}\n---\nConsider this visual insight when answering the user's query, especially where the textual context is ambiguous or limited.",
                                )
                                vlm_response_prompt = injection_tmpl.format(
                                    vlm_response=vlm_response_stripped
                                )
                                vlm_message += [("user", vlm_response_prompt)]
                            else:
                                logger.info("VLM response skipped by reasoning gate.")
                        else:
                            logger.info("VLM response is empty.")
                    except (OSError, ValueError) as e:
                        logger.warning(
                            "VLM processing failed for query='%s', collection='%s': %s",
                            query,
                            collection_name,
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
                            collection_name,
                            e,
                            exc_info=True,
                        )
                        # Provide specific error message for unexpected VLM issues
                        vlm_error_msg = f"Unexpected VLM error: {str(e)}. Please check your VLM configuration and try again."
                        # Don't yield here, let the exception propagate to be caught by the server
                        raise APIError(
                            vlm_error_msg, ErrorCodeMapping.BAD_REQUEST
                        ) from e

            docs = [self.__format_document_with_source(d) for d in context_to_show]

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

            # Add vlm response and user query to prompt
            message += vlm_message
            user_query = [("user", "Query: {question}\n\nAnswer: ")]
            message += user_query

            self.__print_conversation_history(message)
            prompt = ChatPromptTemplate.from_messages(message)

            chain = prompt | llm | StreamingFilterThinkParser | StrOutputParser()

            # Check response groundedness if we still have reflection
            # iterations available
            if (
                os.environ.get("ENABLE_REFLECTION", "false").lower() == "true"
                and reflection_counter.remaining > 0
            ):
                initial_response = chain.invoke({"question": query, "context": docs})
                final_response, is_grounded = check_response_groundedness(
                    query, initial_response, docs, reflection_counter
                )
                if not is_grounded:
                    logger.warning(
                        "Could not generate sufficiently grounded response after %d total reflection attempts",
                        reflection_counter.current_count,
                    )
                return RAGResponse(
                    generate_answer(
                        iter([final_response]),
                        context_to_show,
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        context_reranker_time_ms=context_reranker_time_ms,
                        retrieval_time_ms=retrieval_time_ms,
                        rag_start_time_sec=rag_start_time_sec,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.SUCCESS,
                )
            else:
                return RAGResponse(
                    generate_answer(
                        chain.stream(
                            {"question": query, "context": docs},
                            config={"run_name": "llm-stream"},
                        ),
                        context_to_show,
                        model=model,
                        collection_name=collection_name,
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
                generate_answer(
                    iter(
                        [
                            "Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."
                        ]
                    ),
                    [],
                    model=model,
                    collection_name=collection_name,
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.REQUEST_TIMEOUT,
            )

        except requests.exceptions.ConnectionError as e:
            if "HTTPConnectionPool" in str(e):
                logger.exception(
                    "Connection pool error while connecting to service: %s", e
                )
                return RAGResponse(
                    generate_answer(
                        iter(
                            [
                                "Connection error: Failed to connect to service. Please verify if all required NIMs are running and accessible."
                            ]
                        ),
                        [],
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
                )

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning(
                    "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                )
                return RAGResponse(
                    generate_answer(
                        iter(
                            [
                                "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                            ]
                        ),
                        [],
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.FORBIDDEN,
                )
            elif "[404] Not Found" in str(e):
                # Check if this is a VLM-related error
                if enable_vlm_inference and vlm_model:
                    error_msg = f"VLM model '{vlm_model}' not found. Please verify the VLM model name and ensure it's available in your NVIDIA API account."
                    logger.warning(f"VLM model not found: {error_msg}")
                else:
                    error_msg = "Model or endpoint not found. Please verify the API endpoint and your payload. Ensure that the model name is valid."
                    logger.warning(f"Model not found: {error_msg}")

                return RAGResponse(
                    generate_answer(
                        iter([error_msg]),
                        [],
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.NOT_FOUND,
                )
            else:
                return RAGResponse(
                    generate_answer(
                        iter(
                            [
                                f"{str(e)}"
                            ]
                        ),
                        [],
                        model=model,
                        collection_name=collection_name,
                        enable_citations=enable_citations,
                        otel_metrics_client=metrics,
                    ),
                    status_code=ErrorCodeMapping.BAD_REQUEST,
                )

    def __print_conversation_history(
        self, conversation_history: list[str] = None, query: str | None = None
    ):
        if conversation_history is not None:
            for role, content in conversation_history:
                logger.debug("Role: %s", role)
                logger.debug("Content: %s\n", content)

    def __normalize_relevance_scores(
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

    def __format_document_with_source(self, doc) -> str:
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

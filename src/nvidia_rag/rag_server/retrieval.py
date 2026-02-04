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

"""Document retrieval, reranking, and filter processing logic.

This module provides standalone functions for document retrieval operations:
- Document retrieval from vector databases
- Reranking of retrieved documents
- Filter expression processing (natural language to structured)
- Query rewriting using LLM
- Multi-collection result aggregation

All functions are designed to be independent of class state and take explicit parameters.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from opentelemetry import context as otel_context

from nvidia_rag.rag_server.document_formatter import _normalize_relevance_scores
from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.common import (
    filter_documents_by_confidence,
    process_filter_expr,
    validate_filter_expr,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.filter_expression_generator import (
    generate_filter_from_natural_language,
)
from nvidia_rag.utils.reranker import get_ranking_model
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


def process_filter_expressions(
    filter_expr: str | dict,
    collection_names: list[str],
    metadata_schemas: dict[str, Any],
    config: NvidiaRAGConfig,
) -> tuple[dict[str, str], list[str]]:
    """Process filter expressions for multiple collections.

    Validates and processes filter expressions for each collection, handling
    metadata schemas and collection-specific requirements.

    Args:
        filter_expr: Filter expression (string or dict) to process
        collection_names: List of collection names to process
        metadata_schemas: Metadata schemas for each collection
        config: Configuration object

    Returns:
        Tuple of (collection_filter_mapping, validated_collections):
            - collection_filter_mapping: Dict mapping collection name to processed filter
            - validated_collections: List of collections that support the filter

    Raises:
        APIError: If filter expression validation fails
    """
    if not filter_expr or (isinstance(filter_expr, str) and filter_expr.strip() == ""):
        validation_result = {
            "status": True,
            "validated_collections": collection_names,
        }
    else:
        validation_result = validate_filter_expr(
            filter_expr, collection_names, metadata_schemas, config=config
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
            name for name in collection_names if name not in validated_collections
        ]
        logger.info(
            f"Collections {skipped_collections} do not support the filter expression and will be skipped"
        )

    if not filter_expr or (isinstance(filter_expr, str) and filter_expr.strip() == ""):
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
                config=config,
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
                collection_filter_mapping[collection_name] = processed_filter_expr

    return collection_filter_mapping, validated_collections


def generate_filter_expressions(
    query: str,
    validated_collections: list[str],
    metadata_schemas: dict[str, Any],
    filter_generator_llm: Any,
    prompts: dict[str, Any],
    existing_filter_expr: str | dict,
    config: NvidiaRAGConfig,
) -> dict[str, str]:
    """Generate filter expressions from natural language query.

    Uses LLM to generate structured filter expressions from natural language
    queries, processing each collection in parallel.

    Args:
        query: Natural language query to generate filters from
        validated_collections: List of collection names to generate filters for
        metadata_schemas: Metadata schemas for each collection
        filter_generator_llm: LLM instance for filter generation
        prompts: Prompt templates dictionary
        existing_filter_expr: Existing filter expression to merge with
        config: Configuration object

    Returns:
        Dict mapping collection name to generated filter expression

    Raises:
        APIError: If filter generator LLM is unavailable
    """
    if config.vector_store.name != "milvus":
        logger.warning(
            f"Filter expression generator is currently only supported for Milvus. "
            f"Current vector store: {config.vector_store.name}. Skipping filter generation."
        )
        return dict.fromkeys(validated_collections, "")

    if filter_generator_llm is None:
        raise APIError(
            "Filter expression generator is enabled but the filter generator NIM is unavailable. "
            f"Please verify the service is running at {config.filter_expression_generator.server_url}.",
            ErrorCodeMapping.SERVICE_UNAVAILABLE,
        )

    logger.debug(
        "Filter expression generator enabled, attempting to generate filter from query"
    )

    collection_filter_mapping = {}

    def generate_filter_for_collection(collection_name):
        try:
            metadata_schema_data = metadata_schemas.get(collection_name)

            generated_filter = generate_filter_from_natural_language(
                user_request=query,
                collection_name=collection_name,
                metadata_schema=metadata_schema_data,
                prompt_template=prompts.get("filter_expression_generator_prompt"),
                llm=filter_generator_llm,
                existing_filter_expr=existing_filter_expr,
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
                    config=config,
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

    try:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_filter_for_collection, collection_name)
                for collection_name in validated_collections
            ]

            for future in futures:
                collection_name, processed_filter_expr = future.result()
                collection_filter_mapping[collection_name] = processed_filter_expr

        generated_count = len([f for f in collection_filter_mapping.values() if f])
        if generated_count > 0:
            logger.info(
                f"Generated filter expressions for {generated_count}/{len(validated_collections)} collections"
            )
        else:
            logger.info("No filter expressions generated for any collection")

    except Exception as e:
        logger.error(f"Error generating filter expression: {str(e)}")
        # Return empty filters for all collections on error
        return dict.fromkeys(validated_collections, "")

    return collection_filter_mapping


async def perform_query_rewriting(
    query: str,
    messages: list[dict[str, Any]],
    query_rewriter_llm: Any,
    streaming_filter_think_parser: Any,
    prompts: dict[str, Any],
    config: NvidiaRAGConfig,
    conversation_history_count: int,
) -> str:
    """Rewrite query using conversation history for better retrieval.

    Takes a query and conversation history, reformulates the query into a
    standalone question that can be understood without the chat history.

    Args:
        query: Original query to rewrite
        messages: Conversation history messages
        query_rewriter_llm: LLM instance for query rewriting
        streaming_filter_think_parser: Parser for streaming output
        prompts: Prompt templates dictionary
        config: Configuration object
        conversation_history_count: Number of conversation turns to include

    Returns:
        Rewritten query string

    Raises:
        APIError: If query rewriter LLM is unavailable
    """
    # Skip query rewriting if conversation history is disabled
    if conversation_history_count == 0:
        logger.warning(
            "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0. "
            "Query rewriting requires conversation history to work effectively. "
            "Skipping query rewriting. Set CONVERSATION_HISTORY > 0 to enable query rewriting."
        )
        return query

    # Check if query rewriter is available
    if query_rewriter_llm is None:
        raise APIError(
            "Query rewriting is enabled but the query rewriter NIM is unavailable. "
            f"Please verify the service is running at {config.query_rewriter.server_url}.",
            ErrorCodeMapping.SERVICE_UNAVAILABLE,
        )

    # Extract conversation history
    # conversation is tuple so it should be multiple of two
    # -1 is to keep last k conversation
    history_count = conversation_history_count * 2 * -1
    messages = messages[history_count:]
    conversation_history = []

    for message in messages:
        if message.get("role") != "system":
            conversation_history.append((message.get("role"), message.get("content")))

    # Based on conversation history recreate query for better document retrieval
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    query_rewriter_prompt_config = prompts.get("query_rewriter_prompt", {})
    system_prompt = query_rewriter_prompt_config.get(
        "system", contextualize_q_system_prompt
    )
    human_prompt = query_rewriter_prompt_config.get("human", "{input}")

    # Format conversation history as a string
    formatted_history = ""
    if conversation_history:
        formatted_history = "\n".join(
            [f"{role.capitalize()}: {content}" for role, content in conversation_history]
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
        | streaming_filter_think_parser
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
        rewritten_query = await q_prompt.ainvoke(
            {"input": query, "chat_history": formatted_history}
        )
    except (ConnectionError, OSError, Exception) as e:
        # Wrap connection errors from query rewriter LLM
        if isinstance(e, APIError):
            raise
        query_rewriter_url = config.query_rewriter.server_url
        endpoint_msg = f" at {query_rewriter_url}" if query_rewriter_url else ""
        raise APIError(
            f"Query rewriter LLM NIM unavailable{endpoint_msg}. Please verify the service is running and accessible or disable query rewriting.",
            ErrorCodeMapping.SERVICE_UNAVAILABLE,
        ) from e

    logger.info("Rewritten Query: %s", rewritten_query)
    return rewritten_query


def aggregate_multi_collection_results(
    vdb_op: VDBRag,
    query: str,
    validated_collections: list[str],
    collection_filter_mapping: dict[str, str],
    top_k: int,
    otel_ctx: Any = None,
) -> list[Document]:
    """Retrieve and aggregate documents from multiple collections.

    Performs parallel retrieval from multiple vector store collections and
    aggregates the results.

    Args:
        vdb_op: Vector database operator instance
        query: Query string for retrieval
        validated_collections: List of collection names to retrieve from
        collection_filter_mapping: Dict mapping collection name to filter expression
        top_k: Number of documents to retrieve per collection
        otel_ctx: OpenTelemetry context for tracing (optional)

    Returns:
        List of retrieved documents from all collections
    """
    docs = []
    vectorstores = []
    for collection_name in validated_collections:
        vectorstores.append(vdb_op.get_langchain_vectorstore(collection_name))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                vdb_op.retrieval_langchain,
                query=query,
                collection_name=collection_name,
                vectorstore=vectorstore,
                top_k=top_k,
                filter_expr=collection_filter_mapping.get(collection_name, ""),
                otel_ctx=otel_ctx,
            )
            for collection_name, vectorstore in zip(
                validated_collections, vectorstores, strict=False
            )
        ]
        for future in futures:
            docs.extend(future.result())

    return docs


async def retrieve_and_rerank_documents(
    vdb_op: VDBRag,
    query: str,
    validated_collections: list[str],
    collection_filter_mapping: dict[str, str],
    ranker: Any,
    enable_reranker: bool,
    top_k: int,
    reranker_top_k: int,
    confidence_threshold: float,
    is_image_query: bool = False,
    config: NvidiaRAGConfig = None,
) -> list[Document]:
    """Retrieve documents from vector database and optionally rerank them.

    Handles document retrieval with support for:
    - Single or multiple collections
    - Optional reranking with score normalization
    - Confidence threshold filtering
    - Image query handling

    Args:
        vdb_op: Vector database operator instance
        query: Query string for retrieval
        validated_collections: List of collection names to retrieve from
        collection_filter_mapping: Dict mapping collection name to filter expression
        ranker: Reranker model instance (can be None if reranking disabled)
        enable_reranker: Whether to enable reranking
        top_k: Number of documents to retrieve from VDB
        reranker_top_k: Number of documents to keep after reranking
        confidence_threshold: Minimum confidence score for documents (0.0 to disable)
        is_image_query: Whether this is an image query
        config: Configuration object (optional, for error handling)

    Returns:
        List of retrieved and optionally reranked documents

    Raises:
        APIError: If reranker service is unavailable
    """
    otel_ctx = otel_context.get_current()

    # Handle case where reranker is disabled or image query
    if not ranker or not enable_reranker or is_image_query:
        if is_image_query:
            docs = vdb_op.retrieval_image_langchain(
                query=query,
                collection_name=validated_collections[0],
                vectorstore=vdb_op.get_langchain_vectorstore(validated_collections[0]),
                top_k=top_k,
                reranker_top_k=reranker_top_k,
            )
        else:
            docs = vdb_op.retrieval_langchain(
                query=query,
                collection_name=validated_collections[0],
                vectorstore=vdb_op.get_langchain_vectorstore(validated_collections[0]),
                top_k=top_k,
                filter_expr=collection_filter_mapping.get(validated_collections[0], ""),
                otel_ctx=otel_ctx,
            )
        return docs

    # Reranking is enabled
    logger.info(
        "Narrowing the collection from %s results and further narrowing it to %s with the reranker",
        top_k,
        reranker_top_k,
    )
    logger.info("Setting ranker top n as: %s.", reranker_top_k)
    # Update number of document to be retrieved by ranker
    ranker.top_n = reranker_top_k

    context_reranker = RunnableAssign(
        {
            "context": lambda input: ranker.compress_documents(
                query=input["question"], documents=input["context"]
            )
        }
    )

    # Perform parallel retrieval from all vector stores with their specific filter expressions
    docs = aggregate_multi_collection_results(
        vdb_op=vdb_op,
        query=query,
        validated_collections=validated_collections,
        collection_filter_mapping=collection_filter_mapping,
        top_k=top_k,
        otel_ctx=otel_ctx,
    )

    context_reranker_start_time = time.time()
    try:
        docs = await context_reranker.ainvoke(
            {"context": docs, "question": query},
            config={"run_name": "context_reranker"},
        )
    except (
        requests.exceptions.ConnectionError,
        ConnectionError,
        OSError,
    ) as e:
        if config:
            reranker_url = config.ranking.server_url
            error_msg = f"Reranker NIM unavailable at {reranker_url}. Please verify the service is running and accessible."
        else:
            error_msg = (
                "Reranker NIM unavailable. Please verify the service is running."
            )
        logger.error("Connection error in reranker: %s", e)
        raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e

    logger.info(
        "    == Context reranker time: %.2f ms ==",
        (time.time() - context_reranker_start_time) * 1000,
    )

    # Normalize scores to 0-1 range
    docs = _normalize_relevance_scores(docs.get("context", []))

    # Apply confidence threshold filtering if enabled
    if confidence_threshold > 0.0:
        docs = filter_documents_by_confidence(
            documents=docs,
            confidence_threshold=confidence_threshold,
        )

    return docs


def initialize_reranker(
    enable_reranker: bool,
    reranker_model: str,
    reranker_endpoint: str,
    reranker_top_k: int,
    config: NvidiaRAGConfig,
) -> Any:
    """Initialize reranker model if enabled.

    Args:
        enable_reranker: Whether to enable reranking
        reranker_model: Reranker model name
        reranker_endpoint: Reranker endpoint URL
        reranker_top_k: Number of documents to keep after reranking
        config: Configuration object

    Returns:
        Reranker model instance or None if disabled

    Raises:
        APIError: If reranker service is unavailable
    """
    if not enable_reranker:
        return None

    try:
        return get_ranking_model(
            model=reranker_model,
            url=reranker_endpoint,
            top_n=reranker_top_k,
            config=config,
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
        reranker_url = reranker_endpoint or config.ranking.server_url
        error_msg = f"Reranker NIM unavailable at {reranker_url}. Please verify the service is running and accessible."
        logger.error("Connection error in reranker initialization: %s", e)
        raise APIError(
            error_msg,
            ErrorCodeMapping.SERVICE_UNAVAILABLE,
        ) from e

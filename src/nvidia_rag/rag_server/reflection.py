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
"""
This module contains the reflection logic for the RAG server.

Reflection is a technique used to improve the quality of the generated response by checking
the relevance of the retrieved context and the groundedness of the generated response.
The module uses deterministic sampling parameters and structured prompts for consistent,
reproducible reflection results.

Components:
1. ReflectionCounter: A class that tracks the number of reflection iterations across query rewrites and response regeneration.
2. check_context_relevance: Check relevance of retrieved context and optionally rewrite query for better results.
3. check_response_groundedness: Check groundedness of generated response against retrieved context.
4. _retry_score_generation: Helper method to retry score generation with error handling.

Features:
- Deterministic reflection using temperature=0 and low top_p for reproducible results
- Large context window support (32K tokens) for comprehensive analysis
- Structured prompts with system and human message pairs for precise instruction following
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from opentelemetry import context as otel_context

from nvidia_rag.utils.common import get_env_variable
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)
prompts = get_prompts()


def _retry_score_generation(
    chain, inputs: dict[str, Any], max_retries: int = 3, config: dict[str, Any] = None
) -> int:
    """Helper method to retry score generation with error handling.

    Args:
        chain: The chain to execute
        inputs: Input dictionary for the chain
        max_retries: Maximum number of retry attempts

    Returns:
        int: Generated score (0, 1, or 2), or 0 if all retries fail
    """
    if config is None:
        config = {}
    for retry in range(max_retries):
        try:
            response = chain.invoke(inputs, config=config)
            # Extract numeric score from response
            for score in [2, 1, 0]:
                if str(score) in response:
                    return score
        except Exception as e:
            logger.warning(f"Retry {retry + 1}/{max_retries} failed: {str(e)}")
            if retry == max_retries - 1:
                logger.error("All retries failed for score generation")
                return 0
            continue
    return 0


class ReflectionCounter:
    """Tracks the number of reflection iterations across query rewrites and response regeneration."""

    def __init__(self, max_loops: int):
        self.max_loops = max_loops
        self.current_count = 0

    def increment(self) -> bool:
        """Increment counter and return whether we can continue."""
        if self.current_count >= self.max_loops:
            return False
        self.current_count += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_loops - self.current_count)


def check_context_relevance(
    vdb_op: VDBRag,
    retriever_query: str,
    collection_names: list[str],
    ranker,
    reflection_counter: ReflectionCounter,
    top_k: int = 10,
    enable_reranker: bool = True,
    collection_filter_mapping: str | list[dict[str, Any]] = "",
) -> tuple[list[str], bool]:
    """Check relevance of retrieved context and optionally rewrite query for better results.

    This function evaluates the relevance of retrieved context documents using a deterministic
    reflection LLM. If the context doesn't meet the relevance threshold, it attempts to rewrite
    the query for better retrieval results using structured prompts with both system and human
    message components.

    Args:
        vdb_op (VDBRag): Vector database operations instance
        retriever_query (str): Current query to use for retrieval
        collection_names (list[str]): List of collection names to search
        ranker: Optional document ranker instance for reranking results
        reflection_counter (ReflectionCounter): Instance to track reflection iteration count
        top_k (int): Number of top documents to retrieve (default: 10)
        enable_reranker (bool): Whether to use the reranker if available (default: True)
        collection_filter_mapping: Filter expressions for filtering documents from collections

    Returns:
        Tuple[List[str], bool]: A tuple containing:
            - List of retrieved document objects
            - Boolean indicating whether they meet the relevance threshold

    Note:
        - Uses deterministic sampling (temperature=0, top_p=0.1) for reproducible results
        - Supports large context windows (32768 tokens) for comprehensive evaluation
        - Employs structured prompts with system and human message pairs for both
          relevance checking and query rewriting
    """
    relevance_threshold = int(os.environ.get("CONTEXT_RELEVANCE_THRESHOLD", 1))
    reflection_llm_name = (
        get_env_variable(
            variable_name="REFLECTION_LLM",
            default_value="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        )
        .strip('"')
        .strip("'")
    )
    reflection_llm_endpoint = (
        os.environ.get("REFLECTION_LLM_SERVERURL", "").strip('"').strip("'")
    )

    llm_params = {
        "model": reflection_llm_name,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": 32768,
    }

    if reflection_llm_endpoint:
        llm_params["llm_endpoint"] = reflection_llm_endpoint

    reflection_llm = get_llm(**llm_params)

    relevance_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["reflection_relevance_check_prompt"]["system"]),
            ("human", prompts["reflection_relevance_check_prompt"]["human"]),
        ]
    )

    query_rewrite_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["reflection_query_rewriter_prompt"]["system"]),
            ("human", prompts["reflection_query_rewriter_prompt"]["human"]),
        ]
    )

    current_query = retriever_query
    # Get relevant documents with optional reflection
    otel_ctx = otel_context.get_current()

    while reflection_counter.remaining > 0:
        # Get documents using current query
        if ranker and enable_reranker:
            context_reranker = RunnableAssign(
                {
                    "context": lambda input: ranker.compress_documents(
                        query=input["question"], documents=input["context"]
                    )
                }
            )

            # Perform parallel retrieval from all vector stores with their specific filter expressions
            docs = []
            vectorstores = []
            for collection_name in collection_names:
                vectorstores.append(vdb_op.get_langchain_vectorstore(collection_name))

            with ThreadPoolExecutor() as executor:
                futures = []
                for collection_name, vectorstore in zip(
                    collection_names, vectorstores, strict=False
                ):
                    futures.append(
                        executor.submit(
                            vdb_op.retrieval_langchain,
                            query=current_query,
                            collection_name=collection_name,
                            vectorstore=vectorstore,
                            top_k=top_k,
                            filter_expr=collection_filter_mapping.get(
                                collection_name, ""
                            ),
                            otel_ctx=otel_ctx,
                        )
                    )
                for future in futures:
                    docs.extend(future.result())

            docs = context_reranker.invoke(
                {"context": docs, "question": current_query},
                config={"run_name": "context_reranker"},
            )
            original_docs = docs.get("context", [])
        else:
            # Perform sequential retrieval from the first vector store
            original_docs = vdb_op.retrieval_langchain(
                query=current_query,
                collection_name=collection_names[0],
                vectorstore=vdb_op.get_langchain_vectorstore(collection_names[0]),
                top_k=top_k,
                filter_expr=collection_filter_mapping.get(collection_names[0], ""),
                otel_ctx=otel_ctx,
            )

        docs = [d.page_content for d in original_docs]

        context_text = "\n".join(docs)
        relevance_chain = relevance_template | reflection_llm | StrOutputParser()
        relevance_score = _retry_score_generation(
            relevance_chain,
            {"query": current_query, "context": context_text},
            config={"run_name": "relevance-checker"},
        )

        logger.info(
            f"Context relevance score: {relevance_score} (threshold: {relevance_threshold})"
        )
        reflection_counter.increment()

        if relevance_score >= relevance_threshold:
            return original_docs, True

        if reflection_counter.remaining > 0:
            rewrite_chain = query_rewrite_template | reflection_llm | StrOutputParser()
            current_query = rewrite_chain.invoke(
                {"query": current_query}, config={"run_name": "query-rewriter"}
            )
            logger.info(
                f"Rewritten query (iteration {reflection_counter.current_count}): {current_query}"
            )

    return original_docs, False


def check_response_groundedness(
    query: str,
    response: str,
    context: list[str],
    reflection_counter: ReflectionCounter,
) -> tuple[str, bool]:
    """Check groundedness of generated response against retrieved context.

    This function evaluates whether the generated response is well-grounded in the
    provided context documents using a deterministic reflection LLM. If the response
    doesn't meet the groundedness threshold, it attempts to regenerate a better response
    using structured prompts with both system and human message components.

    The function uses deterministic parameters (temperature=0, low top_p) for consistent
    evaluation and supports large context windows (32K tokens) for comprehensive analysis.

    Args:
        query (str): The original user query
        response (str): Generated response to check for groundedness
        context (List[str]): List of context documents used for grounding evaluation
        reflection_counter (ReflectionCounter): Instance to track reflection iteration count

    Returns:
        Tuple[str, bool]: A tuple containing:
            - str: The final response (original or regenerated)
            - bool: Whether the response meets the groundedness threshold (True) or not (False)

    Environment Variables:
        RESPONSE_GROUNDEDNESS_THRESHOLD: Minimum score required for response to be considered grounded (default: 1)
        REFLECTION_LLM: Model name for the reflection LLM (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)
        REFLECTION_LLM_SERVERURL: Optional custom endpoint for the reflection LLM

    Note:
        - Uses deterministic sampling (temperature=0, top_p=0.1) for reproducible results
        - Supports large context windows (32768 tokens) for comprehensive evaluation
        - Employs structured prompts with system and human message pairs
    """
    # Get configuration values for groundedness evaluation
    groundedness_threshold = int(os.environ.get("RESPONSE_GROUNDEDNESS_THRESHOLD", 1))

    # Configure the reflection LLM for groundedness checking and response regeneration
    reflection_llm_name = (
        get_env_variable(
            variable_name="REFLECTION_LLM",
            default_value="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        )
        .strip('"')
        .strip("'")
    )
    reflection_llm_endpoint = (
        os.environ.get("REFLECTION_LLM_SERVERURL", "").strip('"').strip("'")
    )

    # Set deterministic LLM parameters for consistent and reproducible reflection
    llm_params = {
        "model": reflection_llm_name,
        "temperature": 0,  # Deterministic sampling for reproducible results
        "top_p": 0.1,  # Very low top_p for focused, deterministic responses
        "max_tokens": 32768,  # Large token limit for comprehensive analysis and long responses
    }

    if reflection_llm_endpoint:
        llm_params["llm_endpoint"] = reflection_llm_endpoint

    reflection_llm = get_llm(**llm_params)

    # Prepare structured prompt template for groundedness evaluation
    # Uses both system and human messages for more precise instruction following
    groundedness_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["reflection_groundedness_check_prompt"]["system"]),
            ("human", prompts["reflection_groundedness_check_prompt"]["human"]),
        ]
    )

    # Prepare context and response for evaluation
    context_text = "\n".join(context)
    current_response = response

    # Main reflection loop: evaluate groundedness and regenerate if needed
    while reflection_counter.remaining > 0:
        # Evaluate how well the current response is grounded in the provided context
        groundedness_chain = groundedness_template | reflection_llm | StrOutputParser()
        groundedness_score = _retry_score_generation(
            groundedness_chain, {"context": context_text, "response": current_response}
        )

        logger.info(
            f"Response groundedness score: {groundedness_score} (threshold: {groundedness_threshold})"
        )
        reflection_counter.increment()

        # If response meets the groundedness threshold, return it as successful
        if groundedness_score >= groundedness_threshold:
            return current_response, True

        # If we have remaining iterations, attempt to regenerate a better response
        if reflection_counter.remaining > 0:
            # Prepare structured prompt template for response regeneration
            # Uses both system and human messages for comprehensive instruction delivery
            regen_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        prompts["reflection_response_regeneration_prompt"]["system"],
                    ),
                    (
                        "human",
                        prompts["reflection_response_regeneration_prompt"]["human"],
                    ),
                ]
            )

            # Generate a new response using the deterministic reflection LLM
            regen_chain = regen_prompt | reflection_llm | StrOutputParser()
            current_response = regen_chain.invoke(
                {"query": query, "context": context_text},
                config={"run_name": "response-regenerator"},
            )

            # Check if the model indicates the query is out of context
            # If so, return the original response and stop the reflection process
            if "OUT OF CONTEXT" in current_response:
                return response, False

            logger.info(
                f"Regenerated response (iteration {reflection_counter.current_count})"
            )

    # Return the final response after all reflection iterations are exhausted
    # The boolean False indicates that the groundedness threshold was not met
    return current_response, False

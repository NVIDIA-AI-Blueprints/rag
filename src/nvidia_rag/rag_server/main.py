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

""" This defines the main modules for RAG server which manages the core functionality.
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

import logging
import os
import time
import requests
import math
from traceback import print_exc
from typing import Any, Dict, Generator, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign, RunnablePassthrough
from requests import ConnectTimeout
from opentelemetry import context as otel_context

from nvidia_rag.utils.common import get_config, validate_filter_expr
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.rag_server.response_generator import prepare_llm_request, generate_answer, prepare_citations, Citations, retrieve_summary
from nvidia_rag.utils.vectorstore import create_vectorstore_langchain, get_vectorstore, retreive_docs_from_retriever
from nvidia_rag.utils.llm import get_llm, get_prompts, get_streaming_filter_think_parser
from nvidia_rag.utils.reranker import get_ranking_model
from nvidia_rag.rag_server.reflection import ReflectionCounter, check_context_relevance, check_response_groundedness
from nvidia_rag.rag_server.health import check_all_services_health
from nvidia_rag.rag_server.vlm import VLM
from nvidia_rag.rag_server.validation import validate_model_info, validate_use_knowledge_base, validate_temperature, validate_top_p, validate_reranker_k

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Get the model parameters from the config
model_params = CONFIG.llm.get_model_parameters()
default_max_tokens = model_params["max_tokens"]
default_temperature = model_params["temperature"]
default_top_p = model_params["top_p"]

document_embedder = get_embedding_model(model=CONFIG.embeddings.model_name, url=CONFIG.embeddings.server_url)
ranker = get_ranking_model(model=CONFIG.ranking.model_name, url=CONFIG.ranking.server_url, top_n=CONFIG.retriever.top_k)
query_rewriter_llm_config = {"temperature": 0.7, "top_p": 0.2, "max_tokens": 1024}
logger.info("Query rewriter llm config: model name %s, url %s, config %s", CONFIG.query_rewriter.model_name, CONFIG.query_rewriter.server_url, query_rewriter_llm_config)
query_rewriter_llm = get_llm(model=CONFIG.query_rewriter.model_name, llm_endpoint=CONFIG.query_rewriter.server_url, **query_rewriter_llm_config)
prompts = get_prompts()
vdb_top_k = int(CONFIG.retriever.vdb_top_k)

try:
    VECTOR_STORE = create_vectorstore_langchain(document_embedder=document_embedder)
except Exception as ex:
    VECTOR_STORE = None
    logger.error("Unable to connect to vector store during initialization: %s", ex)

MAX_COLLECTION_NAMES = 5

# Get a StreamingFilterThinkParser based on configuration
StreamingFilterThinkParser = get_streaming_filter_think_parser()

class APIError(Exception):
    """Custom exception class for API errors."""
    def __init__(self, message: str, code: int = 400):
        logger.error("APIError occurred: %s with HTTP status: %d", message, code)
        print_exc()
        self.message = message
        self.code = code
        super().__init__(message)

class NvidiaRAG():

    async def health(self, check_dependencies: bool = False) -> Dict[str, Any]:
        """Check the health of the RAG server."""
        response_message = "Service is up."
        health_results = {}
        health_results["message"] = response_message

        if check_dependencies:
            dependencies_results = await check_all_services_health()
            health_results.update(dependencies_results)
        return health_results


    def generate(
        self,
        messages: List[Dict[str, str]],
        use_knowledge_base: bool = True,
        temperature: float = default_temperature,
        top_p: float = default_top_p,
        max_tokens: int = default_max_tokens,
        stop: List[str] = None,
        reranker_top_k: int = int(CONFIG.retriever.top_k),
        vdb_top_k: int = int(CONFIG.retriever.vdb_top_k),
        vdb_endpoint: str = CONFIG.vector_store.url,
        collection_name: str = "",
        collection_names: List[str] = [CONFIG.vector_store.default_collection_name],
        enable_query_rewriting: bool = CONFIG.query_rewriter.enable_query_rewriter,
        enable_reranker: bool = CONFIG.ranking.enable_reranker,
        enable_guardrails: bool = CONFIG.enable_guardrails,
        enable_citations: bool = CONFIG.enable_citations,
        enable_vlm_inference: bool = CONFIG.enable_vlm_inference,
        model: str = CONFIG.llm.model_name,
        llm_endpoint: str = CONFIG.llm.server_url,
        embedding_model: str = CONFIG.embeddings.model_name,
        embedding_endpoint: Optional[str] = CONFIG.embeddings.server_url,
        reranker_model: str = CONFIG.ranking.model_name,
        reranker_endpoint: str = CONFIG.ranking.server_url,
        vlm_model: str = CONFIG.vlm.model_name,
        vlm_endpoint: str = CONFIG.vlm.server_url,
        filter_expr: Optional[str] = '',
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
            vdb_endpoint: Vector database endpoint URL
            collection_name: Name of the collection to use
            collection_names: List of collection names to use
            enable_query_rewriting: Whether to enable query rewriting
            enable_reranker: Whether to enable reranking
            enable_guardrails: Whether to enable guardrails
            enable_citations: Whether to enable citations
            model: Name of the LLM model
            llm_endpoint: LLM server endpoint URL
            embedding_model: Name of the embedding model
            embedding_endpoint: Embedding server endpoint URL
            reranker_model: Name of the reranker model
            reranker_endpoint: Reranker server endpoint URL
            filter_expr: Filter expression to filter document from vector DB
        """

        # Validate boolean and float parameters
        use_knowledge_base = validate_use_knowledge_base(use_knowledge_base)
        temperature = validate_temperature(temperature)
        top_p = validate_top_p(top_p)

        # Validate top_k parameters
        reranker_top_k = validate_reranker_k(reranker_top_k, vdb_top_k)

        # Normalize all model and endpoint values using validation functions
        model, llm_endpoint, embedding_model, embedding_endpoint, reranker_model, reranker_endpoint, vlm_model, vlm_endpoint = map(
            lambda x: validate_model_info(x[0], x[1]),
            [
                (model, "model"),
                (llm_endpoint, "llm_endpoint"),
                (embedding_model, "embedding_model"),
                (embedding_endpoint, "embedding_endpoint"),
                (reranker_model, "reranker_model"),
                (reranker_endpoint, "reranker_endpoint"),
                (vlm_model, "vlm_model"),
                (vlm_endpoint, "vlm_endpoint")
            ]
        )

        query, chat_history = prepare_llm_request(messages)
        llm_settings = {
            "model": model,
            "llm_endpoint": llm_endpoint,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "enable_guardrails": enable_guardrails,
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
                embedding_model=embedding_model,
                embedding_endpoint=embedding_endpoint,
                vdb_endpoint=vdb_endpoint,
                enable_reranker=enable_reranker,
                reranker_model=reranker_model,
                reranker_endpoint=reranker_endpoint,
                enable_vlm_inference=enable_vlm_inference,
                vlm_model=vlm_model,
                vlm_endpoint=vlm_endpoint,
                model=model,
                enable_query_rewriting=enable_query_rewriting,
                enable_citations=enable_citations,
                filter_expr=filter_expr
            )
        else:
            logger.info("Using LLM to generate response directly without knowledge base.")
            return self.__llm_chain(
                llm_settings=llm_settings,
                query=query,
                chat_history=chat_history,
                model=model,
                collection_name=collection_name,
                enable_citations=enable_citations
            )


    def search(
        self,
        query: str,
        messages: List[Dict[str, str]] = [],
        reranker_top_k: int = int(CONFIG.retriever.top_k),
        vdb_top_k: int = int(CONFIG.retriever.vdb_top_k),
        collection_name: str = "",
        collection_names: List[str] = [CONFIG.vector_store.default_collection_name],
        vdb_endpoint: str = CONFIG.vector_store.url,
        enable_query_rewriting: bool = CONFIG.query_rewriter.enable_query_rewriter,
        enable_reranker: bool = CONFIG.ranking.enable_reranker,
        embedding_model: str = CONFIG.embeddings.model_name,
        embedding_endpoint: Optional[str] = CONFIG.embeddings.server_url,
        reranker_model: str = CONFIG.ranking.model_name,
        reranker_endpoint: Optional[str] = CONFIG.ranking.server_url,
        filter_expr: Optional[str] = '',
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
            filter_expr (Optional[str]): Filter expression to filter document from vector DB
        Returns:
            Citations: Retrieved documents.
        """

        logger.info("Searching relevant document for the query: %s", query)

        # Validate top_k parameters
        reranker_top_k = validate_reranker_k(reranker_top_k, vdb_top_k)

        # Normalize all model and endpoint values using validation functions
        embedding_model, embedding_endpoint, reranker_model, reranker_endpoint = map(
            lambda x: validate_model_info(x[0], x[1]),
            [
                (embedding_model, "embedding_model"),
                (embedding_endpoint, "embedding_endpoint"),
                (reranker_model, "reranker_model"),
                (reranker_endpoint, "reranker_endpoint"),
            ]
        )

        try:
            if collection_name: # Would be deprecated in the future
                logger.warning("'collection_name' parameter is provided. This will be deprecated in the future. Use 'collection_names' instead.")
                collection_names = [collection_name]

            if not collection_names:
                raise APIError("Collection names are not provided.", 400)

            if len(collection_names) > 1 and not enable_reranker:
                raise APIError("Reranking is not enabled but multiple collection names are provided.", 400)
            
            if not validate_filter_expr(filter_expr):
                raise APIError("Invalid filter expression.", 400)

            if len(collection_names) > MAX_COLLECTION_NAMES:
                raise APIError(f"Only {MAX_COLLECTION_NAMES} collections are supported at a time.", 400)

            document_embedder = get_embedding_model(model=embedding_model, url=embedding_endpoint)
            # Initialize vector stores for each collection name
            vector_stores = []
            for collection_name in collection_names:
                vector_stores.append(get_vectorstore(document_embedder, collection_name, vdb_endpoint))

            # Check if all vector stores are initialized properly
            for vs in vector_stores:
                if vs is None:
                    raise APIError("Vector store not initialized properly. Please check if the vector DB is up and running.", 500)

            docs = []
            local_ranker = get_ranking_model(model=reranker_model, url=reranker_endpoint, top_n=reranker_top_k)
            top_k = vdb_top_k if local_ranker and enable_reranker else reranker_top_k
            logger.info("Setting top k as: %s.", top_k)
            # Initialize retrievers for each vector store
            retrievers = []
            for vs in vector_stores:
                retrievers.append(vs.as_retriever(search_kwargs={"k": top_k}))


            retriever_query = query
            if messages:
                if enable_query_rewriting:
                    # conversation is tuple so it should be multiple of two
                    # -1 is to keep last k conversation
                    history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
                    messages = messages[history_count:]
                    conversation_history = []

                    for message in messages:
                        if message.get("role") !=  "system":
                            conversation_history.append((message.get("role"), message.get("content")))

                    # Based on conversation history recreate query for better document retrieval
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                    query_rewriter_prompt = prompts.get("query_rewriter_prompt", contextualize_q_system_prompt)
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [("system", query_rewriter_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
                    )
                    q_prompt = contextualize_q_prompt | query_rewriter_llm | StreamingFilterThinkParser | StrOutputParser()
                    # query to be used for document retrieval
                    logger.info("Query rewriter prompt: %s", contextualize_q_prompt)
                    retriever_query = q_prompt.invoke({"input": query, "chat_history": conversation_history})
                    logger.info("Rewritten Query: %s %s", retriever_query, len(retriever_query))
                    if retriever_query.replace('"', "'") == "''" or len(retriever_query) == 0:
                        return Citations()
                else:
                    # Use previous user queries and current query to form a single query for document retrieval
                    user_queries = [msg.get("content") for msg in messages if msg.get("role") == "user"]
                    retriever_query = ". ".join([*user_queries, query])
                    logger.info("Combined retriever query: %s", retriever_query)
            # Get relevant documents with optional reflection
            otel_ctx = otel_context.get_current()
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)
                docs, is_relevant = check_context_relevance(query, retrievers, local_ranker, reflection_counter, enable_reranker, filter_expr=filter_expr)
                # Normalize scores to 0-1 range (was missing!)
                if local_ranker and enable_reranker:
                    docs = self.__normalize_relevance_scores(docs)
                if not is_relevant:
                    logger.warning("Could not find sufficiently relevant context after maximum attempts")
                return prepare_citations(retrieved_documents=docs,
                                         force_citations=True)
            else:
                if local_ranker and enable_reranker:
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to %s with the reranker for rag"
                        " chain.",
                        top_k,
                        reranker_top_k)
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)
                    # Update number of document to be retriever by ranker
                    local_ranker.top_n = reranker_top_k

                    context_reranker = RunnableAssign({
                        "context":
                            lambda input: local_ranker.compress_documents(query=input['question'],
                                                                        documents=input['context'])
                    })

                    # Perform parallel retrieval from all vector stores
                    docs = []
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(retreive_docs_from_retriever, retriever=retriever, retriever_query=retriever_query, expr=filter_expr, otel_ctx=otel_ctx) for retriever in retrievers]
                        for future in futures:
                            docs.extend(future.result())

                    start_time = time.time()
                    docs = context_reranker.invoke({"context": docs, "question": retriever_query}, config={'run_name':'context_reranker'})
                    logger.info("    == Context reranker time: %.2f ms ==", (time.time() - start_time) * 1000)

                    # Normalize scores to 0-1 range"
                    docs = self.__normalize_relevance_scores(docs.get("context", []))

                    return prepare_citations(retrieved_documents=docs,
                                             force_citations=True)

            # Multiple retrievers are not supported when reranking is disabled
            docs = retreive_docs_from_retriever(retriever=retrievers[0], retriever_query=retriever_query, expr=filter_expr, otel_ctx=otel_ctx)
            # TODO: Check how to get the relevance score from milvus
            return prepare_citations(retrieved_documents=docs,
                                     force_citations=True)

        except Exception as e:
            raise APIError(f"Failed to search documents. {str(e)}") from e


    async def get_summary(
        self,
        collection_name: str,
        file_name: str,
        blocking: bool = False,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Get the summary of a document."""

        summary_response = await retrieve_summary(collection_name=collection_name, file_name=file_name, wait=blocking, timeout=timeout)
        return summary_response


    def __llm_chain(
        self,
        llm_settings: Dict[str, Any],
        query: str,
        chat_history: List[Dict[str, str]],
        model: str = "",
        collection_name: str = "",
        enable_citations: bool = True
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
            system_message = []
            conversation_history = []
            user_message = []
            nemotron_message = []
            system_prompt = ""

            system_prompt += prompts.get("chat_template", "")

            if "llama-3.3-nemotron-super-49b" in str(model):
                if os.environ.get("ENABLE_NEMOTRON_THINKING", "false").lower() == "true":
                    logger.info("Setting system prompt as detailed thinking on")
                    system_prompt = "detailed thinking on"
                else:
                    logger.info("Setting system prompt as detailed thinking off")
                    system_prompt = "detailed thinking off"
                nemotron_message += [("user", prompts.get("chat_template", ""))]

            for message in chat_history:
                if message.get("role") ==  "system":
                    system_prompt = system_prompt + " " + message.get("content")
                else:
                    conversation_history.append((message.get("role"), message.get("content")))

            system_message = [("system", system_prompt)]

            logger.info("Query is: %s", query)
            if query is not None and query != "":
                user_message += [("user", "{question}")]

            # Prompt template with system message, conversation history and user query
            message = system_message + nemotron_message + conversation_history + user_message

            self.__print_conversation_history(message, query)

            prompt_template = ChatPromptTemplate.from_messages(message)
            llm = get_llm(**llm_settings)

            chain = prompt_template | llm | StreamingFilterThinkParser | StrOutputParser()
            return generate_answer(chain.stream({"question": query}, config={'run_name':'llm-stream'}), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
        except ConnectTimeout as e:
            logger.warning("Connection timed out while making a request to the LLM endpoint: %s", e)
            return generate_answer(iter([f"Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning("Authentication or permission error: Verify the validity and permissions of your NVIDIA API key.")
                return generate_answer(iter([f"Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
            elif "[404] Not Found" in str(e):
                logger.warning("Please verify the API endpoint and your payload. Ensure that the model name is valid.")
                return generate_answer(iter([f"Please verify the API endpoint and your payload. Ensure that the model name is valid."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
            else:
                return generate_answer(iter([f"Failed to generate RAG chain response. {str(e)}"]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)

    def __rag_chain(
        self,
        llm_settings: Dict[str, Any],
        query: str,
        chat_history: List[Dict[str, str]],
        reranker_top_k: int = 10,
        vdb_top_k: int = 40,
        collection_name: str = "",
        collection_names: List[str] = [CONFIG.vector_store.default_collection_name],
        embedding_model: str = "",
        embedding_endpoint: Optional[str] = None,
        vdb_endpoint: str = "http://localhost:19530",
        enable_reranker: bool = True,
        reranker_model: str = "",
        reranker_endpoint: Optional[str] = None,
        enable_vlm_inference: bool = False,
        vlm_model: str = "",
        vlm_endpoint: str = "",
        model: str = "",
        enable_query_rewriting: bool = False,
        enable_citations: bool = True,
        filter_expr: Optional[str] = '',
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
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
        """
        logger.info("Using multiturn rag to generate response from document for the query: %s", query)

        try:
            # If collection_name is provided, use it as the collection name, Otherwise, use the collection names from the kwargs
            if collection_name: # Would be deprecated in the future
                logger.warning("'collection_name' parameter is provided. This will be deprecated in the future. Use 'collection_names' instead.")
                collection_names = [collection_name]
            # Check if collection names are provided
            if not collection_names:
                raise APIError("Collection names are not provided.", 400)
            if len(collection_names) > 1 and not enable_reranker:
                raise APIError("Reranking is not enabled but multiple collection names are provided.", 400)
            if len(collection_names) > MAX_COLLECTION_NAMES:
                raise APIError(f"Only {MAX_COLLECTION_NAMES} collections are supported at a time.", 400)
            if not validate_filter_expr(filter_expr):
                raise APIError("Invalid filter expression.", 400)

            document_embedder = get_embedding_model(model=embedding_model, url=embedding_endpoint)
            # Initialize vector stores for each collection name
            vector_stores = []
            for collection_name in collection_names:
                vector_stores.append(get_vectorstore(document_embedder, collection_name, vdb_endpoint))

            # Check if all vector stores are initialized properly
            for vs in vector_stores:
                if vs is None:
                    raise APIError("Vector store not initialized properly. Please check if the vector DB is up and running.", 500)

            llm = get_llm(**llm_settings)
            logger.info("Ranker enabled: %s", enable_reranker)
            ranker = get_ranking_model(model=reranker_model, url=reranker_endpoint, top_n=reranker_top_k)
            top_k = vdb_top_k if ranker and enable_reranker else reranker_top_k
            logger.info("Setting retriever top k as: %s.", top_k)
            # Initialize retrievers for each vector store
            retrievers = []
            for vs in vector_stores:
                retrievers.append(vs.as_retriever(search_kwargs={"k": top_k}))

            # conversation is tuple so it should be multiple of two
            # -1 is to keep last k conversation
            history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
            chat_history = chat_history[history_count:]
            system_prompt = ""
            conversation_history = []
            system_prompt += prompts.get("rag_template", "")
            user_message = []

            if "llama-3.3-nemotron-super-49b" in str(model):
                if os.environ.get("ENABLE_NEMOTRON_THINKING", "false").lower() == "true":
                    logger.info("Setting system prompt as detailed thinking on")
                    system_prompt = "detailed thinking on"
                else:
                    logger.info("Setting system prompt as detailed thinking off")
                    system_prompt = "detailed thinking off"
                user_message += [("user", prompts.get("rag_template", ""))]

            for message in chat_history:
                if message.get("role") ==  "system":
                    system_prompt = system_prompt + " " + message.get("content")
                else:
                    conversation_history.append((message.get("role"), message.get("content")))

            system_message = [("system", system_prompt)]
            retriever_query = query
            if chat_history:
                if enable_query_rewriting:
                    # Based on conversation history recreate query for better document retrieval
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                    query_rewriter_prompt = prompts.get("query_rewriter_prompt", contextualize_q_system_prompt)
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [("system", query_rewriter_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
                    )
                    q_prompt = contextualize_q_prompt | query_rewriter_llm | StreamingFilterThinkParser | StrOutputParser()
                    # query to be used for document retrieval
                    logger.info("Query rewriter prompt: %s", contextualize_q_prompt)
                    retriever_query = q_prompt.invoke({"input": query, "chat_history": conversation_history}, config={'run_name':'query-rewriter'})
                    logger.info("Rewritten Query: %s %s", retriever_query, len(retriever_query))
                    if retriever_query.replace('"', "'") == "''" or len(retriever_query) == 0:
                        return generate_answer(iter([""]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
                else:
                    # Use previous user queries and current query to form a single query for document retrieval
                    user_queries = [msg.get("content") for msg in chat_history if msg.get("role") == "user"]
                    # TODO: Find a better way to join this when queries already have punctuation
                    retriever_query = ". ".join([*user_queries, query])
                    logger.info("Combined retriever query: %s", retriever_query)

            # Get relevant documents with optional reflection
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)

                context_to_show, is_relevant = check_context_relevance(
                    retriever_query,
                    retrievers,
                    ranker,
                    reflection_counter,
                    filter_expr=filter_expr
                )
                
                # Normalize scores to 0-1 range (was missing!)
                if ranker and enable_reranker:
                    context_to_show = self.__normalize_relevance_scores(context_to_show)

                if not is_relevant:
                    logger.warning("Could not find sufficiently relevant context after %d attempts",
                                  reflection_counter.current_count)
            else:
                otel_ctx = otel_context.get_current()
                if ranker and enable_reranker:
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to "
                        "%s with the reranker for rag chain.",
                        top_k,
                        reranker_top_k)
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)
                    context_reranker = RunnableAssign({
                        "context":
                            lambda input: ranker.compress_documents(query=input['question'], documents=input['context'])
                    })

                    # Perform parallel retrieval from all vector stores
                    docs = []
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(retreive_docs_from_retriever, retriever=retriever, retriever_query=query, expr=filter_expr, otel_ctx=otel_ctx) for retriever in retrievers]
                        for future in futures:
                            docs.extend(future.result())

                    start_time = time.time()
                    docs = context_reranker.invoke({"context": docs, "question": query}, config={'run_name':'context_reranker'})
                    logger.info("    == Context reranker time: %.2f ms ==", (time.time() - start_time) * 1000)
                    context_to_show = docs.get("context", [])
                    # Normalize scores to 0-1 range
                    context_to_show = self.__normalize_relevance_scores(context_to_show)
                else:
                    # Multiple retrievers are not supported when reranking is disabled
                    docs = retreive_docs_from_retriever(retriever=retrievers[0], retriever_query=query, expr=filter_expr, otel_ctx=otel_ctx)
                    context_to_show = docs

            if enable_vlm_inference:
                logger.info("Calling VLM to analyze images cited in the context")
                vlm_response: str = ""
                try:
                    vlm = VLM(vlm_model, vlm_endpoint)
                    vlm_response = vlm.analyze_images_from_context(
                        context_to_show, query
                    )
                    if vlm_response and vlm.reason_on_vlm_response(
                        query, vlm_response, context_to_show, llm_settings
                    ):
                        logger.info("VLM response validated and added to prompt: %s", vlm_response)
                        vlm_response_prompt = (
                            "The following is an answer generated by a Vision-Language Model (VLM) based solely on images cited in the context:\n"
                            f"---\n{vlm_response.strip()}\n---\n"
                            "Consider this visual insight when answering the user's query, especially where the textual context is ambiguous or limited."
                        )
                        user_message += [("user", vlm_response_prompt)]
                    else:
                        logger.info("VLM response skipped after reasoning or was empty.")
                except (ValueError, EnvironmentError) as e:
                    logger.warning(
                        "VLM processing failed for query='%s', collection='%s': %s",
                        query, collection_name, e, exc_info=True
                    )

                except Exception as e:
                    logger.error(
                        "Unexpected error during VLM processing for query='%s', collection='%s': %s",
                        query, collection_name, e, exc_info=True
                    )

            docs = [self.__format_document_with_source(d) for d in context_to_show]

            # Prompt for response generation based on context
            user_message += [("user", "{question}")]
            message = system_message + conversation_history + user_message
            self.__print_conversation_history(message)
            prompt = ChatPromptTemplate.from_messages(message)

            chain = prompt | llm | StreamingFilterThinkParser | StrOutputParser()

            # Check response groundedness if we still have reflection iterations available
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true" and reflection_counter.remaining > 0:
                initial_response = chain.invoke({"question": query, "context": docs})
                final_response, is_grounded = check_response_groundedness(
                    initial_response,
                    docs,
                    reflection_counter
                )
                if not is_grounded:
                    logger.warning("Could not generate sufficiently grounded response after %d total reflection attempts",
                                    reflection_counter.current_count)
                return generate_answer(iter([final_response]), context_to_show, model=model, collection_name=collection_name, enable_citations=enable_citations)
            else:
                return generate_answer(chain.stream({"question": query, "context": docs}, config={'run_name':'llm-stream'}), context_to_show, model=model, collection_name=collection_name, enable_citations=enable_citations)

        except ConnectTimeout as e:
            logger.warning("Connection timed out while making a request to the LLM endpoint: %s", e)
            return generate_answer(iter([f"Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)

        except requests.exceptions.ConnectionError as e:
            if "HTTPConnectionPool" in str(e):
                logger.error("Connection pool error while connecting to service: %s", e)
                return generate_answer(iter([f"Connection error: Failed to connect to service. Please verify if all required NIMs are running and accessible."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning("Authentication or permission error: Verify the validity and permissions of your NVIDIA API key.")
                return generate_answer(iter([f"Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
            elif "[404] Not Found" in str(e):
                logger.warning("Please verify the API endpoint and your payload. Ensure that the model name is valid.")
                return generate_answer(iter([f"Please verify the API endpoint and your payload. Ensure that the model name is valid."]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)
            else:
                return generate_answer(iter([f"Failed to generate RAG chain with multi-turn response. {str(e)}"]), [], model=model, collection_name=collection_name, enable_citations=enable_citations)


    def __print_conversation_history(self, conversation_history: List[str] = None, query: str | None = None):
        if conversation_history is not None:
            for role, content in conversation_history:
                logger.debug("Role: %s", role)
                logger.debug("Content: %s\n", content)


    def __normalize_relevance_scores(self, documents: List["Document"]) -> List["Document"]:
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
            if 'relevance_score' in doc.metadata:
                original_score = doc.metadata['relevance_score']
                scaled_score = original_score * 0.1
                normalized_score = 1 / (1 + math.exp(-scaled_score))
                doc.metadata['relevance_score'] = normalized_score

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
        enable_metadata = os.getenv('ENABLE_SOURCE_METADATA', 'True').lower() == 'true'

        # Return just content if metadata is disabled or doc has no metadata
        if not enable_metadata or not hasattr(doc, 'metadata'):
            result = doc.page_content
            logger.debug(f"After format_document_with_source (metadata disabled) - Result: {result}")
            return result

        # Handle nested metadata structure
        source = doc.metadata.get('source', {})
        source_path = source.get('source_name', '') if isinstance(source, dict) else source

        # If no source path is found, return just the content
        if not source_path:
            result = doc.page_content
            logger.debug(f"After format_document_with_source (no source path) - Result: {result}")
            return result

        filename = os.path.splitext(os.path.basename(source_path))[0]
        logger.debug(f"Before format_document_with_source - Filename: {filename}")
        result = f"File: {filename}\nContent: {doc.page_content}"

        # Debug log after formatting
        logger.debug(f"After format_document_with_source - Result: {result}")

        return result

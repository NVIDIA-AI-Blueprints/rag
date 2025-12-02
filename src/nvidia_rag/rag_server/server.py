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

"""The definition of the NVIDIA RAG server which exposes the endpoints for the RAG server.
Endpoints:
1. /health: Check the health of the RAG server and its dependencies.
2. /generate: Generate a response using the RAG chain.
3. /search: Search for the most relevant documents for the given search parameters.
4. /chat/completions: Just an alias function to /generate endpoint which is openai compatible
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Generator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
from pydantic import BaseModel, Field, constr, model_validator
from starlette.responses import Response
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from nvidia_rag.rag_server.health import print_health_report
from nvidia_rag.rag_server.main import APIError, NvidiaRAG
from nvidia_rag.rag_server.response_generator import (
    ChainResponse,
    Citations,
    ErrorCodeMapping,
    ImageContent,
    Message,
    TextContent,
    error_response_generator,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import (
    DatabaseHealthInfo,
    NIMServiceHealthInfo,
    RAGHealthResponse,
    StorageHealthInfo,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# Create config instance - loads from environment variables and defaults
CONFIG = NvidiaRAGConfig()

logger.info("Configuration loaded successfully")
logger.debug(f"Configuration:\n{CONFIG}")

# Extract model parameters for Field defaults
model_params = CONFIG.llm.get_model_parameters()
default_min_tokens = model_params["min_tokens"]
default_ignore_eos = model_params["ignore_eos"]
default_max_tokens = model_params["max_tokens"]
default_temperature = model_params["temperature"]
default_top_p = model_params["top_p"]
default_min_thinking_tokens = model_params.get("min_thinking_tokens", 1)
default_max_thinking_tokens = model_params.get("max_thinking_tokens", 8192)

logger.debug("Default LLM parameters:")
logger.debug(f"  min_tokens: {default_min_tokens}")
logger.debug(f"  ignore_eos: {default_ignore_eos}")
logger.debug(f"  max_tokens: {default_max_tokens}")
logger.debug(f"  temperature: {default_temperature}")
logger.debug(f"  top_p: {default_top_p}")
logger.debug(f"  min_thinking_tokens: {default_min_thinking_tokens}")
logger.debug(f"  max_thinking_tokens: {default_max_thinking_tokens}")

tags_metadata = [
    {
        "name": "Health APIs",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {
        "name": "Retrieval APIs",
        "description": "APIs for retrieving document chunks for a query.",
    },
    {"name": "RAG APIs", "description": "APIs for retrieval followed by generation."},
]

# create the FastAPI server
app = FastAPI(
    root_path="/v1",
    title="APIs for NVIDIA RAG Server",
    description="This API schema describes all the retriever endpoints exposed for NVIDIA RAG server Blueprint",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
)

# Allow access in browser from RAG UI and Storybook (development)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create NvidiaRAG instance with config
NVIDIA_RAG = NvidiaRAG(config=CONFIG)

metrics = None
if CONFIG.tracing.enabled:
    from .tracing import instrument

    metrics = instrument(app, CONFIG)


def validate_confidence_threshold_field(confidence_threshold: float) -> float:
    """Shared validation logic for confidence_threshold."""
    if confidence_threshold < 0.0:
        raise ValueError(
            f"confidence_threshold must be >= 0.0, got {confidence_threshold}. "
            "The confidence threshold represents the minimum relevance score required for documents to be included."
        )
    if confidence_threshold > 1.0:
        raise ValueError(
            f"confidence_threshold must be <= 1.0, got {confidence_threshold}. "
            "The confidence threshold represents the minimum relevance score required for documents to be included. "
            "Values range from 0.0 (no filtering) to 1.0 (only perfect matches)."
        )
    return confidence_threshold


def _extract_vdb_auth_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header (e.g., 'Bearer <token>')."""
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    messages: list[Message] = Field(
        ...,
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    use_knowledge_base: bool = Field(
        default=True, description="Whether to use a knowledge base"
    )
    temperature: float = Field(
        default_temperature,
        description="The sampling temperature to use for text generation. "
        "The higher the temperature value is, the less deterministic the output text will be. "
        "It is not recommended to modify both temperature and top_p in the same call.",
        ge=0.0,
        le=1.0,
    )
    top_p: float = Field(
        default_top_p,
        description="The top-p sampling mass used for text generation. "
        "The top-p value determines the probability mass that is sampled at sampling time. "
        "For example, if top_p = 0.2, only the most likely tokens "
        "(summing to 0.2 cumulative probability) will be sampled. "
        "It is not recommended to modify both temperature and top_p in the same call.",
        ge=0.1,
        le=1.0,
    )
    min_tokens: int = Field(
        default_min_tokens,
        description="The minimum number of tokens to generate in any given call",
    )

    ignore_eos: bool = Field(
        default_ignore_eos,
        description="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated",
    )

    max_tokens: int = Field(
        default_max_tokens,
        description="The maximum number of tokens to generate in any given call. "
        "Note that the model is not aware of this value, "
        " and generation will simply stop at the number of tokens specified.",
        ge=0,
        le=128000,
        format="int64",
    )
    min_thinking_tokens: int = Field(
        default=default_min_thinking_tokens,
        description="Minimum number of thinking tokens to allocate for reasoning models. "
        "Enable thinking mode if either min_thinking_tokens or max_thinking_tokens is provided.",
        ge=0,
        format="int64",
    )
    max_thinking_tokens: int = Field(
        default=default_max_thinking_tokens,
        description="Maximum number of thinking tokens to allocate for reasoning models. "
        "Enable thinking mode if either min_thinking_tokens or max_thinking_tokens is provided.",
        ge=0,
        format="int64",
    )
    reranker_top_k: int = Field(
        description="The maximum number of documents to return in the response.",
        default=CONFIG.retriever.top_k,
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=CONFIG.retriever.vdb_top_k,
        ge=0,
        le=400,
        format="int64",
    )
    # Reserved for future use
    # vdb_search_type: str = Field(
    #     description="Search type for the vector space. Can be one of dense or hybrid",
    #     default=os.getenv("APP_VECTORSTORE_SEARCHTYPE", "dense")
    # )
    vdb_endpoint: str = Field(
        description="Endpoint url of the vector database server.",
        default=CONFIG.vector_store.url,
    )
    # TODO: Remove this field in the future
    collection_name: str = Field(
        description="Name of collection to be used for inference.",
        default="",
        max_length=4096,
        pattern=r"[\s\S]*",
        deprecated=True,
    )
    collection_names: list[str] = Field(
        default=[CONFIG.vector_store.default_collection_name],
        description="Name of the collections in the vector database.",
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=CONFIG.query_rewriter.enable_query_rewriter,
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=CONFIG.ranking.enable_reranker,
    )
    enable_guardrails: bool = Field(
        description="Enable or disable guardrailing of queries/responses.",
        default=CONFIG.enable_guardrails,
    )
    enable_citations: bool = Field(
        description="Enable or disable citations as part of response.",
        default=CONFIG.enable_citations,
    )
    enable_vlm_inference: bool = Field(
        description="Enable or disable VLM inference.",
        default=CONFIG.enable_vlm_inference,
    )
    enable_filter_generator: bool = Field(
        description="Enable or disable automatic filter expression generation from natural language.",
        default=CONFIG.filter_expression_generator.enable_filter_generator,
    )
    model: str = Field(
        description="Name of NIM LLM model to be used for inference.",
        default=CONFIG.llm.model_name.strip('"'),
        max_length=4096,
        pattern=r"[\s\S]*",
    )
    llm_endpoint: str = Field(
        description="Endpoint URL for the llm model server.",
        default=CONFIG.llm.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=CONFIG.embeddings.model_name.strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: str | None = Field(
        description="Endpoint URL for the embedding model server.",
        default=CONFIG.embeddings.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=CONFIG.ranking.model_name.strip('"'),
        max_length=256,
    )
    reranker_endpoint: str | None = Field(
        description="Endpoint URL for the reranker model server.",
        default=CONFIG.ranking.server_url.strip('"'),
        max_length=2048,
    )
    vlm_model: str = Field(
        description="Name of the VLM model used for inference.",
        default=CONFIG.vlm.model_name.strip('"'),
        max_length=256,
    )
    vlm_endpoint: str | None = Field(
        description="Endpoint URL for the VLM model server.",
        default=CONFIG.vlm.server_url.strip('"'),
        max_length=2048,
    )

    # seed: int = Field(42, description="If specified, our system will make a best effort to sample deterministically,
    #       such that repeated requests with the same seed and parameters should return the same result.")
    # bad: List[str] = Field(None, description="A word or list of words not to use. The words are case sensitive.")
    stop: list[constr(max_length=256)] = Field(
        description="A string or a list of strings where the API will stop generating further tokens."
        "The returned text will not contain the stop sequence.",
        max_items=256,
        default=[],
    )
    # stream: bool = Field(True, description="If set, partial message deltas will be sent.
    #           Tokens will be sent as data-only server-sent events (SSE) as they become available
    #           (JSON responses are prefixed by data:), with the stream terminated by a data: [DONE] message.")

    filter_expr: str | list[dict[str, Any]] = Field(
        default="",
        description="Filter expression to filter documents from vector database. "
        "Can be a string or a list of dictionaries with filter conditions.",
    )
    confidence_threshold: float = Field(
        default=CONFIG.default_confidence_threshold,
        description="Minimum confidence score threshold for filtering chunks. "
        "Only chunks with relevance scores >= this threshold will be included. "
        "Range: 0.0 to 1.0. Default: 0.0 (no filtering). "
        "Note: Requires enable_reranker=True to generate relevance scores.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_confidence_threshold(cls, values):
        """Custom validator for confidence_threshold to provide better error messages."""
        validate_confidence_threshold_field(values.confidence_threshold)
        return values

    # Validator to check chat message structure
    @model_validator(mode="after")
    def validate_messages_structure(cls, values):
        messages = values.messages
        if not messages:
            raise ValueError("At least one message is required")

        # Check for at least one user message
        if not any(msg.role == "user" for msg in messages):
            raise ValueError("At least one message must have role='user'")

        # Validate last message role is user
        if messages[-1].role != "user":
            raise ValueError("The last message must have role='user'")
        return values


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    query: str | list[TextContent | ImageContent] = Field(
        description="The content or keywords to search for within documents. "
        "Can be a string for text-only queries, or an array of content objects for multimodal queries containing text and/or images.",
        default="Tell me something interesting",
    )
    reranker_top_k: int = Field(
        description="Number of document chunks to retrieve.",
        default=int(CONFIG.retriever.top_k),
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=CONFIG.retriever.vdb_top_k,
        ge=0,
        le=400,
        format="int64",
    )
    vdb_endpoint: str = Field(
        description="Endpoint url of the vector database server.",
        default=CONFIG.vector_store.url,
    )
    # Reserved for future use
    # vdb_search_type: str = Field(
    #     description="Search type for the vector space. Can be one of dense or hybrid",
    #     default=os.getenv("APP_VECTORSTORE_SEARCHTYPE", "dense")
    # )
    # TODO: Remove this field in the future
    collection_name: str = Field(
        description="Name of collection to be used for searching document.",
        default="",
        max_length=4096,
        pattern=r"[\s\S]*",
        deprecated=True,
    )
    collection_names: list[str] = Field(
        default=[CONFIG.vector_store.default_collection_name],
        description="Name of the collections in the vector database.",
    )
    messages: list[Message] = Field(
        default=[],
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=CONFIG.query_rewriter.enable_query_rewriter,
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=CONFIG.ranking.enable_reranker,
    )
    enable_filter_generator: bool = Field(
        description="Enable or disable automatic filter expression generation from natural language.",
        default=CONFIG.filter_expression_generator.enable_filter_generator,
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=CONFIG.embeddings.model_name.strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: str = Field(
        description="Endpoint URL for the embedding model server.",
        default=CONFIG.embeddings.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=CONFIG.ranking.model_name.strip('"'),
        max_length=256,
    )
    reranker_endpoint: str | None = Field(
        description="Endpoint URL for the reranker model server.",
        default=CONFIG.ranking.server_url.strip('"'),
        max_length=2048,
    )

    filter_expr: str | list[dict[str, Any]] = Field(
        description="Filter expression to filter the retrieved documents from Milvus collection.",
        default="",
        # max_length=4096,
        # pattern=r"[\s\S]*",
    )
    confidence_threshold: float = Field(
        default=CONFIG.default_confidence_threshold,
        description="Minimum confidence score threshold for filtering chunks. "
        "Only chunks with relevance scores >= this threshold will be included. "
        "Range: 0.0 to 1.0. Default: 0.0 (no filtering). "
        "Note: Requires enable_reranker=True to generate relevance scores.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_confidence_threshold(cls, values):
        """Custom validator for confidence_threshold to provide better error messages."""
        validate_confidence_threshold_field(values.confidence_threshold)
        return values

    @model_validator(mode="after")
    def sanitize_query_content(cls, values):
        """Sanitize query content similar to Message content validation."""
        import bleach

        query = values.query
        if isinstance(query, str):
            values.query = bleach.clean(query, strip=True)
        elif isinstance(query, list):
            # For list content, sanitize text content but leave image URLs as-is
            sanitized_content = []
            for item in query:
                if isinstance(item, TextContent):
                    item.text = bleach.clean(item.text, strip=True)
                sanitized_content.append(item)
            values.query = sanitized_content
        return values

    # Validator to check chat message structure
    @model_validator(mode="after")
    def validate_messages_structure(cls, values):
        messages = values.messages
        if not messages:
            # If no messages are provided, don't raise an error
            return values

        # Check for at least one user message
        if not any(msg.role == "user" for msg in messages):
            raise ValueError("At least one message must have role='user'")

        # Validate last message role is user
        if messages[-1].role != "user":
            raise ValueError("The last message must have role='user'")
        return values


# Define the summary response model
class SummaryResponse(BaseModel):
    """Represents a summary of a document with status tracking."""

    message: str = Field(
        default="", description="Message describing the summary status"
    )
    status: str = Field(
        default="",
        description="Status of the summary: SUCCESS (completed), PENDING (queued), IN_PROGRESS (generating), FAILED (error occurred), or NOT_FOUND (never requested)",
    )
    summary: str = Field(
        default="", description="Summary content (only present if status is SUCCESS)"
    )
    file_name: str = Field(default="", description="Name of the document")
    collection_name: str = Field(default="", description="Name of the collection")
    error: str | None = Field(
        default=None, description="Error details if status is FAILED"
    )
    started_at: str | None = Field(
        default=None, description="ISO format timestamp when generation started"
    )
    completed_at: str | None = Field(
        default=None, description="ISO format timestamp when generation completed"
    )
    updated_at: str | None = Field(
        default=None, description="ISO format timestamp of last status update"
    )
    progress: dict[str, Any] | None = Field(
        default=None, description="Progress information if status is IN_PROGRESS"
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors(), exclude={"input"})},
    )


@app.get(
    "/health",
    response_model=RAGHealthResponse,
    tags=["Health APIs"],
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        }
    },
)
async def health_check(check_dependencies: bool = False):
    """
    Perform a Health Check

    Args:
        check_dependencies: If True, check health of all dependent services.
                           If False (default), only report that the API service is up.

    Returns 200 when service is up and includes health status of all dependent services when requested.
    """

    logger.info("Checking service health...")
    response = await NVIDIA_RAG.health(check_dependencies)

    # Only perform detailed service checks if requested
    if check_dependencies:
        try:
            print_health_report(response)
        except Exception as e:
            logger.error(f"Error during dependency health checks: {str(e)}")
    else:
        logger.info("Skipping dependency health checks as check_dependencies=False")

    return response


@app.get("/metrics")
def metrics_endpoint():
    """Exposes aggregated metrics for Multi-worker setup across all workers."""
    try:
        # Create a new registry to collect metrics from all workers
        registry = CollectorRegistry()
        # Use multi-process collector to aggregate metrics from all workers
        MultiProcessCollector(registry)
        metrics_data = generate_latest(registry)
        logger.debug(f"Generated {len(metrics_data)} bytes of aggregated metrics data")
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {e}\n", media_type="text/plain"
        )


@app.post(
    "/generate",
    tags=["RAG APIs"],
    response_model=ChainResponse,
    responses={
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {"detail": "The client cancelled the request"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        },
    },
)
async def generate_answer(request: Request, prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""
    generate_start_time = time.time()

    # Helper function to sanitize message content for logging
    def sanitize_content_for_logging(content):
        """Remove image data from content for cleaner logging."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            sanitized_content = []
            for item in content:
                if hasattr(item, "type") and item.type == "image_url":
                    # Replace image data with placeholder for logging
                    sanitized_content.append(
                        {
                            "type": "image_url",
                            "image_url": "<base64 image data omitted>",
                        }
                    )
                elif hasattr(item, "type") and item.type == "image":
                    # Handle 'image' type as well
                    sanitized_content.append(
                        {
                            "type": "image",
                            "image_url": "<base64 image data omitted>",
                        }
                    )
                else:
                    # Keep text content as is
                    sanitized_content.append(
                        item.dict() if hasattr(item, "dict") else item
                    )
            return sanitized_content
        return content

    request_data = {
        "messages": [
            {"role": msg.role, "content": sanitize_content_for_logging(msg.content)}
            for msg in prompt.messages
        ],
        "use_knowledge_base": prompt.use_knowledge_base,
        "temperature": prompt.temperature,
        "top_p": prompt.top_p,
        "max_tokens": prompt.max_tokens,
        "min_tokens": prompt.min_tokens,
        "ignore_eos": prompt.ignore_eos,
        "min_thinking_tokens": prompt.min_thinking_tokens,
        "max_thinking_tokens": prompt.max_thinking_tokens,
        "stop": prompt.stop,
        "reranker_top_k": prompt.reranker_top_k,
        "vdb_top_k": prompt.vdb_top_k,
        "vdb_endpoint": prompt.vdb_endpoint,
        "collection_name": prompt.collection_name,
        "collection_names": prompt.collection_names,
        "enable_query_rewriting": prompt.enable_query_rewriting,
        "enable_reranker": prompt.enable_reranker,
        "enable_guardrails": prompt.enable_guardrails,
        "enable_citations": prompt.enable_citations,
        "enable_vlm_inference": prompt.enable_vlm_inference,
        "enable_filter_generator": prompt.enable_filter_generator,
        "model": prompt.model,
        "llm_endpoint": prompt.llm_endpoint,
        "embedding_model": prompt.embedding_model,
        "embedding_endpoint": prompt.embedding_endpoint,
        "reranker_model": prompt.reranker_model,
        "reranker_endpoint": prompt.reranker_endpoint,
        "vlm_model": prompt.vlm_model,
        "vlm_endpoint": prompt.vlm_endpoint,
        "filter_expr": prompt.filter_expr,
        "confidence_threshold": prompt.confidence_threshold,
    }
    logger.info(
        f"📥 Incoming request to /generate endpoint:\n{json.dumps(request_data, indent=2)}"
    )

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        # Convert messages to list of dicts
        messages_dict = []
        for msg in prompt.messages:
            if isinstance(msg.content, str):
                # Simple string content
                messages_dict.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                # Array content with text and/or images
                content_list = []
                for content_item in msg.content:
                    if hasattr(content_item, "type"):
                        if content_item.type == "text":
                            content_list.append(
                                {"type": "text", "text": content_item.text}
                            )
                        elif content_item.type == "image_url":
                            content_list.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": content_item.image_url.url,
                                        "detail": content_item.image_url.detail,
                                    },
                                }
                            )
                    else:
                        # Fallback for dict-like content
                        content_list.append(content_item)
                # share input as a single string
                messages_dict.append({"role": msg.role, "content": content_list})
            else:
                # Fallback for other content types
                messages_dict.append({"role": msg.role, "content": msg.content})

        # Extract bearer token from Authorization header (e.g., "Bearer <token>")
        vdb_auth_token = _extract_vdb_auth_token(request)

        rag_response = NVIDIA_RAG.generate(
            messages=messages_dict,
            use_knowledge_base=prompt.use_knowledge_base,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            vdb_auth_token=vdb_auth_token,
            min_tokens=prompt.min_tokens,
            ignore_eos=prompt.ignore_eos,
            max_tokens=prompt.max_tokens,
            min_thinking_tokens=prompt.min_thinking_tokens,
            max_thinking_tokens=prompt.max_thinking_tokens,
            stop=prompt.stop,
            reranker_top_k=prompt.reranker_top_k,
            vdb_top_k=prompt.vdb_top_k,
            vdb_endpoint=prompt.vdb_endpoint,
            collection_name=prompt.collection_name,
            collection_names=prompt.collection_names,
            enable_query_rewriting=prompt.enable_query_rewriting,
            enable_reranker=prompt.enable_reranker,
            enable_guardrails=prompt.enable_guardrails,
            enable_citations=prompt.enable_citations,
            enable_vlm_inference=prompt.enable_vlm_inference,
            enable_filter_generator=prompt.enable_filter_generator,
            model=prompt.model,
            llm_endpoint=prompt.llm_endpoint,
            embedding_model=prompt.embedding_model,
            embedding_endpoint=prompt.embedding_endpoint,
            reranker_model=prompt.reranker_model,
            reranker_endpoint=prompt.reranker_endpoint,
            vlm_model=prompt.vlm_model,
            vlm_endpoint=prompt.vlm_endpoint,
            filter_expr=prompt.filter_expr,
            confidence_threshold=prompt.confidence_threshold,
            rag_start_time_sec=generate_start_time,
            metrics=metrics,
        )

        # Extract generator and status code from RAGResponse
        response_generator = rag_response.generator
        status_code = rag_response.status_code

        # Return streaming response with appropriate status code
        return StreamingResponse(
            response_generator, media_type="text/event-stream", status_code=status_code
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during response generation. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."},
            status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
        )

    except ValueError as e:
        # Handle validation errors with specific messages
        logger.warning("Validation error in /generate endpoint: %s", e)
        error_message = str(e)
        return StreamingResponse(
            error_response_generator(error_message),
            media_type="text/event-stream",
            status_code=ErrorCodeMapping.BAD_REQUEST,
        )

    except APIError as e:
        # Handle API errors with specific messages (metadata filtering, etc.)
        logger.warning("API error in /generate endpoint: %s", e)
        error_message = str(e)
        # Use the status code from the APIError if available, otherwise use centralized mapping
        status_code = getattr(e, "code", ErrorCodeMapping.BAD_REQUEST)
        return StreamingResponse(
            error_response_generator(error_message),
            media_type="text/event-stream",
            status_code=status_code,
        )

    except Exception as e:
        logger.error(
            "Error from /generate endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return StreamingResponse(
            error_response_generator(
                "Sorry, there was an error processing your request. Please check the server logs for more details."
            ),
            media_type="text/event-stream",
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )


# Alias function to /generate endpoint OpenAI API compatibility
@app.post(
    "/chat/completions",
    tags=["RAG APIs"],
    response_model=ChainResponse,
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        }
    },
)
async def v1_chat_completions(request: Request, prompt: Prompt) -> StreamingResponse:
    """Just an alias function to /generate endpoint which is openai compatible"""

    response = await generate_answer(request, prompt)
    return response


@app.post(
    "/search",
    tags=["Retrieval APIs"],
    response_model=Citations,
    responses={
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {"detail": "The client cancelled the request"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        },
    },
)
async def document_search(
    request: Request, data: DocumentSearch
) -> dict[str, list[dict[str, Any]]]:
    """Search for the most relevant documents for the given search parameters."""

    # Helper function to sanitize query content for logging
    def sanitize_query_for_logging(query):
        """Remove image data from query for cleaner logging."""
        if isinstance(query, str):
            return query
        elif isinstance(query, list):
            sanitized_query = []
            for item in query:
                if hasattr(item, "type") and item.type == "image_url":
                    # Replace image data with placeholder for logging
                    sanitized_query.append(
                        {
                            "type": "image_url",
                            "image_url": "[IMAGE_DATA_REMOVED_FOR_LOGGING]",
                        }
                    )
                else:
                    # Keep text content as is
                    sanitized_query.append(
                        item.dict() if hasattr(item, "dict") else item
                    )
            return sanitized_query
        return query

    request_data = {
        "query": sanitize_query_for_logging(data.query),
        "reranker_top_k": data.reranker_top_k,
        "vdb_top_k": data.vdb_top_k,
        "collection_names": data.collection_names,
        "enable_reranker": data.enable_reranker,
        "enable_query_rewriting": data.enable_query_rewriting,
        "enable_filter_generator": data.enable_filter_generator,
        "confidence_threshold": data.confidence_threshold,
    }
    logger.info(
        f"📥 Incoming request to /search endpoint:\n{json.dumps(request_data, indent=2)}"
    )

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        messages_dict = [
            {"role": msg.role, "content": msg.content} for msg in data.messages
        ]

        # Process query to handle multimodal content similar to generate endpoint
        query_processed = data.query
        if isinstance(data.query, list):
            # Convert multimodal query to the format expected by the main search method
            content_list = []
            for content_item in data.query:
                if hasattr(content_item, "type"):
                    if content_item.type == "text":
                        content_list.append({"type": "text", "text": content_item.text})
                    elif content_item.type == "image_url":
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": content_item.image_url.url,
                                    "detail": content_item.image_url.detail,
                                },
                            }
                        )
                else:
                    # Fallback for dict-like content
                    content_list.append(content_item)
            query_processed = content_list

        # Extract bearer token from Authorization header (e.g., "Bearer <token>")
        vdb_auth_token = _extract_vdb_auth_token(request)

        return NVIDIA_RAG.search(
            query=query_processed,
            messages=messages_dict,
            vdb_auth_token=vdb_auth_token,
            reranker_top_k=data.reranker_top_k,
            vdb_top_k=data.vdb_top_k,
            collection_name=data.collection_name,
            collection_names=data.collection_names,
            vdb_endpoint=data.vdb_endpoint,
            enable_query_rewriting=data.enable_query_rewriting,
            enable_reranker=data.enable_reranker,
            enable_filter_generator=data.enable_filter_generator,
            embedding_model=data.embedding_model,
            embedding_endpoint=data.embedding_endpoint,
            reranker_model=data.reranker_model,
            reranker_endpoint=data.reranker_endpoint,
            filter_expr=data.filter_expr,
            confidence_threshold=data.confidence_threshold,
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during document search. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."},
            status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
        )
    except APIError as e:
        # Handle APIError with specific status codes
        status_code = getattr(e, "code", ErrorCodeMapping.INTERNAL_SERVER_ERROR)
        logger.error("API Error from POST /search endpoint. Error details: %s", e)
        return JSONResponse(content={"message": str(e)}, status_code=status_code)
    except Exception as e:
        logger.error(
            "Error from POST /search endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return JSONResponse(
            content={"message": "Error occurred while searching documents. " + str(e)},
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )


@app.get(
    "/summary",
    tags=["Retrieval APIs"],
    response_model=SummaryResponse,
    responses={
        400: {
            "description": "Bad request (invalid timeout value)",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Invalid timeout value. Timeout must be a non-negative integer.",
                        "error": "Provided timeout value: -1",
                    }
                }
            },
        },
        404: {
            "description": "Summary not found (non-blocking mode)",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Summary for example.pdf not found. Set wait=true to wait for generation.",
                        "status": "pending",
                    }
                }
            },
        },
        408: {
            "description": "Request timeout (blocking mode)",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Timeout waiting for summary generation for example.pdf",
                        "status": "timeout",
                    }
                }
            },
        },
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {"detail": "The client cancelled the request"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Error occurred while getting summary.",
                        "error": "Internal server error details",
                    }
                }
            },
        },
    },
)
async def get_summary(
    request: Request,
    collection_name: str,
    file_name: str,
    blocking: bool = False,
    timeout: float = 300,
) -> JSONResponse:
    """
    Retrieve document summary from the collection.

    This endpoint fetches the pre-generated summary of a document. It supports both
    blocking and non-blocking behavior through the 'wait' parameter.

    Args:
        request (Request): FastAPI request object
        collection_name (str): Name of the document collection
        file_name (str): Name of the file to get summary for
        blocking (bool, optional): If True, waits for summary generation. Defaults to False
        timeout (float, optional): Maximum time to wait in seconds. Will be converted to int. Defaults to 300

    Returns:
        JSONResponse: Contains either:
            - Summary data: {"summary": str, "file_name": str, "collection_name": str}
            - Error message: {"message": str, "status": str}

    Status Codes:
        400: Bad request (invalid timeout value)
        404: Summary not found (non-blocking mode)
        408: Timeout waiting for summary (blocking mode)
        499: Client Closed Request
        500: Internal server error
    """

    # Convert float timeout to int and validate to avoid negative values
    timeout = int(timeout)
    if timeout < 0:
        return JSONResponse(
            content={
                "message": "Invalid timeout value. Timeout must be a non-negative integer.",
                "error": f"Provided timeout value: {timeout}",
            },
            status_code=ErrorCodeMapping.BAD_REQUEST,
        )

    try:
        response = await NVIDIA_RAG.get_summary(
            collection_name=collection_name,
            file_name=file_name,
            blocking=blocking,
            timeout=timeout,
        )

        status = response.get("status")

        # Map status to appropriate HTTP status codes
        if status == "SUCCESS":
            return JSONResponse(content=response, status_code=ErrorCodeMapping.SUCCESS)
        elif status == "NOT_FOUND":
            return JSONResponse(
                content=response, status_code=ErrorCodeMapping.NOT_FOUND
            )
        elif status in ["PENDING", "IN_PROGRESS"]:
            # Return 202 Accepted for in-progress tasks
            return JSONResponse(content=response, status_code=ErrorCodeMapping.ACCEPTED)
        elif status == "FAILED":
            # Check if it's a timeout vs other failure
            error = response.get("error", "")
            if "timeout" in error.lower():
                return JSONResponse(
                    content=response, status_code=ErrorCodeMapping.REQUEST_TIMEOUT
                )
            else:
                return JSONResponse(
                    content=response, status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR
                )
        else:
            # Unknown status
            logger.warning(f"Unknown summary status: {status}")
            return JSONResponse(
                content=response, status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR
            )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while getting summary: {e}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client"},
            status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
        )
    except Exception as e:
        logger.error(
            "Error from GET /summary endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return JSONResponse(
            content={
                "message": "Error occurred while getting summary.",
                "error": str(e),
            },
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )

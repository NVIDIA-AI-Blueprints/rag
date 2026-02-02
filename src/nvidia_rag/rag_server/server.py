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
2. /configuration: Get the server's default configuration values.
3. /generate: Generate a response using the RAG chain.
4. /search: Search for the most relevant documents for the given search parameters.
5. /chat/completions: Just an alias function to /generate endpoint which is openai compatible
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
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

logger.debug("Default LLM parameters:")
logger.debug(f"  min_tokens: {default_min_tokens}")
logger.debug(f"  ignore_eos: {default_ignore_eos}")
logger.debug(f"  max_tokens: {default_max_tokens}")
logger.debug(f"  temperature: {default_temperature}")
logger.debug(f"  top_p: {default_top_p}")

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
    title="APIs for NVIDIA RAG Server",
    description="This API schema describes all the retriever endpoints exposed for NVIDIA RAG server Blueprint",
    version="1.0.0",
    docs_url=None,  # Custom docs per version
    redoc_url=None,
    openapi_url=None,  # Custom openapi per version
    openapi_tags=tags_metadata,
)

# Create v1 router for all standard endpoints (with /v1 prefix)
v1_router = APIRouter(prefix="/v1")

# Create v2 router for OpenAI-compatible vector_stores endpoint only
v2_router = APIRouter(prefix="/v2")

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
# Explicitly pass the prompts configuration file to maintain consistency
# between server mode and library mode. Only pass if file exists to avoid
# unnecessary warnings.
PROMPT_CONFIG_FILE = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")
NVIDIA_RAG = NvidiaRAG(
    config=CONFIG,
    prompts=PROMPT_CONFIG_FILE if Path(PROMPT_CONFIG_FILE).is_file() else None,
)

# Register routers
# Routers will be mounted at the end of the file after all routes are defined

metrics = None
if CONFIG.tracing.enabled:
    from nvidia_rag.utils.observability.tracing import instrument

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
    auth_header = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )
    if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


def _convert_openai_filter_to_milvus_string(filter_obj: Any) -> str:
    """
    Convert OpenAI filter format to Milvus filter expression string.

    Args:
        filter_obj: ComparisonFilter or CompoundFilter object

    Returns:
        String representing the filter in Milvus format (e.g., 'content_metadata["key"] == "value"')
    """
    # Debug logging
    logger.debug(f"Converting filter object type: {type(filter_obj)}")
    logger.debug(
        f"Has 'key': {hasattr(filter_obj, 'key')}, Has 'type': {hasattr(filter_obj, 'type')}, Has 'value': {hasattr(filter_obj, 'value')}, Has 'filters': {hasattr(filter_obj, 'filters')}"
    )

    # Handle ComparisonFilter
    if (
        hasattr(filter_obj, "key")
        and hasattr(filter_obj, "type")
        and hasattr(filter_obj, "value")
    ):
        key = filter_obj.key
        op = filter_obj.type
        value = filter_obj.value

        # Format the key - if it doesn't contain brackets, wrap it in content_metadata
        # Standard metadata fields: document_id, document_name, document_type, language,
        # date_created, last_modified, page_number, etc.
        # Custom fields go in content_metadata["field_name"]
        standard_fields = [
            "document_id",
            "document_name",
            "document_type",
            "language",
            "date_created",
            "last_modified",
            "page_number",
            "description",
            "height",
            "width",
        ]

        if "[" not in key:  # Simple field name without brackets
            if key not in standard_fields:
                # Custom field - wrap in content_metadata
                formatted_key = f'content_metadata["{key}"]'
            else:
                # Standard field - use as-is
                formatted_key = key
        else:
            # Already formatted (e.g., 'content_metadata["field"]')
            formatted_key = key

        # Map OpenAI operators to Milvus operators
        if op == "eq":
            if isinstance(value, str):
                return f'{formatted_key} == "{value}"'
            elif isinstance(value, bool):
                return f"{formatted_key} == {str(value).lower()}"
            else:
                return f"{formatted_key} == {value}"
        elif op == "ne":
            if isinstance(value, str):
                return f'{formatted_key} != "{value}"'
            elif isinstance(value, bool):
                return f"{formatted_key} != {str(value).lower()}"
            else:
                return f"{formatted_key} != {value}"
        elif op == "gt":
            return f"{formatted_key} > {value}"
        elif op == "gte":
            return f"{formatted_key} >= {value}"
        elif op == "lt":
            return f"{formatted_key} < {value}"
        elif op == "lte":
            return f"{formatted_key} <= {value}"
        elif op == "in":
            if isinstance(value, list):
                # Format list values for 'in' operator
                if value and isinstance(value[0], str):
                    formatted_values = ", ".join([f'"{v}"' for v in value])
                else:
                    formatted_values = ", ".join([str(v) for v in value])
                return f"{formatted_key} in [{formatted_values}]"
            else:
                return f"{formatted_key} in [{value}]"
        elif op == "nin":
            if isinstance(value, list):
                # Format list values for 'not in' operator
                if value and isinstance(value[0], str):
                    formatted_values = ", ".join([f'"{v}"' for v in value])
                else:
                    formatted_values = ", ".join([str(v) for v in value])
                return f"{formatted_key} not in [{formatted_values}]"
            else:
                return f"{formatted_key} not in [{value}]"
        else:
            logger.warning(f"Unknown operator: {op}")
            return ""

    # Handle CompoundFilter
    elif hasattr(filter_obj, "filters") and hasattr(filter_obj, "type"):
        filter_type = filter_obj.type
        sub_filters = [
            _convert_openai_filter_to_milvus_string(f) for f in filter_obj.filters
        ]
        # Remove empty sub-filters
        sub_filters = [f for f in sub_filters if f]

        if not sub_filters:
            return ""

        if filter_type == "and":
            return f"({' and '.join(sub_filters)})"
        elif filter_type == "or":
            return f"({' or '.join(sub_filters)})"
        else:
            logger.warning(f"Unknown compound filter type: {filter_type}")
            return ""

    return ""


def _convert_openai_filter_to_elasticsearch(filter_obj: Any) -> list[dict[str, Any]]:
    """
    Convert OpenAI filter format to Elasticsearch query DSL format.

    Args:
        filter_obj: ComparisonFilter or CompoundFilter object

    Returns:
        List of dictionaries in Elasticsearch query DSL format
    """
    # Handle ComparisonFilter
    if (
        hasattr(filter_obj, "key")
        and hasattr(filter_obj, "type")
        and hasattr(filter_obj, "value")
    ):
        key = filter_obj.key
        op = filter_obj.type
        value = filter_obj.value

        # Map OpenAI operators to Elasticsearch query DSL
        if op == "eq":
            return [{"term": {key: value}}]
        elif op == "ne":
            return [{"bool": {"must_not": [{"term": {key: value}}]}}]
        elif op == "gt":
            return [{"range": {key: {"gt": value}}}]
        elif op == "gte":
            return [{"range": {key: {"gte": value}}}]
        elif op == "lt":
            return [{"range": {key: {"lt": value}}}]
        elif op == "lte":
            return [{"range": {key: {"lte": value}}}]
        elif op == "in":
            return [{"terms": {key: value if isinstance(value, list) else [value]}}]
        elif op == "nin":
            return [
                {
                    "bool": {
                        "must_not": [
                            {
                                "terms": {
                                    key: value if isinstance(value, list) else [value]
                                }
                            }
                        ]
                    }
                }
            ]
        else:
            logger.warning(f"Unknown operator: {op}")
            return []

    # Handle CompoundFilter
    elif hasattr(filter_obj, "filters") and hasattr(filter_obj, "type"):
        filter_type = filter_obj.type
        sub_filters = []
        for f in filter_obj.filters:
            sub_filters.extend(_convert_openai_filter_to_elasticsearch(f))

        if not sub_filters:
            return []

        if filter_type == "and":
            return [{"bool": {"must": sub_filters}}]
        elif filter_type == "or":
            return [{"bool": {"should": sub_filters, "minimum_should_match": 1}}]
        else:
            logger.warning(f"Unknown compound filter type: {filter_type}")
            return []

    return []


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
        default=None,
        description="Minimum number of thinking tokens to allocate for reasoning models. "
        "Enable thinking mode if either min_thinking_tokens or max_thinking_tokens is provided.",
        format="int64",
    )
    max_thinking_tokens: int = Field(
        default=None,
        description="Maximum number of thinking tokens to allocate for reasoning models. "
        "Enable thinking mode if either min_thinking_tokens or max_thinking_tokens is provided.",
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
    vlm_temperature: float = Field(
        default=CONFIG.vlm.temperature,
        description="The sampling temperature to use for VLM text generation.",
        ge=0.0,
        le=1.0,
    )
    vlm_top_p: float = Field(
        default=CONFIG.vlm.top_p,
        description="The top-p sampling mass used for VLM text generation.",
        ge=0.1,
        le=1.0,
    )
    vlm_max_tokens: int = Field(
        default=CONFIG.vlm.max_tokens,
        description="The maximum number of tokens to generate by the VLM.",
        ge=0,
        le=128000,
        format="int64",
    )
    vlm_max_total_images: int = Field(
        default=CONFIG.vlm.max_total_images,
        description="Maximum total images sent to VLM per request (query + context).",
        ge=0,
        le=64,
        format="int64",
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
    enable_citations: bool = Field(
        description="Enable or disable image/table/chart citations as part of response.",
        default=CONFIG.enable_citations,
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


# Configuration response models
class RagConfigurationDefaults(BaseModel):
    """Default values for RAG configuration parameters."""

    temperature: float = Field(
        description="Default sampling temperature for generation"
    )
    top_p: float = Field(description="Default top-p sampling mass")
    max_tokens: int = Field(description="Default maximum tokens to generate")
    vdb_top_k: int = Field(
        description="Default number of documents to retrieve from vector DB"
    )
    reranker_top_k: int = Field(
        description="Default number of documents after reranking"
    )
    confidence_threshold: float = Field(
        description="Default confidence score threshold"
    )


class FeatureTogglesDefaults(BaseModel):
    """Default values for feature toggles."""

    enable_reranker: bool = Field(description="Whether reranker is enabled by default")
    enable_citations: bool = Field(
        description="Whether citations are enabled by default"
    )
    enable_guardrails: bool = Field(
        description="Whether guardrails are enabled by default"
    )
    enable_query_rewriting: bool = Field(
        description="Whether query rewriting is enabled by default"
    )
    enable_vlm_inference: bool = Field(
        description="Whether VLM inference is enabled by default"
    )
    enable_filter_generator: bool = Field(
        description="Whether filter generator is enabled by default"
    )


class ModelsDefaults(BaseModel):
    """Default model names."""

    llm_model: str = Field(description="Default LLM model name")
    embedding_model: str = Field(description="Default embedding model name")
    reranker_model: str = Field(description="Default reranker model name")
    vlm_model: str = Field(description="Default VLM model name")


class EndpointsDefaults(BaseModel):
    """Default endpoint URLs."""

    llm_endpoint: str = Field(description="Default LLM endpoint URL")
    embedding_endpoint: str = Field(description="Default embedding endpoint URL")
    reranker_endpoint: str = Field(description="Default reranker endpoint URL")
    vlm_endpoint: str = Field(description="Default VLM endpoint URL")
    vdb_endpoint: str = Field(description="Default vector database endpoint URL")


class ConfigurationResponse(BaseModel):
    """Response containing all server default configuration values."""

    rag_configuration: RagConfigurationDefaults = Field(
        description="Default RAG configuration parameters"
    )
    feature_toggles: FeatureTogglesDefaults = Field(
        description="Default feature toggle states"
    )
    models: ModelsDefaults = Field(description="Default model names")
    endpoints: EndpointsDefaults = Field(description="Default endpoint URLs")


# OpenAI-compatible vector store search models


class RankingOptions(BaseModel):
    """Ranking options for vector store search."""

    ranker: str = Field(
        default="auto",
        description="Control re-ranking behavior. "
        "To enable: 'auto', 'true', 'on', 'enabled', 'yes', '1'. "
        "To disable (reduces latency): 'none', 'false', 'off', 'disabled', 'no', '0'. "
        "Case-insensitive.",
        examples=["auto", "none", "false"],
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum score threshold for filtering results. Only results with scores >= this value will be returned.",
        ge=0.0,
        le=1.0,
        examples=[0.0, 0.5, 0.75],
    )

    model_config = {
        "json_schema_extra": {"examples": [{"ranker": "auto", "score_threshold": 0.5}]}
    }


class ComparisonFilter(BaseModel):
    """A filter used to compare a specified attribute key to a given value using a defined comparison operation."""

    key: str = Field(
        ...,
        description="The key to compare against the value.",
        examples=["author", "page_number", "category"],
    )
    type: str = Field(
        ...,
        description="Specifies the comparison operator: eq (equals), ne (not equal), gt (greater than), "
        "gte (greater than or equal), lt (less than), lte (less than or equal), in, nin (not in).",
        examples=["eq", "gt", "in"],
    )
    value: str | int | float | bool | list[str | int | float | bool] = Field(
        ...,
        description="The value to compare against the attribute key; supports string, number, boolean, or array types.",
        examples=["John Doe", 5, True, ["tech", "science"]],
    )

    @model_validator(mode="after")
    def validate_type(cls, values):
        """Validate that type is one of the allowed comparison operators."""
        allowed_types = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
        if values.type not in allowed_types:
            raise ValueError(
                f"Invalid comparison type '{values.type}'. Must be one of: {', '.join(allowed_types)}"
            )
        return values

    model_config = {
        "json_schema_extra": {
            "examples": [{"key": "<KEYNAME>", "type": "eq", "value": "John Doe"}]
        }
    }


class CompoundFilter(BaseModel):
    """Combine multiple filters using 'and' or 'or'."""

    type: str = Field(
        ...,
        description="Type of operation: 'and' or 'or'.",
        examples=["and", "or"],
    )
    filters: list["ComparisonFilter | CompoundFilter"] = Field(
        ...,
        description="Array of filters to combine. Items can be ComparisonFilter or CompoundFilter.",
    )

    @model_validator(mode="after")
    def validate_type(cls, values):
        """Validate that type is either 'and' or 'or'."""
        if values.type not in ["and", "or"]:
            raise ValueError(
                f"Invalid compound filter type '{values.type}'. Must be either 'and' or 'or'."
            )
        return values

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "and",
                    "filters": [
                        {"key": "author", "type": "eq", "value": "John Doe"},
                        {"key": "page_number", "type": "gte", "value": 5},
                    ],
                }
            ]
        }
    }


# Allow recursive filters
CompoundFilter.model_rebuild()


class VectorStoreSearchRequest(BaseModel):
    """OpenAI-compatible vector store search request."""

    query: str | list[TextContent | ImageContent] = Field(
        ...,
        description="A query string for a search or an array of content objects for multimodal queries.",
        examples=["What is the return policy?", "Tell me about machine learning"],
    )
    filters: ComparisonFilter | CompoundFilter | None = Field(
        default=None,
        description="A filter to apply based on file attributes. Can be a comparison filter or compound filter.",
    )
    max_num_results: int = Field(
        default=10,
        description="The maximum number of results to return. This number should be between 1 and 50 inclusive.",
        ge=1,
        le=50,
        examples=[10, 5, 20],
    )
    ranking_options: RankingOptions | None = Field(
        default=None,
        description="Ranking options for search.",
    )
    rewrite_query: bool = Field(
        default=False,
        description="Whether to rewrite the natural language query for vector search.",
        examples=[False, True],
    )

    @model_validator(mode="before")
    @classmethod
    def handle_empty_dicts(cls, values):
        """Convert empty dicts to None for optional fields."""
        if isinstance(values, dict):
            # Handle empty filters
            if "filters" in values and values["filters"] == {}:
                values["filters"] = None
            # Handle empty ranking_options
            if "ranking_options" in values and values["ranking_options"] == {}:
                values["ranking_options"] = None
        return values

    @model_validator(mode="after")
    def sanitize_query_content(cls, values):
        """Sanitize query content similar to DocumentSearch validation."""
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the return policy?",
                    "max_num_results": 10,
                    "ranking_options": {"ranker": "auto", "score_threshold": 0.5},
                    "rewrite_query": False,
                },
                {
                    "query": "machine learning basics",
                    "max_num_results": 5,
                    "ranking_options": {"ranker": "none", "score_threshold": 0.0},
                    "rewrite_query": True,
                },
            ]
        }
    }


class VectorStoreSearchResultContent(BaseModel):
    """Content object in search result."""

    type: str = Field(description="Type of content (e.g., 'text')")
    text: str = Field(description="Text content")


class VectorStoreSearchResultItem(BaseModel):
    """Single search result item in OpenAI format."""

    file_id: str = Field(description="Identifier for the file")
    filename: str = Field(description="Name of the file")
    score: float = Field(description="Relevance score")
    attributes: dict[str, Any] = Field(description="File attributes/metadata")
    content: list[VectorStoreSearchResultContent] = Field(
        description="Content chunks from the file"
    )


class VectorStoreSearchResponse(BaseModel):
    """OpenAI-compatible vector store search response."""

    object: str = Field(
        default="vector_store.search_results.page",
        description="Object type identifier",
    )
    search_query: str = Field(description="The search query that was executed")
    data: list[VectorStoreSearchResultItem] = Field(
        description="List of search results"
    )
    has_more: bool = Field(
        default=False, description="Whether there are more results available"
    )
    next_page: str | None = Field(
        default=None, description="Token for retrieving the next page of results"
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> StreamingResponse:
    """Handle request validation errors by returning streaming response."""
    errors = exc.errors()
    error_parts = []
    for err in errors:
        loc = [str(loc_item) for loc_item in err.get("loc", ()) if loc_item != "body"]
        field = ".".join(loc) if loc else ""
        msg = err.get("msg", "Validation error")
        error_parts.append(f"{field}: {msg}" if field else msg)

    error_message = "Validation error: " + "; ".join(error_parts)
    logger.warning("Request validation error: %s", error_message)

    return StreamingResponse(
        error_response_generator(error_message),
        media_type="text/event-stream",
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
    )


# Custom documentation endpoints for v1
@app.get("/v1/docs", include_in_schema=False)
async def v1_docs():
    """Swagger UI documentation for v1 API."""
    return get_swagger_ui_html(
        openapi_url="/v1/openapi.json", title="NVIDIA RAG API v1"
    )


@app.get("/v1/openapi.json", include_in_schema=False)
async def v1_openapi():
    """OpenAPI schema for v1 API."""
    return get_openapi(
        title="APIs for NVIDIA RAG Server (v1)",
        version="1.0.0",
        description="This API schema describes all the retriever endpoints exposed for NVIDIA RAG server Blueprint",
        routes=v1_router.routes,
        tags=tags_metadata,
    )


# Custom documentation endpoints for v2
@app.get("/v2/docs", include_in_schema=False)
async def v2_docs():
    """Swagger UI documentation for v2 API."""
    return get_swagger_ui_html(
        openapi_url="/v2/openapi.json", title="NVIDIA RAG API v2 (OpenAI Compatible)"
    )


@app.get("/v2/openapi.json", include_in_schema=False)
async def v2_openapi():
    """OpenAPI schema for v2 API."""
    return get_openapi(
        title="APIs for NVIDIA RAG Server (v2) - OpenAI Compatible",
        version="2.0.0",
        description="OpenAI-compatible API endpoints for NVIDIA RAG server Blueprint. This version includes enhanced OpenAI-compatible endpoints.",
        routes=v2_router.routes,
        tags=[
            {
                "name": "Retrieval APIs",
                "description": "OpenAI-compatible APIs for retrieving document chunks.",
            }
        ],
    )


# Default docs redirect to v1 (for backward compatibility)
@app.get("/docs", include_in_schema=False)
async def default_docs():
    """Redirect to v1 docs for backward compatibility."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/v1/docs")


@app.get("/openapi.json", include_in_schema=False)
async def default_openapi():
    """Redirect to v1 openapi for backward compatibility."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/v1/openapi.json")


@app.get(
    "/health",
    response_model=RAGHealthResponse,
    tags=["Health APIs"],
    description="Perform a health check on the RAG server.",
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
    """Perform a health check on the RAG server."""

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


@app.get(
    "/configuration",
    response_model=ConfigurationResponse,
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
async def get_configuration():
    """
    Get Server Default Configuration

    Returns the default configuration values used by the RAG server.
    These values are derived from the server's environment variables and configuration.

    Use this endpoint to:
    - Display actual default values in the UI
    - Understand what values will be used when parameters are not explicitly set
    - Ensure frontend and backend defaults are synchronized
    """
    logger.info("Fetching server configuration defaults...")

    try:
        # Get model parameters from config
        model_params = CONFIG.llm.get_model_parameters()

        return ConfigurationResponse(
            rag_configuration=RagConfigurationDefaults(
                temperature=model_params["temperature"],
                top_p=model_params["top_p"],
                max_tokens=model_params["max_tokens"],
                vdb_top_k=CONFIG.retriever.vdb_top_k,
                reranker_top_k=CONFIG.retriever.top_k,
                confidence_threshold=CONFIG.default_confidence_threshold,
            ),
            feature_toggles=FeatureTogglesDefaults(
                enable_reranker=CONFIG.ranking.enable_reranker,
                enable_citations=CONFIG.enable_citations,
                enable_guardrails=CONFIG.enable_guardrails,
                enable_query_rewriting=CONFIG.query_rewriter.enable_query_rewriter,
                enable_vlm_inference=CONFIG.enable_vlm_inference,
                enable_filter_generator=CONFIG.filter_expression_generator.enable_filter_generator,
            ),
            models=ModelsDefaults(
                llm_model=CONFIG.llm.model_name.strip('"'),
                embedding_model=CONFIG.embeddings.model_name.strip('"'),
                reranker_model=CONFIG.ranking.model_name.strip('"'),
                vlm_model=CONFIG.vlm.model_name.strip('"'),
            ),
            endpoints=EndpointsDefaults(
                llm_endpoint=CONFIG.llm.server_url.strip('"'),
                embedding_endpoint=CONFIG.embeddings.server_url.strip('"'),
                reranker_endpoint=CONFIG.ranking.server_url.strip('"'),
                vlm_endpoint=CONFIG.vlm.server_url.strip('"'),
                vdb_endpoint=CONFIG.vector_store.url,
            ),
        )
    except Exception as e:
        logger.error(f"Error fetching configuration: {str(e)}")
        return JSONResponse(
            content={"detail": f"Error fetching configuration: {str(e)}"},
            status_code=500,
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
        "vlm_temperature": prompt.vlm_temperature,
        "vlm_top_p": prompt.vlm_top_p,
        "vlm_max_tokens": prompt.vlm_max_tokens,
        "vlm_max_total_images": prompt.vlm_max_total_images,
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

        # Get the streaming generator from NVIDIA_RAG.generate
        rag_response = await NVIDIA_RAG.generate(
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
            vlm_temperature=prompt.vlm_temperature,
            vlm_top_p=prompt.vlm_top_p,
            vlm_max_tokens=prompt.vlm_max_tokens,
            vlm_max_total_images=prompt.vlm_max_total_images,
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
        error_message = "Request was cancelled by the client."
        return StreamingResponse(
            error_response_generator(error_message),
            media_type="text/event-stream",
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
        logger.warning("API error in /generate endpoint: %s", e)
        error_message = e.message
        status_code = getattr(e, "status_code", ErrorCodeMapping.BAD_REQUEST)
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
    description="Search for the most relevant documents for the given search parameters.",
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
        "enable_citations": data.enable_citations,
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

        return await NVIDIA_RAG.search(
            query=query_processed,
            messages=messages_dict,
            vdb_auth_token=vdb_auth_token,
            reranker_top_k=data.reranker_top_k,
            vdb_top_k=data.vdb_top_k,
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
            enable_citations=data.enable_citations,
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during document search. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."},
            status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
        )
    except APIError as e:
        status_code = getattr(e, "status_code", ErrorCodeMapping.INTERNAL_SERVER_ERROR)
        logger.exception("API Error from POST /search endpoint. Error details: %s", e)
        return JSONResponse(content={"message": e.message}, status_code=status_code)
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.RequestException,
        ConnectionError,
        OSError,
    ) as e:
        error_msg = "Failed to search documents. Service unavailable. Please verify the NIM services are running and accessible."
        logger.exception("Connection error from POST /search endpoint: %s", e)
        return JSONResponse(
            content={"message": error_msg},
            status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
        )
    except Exception as e:
        logger.error(
            "Error from POST /search endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return JSONResponse(
            content={"message": "Failed to search documents. " + str(e).split("\n")[0]},
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )


async def _vector_store_search_impl(
    request: Request,
    vector_store_id: str,
    search_request: VectorStoreSearchRequest,
) -> JSONResponse:
    """
    OpenAI-compatible vector store search endpoint.

    Search within a vector store using natural language queries.
    This endpoint maps to the existing /search functionality with OpenAI-compatible schema.

    Args:
        request: FastAPI request object
        vector_store_id: The ID of the vector store (collection name) to search
        search_request: Search request parameters

    Returns:
        JSONResponse: Search results in OpenAI-compatible format
    """

    # Helper function to sanitize query content for logging
    def sanitize_query_for_logging(query):
        """Remove image data from query for cleaner logging."""
        if isinstance(query, str):
            return query
        elif isinstance(query, list):
            sanitized_query = []
            for item in query:
                if hasattr(item, "type") and item.type == "image_url":
                    sanitized_query.append(
                        {
                            "type": "image_url",
                            "image_url": "[IMAGE_DATA_REMOVED_FOR_LOGGING]",
                        }
                    )
                else:
                    sanitized_query.append(
                        item.dict() if hasattr(item, "dict") else item
                    )
            return sanitized_query
        return query

    # Helper function to serialize filters/ranking_options for logging
    def serialize_for_logging(obj):
        """Serialize Pydantic models or dicts for logging."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    request_data = {
        "vector_store_id": vector_store_id,
        "query": sanitize_query_for_logging(search_request.query),
        "filters": serialize_for_logging(search_request.filters),
        "max_num_results": search_request.max_num_results,
        "ranking_options": serialize_for_logging(search_request.ranking_options),
        "rewrite_query": search_request.rewrite_query,
    }
    logger.info(
        f"📥 Incoming request to /vector_stores/{vector_store_id}/search endpoint:\n{json.dumps(request_data, indent=2)}"
    )

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)

    try:
        # Map OpenAI request to internal DocumentSearch format
        # Parse ranking options if provided
        enable_reranker = CONFIG.ranking.enable_reranker
        reranker_model = CONFIG.ranking.model_name.strip('"')
        reranker_endpoint = CONFIG.ranking.server_url.strip('"')
        confidence_threshold = CONFIG.default_confidence_threshold
        vdb_top_k = CONFIG.retriever.vdb_top_k

        if search_request.ranking_options:
            # Map ranking_options to internal parameters
            # Handle ranker field: 'none', 'false', 'off', 'disabled' disables reranker
            # 'auto', 'true', 'on', 'enabled' enables reranker
            ranker_value = search_request.ranking_options.ranker.lower()
            if ranker_value in ["none", "false", "off", "disabled", "no", "0"]:
                enable_reranker = False
            elif ranker_value in ["auto", "true", "on", "enabled", "yes", "1"]:
                enable_reranker = True
            # If ranker value is something else, keep the default from CONFIG

            # Map score_threshold to confidence_threshold
            confidence_threshold = search_request.ranking_options.score_threshold

        # Convert filters to filter_expr based on vector store type
        filter_expr: str | list[dict[str, Any]] = ""
        if search_request.filters:
            # Log the original OpenAI filter
            if hasattr(search_request.filters, "model_dump"):
                original_filter = search_request.filters.model_dump()
            elif hasattr(search_request.filters, "dict"):
                original_filter = search_request.filters.dict()
            else:
                original_filter = search_request.filters
            logger.info(f"Original OpenAI filter: {original_filter}")

            # Convert OpenAI filter format to the appropriate format based on vector store
            if CONFIG.vector_store.name == "milvus":
                # Convert to Milvus string format
                filter_expr = _convert_openai_filter_to_milvus_string(
                    search_request.filters
                )
                logger.info(f"✓ Converted filter to Milvus format: {filter_expr}")
            elif CONFIG.vector_store.name == "elasticsearch":
                # Convert to Elasticsearch query DSL format
                filter_expr = _convert_openai_filter_to_elasticsearch(
                    search_request.filters
                )
                logger.info(
                    f"✓ Converted filter to Elasticsearch format: {json.dumps(filter_expr, indent=2)}"
                )
            else:
                logger.warning(
                    f"Unsupported vector store: {CONFIG.vector_store.name}. Filter will be empty."
                )
                filter_expr = ""
        else:
            logger.info("No filters provided in request")

        # Process query to handle multimodal content
        query_processed = search_request.query
        if isinstance(search_request.query, list):
            content_list = []
            for content_item in search_request.query:
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
                    content_list.append(content_item)
            query_processed = content_list

        # Extract bearer token from Authorization header
        vdb_auth_token = _extract_vdb_auth_token(request)

        # Get the original query string for response
        original_query = (
            search_request.query
            if isinstance(search_request.query, str)
            else "multimodal query"
        )

        # Log the parameters being passed to backend
        logger.info(
            f"🔍 Calling backend search with parameters:\n"
            f"  - Collection: {vector_store_id}\n"
            f"  - Query rewriting: {search_request.rewrite_query}\n"
            f"  - Reranker enabled: {enable_reranker}\n"
            f"  - Max results: {search_request.max_num_results}\n"
            f"  - VDB top_k: {vdb_top_k}\n"
            f"  - Confidence threshold: {confidence_threshold}\n"
            f"  - Filter expression: {filter_expr if filter_expr else '(none)'}"
        )

        # Call the internal search method
        internal_response = await NVIDIA_RAG.search(
            query=query_processed,
            messages=[],  # OpenAI format doesn't include conversation history
            vdb_auth_token=vdb_auth_token,
            reranker_top_k=search_request.max_num_results,
            vdb_top_k=vdb_top_k,
            collection_names=[vector_store_id],
            vdb_endpoint=CONFIG.vector_store.url,
            enable_query_rewriting=search_request.rewrite_query,
            enable_reranker=enable_reranker,
            enable_filter_generator=False,  # OpenAI format uses explicit filters
            embedding_model=CONFIG.embeddings.model_name.strip('"'),
            embedding_endpoint=CONFIG.embeddings.server_url.strip('"'),
            reranker_model=reranker_model,
            reranker_endpoint=reranker_endpoint,
            filter_expr=filter_expr,
            confidence_threshold=confidence_threshold,
        )

        # Transform internal response to OpenAI format
        # internal_response is a Citations object (Pydantic model), not a dict
        data = []
        results = (
            internal_response.results if hasattr(internal_response, "results") else []
        )

        logger.info(f"✓ Backend returned {len(results)} results")

        for result in results:
            # Generate file_id from document_id or document_name
            # result is a SourceResult object (Pydantic model)
            file_id = (
                result.document_id
                if result.document_id
                else f"file_{abs(hash(result.document_name))}"
            )

            # Extract content
            content_text = result.content
            content_list = [
                VectorStoreSearchResultContent(type="text", text=content_text)
            ]

            # Map metadata to attributes
            # metadata is a SourceMetadata object (Pydantic model)
            metadata = result.metadata
            attributes = {
                "document_type": result.document_type,
                "page_number": metadata.page_number,
                "language": metadata.language,
                "date_created": metadata.date_created,
                "last_modified": metadata.last_modified,
                "description": metadata.description,
                "height": metadata.height,
                "width": metadata.width,
                "location": metadata.location,
                "location_max_dimensions": metadata.location_max_dimensions,
            }

            # Add content_metadata if present
            if metadata.content_metadata:
                attributes["content_metadata"] = metadata.content_metadata

            search_result_item = VectorStoreSearchResultItem(
                file_id=file_id,
                filename=result.document_name,
                score=result.score,
                attributes=attributes,
                content=content_list,
            )
            data.append(search_result_item)

        # Create OpenAI-compatible response
        openai_response = VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=original_query,
            data=data,
            has_more=False,  # Pagination not implemented yet
            next_page=None,
        )

        return JSONResponse(
            content=openai_response.model_dump(),
            status_code=ErrorCodeMapping.SUCCESS,
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during vector store search. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."},
            status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
        )
    except APIError as e:
        # Handle APIError with specific status codes
        status_code = getattr(e, "code", ErrorCodeMapping.INTERNAL_SERVER_ERROR)
        logger.error(
            "API Error from POST /vector_stores/{vector_store_id}/search endpoint. Error details: %s",
            e,
        )
        return JSONResponse(content={"message": str(e)}, status_code=status_code)
    except ValueError as e:
        # Handle validation errors
        logger.warning(
            "Validation error in /vector_stores/{vector_store_id}/search endpoint: %s",
            e,
        )
        return JSONResponse(
            content={"message": str(e)},
            status_code=ErrorCodeMapping.BAD_REQUEST,
        )
    except Exception as e:
        logger.error(
            "Error from POST /vector_stores/{vector_store_id}/search endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return JSONResponse(
            content={
                "message": "Error occurred while searching vector store. " + str(e)
            },
            status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
        )


# V2 endpoint for vector store search (OpenAI-compatible)
# NOTE: This endpoint is ONLY available in v2, not in v1
@v2_router.post(
    "/vector_stores/{vector_store_id}/search",
    tags=["Retrieval APIs"],
    response_model=VectorStoreSearchResponse,
    responses={
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid request parameters"}
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
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        },
    },
)
async def vector_store_search(
    request: Request,
    vector_store_id: str,
    search_request: VectorStoreSearchRequest,
) -> JSONResponse:
    """
    OpenAI-compatible vector store search endpoint (v2 only).

    This is the primary OpenAI-compatible endpoint for vector store search.
    Search within a vector store using natural language queries with full OpenAI API compatibility.

    **Note:** This endpoint is exclusive to the v2 API and is not available in v1.

    Args:
        request: FastAPI request object
        vector_store_id: The ID of the vector store (collection name) to search
        search_request: Search request parameters in OpenAI format

    Returns:
        JSONResponse: Search results in OpenAI-compatible format
    """
    return await _vector_store_search_impl(request, vector_store_id, search_request)


@app.get(
    "/summary",
    tags=["Retrieval APIs"],
    response_model=SummaryResponse,
    description="Retrieve document summary from the collection. Fetches pre-generated summaries with support for both blocking and non-blocking modes.",
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
    "Retrieve document summary from the collection. Fetches pre-generated summaries with support for both blocking and non-blocking modes."

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


# Manually add all v1 routes to v1_router to make them available at /v1/* paths
# This allows endpoints to be accessible at both root level (/) and with /v1 prefix
v1_router.add_api_route(
    "/health",
    health_check,
    methods=["GET"],
    response_model=RAGHealthResponse,
    tags=["Health APIs"],
)
v1_router.add_api_route(
    "/metrics",
    metrics_endpoint,
    methods=["GET"],
)
v1_router.add_api_route(
    "/configuration",
    get_configuration,
    methods=["GET"],
    response_model=ConfigurationResponse,
    tags=["Health APIs"],
)
v1_router.add_api_route(
    "/generate",
    generate_answer,
    methods=["POST"],
    response_model=ChainResponse,
    tags=["RAG APIs"],
)
v1_router.add_api_route(
    "/chat/completions",
    v1_chat_completions,
    methods=["POST"],
    response_model=ChainResponse,
    tags=["RAG APIs"],
)
v1_router.add_api_route(
    "/search",
    document_search,
    methods=["POST"],
    response_model=Citations,
    tags=["Retrieval APIs"],
)
v1_router.add_api_route(
    "/summary",
    get_summary,
    methods=["GET"],
    response_model=SummaryResponse,
    tags=["Retrieval APIs"],
)

# Mount routers
# v1_router is mounted to provide /v1/* endpoints
app.include_router(v1_router)
# v2_router is mounted to provide /v2/* endpoints (OpenAI-compatible)
app.include_router(v2_router)

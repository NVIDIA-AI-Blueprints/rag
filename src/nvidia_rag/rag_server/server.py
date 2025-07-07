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
import logging
import os
import time
import bleach
from typing import Any, Dict, Optional, List, Generator
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, Field, constr, validator, field_validator, model_validator

from nvidia_rag.rag_server.main import NvidiaRAG
from nvidia_rag.rag_server.response_generator import Message, ChainResponse, Citations
from nvidia_rag.utils.common import get_config
from nvidia_rag.rag_server.health import check_all_services_health, print_health_report

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

settings = get_config()
model_params = settings.llm.get_model_parameters()
default_max_tokens = model_params["max_tokens"]
default_temperature = model_params["temperature"]
default_top_p = model_params["top_p"]

logger.debug(f"default_max_tokens: {default_max_tokens}")
logger.debug(f"default_temperature: {default_temperature}")
logger.debug(f"default_top_p: {default_top_p}")

tags_metadata = [
    {
        "name": "Health APIs",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Retrieval APIs", "description": "APIs for retrieving document chunks for a query."},
    {"name": "RAG APIs", "description": "APIs for retrieval followed by generation."},
]

# create the FastAPI server
app = FastAPI(root_path=f"/v1", title="APIs for NVIDIA RAG Server",
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

NVIDIA_RAG = NvidiaRAG()

settings = get_config()
metrics = None
if settings.tracing.enabled:
    from .tracing import instrument
    metrics = instrument(app, settings)

class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    messages: List[Message] = Field(
        ...,
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    use_knowledge_base: bool = Field(default=True, description="Whether to use a knowledge base")
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
    max_tokens: int = Field(
        default_max_tokens,
        description="The maximum number of tokens to generate in any given call. "
        "Note that the model is not aware of this value, "
        " and generation will simply stop at the number of tokens specified.",
        ge=0,
        le=128000,
        format="int64",
    )
    reranker_top_k: int = Field(
        description="The maximum number of documents to return in the response.",
        default=settings.retriever.top_k,
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=settings.retriever.vdb_top_k,
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
        default=settings.vector_store.url
    )
    # TODO: Remove this field in the future
    collection_name: str = Field(
        description="Name of collection to be used for inference.",
        default="",
        max_length=4096,
        pattern=r'[\s\S]*',
        deprecated=True
    )
    collection_names: List[str] = Field(
        default=[settings.vector_store.default_collection_name],
        description="Name of the collections in the vector database.",
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=settings.query_rewriter.enable_query_rewriter,
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=settings.ranking.enable_reranker,
    )
    enable_guardrails: bool = Field(
        description="Enable or disable guardrailing of queries/responses.",
        default=settings.enable_guardrails,
    )
    enable_citations: bool = Field(
        description="Enable or disable citations as part of response.",
        default=settings.enable_citations,
    )
    enable_vlm_inference: bool = Field(
        description="Enable or disable VLM inference.",
        default=settings.enable_vlm_inference,
    )
    model: str = Field(
        description="Name of NIM LLM model to be used for inference.",
        default=settings.llm.model_name.strip('"'),
        max_length=4096,
        pattern=r'[\s\S]*',
    )
    llm_endpoint: str = Field(
        description="Endpoint URL for the llm model server.",
        default=settings.llm.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=settings.embeddings.model_name.strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: Optional[str] = Field(
        description="Endpoint URL for the embedding model server.",
        default=settings.embeddings.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=settings.ranking.model_name.strip('"'),
        max_length=256,
    )
    reranker_endpoint: Optional[str] = Field(
        description="Endpoint URL for the reranker model server.",
        default=settings.ranking.server_url.strip('"'),
        max_length=2048,
    )
    vlm_model: str = Field(
        description="Name of the VLM model used for inference.",
        default=settings.vlm.model_name.strip('"'),
        max_length=256,
    )
    vlm_endpoint: Optional[str] = Field(
        description="Endpoint URL for the VLM model server.",
        default=settings.vlm.server_url.strip('"'),
        max_length=2048,
    )

    # seed: int = Field(42, description="If specified, our system will make a best effort to sample deterministically,
    #       such that repeated requests with the same seed and parameters should return the same result.")
    # bad: List[str] = Field(None, description="A word or list of words not to use. The words are case sensitive.")
    stop: List[constr(max_length=256)] = Field(
        description="A string or a list of strings where the API will stop generating further tokens."
        "The returned text will not contain the stop sequence.",
        max_items=256,
        default=[],
    )
    # stream: bool = Field(True, description="If set, partial message deltas will be sent.
    #           Tokens will be sent as data-only server-sent events (SSE) as they become available
    #           (JSON responses are prefixed by data:), with the stream terminated by a data: [DONE] message.")

    filter_expr: str = Field(
        description="Filter expression to filter the retrieved documents from Milvus collection.",
        default='',
        max_length=4096,
        pattern=r'[\s\S]*',
    )

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

    query: str = Field(
        description="The content or keywords to search for within documents.",
        max_length=131072,
        pattern=r'[\s\S]*',
        default="Tell me something interesting",
    )
    reranker_top_k: int = Field(
        description="Number of document chunks to retrieve.",
        default=int(settings.retriever.top_k),
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=settings.retriever.vdb_top_k,
        ge=0,
        le=400,
        format="int64",
    )
    vdb_endpoint: str = Field(
        description="Endpoint url of the vector database server.",
        default=settings.vector_store.url
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
        pattern=r'[\s\S]*',
        deprecated=True
    )
    collection_names: List[str] = Field(
        default=[settings.vector_store.default_collection_name],
        description="Name of the collections in the vector database.",
    )
    messages: List[Message] = Field(
        default=[],
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=settings.query_rewriter.enable_query_rewriter,
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=settings.ranking.enable_reranker,
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=settings.embeddings.model_name.strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: str = Field(
        description="Endpoint URL for the embedding model server.",
        default=settings.embeddings.server_url.strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=settings.ranking.model_name.strip('"'),
        max_length=256,
    )
    reranker_endpoint: Optional[str] = Field(
        description="Endpoint URL for the reranker model server.",
        default=settings.ranking.server_url.strip('"'),
        max_length=2048,
    )

    filter_expr: str = Field(
        description="Filter expression to filter the retrieved documents from Milvus collection.",
        default='',
        max_length=4096,
        pattern=r'[\s\S]*',
    )

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
    """Represents a summary of a document."""

    message: str = Field(
        default="",
        description="Message of the summary"
    )

    status: str = Field(
        default="",
        description="Status of the summary"
    )

    summary: str = Field(
        default="",
        description="Summary of the document"
    )
    file_name: str = Field(
        default="",
        description="Name of the document"
    )
    collection_name: str = Field(
        default="",
        description="Name of the collection"
    )


# Define the service health models in server.py
class BaseServiceHealthInfo(BaseModel):
    """Base health info model with common fields for all services"""
    service: str
    url: str
    status: str
    latency_ms: float = 0
    error: Optional[str] = None

class DatabaseHealthInfo(BaseServiceHealthInfo):
    """Health info specific to database services"""
    collections: Optional[int] = None

class StorageHealthInfo(BaseServiceHealthInfo):
    """Health info specific to object storage services"""
    buckets: Optional[int] = None
    message: Optional[str] = None

class NIMServiceHealthInfo(BaseServiceHealthInfo):
    """Health info specific to NIM services (LLM, embeddings, etc.)"""
    message: Optional[str] = None
    http_status: Optional[int] = None

class HealthResponse(BaseModel):
    """Overall health response with specialized fields for each service type"""
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="Service is up.")
    databases: List[DatabaseHealthInfo] = Field(default_factory=list)
    object_storage: List[StorageHealthInfo] = Field(default_factory=list)
    nim: List[NIMServiceHealthInfo] = Field(default_factory=list)  # Unified category for NIM services


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors(), exclude={"input"})},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health APIs"],
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
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
    health_results = await NVIDIA_RAG.health(check_dependencies)
    response = HealthResponse(**health_results)

    # Only perform detailed service checks if requested
    if check_dependencies:
        try:
            print_health_report(health_results)

            # Process databases
            if "databases" in health_results:
                response.databases = [
                    DatabaseHealthInfo(**service)
                    for service in health_results["databases"]
                ]

            # Process object_storage
            if "object_storage" in health_results:
                response.object_storage = [
                    StorageHealthInfo(**service)
                    for service in health_results["object_storage"]
                ]

            # Process nim services
            if "nim" in health_results:
                response.nim = [
                    NIMServiceHealthInfo(**service)
                    for service in health_results["nim"]
                ]

        except Exception as e:
            logger.error(f"Error during dependency health checks: {str(e)}")
    else:
        logger.info("Skipping dependency health checks as check_dependencies=False")

    return response


@app.post(
    "/generate",
    tags=["RAG APIs"],
    response_model=ChainResponse,
    responses={
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The client cancelled the request"
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def generate_answer(request: Request, prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""
    generate_start_time = time.time()

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        # Convert messages to list of dicts
        messages_dict = [{'role': msg.role, 'content': msg.content} for msg in prompt.messages]

        # Get the streaming generator from NVIDIA_RAG.generate
        response_generator = NVIDIA_RAG.generate(
            messages=messages_dict,
            use_knowledge_base=prompt.use_knowledge_base,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            max_tokens=prompt.max_tokens,
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
            model=prompt.model,
            llm_endpoint=prompt.llm_endpoint,
            embedding_model=prompt.embedding_model,
            embedding_endpoint=prompt.embedding_endpoint,
            reranker_model=prompt.reranker_model,
            reranker_endpoint=prompt.reranker_endpoint,
            vlm_model=prompt.vlm_model,
            vlm_endpoint=prompt.vlm_endpoint,
            filter_expr=prompt.filter_expr,
        )

        # Wrap the generator with TTFT calculation and buffering fixes
        ttft_generator = optimized_streaming_wrapper(response_generator, generate_start_time)

        # Return streaming response with proper headers to prevent buffering
        return StreamingResponse(
            ttft_generator,
            media_type="text/event-stream"
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during response generation. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)

    except Exception as e:
        logger.error("Error from /generate endpoint. Error details: %s", e,
                     exc_info=logger.getEffectiveLevel() <= logging.DEBUG)

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
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def v1_chat_completions(request: Request, prompt: Prompt) -> StreamingResponse:
    """ Just an alias function to /generate endpoint which is openai compatible """

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
                    "example": {
                        "detail": "The client cancelled the request"
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def document_search(request: Request, data: DocumentSearch) -> Dict[str, List[Dict[str, Any]]]:
    """Search for the most relevant documents for the given search parameters."""

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        messages_dict = [{'role': msg.role, 'content': msg.content} for msg in data.messages]
        return NVIDIA_RAG.search(
            query=data.query,
            messages=messages_dict,
            reranker_top_k=data.reranker_top_k,
            vdb_top_k=data.vdb_top_k,
            collection_name=data.collection_name,
            collection_names=data.collection_names,
            vdb_endpoint=data.vdb_endpoint,
            enable_query_rewriting=data.enable_query_rewriting,
            enable_reranker=data.enable_reranker,
            embedding_model=data.embedding_model,
            embedding_endpoint=data.embedding_endpoint,
            reranker_model=data.reranker_model,
            reranker_endpoint=data.reranker_endpoint,
            filter_expr=data.filter_expr,
        )

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during document search. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from POST /search endpoint. Error details: %s", e,
                     exc_info=logger.getEffectiveLevel() <= logging.DEBUG)
        return JSONResponse(content={"message": "Error occurred while searching documents. " + str(e)}, status_code=500)


@app.get(
    "/summary",
    tags=["Retrieval APIs"],
    response_model=SummaryResponse,
    responses={
        404: {
            "description": "Summary not found (non-blocking mode)",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Summary for example.pdf not found. Set wait=true to wait for generation.",
                        "status": "pending"
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
                        "status": "timeout"
                    }
                }
            },
        },
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The client cancelled the request"
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Error occurred while getting summary.",
                        "error": "Internal server error details"
                    }
                }
            },
        }
    },
)
async def get_summary(
    request: Request,
    collection_name: str,
    file_name: str,
    blocking: bool = False,
    timeout: int = 300
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
        timeout (int, optional): Maximum time to wait in seconds. Defaults to 300

    Returns:
        JSONResponse: Contains either:
            - Summary data: {"summary": str, "file_name": str, "collection_name": str}
            - Error message: {"message": str, "status": str}

    Status Codes:
        404: Summary not found (non-blocking mode)
        408: Timeout waiting for summary (blocking mode)
        499: Client Closed Request
        500: Internal server error
    """

    try:
        response = await NVIDIA_RAG.get_summary(collection_name=collection_name, file_name=file_name, blocking=blocking, timeout=timeout)

        if response.get("status") == "FAILED":
            return JSONResponse(content=response, status_code=404)
        elif response.get("status") == "TIMEOUT":
            return JSONResponse(content=response, status_code=408)
        elif response.get("status") == "SUCCESS":
            return JSONResponse(content=response, status_code=200)
        elif response.get("status") == "ERROR":
            return JSONResponse(content=response, status_code=500)

    except Exception as e:
        logger.error("Error from GET /summary endpoint. Error details: %s", e)
        return JSONResponse(
            content={
                "message": "Error occurred while getting summary.",
                "error": str(e)
            },
            status_code=500
        )


async def optimized_streaming_wrapper(
        generator: Generator,
        start_time: float
    ):
    """
    Optimized wrapper for streaming generator to calculate TTFT with minimal buffering.
    
    Args:
        generator: The streaming generator from NVIDIA_RAG.generate()
        start_time: The timestamp when the request started
        
    Yields:
        The same chunks as the original generator, but with proper flushing and timing
    """
    token_count = 0
    
    async for chunk in generator:
        current_time = time.time()
        token_count += 1
        
        if token_count == 1:
            ttft = (current_time - start_time) * 1000  # Convert to milliseconds
            logger.info("    == RAG Time to First Token (TTFT): %.2f ms ==", ttft)
            
        # Yield the chunk immediately without additional processing
        yield chunk

        await asyncio.sleep(0)  # Allow event loop to process

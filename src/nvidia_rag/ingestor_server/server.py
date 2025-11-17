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

"""The definition of the NVIDIA RAG Ingestion server.
POST /documents: Upload documents to the vector store.
GET /status: Get the status of an ingestion task.
PATCH /documents: Update documents in the vector store.
GET /documents: Get documents in the vector store.
DELETE /documents: Delete documents from the vector store.
GET /collections: Get collections in the vector store.
POST /collections: Create collections in the vector store.
DELETE /collections: Delete collections in the vector store.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from nvidia_rag.ingestor_server.main import Mode, NvidiaRAGIngestor
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import (
    DatabaseHealthInfo,
    IngestorHealthResponse,
    NIMServiceHealthInfo,
    ProcessingHealthInfo,
    StorageHealthInfo,
    TaskManagementHealthInfo,
)
from nvidia_rag.utils.metadata_validation import MetadataField

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health APIs",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {
        "name": "Ingestion APIs",
        "description": "APIs for uploading, deletion and listing documents.",
    },
    {
        "name": "Vector DB APIs",
        "description": "APIs for managing collections in vector database.",
    },
]


# create the FastAPI server
app = FastAPI(
    root_path="/v1",
    title="APIs for NVIDIA RAG Ingestion Server",
    description="This API schema describes all the Ingestion endpoints exposed for NVIDIA RAG server Blueprint",
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


EXAMPLE_DIR = "./"

# Initialize configuration and ingestor
CONFIG = NvidiaRAGConfig()
NV_INGEST_INGESTOR = NvidiaRAGIngestor(mode=Mode.SERVER, config=CONFIG)


class SplitOptions(BaseModel):
    """Options for splitting the document into smaller chunks."""

    chunk_size: int = Field(
        CONFIG.nv_ingest.chunk_size, description="Number of units per split."
    )
    chunk_overlap: int = Field(
        CONFIG.nv_ingest.chunk_overlap,
        description="Number of overlapping units between consecutive splits.",
    )

def _extract_vdb_auth_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header (e.g., 'Bearer <token>')."""
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


class CustomMetadata(BaseModel):
    """Custom metadata to be added to the document."""

    filename: str = Field(..., description="Name of the file.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata to be added to the document."
    )


class DocumentCatalogMetadata(BaseModel):
    """Catalog metadata for a specific document during upload."""

    filename: str = Field(..., description="Name of the file to apply metadata to.")
    description: str | None = Field(
        None, description="Description of the document for catalog purposes."
    )
    tags: list[str] | None = Field(
        None, description="Tags for categorizing and discovering the document."
    )


class SummaryOptions(BaseModel):
    """Advanced options for summary generation (used with generate_summary=True).

    Page Filter formats:
    - Ranges: [[1, 10], [20, 30]] for pages 1-10 and 20-30
    - Negative ranges: [[-10, -1]] for last 10 pages (Pythonic indexing where -1 is last page)
    - Even/odd: "even" or "odd" for all even or odd pages

    Examples:
    - [[1, 10], [-5, -1]] selects first 10 pages and last 5 pages
    - [[-1, -1]] selects only the last page
    """

    page_filter: list[list[int]] | str | None = Field(
        None,
        description=(
            "Page selection specification for summarization. Supports: "
            "list[list[int]] (ranges as [start,end] with negative indexing supported), "
            "str ('even' or 'odd'). Only applicable when generate_summary is enabled."
        ),
        examples=[
            [[1, 10]],
            [[1, 10], [20, 30]],
            [[-10, -1]],
            [[1, 10], [-5, -1]],
            "even",
            "odd",
        ],
    )

    shallow_summary: bool = Field(
        default=False,
        description=(
            "Enable fast summary generation using text-only extraction. "
            "When True, performs text-only NV-Ingest extraction first to generate summaries quickly, "
            "then continues with full multimodal ingestion (tables, images, charts) for VDB. "
            "Summary generation starts immediately with text-only results while full ingestion proceeds in parallel. "
            "Default: False (summary generated after full multimodal ingestion)."
        ),
    )

    summarization_strategy: str | None = Field(
        default=None,
        description=(
            "Summarization strategy for combining document chunks. "
            "'single': Summarize entire document in one pass (truncates if exceeds max_chunk_length). "
            "'hierarchical': Parallel tree-based summarization (fastest for large documents). "
            "If not specified, uses default sequential iterative processing."
        ),
    )

    @model_validator(mode="after")
    def validate_page_filter_and_strategy(self) -> "SummaryOptions":
        """Validate page_filter format and summarization_strategy."""
        # Validate page_filter
        if self.page_filter is not None:
            page_filter = self.page_filter

            if isinstance(page_filter, str):
                if page_filter.lower() not in ["even", "odd"]:
                    raise ValueError(
                        f"Invalid page_filter string '{page_filter}'. Supported: 'even', 'odd'"
                    )
                self.page_filter = page_filter.lower()

            elif isinstance(page_filter, list):
                if not page_filter:
                    raise ValueError("Page filter range list cannot be empty")

                # Must be list of lists (ranges)
                if not all(isinstance(item, list) for item in page_filter):
                    raise ValueError(
                        "Page filter must contain ranges as [start, end]. "
                        "Got mixed types or non-list items."
                    )

                for i, range_item in enumerate(page_filter):
                    if len(range_item) != 2:
                        raise ValueError(
                            f"Range {i} must have exactly 2 elements [start, end], got {len(range_item)}"
                        )
                    start, end = range_item
                    if not isinstance(start, int) or not isinstance(end, int):
                        raise ValueError(
                            f"Range {i} must contain integers, got [{type(start).__name__}, {type(end).__name__}]"
                        )
                    # Validate page numbers
                    if start == 0 or end == 0:
                        raise ValueError(
                            f"Range {i}: page numbers cannot be 0. Use 1-based indexing or negative for last pages."
                        )
                    # For negative ranges: start must be <= end (e.g., [-10, -1] is valid, [-1, -10] is not)
                    if start < 0 and end < 0 and start > end:
                        raise ValueError(
                            f"Range {i}: invalid negative range [{start}, {end}]. "
                            f"Use [-10, -1] for last 10 pages, not [-1, -10]."
                        )
                    # For positive ranges: start must be <= end
                    if start > 0 and end > 0 and start > end:
                        raise ValueError(
                            f"Range {i}: start must be <= end, got [{start}, {end}]"
                        )
                    # Mixed positive/negative not allowed
                    if (start < 0 and end > 0) or (start > 0 and end < 0):
                        raise ValueError(
                            f"Range {i}: cannot mix positive and negative indexing in same range. Got [{start}, {end}]"
                        )
            else:
                raise ValueError(
                    f"Invalid page_filter type: {type(page_filter).__name__}. "
                    f"Expected: list[list[int]] (ranges) or str ('even'/'odd')"
                )

        # Validate summarization_strategy
        if self.summarization_strategy is not None:
            allowed_strategies = ["single", "hierarchical"]
            if self.summarization_strategy not in allowed_strategies:
                raise ValueError(
                    f"Invalid summarization_strategy: '{self.summarization_strategy}'. "
                    f"Allowed values: {allowed_strategies}"
                )

        return self


class DocumentUploadRequest(BaseModel):
    """Request model for uploading and processing documents."""

    vdb_endpoint: str = Field(
        os.getenv("APP_VECTORSTORE_URL", "http://localhost:19530"),
        description="URL of the vector database endpoint.",
        exclude=True,  # WAR to hide it from openapi schema
    )

    @model_validator(mode="after")
    def validate_summary_configuration(self) -> "DocumentUploadRequest":
        """Validate that summary_options is only used when generate_summary is True."""
        if self.summary_options and not self.generate_summary:
            raise ValueError(
                "summary_options can only be provided when generate_summary=True. "
                "Either set generate_summary=True or remove summary_options."
            )
        return self

    collection_name: str = Field(
        "multimodal_data", description="Name of the collection in the vector database."
    )

    blocking: bool = Field(False, description="Enable/disable blocking ingestion.")

    split_options: SplitOptions = Field(
        default_factory=SplitOptions,
        description="Options for splitting documents into smaller parts before embedding.",
    )

    custom_metadata: list[CustomMetadata] = Field(
        default_factory=list, description="Custom metadata to be added to the document."
    )

    generate_summary: bool = Field(
        default=False,
        description="Enable/disable summary generation for each uploaded document.",
    )

    documents_catalog_metadata: list[DocumentCatalogMetadata] = Field(
        default_factory=list,
        description="Catalog metadata (description, tags) for specific documents. Optional per-document catalog information.",
    )

    summary_options: SummaryOptions | None = Field(
        None,
        description="Advanced options for summary generation (e.g., page filtering). Only used when generate_summary is True.",
    )

    # Reserved for future use
    # embedding_model: str = Field(
    #     os.getenv("APP_EMBEDDINGS_MODELNAME", ""),
    #     description="Identifier for the embedding model to be used."
    # )

    # embedding_endpoint: str = Field(
    #     os.getenv("APP_EMBEDDINGS_SERVERURL", ""),
    #     description="URL of the embedding service endpoint."
    # )


class UploadedDocument(BaseModel):
    """Model representing an individual uploaded document."""

    # Reserved for future use
    # document_id: str = Field("", description="Unique identifier for the document.")
    document_name: str = Field("", description="Name of the document.")
    # Reserved for future use
    # size_bytes: int = Field(0, description="Size of the document in bytes.")
    metadata: dict[str, Any] = Field({}, description="Metadata of the document.")
    document_info: dict[str, Any] = Field({}, description="Document information.")


class FailedDocument(BaseModel):
    """Model representing an individual uploaded document."""

    document_name: str = Field("", description="Name of the document.")
    error_message: str = Field(
        "", description="Error message from the ingestion process."
    )


class UploadDocumentResponse(BaseModel):
    """Response model for uploading a document."""

    message: str = Field(
        "", description="Message indicating the status of the request."
    )
    total_documents: int = Field(0, description="Total number of documents uploaded.")
    documents_completed: int = Field(0, description="Number of documents completed.")
    batches_completed: int = Field(0, description="Number of batches completed.")
    documents: list[UploadedDocument] = Field(
        [], description="List of uploaded documents."
    )
    failed_documents: list[FailedDocument] = Field(
        [], description="List of failed documents."
    )
    validation_errors: list[dict[str, Any]] = Field(
        [], description="List of validation errors."
    )


class IngestionTaskResponse(BaseModel):
    """Response model for uploading a document."""

    message: str = Field(
        "", description="Message indicating the status of the request."
    )
    task_id: str = Field("", description="Task ID of the ingestion process.")


class IngestionTaskStatusResponse(BaseModel):
    """Response model for getting the status of an ingestion task."""

    state: str = Field("", description="State of the ingestion task.")
    result: UploadDocumentResponse = Field(
        ..., description="Result of the ingestion task."
    )


class DocumentListResponse(BaseModel):
    """Response model for uploading a document."""

    message: str = Field(
        "", description="Message indicating the status of the request."
    )
    total_documents: int = Field(0, description="Total number of documents uploaded.")
    documents: list[UploadedDocument] = Field(
        [], description="List of uploaded documents."
    )


class UploadedCollection(BaseModel):
    """Model representing an individual uploaded document."""

    collection_name: str = Field("", description="Name of the collection.")
    num_entities: int = Field(
        0, description="Number of rows or entities in the collection."
    )
    metadata_schema: list[dict[str, Any]] = Field(
        [], description="Metadata schema of the collection."
    )
    collection_info: dict[str, Any] = Field(
        {}, description="Collection info of the collection."
    )


class CollectionListResponse(BaseModel):
    """Response model for uploading a document."""

    message: str = Field(
        "", description="Message indicating the status of the request."
    )
    total_collections: int = Field(
        0, description="Total number of collections uploaded."
    )
    collections: list[UploadedCollection] = Field(
        [], description="List of uploaded collections."
    )


class CreateCollectionRequest(BaseModel):
    """Request model for creating a collection."""

    vdb_endpoint: str = Field(
        os.getenv("APP_VECTORSTORE_URL", ""),
        description="Endpoint of the vector database.",
    )
    collection_name: str = Field(
        os.getenv("COLLECTION_NAME", ""), description="Name of the collection."
    )
    embedding_dimension: int = Field(
        os.getenv("APP_EMBEDDINGS_DIMENSIONS", 2048),
        description="Embedding dimension of the collection.",
    )
    metadata_schema: list[MetadataField] = Field(
        [], description="Metadata schema of the collection."
    )
    description: str = Field(
        "", description="Human-readable description of the collection"
    )
    tags: list[str] = Field([], description="Tags for categorization and search")
    owner: str = Field("", description="Owner team or person")
    created_by: str = Field("", description="Username/email of creator")
    business_domain: str = Field(
        "", description="Business domain (Finance, Engineering, HR, Legal, etc.)"
    )
    status: str = Field(
        "Active", description="Collection status (Active, Archived, Stale, Pending)"
    )


class FailedCollection(BaseModel):
    """Model representing a collection that failed to be created or deleted."""

    collection_name: str = Field("", description="Name of the collection.")
    error_message: str = Field(
        "",
        description="Error message from the collection creation or deletion process.",
    )


class CollectionsResponse(BaseModel):
    """Response model for creation or deletion of collections in Milvus."""

    message: str = Field(..., description="Status message of the process.")
    successful: list[str] = Field(
        default_factory=list,
        description="List of successfully created or deleted collections.",
    )
    failed: list[FailedCollection] = Field(
        default_factory=list,
        description="List of collections that failed to be created or deleted.",
    )
    total_success: int = Field(
        0, description="Total number of collections successfully created or deleted."
    )
    total_failed: int = Field(
        0,
        description="Total number of collections that failed to be created or deleted.",
    )


class CreateCollectionResponse(BaseModel):
    """Response model for creation or deletion of a collection in Milvus."""

    message: str = Field(..., description="Status message of the process.")
    collection_name: str = Field(..., description="Name of the collection.")


class UpdateCollectionMetadataRequest(BaseModel):
    """Request model for updating collection metadata."""

    description: str | None = Field(None, description="Updated description")
    tags: list[str] | None = Field(None, description="Updated tags")
    owner: str | None = Field(None, description="Updated owner")
    business_domain: str | None = Field(None, description="Updated business domain")
    status: str | None = Field(None, description="Updated status")


class UpdateDocumentMetadataRequest(BaseModel):
    """Request model for updating document metadata."""

    description: str | None = Field(None, description="Updated description")
    tags: list[str] | None = Field(None, description="Updated tags")


class UpdateMetadataResponse(BaseModel):
    """Response model for metadata update operations."""

    message: str = Field(..., description="Status message")
    collection_name: str = Field(..., description="Collection name")


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    try:
        body = await request.json()
        logger.warning("Invalid incoming Request Body:", body)
    except Exception as e:
        print("Failed to read request body:", e)
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors(), exclude={"input"})},
    )


@app.get(
    "/health",
    response_model=IngestorHealthResponse,
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
    response = await NV_INGEST_INGESTOR.health(check_dependencies)

    # Only perform detailed service checks if requested
    if check_dependencies:
        try:
            from nvidia_rag.ingestor_server.health import print_health_report

            print_health_report(response)
        except Exception as e:
            logger.error(f"Error during dependency health checks: {str(e)}")
    else:
        logger.info("Skipping dependency health checks as check_dependencies=False")

    return response


async def parse_json_data(
    data: str = Form(
        ...,
        description="JSON data in string format containing metadata about the documents which needs to be uploaded.",
        examples=[json.dumps(DocumentUploadRequest().model_dump())],
        media_type="application/json",
    ),
) -> DocumentUploadRequest:
    try:
        json_data = json.loads(data)
        return DocumentUploadRequest(**json_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=UploadDocumentResponse,
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
        200: {
            "description": "Background Ingestion Started",
            "model": IngestionTaskResponse,
        },
    },
)
async def upload_document(
    request: Request,
    documents: list[UploadFile] = File(...),
    payload: DocumentUploadRequest = Depends(parse_json_data),
) -> UploadDocumentResponse | IngestionTaskResponse:
    """Upload a document to the vector store."""

    if not len(documents):
        raise Exception("No files provided for uploading.")

    try:
        # Extract bearer token from Authorization header (e.g., "Bearer <token>")
        vdb_auth_token = _extract_vdb_auth_token(request)

        # Store all provided file paths in a temporary directory (only unique files)
        all_file_paths, duplicate_validation_errors = await process_file_paths(
            documents, payload.collection_name
        )

        response_dict = await NV_INGEST_INGESTOR.upload_documents(
            filepaths=all_file_paths,
            vdb_auth_token=vdb_auth_token,
            **payload.model_dump(),
            additional_validation_errors=duplicate_validation_errors,
        )
        if not payload.blocking:
            return JSONResponse(
                content=IngestionTaskResponse(**response_dict).model_dump(),
                status_code=200,
            )

        return UploadDocumentResponse(**response_dict)
    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while uploading document {e}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client"}, status_code=499
        )
    except Exception as e:
        logger.error(
            f"Error from POST /documents endpoint. Ingestion of file failed with error: {e}"
        )
        return JSONResponse(
            content={"message": f"Ingestion of files failed with error: {e}"},
            status_code=500,
        )


@app.get(
    "/status",
    tags=["Ingestion APIs"],
    response_model=IngestionTaskStatusResponse,
)
async def get_task_status(task_id: str):
    """Get the status of an ingestion task."""

    logger.info(f"Getting status of task {task_id}")
    try:
        result = await NV_INGEST_INGESTOR.status(task_id)
        return IngestionTaskStatusResponse(
            state=result.get("state", "UNKNOWN"), result=result.get("result", {})
        )
    except KeyError as e:
        logger.error(f"Task {task_id} not found with error: {e}")
        return IngestionTaskStatusResponse(
            state="UNKNOWN", result={"message": "Task not found"}
        )


@app.patch(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=DocumentListResponse,
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
async def update_documents(
    request: Request,
    documents: list[UploadFile] = File(...),
    payload: DocumentUploadRequest = Depends(parse_json_data),
) -> DocumentListResponse:
    """Upload a document to the vector store. If the document already exists, it will be replaced."""

    try:
        # Extract bearer token from Authorization header (e.g., "Bearer <token>")
        vdb_auth_token = _extract_vdb_auth_token(request)

        # Store all provided file paths in a temporary directory (only unique files)
        all_file_paths, duplicate_validation_errors = await process_file_paths(
            documents, payload.collection_name
        )

        response_dict = await NV_INGEST_INGESTOR.update_documents(
            filepaths=all_file_paths,
            vdb_auth_token=vdb_auth_token,
            **payload.model_dump(),
            additional_validation_errors=duplicate_validation_errors,
        )
        if not payload.blocking:
            return JSONResponse(
                content=IngestionTaskResponse(**response_dict).model_dump(),
                status_code=200,
            )

        return UploadDocumentResponse(**response_dict)

    except asyncio.CancelledError:
        logger.error("Request cancelled while deleting and uploading document")
        return JSONResponse(
            content={"message": "Request was cancelled by the client"}, status_code=499
        )
    except Exception as e:
        logger.error(
            f"Error from PATCH /documents endpoint. Ingestion failed with error: {e}"
        )
        return JSONResponse(
            content={"message": f"Ingestion of files failed with error. {e}"},
            status_code=500,
        )


@app.get(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=DocumentListResponse,
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
async def get_documents(
    request: Request,
    collection_name: str = os.getenv("COLLECTION_NAME", ""),
    vdb_endpoint: str = Query(
        default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False
    ),
) -> DocumentListResponse:
    """Get list of document ingested in vectorstore."""
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        documents = NV_INGEST_INGESTOR.get_documents(collection_name, vdb_endpoint, vdb_auth_token)
        return DocumentListResponse(**documents)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching documents. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from GET /documents endpoint. Error details: %s", e)
        return JSONResponse(
            content={"message": f"Error occurred while fetching documents: {e}"},
            status_code=500,
        )


@app.delete(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=DocumentListResponse,
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
async def delete_documents(
    request: Request,
    document_names: list[str] | None = None,
    collection_name: str = os.getenv("COLLECTION_NAME"),
    vdb_endpoint: str = Query(
        default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False
    ),
) -> DocumentListResponse:
    if document_names is None:
        document_names = []
    """Delete a document from vectorstore."""
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        response = NV_INGEST_INGESTOR.delete_documents(
            document_names=document_names,
            collection_name=collection_name,
            vdb_endpoint=vdb_endpoint,
            include_upload_path=True,
            vdb_auth_token=vdb_auth_token,
        )
        return DocumentListResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(
            f"Request cancelled while deleting document: {document_names}, {str(e)}"
        )
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from DELETE /documents endpoint. Error details: %s", e)
        return JSONResponse(
            content={"message": f"Error deleting document {document_names}: {e}"},
            status_code=500,
        )


@app.get(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionListResponse,
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
async def get_collections(
    request: Request,
    vdb_endpoint: str = Query(
        default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False
    ),
) -> CollectionListResponse:
    """
    Endpoint to get a list of collection names from the Milvus server.
    Returns a list of collection names.
    """
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        response = NV_INGEST_INGESTOR.get_collections(vdb_endpoint, vdb_auth_token)
        return CollectionListResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from GET /collections endpoint. Error details: %s", e)
        return JSONResponse(
            content={
                "message": f"Error occurred while fetching collections. Error: {e}"
            },
            status_code=500,
        )


@app.post(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionsResponse,
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
    deprecated=True,
    description="This endpoint is deprecated. Use POST /collection instead. Custom metadata is not supported in this endpoint.",
)
async def create_collections(
    request: Request,
    vdb_endpoint: str = Query(
        default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False
    ),
    collection_names: list[str] | None = None,
    collection_type: str = "text",
    embedding_dimension: int = 2048,
) -> CollectionsResponse:
    if collection_names is None:
        collection_names = [os.getenv("COLLECTION_NAME")]
    """
    Endpoint to create a collection from the Milvus server.
    Returns status message.
    """
    logger.warning(
        "The endpoint POST /collections is deprecated and will be removed in a future release. "
        "Please use POST /collection instead. Custom metadata is not supported in this endpoint."
    )
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        response = NV_INGEST_INGESTOR.create_collections(
            collection_names, vdb_endpoint, embedding_dimension, vdb_auth_token
        )
        return CollectionsResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from POST /collections endpoint. Error details: %s", e)
        return JSONResponse(
            content={
                "message": f"Error occurred while creating collections. Error: {e}"
            },
            status_code=500,
        )


@app.post(
    "/collection",
    tags=["Vector DB APIs"],
    response_model=CreateCollectionResponse,
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

async def create_collection(request: Request, data: CreateCollectionRequest) -> CreateCollectionResponse:
    """
    Endpoint to create a collection with catalog metadata.
    Returns status message.
    """
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        response = NV_INGEST_INGESTOR.create_collection(
            collection_name=data.collection_name,
            vdb_endpoint=data.vdb_endpoint,
            embedding_dimension=data.embedding_dimension,
            metadata_schema=[field.model_dump() for field in data.metadata_schema],
            description=data.description,
            tags=data.tags,
            owner=data.owner,
            created_by=data.created_by,
            business_domain=data.business_domain,
            status=data.status,
            vdb_auth_token=vdb_auth_token,
        )
        return CreateCollectionResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from POST /collection endpoint. Error details: %s", e)
        return JSONResponse(
            content={
                "message": f"Error occurred while creating collection. Error: {e}"
            },
            status_code=500,
        )


@app.patch(
    "/collections/{collection_name}/metadata",
    tags=["Vector DB APIs"],
    response_model=UpdateMetadataResponse,
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
async def update_collection_metadata(
    collection_name: str,
    data: UpdateCollectionMetadataRequest,
) -> UpdateMetadataResponse:
    """Endpoint to update collection catalog metadata."""
    try:
        response = NV_INGEST_INGESTOR.update_collection_metadata(
            collection_name=collection_name,
            description=data.description,
            tags=data.tags,
            owner=data.owner,
            business_domain=data.business_domain,
            status=data.status,
        )
        return UpdateMetadataResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(
            f"Request cancelled while updating collection metadata. {str(e)}"
        )
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error(
            "Error from PATCH /collections/{collection_name}/metadata endpoint. Error: %s",
            e,
        )
        return JSONResponse(
            content={"message": f"Error updating collection metadata. Error: {e}"},
            status_code=500,
        )


@app.patch(
    "/collections/{collection_name}/documents/{document_name}/metadata",
    tags=["Vector DB APIs"],
    response_model=UpdateMetadataResponse,
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
async def update_document_metadata(
    collection_name: str,
    document_name: str,
    data: UpdateDocumentMetadataRequest,
) -> UpdateMetadataResponse:
    """Endpoint to update document catalog metadata."""
    try:
        response = NV_INGEST_INGESTOR.update_document_metadata(
            collection_name=collection_name,
            document_name=document_name,
            description=data.description,
            tags=data.tags,
        )
        return UpdateMetadataResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while updating document metadata. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error(
            "Error from PATCH /collections/{collection_name}/documents/{document_name}/metadata endpoint. Error: %s",
            e,
        )
        return JSONResponse(
            content={"message": f"Error updating document metadata. Error: {e}"},
            status_code=500,
        )


@app.delete(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionsResponse,
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
async def delete_collections(
    request: Request,
    vdb_endpoint: str = Query(
        default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False
    ),
    collection_names: list[str] = Query(default=None),
) -> CollectionsResponse:
    if collection_names is None:
        collection_names = [os.getenv("COLLECTION_NAME")]
    """
    Endpoint to delete a collection from the Milvus server.
    Returns status message.
    """
    try:
        # Extract vdb auth token and pass through to backend
        vdb_auth_token = _extract_vdb_auth_token(request)
        response = NV_INGEST_INGESTOR.delete_collections(
            collection_names=collection_names, vdb_endpoint=vdb_endpoint, vdb_auth_token=vdb_auth_token
        )
        return CollectionsResponse(**response)

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(
            content={"message": "Request was cancelled by the client."}, status_code=499
        )
    except Exception as e:
        logger.error("Error from DELETE /collections endpoint. Error details: %s", e)
        return JSONResponse(
            content={
                "message": f"Error occurred while deleting collections. Error: {e}"
            },
            status_code=500,
        )


async def process_file_paths(filepaths: list[UploadFile], collection_name: str):
    """Process the uploaded files and return the list of file paths.

    Args:
        filepaths: List of UploadFile objects from the client
        collection_name: Name of the collection to store files for

    Returns:
        tuple: (all_file_paths, duplicate_validation_errors)
            - all_file_paths: List of unique file paths (strings) where files are saved
            - duplicate_validation_errors: List of validation error dicts for duplicate files
    """

    base_upload_folder = Path(
        os.path.join(CONFIG.temp_dir, f"uploaded_files/{collection_name}")
    )
    base_upload_folder.mkdir(parents=True, exist_ok=True)
    all_file_paths = []

    # Track filenames to detect duplicates
    filename_counts = {}
    processed_filenames = set()

    for file in filepaths:
        upload_file = os.path.basename(file.filename)

        if not upload_file:
            raise RuntimeError("Error parsing uploaded filename.")

        # Count occurrences of each filename
        filename_counts[upload_file] = filename_counts.get(upload_file, 0) + 1

        # Only process the first occurrence of each filename
        if upload_file in processed_filenames:
            continue

        processed_filenames.add(upload_file)

        # Create a unique directory for each file
        unique_dir = base_upload_folder  # / str(uuid4())
        unique_dir.mkdir(parents=True, exist_ok=True)

        file_path = unique_dir / upload_file
        all_file_paths.append(str(file_path))

        # Copy uploaded file to upload_dir directory and pass that file path to
        # ingestor server
        with open(file_path, "wb") as f:
            file.file.seek(0)
            shutil.copyfileobj(file.file, f)

    # Create validation errors for duplicates
    duplicate_validation_errors = []
    duplicates = {name: count for name, count in filename_counts.items() if count > 1}

    if duplicates:
        logger.warning(
            f"Duplicate files detected: {len(duplicates)} unique filenames had duplicates. "
            f"Total files submitted: {len(filepaths)}, "
            f"Unique files to process: {len(all_file_paths)}"
        )
        for filename, count in duplicates.items():
            duplicate_count = count - 1  # Subtract 1 since we keep one copy
            logger.warning(
                f"File '{filename}' submitted {count} times, processing only 1 copy"
            )
            duplicate_validation_errors.append(
                {
                    "error": f"File '{filename}': Total of {duplicate_count} duplicate(s) found. Duplicates were discarded and 1 file is being processed.",
                    "metadata": {
                        "filename": filename,
                        "duplicate_count": duplicate_count,
                        "total_occurrences": count,
                    },
                }
            )

    return all_file_paths, duplicate_validation_errors

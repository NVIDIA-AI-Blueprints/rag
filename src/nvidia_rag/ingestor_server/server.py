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
import logging
import os
import json
import shutil
from pathlib import Path
from typing import List, Union, Dict, Any, Literal
from fastapi import UploadFile, Request, File, FastAPI, Form, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from nvidia_rag.utils.common import get_config
from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health APIs",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Ingestion APIs", "description": "APIs for uploading, deletion and listing documents."},
    {"name": "Vector DB APIs", "description": "APIs for managing collections in vector database."}
]


# create the FastAPI server
app = FastAPI(root_path=f"/v1", title="APIs for NVIDIA RAG Ingestion Server",
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

# Initialize the NVIngestIngestor class
NV_INGEST_INGESTOR = NvidiaRAGIngestor()
CONFIG = get_config()

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")


class SplitOptions(BaseModel):
    """Options for splitting the document into smaller chunks."""
    chunk_size: int = Field(CONFIG.nv_ingest.chunk_size, description="Number of units per split.")
    chunk_overlap: int = Field(CONFIG.nv_ingest.chunk_overlap, description="Number of overlapping units between consecutive splits.")


class CustomMetadata(BaseModel):
    """Custom metadata to be added to the document."""
    filename: str = Field(..., description="Name of the file.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to be added to the document.")


class DocumentUploadRequest(BaseModel):
    """Request model for uploading and processing documents."""

    vdb_endpoint: str = Field(
        os.getenv("APP_VECTORSTORE_URL", "http://localhost:19530"),
        description="URL of the vector database endpoint.",
        exclude=True # WAR to hide it from openapi schema
    )

    collection_name: str = Field(
        "multimodal_data",
        description="Name of the collection in the vector database."
    )

    blocking: bool = Field(
        False,
        description="Enable/disable blocking ingestion."
    )

    split_options: SplitOptions = Field(
        default_factory=SplitOptions,
        description="Options for splitting documents into smaller parts before embedding."
    )

    custom_metadata: List[CustomMetadata] = Field(
        default_factory=list,
        description="Custom metadata to be added to the document."
    )

    generate_summary: bool = Field(
        default=False,
        description="Enable/disable summary generation for each uploaded document."
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
    metadata: Dict[str, Any] = Field({}, description="Metadata of the document.")

class FailedDocument(BaseModel):
    """Model representing an individual uploaded document."""
    document_name: str = Field("", description="Name of the document.")
    error_message: str = Field("", description="Error message from the ingestion process.")

class UploadDocumentResponse(BaseModel):
    """Response model for uploading a document."""
    message: str = Field("", description="Message indicating the status of the request.")
    total_documents: int = Field(0, description="Total number of documents uploaded.")
    documents: List[UploadedDocument] = Field([], description="List of uploaded documents.")
    failed_documents: List[FailedDocument] = Field([], description="List of failed documents.")
    validation_errors: List[Dict[str, Any]] = Field([], description="List of validation errors.")

class IngestionTaskResponse(BaseModel):
    """Response model for uploading a document."""
    message: str = Field("", description="Message indicating the status of the request.")
    task_id: str = Field("", description="Task ID of the ingestion process.")

class IngestionTaskStatusResponse(BaseModel):
    """Response model for getting the status of an ingestion task."""
    state: str = Field("", description="State of the ingestion task.")
    result: UploadDocumentResponse = Field(..., description="Result of the ingestion task.")

class DocumentListResponse(BaseModel):
    """Response model for uploading a document."""
    message: str = Field("", description="Message indicating the status of the request.")
    total_documents: int = Field(0, description="Total number of documents uploaded.")
    documents: List[UploadedDocument] = Field([], description="List of uploaded documents.")

class MetadataField(BaseModel):
    """Model representing a metadata field."""
    name: str = Field("", description="Name of the metadata field.")
    type: Literal["string", "datetime"] = Field("", description="Type of the metadata field from the following: 'string', 'datetime'.")
    description: str = Field("", description="Optional description of the metadata field.")

class UploadedCollection(BaseModel):
    """Model representing an individual uploaded document."""
    collection_name: str = Field("", description="Name of the collection.")
    num_entities: int = Field(0, description="Number of rows or entities in the collection.")
    metadata_schema: List[MetadataField] = Field([], description="Metadata schema of the collection.")

class CollectionListResponse(BaseModel):
    """Response model for uploading a document."""
    message: str = Field("", description="Message indicating the status of the request.")
    total_collections: int = Field(0, description="Total number of collections uploaded.")
    collections: List[UploadedCollection] = Field([], description="List of uploaded collections.")

class CreateCollectionRequest(BaseModel):
    """Request model for creating a collection."""
    vdb_endpoint: str = Field(os.getenv("APP_VECTORSTORE_URL", ""), description="Endpoint of the vector database.")
    collection_name: str = Field(os.getenv("COLLECTION_NAME", ""), description="Name of the collection.")
    embedding_dimension: int = Field(2048, description="Embedding dimension of the collection.")
    metadata_schema: List[MetadataField] = Field([], description="Metadata schema of the collection.")

class FailedCollection(BaseModel):
    """Model representing a collection that failed to be created or deleted."""
    collection_name: str = Field("", description="Name of the collection.")
    error_message: str = Field("", description="Error message from the collection creation or deletion process.")

class CollectionsResponse(BaseModel):
    """Response model for creation or deletion of collections in Milvus."""
    message: str = Field(..., description="Status message of the process.")
    successful: List[str] = Field(default_factory=list, description="List of successfully created or deleted collections.")
    failed: List[FailedCollection] = Field(default_factory=list, description="List of collections that failed to be created or deleted.")
    total_success: int = Field(0, description="Total number of collections successfully created or deleted.")
    total_failed: int = Field(0, description="Total number of collections that failed to be created or deleted.")

class CreateCollectionResponse(BaseModel):
    """Response model for creation or deletion of a collection in Milvus."""
    message: str = Field(..., description="Status message of the process.")
    collection_name: str = Field(..., description="Name of the collection.")


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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
async def health_check():
    """
    Perform a Health Check

    Returns 200 when service is up. This does not check the health of downstream services.
    """

    response_message = "Ingestion Service is up."
    return HealthResponse(message=response_message)


async def parse_json_data(
    data: str = Form(
        ...,
        description="JSON data in string format containing metadata about the documents which needs to be uploaded.",
        examples=[json.dumps(DocumentUploadRequest().model_dump())],
        media_type="application/json"
    )
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
        },
        200: {
            "description": "Background Ingestion Started",
            "model": IngestionTaskResponse
        }
    }
)
async def upload_document(documents: List[UploadFile] = File(...),
    request: DocumentUploadRequest = Depends(parse_json_data)) -> Union[UploadDocumentResponse, IngestionTaskResponse]:
    """Upload a document to the vector store."""

    if not len(documents):
        raise Exception("No files provided for uploading.")

    try:
        # Store all provided file paths in a temporary directory
        all_file_paths = process_file_paths(documents, request.collection_name)
        response_dict = await NV_INGEST_INGESTOR.upload_documents(
            filepaths=all_file_paths,
            delete_files_after_ingestion=True,
            **request.model_dump()
        )
        if not request.blocking:
            return JSONResponse(
                content=IngestionTaskResponse(**response_dict).model_dump(),
                status_code=200)

        return UploadDocumentResponse(**response_dict)
    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while uploading document {e}")
        return JSONResponse(content={"message": "Request was cancelled by the client"}, status_code=499)
    except Exception as e:
        logger.error(f"Error from POST /documents endpoint. Ingestion of file failed with error: {e}")
        return JSONResponse(content={"message": f"Ingestion of files failed with error: {e}"}, status_code=500)


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
            state=result.get("state", "UNKNOWN"),
            result=result.get("result", {})
        )
    except KeyError as e:
        logger.error(f"Task {task_id} not found with error: {e}")
        return IngestionTaskStatusResponse(
            state="UNKNOWN",
            result={"message": "Task not found"}
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
        },
    }
)
async def update_documents(documents: List[UploadFile] = File(...),
    request: DocumentUploadRequest = Depends(parse_json_data)) -> DocumentListResponse:

    """Upload a document to the vector store. If the document already exists, it will be replaced."""

    try:
        # Store all provided file paths in a temporary directory
        all_file_paths = process_file_paths(documents, request.collection_name)
        response_dict = await NV_INGEST_INGESTOR.update_documents(
            filepaths=all_file_paths,
            delete_files_after_ingestion=True,
            **request.model_dump()
        )
        if not request.blocking:
            return JSONResponse(
                content=IngestionTaskResponse(**response_dict).model_dump(),
                status_code=200)

        return UploadDocumentResponse(**response_dict)

    except asyncio.CancelledError as e:
        logger.error(f"Request cancelled while deleting and uploading document")
        return JSONResponse(content={"message": "Request was cancelled by the client"}, status_code=499)
    except Exception as e:
        logger.error(f"Error from PATCH /documents endpoint. Ingestion failed with error: {e}")
        return JSONResponse(content={"message": f"Ingestion of files failed with error. {e}"}, status_code=500)


@app.get(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=DocumentListResponse,
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
async def get_documents(
    _: Request,
    collection_name: str = os.getenv("COLLECTION_NAME", ""),
    vdb_endpoint: str = Query(default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False)
) -> DocumentListResponse:
    """Get list of document ingested in vectorstore."""
    try:
        if hasattr(NV_INGEST_INGESTOR, "get_documents") and callable(NV_INGEST_INGESTOR.get_documents):
            documents = NV_INGEST_INGESTOR.get_documents(collection_name, vdb_endpoint)
            return DocumentListResponse(**documents)
        raise NotImplementedError("Example class has not implemented the get_documents method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching documents. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from GET /documents endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error occurred while fetching documents: {e}"}, status_code=500)


@app.delete(
    "/documents",
    tags=["Ingestion APIs"],
    response_model=DocumentListResponse,
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
async def delete_documents(_: Request, document_names: List[str] = [], collection_name: str = os.getenv("COLLECTION_NAME"), vdb_endpoint: str = Query(default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False)) -> DocumentListResponse:
    """Delete a document from vectorstore."""
    try:
        if hasattr(NV_INGEST_INGESTOR, "delete_documents") and callable(NV_INGEST_INGESTOR.delete_documents):
            response = NV_INGEST_INGESTOR.delete_documents(document_names=document_names, collection_name=collection_name, vdb_endpoint=vdb_endpoint, include_upload_path=True)
            return DocumentListResponse(**response)

        raise NotImplementedError("Example class has not implemented the delete_document method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while deleting document:, {document_names}, {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from DELETE /documents endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error deleting document {document_names}: {e}"}, status_code=500)


@app.get(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionListResponse,
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
        },
    },
)
async def get_collections(vdb_endpoint: str = Query(default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False)) -> CollectionListResponse:
    """
    Endpoint to get a list of collection names from the Milvus server.
    Returns a list of collection names.
    """
    try:
        if hasattr(NV_INGEST_INGESTOR, "get_collections") and callable(NV_INGEST_INGESTOR.get_collections):
            response = NV_INGEST_INGESTOR.get_collections(vdb_endpoint)
            return CollectionListResponse(**response)
        raise NotImplementedError("Example class has not implemented the get_collections method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from GET /collections endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error occurred while fetching collections. Error: {e}"}, status_code=500)


@app.post(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionsResponse,
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
        },
    },
    deprecated=True,
    description="This endpoint is deprecated. Use POST /collection instead. Custom metadata is not supported in this endpoint."
)
async def create_collections(
    vdb_endpoint: str = Query(default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False),
    collection_names: List[str] = [os.getenv("COLLECTION_NAME")],
    collection_type: str = "text",
    embedding_dimension: int = 2048
) -> CollectionsResponse:
    """
    Endpoint to create a collection from the Milvus server.
    Returns status message.
    """
    try:
        if hasattr(NV_INGEST_INGESTOR, "create_collections") and callable(NV_INGEST_INGESTOR.create_collections):
            response = NV_INGEST_INGESTOR.create_collections(collection_names, vdb_endpoint, embedding_dimension)
            return CollectionsResponse(**response)
        raise NotImplementedError("Example class has not implemented the create_collections method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from POST /collections endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error occurred while creating collections. Error: {e}"}, status_code=500)

@app.post(
    "/collection",
    tags=["Vector DB APIs"],
    response_model=CreateCollectionResponse,
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
        },
    },
)
async def create_collection(
    data: CreateCollectionRequest
) -> CreateCollectionResponse:
    """
    Endpoint to create a collection from the Milvus server.
    Returns status message.
    """
    try:
        if hasattr(NV_INGEST_INGESTOR, "create_collection") and callable(NV_INGEST_INGESTOR.create_collection):
            response = NV_INGEST_INGESTOR.create_collection(
                collection_name=data.collection_name,
                vdb_endpoint=data.vdb_endpoint,
                embedding_dimension=data.embedding_dimension,
                metadata_schema=[field.model_dump() for field in data.metadata_schema]
            )
            return CreateCollectionResponse(**response)
        raise NotImplementedError("Example class has not implemented the create_collection method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from POST /collections endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error occurred while creating collections. Error: {e}"}, status_code=500)

@app.delete(
    "/collections",
    tags=["Vector DB APIs"],
    response_model=CollectionsResponse,
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
        },
    },
)
async def delete_collections(vdb_endpoint: str = Query(default=os.getenv("APP_VECTORSTORE_URL"), include_in_schema=False), collection_names: List[str] = [os.getenv("COLLECTION_NAME")]) -> CollectionsResponse:
    """
    Endpoint to delete a collection from the Milvus server.
    Returns status message.
    """
    try:
        if hasattr(NV_INGEST_INGESTOR, "delete_collections") and callable(NV_INGEST_INGESTOR.delete_collections):
            response = NV_INGEST_INGESTOR.delete_collections(vdb_endpoint, collection_names)
            return CollectionsResponse(**response)
        raise NotImplementedError("Example class has not implemented the delete_collections method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled while fetching collections. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from DELETE /collections endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error occurred while deleting collections. Error: {e}"}, status_code=500)


def process_file_paths(filepaths: List[str], collection_name: str):
    """Process the file paths and return the list of file paths."""

    base_upload_folder = Path(os.path.join(CONFIG.temp_dir,
                                           f"uploaded_files/{collection_name}"))
    base_upload_folder.mkdir(parents=True, exist_ok=True)
    all_file_paths = []

    for file in filepaths:
        upload_file = os.path.basename(file.filename)

        if not upload_file:
            raise RuntimeError("Error parsing uploaded filename.")

        # Create a unique directory for each file
        unique_dir = base_upload_folder #/ str(uuid4())
        unique_dir.mkdir(parents=True, exist_ok=True)

        file_path = unique_dir / upload_file
        all_file_paths.append(str(file_path))

        # Copy uploaded file to upload_dir directory and pass that file path to ingestor server
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    return all_file_paths

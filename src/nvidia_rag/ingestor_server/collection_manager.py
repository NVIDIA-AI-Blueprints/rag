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
Collection management module for RAG ingestion pipeline.

This module provides functions for managing collections in the vector database:
1. create_collection: Create a new collection with metadata schema
2. update_collection_metadata: Update collection catalog metadata
3. create_collections: Create multiple collections in bulk
4. delete_collections: Delete multiple collections
5. get_collections: Retrieve all collections with their metadata
"""

import logging
import os
from typing import Any

from nvidia_rag.rag_server.main import APIError
from nvidia_rag.utils.common import create_catalog_metadata
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.metadata_validation import (
    SYSTEM_MANAGED_FIELDS,
    MetadataField,
)
from nvidia_rag.utils.minio_operator import (
    get_unique_thumbnail_id_collection_prefix,
)
from nvidia_rag.utils.observability.tracing import (
    get_tracer,
    trace_function,
)
from nvidia_rag.utils.vdb import _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag

# Initialize logger
logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.collection_manager")


@trace_function("ingestor.collection_manager.create_collection", tracer=TRACER)
def create_collection(
    vdb_op: VDBRag,
    config: NvidiaRAGConfig,
    collection_name: str | None = None,
    vdb_endpoint: str | None = None,
    metadata_schema: list[dict[str, str]] | None = None,
    description: str = "",
    tags: list[str] | None = None,
    owner: str = "",
    created_by: str = "",
    business_domain: str = "",
    status: str = "Active",
    vdb_auth_token: str = "",
) -> str:
    """
    Main function called by ingestor server to create a new collection in vector-DB
    """
    # Apply defaults from config if not provided
    if vdb_endpoint is None:
        vdb_endpoint = config.vector_store.url
    embedding_dimension = config.embeddings.dimensions

    vdb_op = _get_vdb_op(
        vdb_endpoint=vdb_endpoint,
        collection_name=collection_name,
        custom_metadata=None,
        all_file_paths=None,
        metadata_schema=None,
        config=config,
        vdb_auth_token=vdb_auth_token,
    )
    # Get the collection name from vdb_op if it was None
    if collection_name is None:
        collection_name = vdb_op.collection_name

    if metadata_schema is None:
        metadata_schema = []

    existing_field_names = {field.get("name") for field in metadata_schema}

    for field_name, field_def in SYSTEM_MANAGED_FIELDS.items():
        # Skip reserved fields - they are managed by NV-Ingest and should not be in the schema
        if field_def.get("reserved", False):
            continue

        if field_name not in existing_field_names:
            metadata_schema.append(
                {
                    "name": field_name,
                    "type": field_def["type"],
                    "description": field_def["description"],
                    "required": False,
                    "user_defined": field_def["rag_managed"],
                    "support_dynamic_filtering": field_def[
                        "support_dynamic_filtering"
                    ],
                }
            )

    try:
        vdb_op.create_metadata_schema_collection()
        vdb_op.create_document_info_collection()

        existing_collections = vdb_op.get_collection()
        if collection_name in [f["collection_name"] for f in existing_collections]:
            return {
                "message": f"Collection {collection_name} already exists.",
                "collection_name": collection_name,
            }
        logger.info(f"Creating collection {collection_name}")
        vdb_op.create_collection(collection_name, embedding_dimension)

        if metadata_schema:
            validated_schema = []
            for field_dict in metadata_schema:
                try:
                    field = MetadataField(**field_dict)
                    validated_schema.append(field.model_dump())
                except Exception as e:
                    logger.error(
                        f"Invalid metadata field: {field_dict}, error: {e}"
                    )
                    raise Exception(
                        f"Invalid metadata field '{field_dict.get('name', 'unknown')}': {str(e)}"
                    ) from e

            vdb_op.add_metadata_schema(collection_name, validated_schema)

        catalog_metadata = create_catalog_metadata(
            description=description,
            tags=tags,
            owner=owner,
            created_by=created_by,
            business_domain=business_domain,
            status=status,
        )

        vdb_op.add_document_info(
            info_type="catalog",
            collection_name=collection_name,
            document_name="NA",
            info_value=catalog_metadata,
        )

        return {
            "message": f"Collection {collection_name} created successfully.",
            "collection_name": collection_name,
        }
    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        logger.exception(f"Failed to create collection: {e}")
        raise Exception(f"Failed to create collection: {e}") from e


@trace_function("ingestor.collection_manager.update_collection_metadata", tracer=TRACER)
def update_collection_metadata(
    vdb_op: VDBRag,
    config: NvidiaRAGConfig,
    collection_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
    owner: str | None = None,
    business_domain: str | None = None,
    status: str | None = None,
) -> dict:
    """Update collection catalog metadata at runtime.

    Args:
        vdb_op (VDBRag): Vector database operator
        config (NvidiaRAGConfig): Configuration object
        collection_name (str): Name of the collection
        description (str, optional): Updated description
        tags (list[str], optional): Updated tags list
        owner (str, optional): Updated owner
        business_domain (str, optional): Updated business domain
        status (str, optional): Updated status
    """
    vdb_op = _get_vdb_op(
        vdb_endpoint=config.vector_store.url,
        collection_name=collection_name,
        custom_metadata=None,
        all_file_paths=None,
        metadata_schema=None,
        config=config,
        vdb_auth_token="",
    )
    # Get the collection name from vdb_op if needed
    collection_name = vdb_op.collection_name

    if not vdb_op.check_collection_exists(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")

    updates = {}
    if description is not None:
        updates["description"] = description
    if tags is not None:
        updates["tags"] = tags
    if owner is not None:
        updates["owner"] = owner
    if business_domain is not None:
        updates["business_domain"] = business_domain
    if status is not None:
        updates["status"] = status

    if not updates:
        return {
            "message": "No fields to update.",
            "collection_name": collection_name,
        }

    try:
        # Ensure document-info collection exists
        vdb_op.create_document_info_collection()
        vdb_op.update_catalog_metadata(collection_name, updates)

        return {
            "message": f"Collection {collection_name} metadata updated successfully.",
            "collection_name": collection_name,
        }
    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        logger.exception(f"Failed to update collection metadata: {e}")
        raise Exception(f"Failed to update collection metadata: {e}") from e


@trace_function("ingestor.collection_manager.create_collections", tracer=TRACER)
def create_collections(
    vdb_op: VDBRag,
    config: NvidiaRAGConfig,
    collection_names: list[str],
    vdb_endpoint: str | None = None,
    embedding_dimension: int | None = None,
    collection_type: str = "text",
    vdb_auth_token: str = "",
) -> dict[str, Any]:
    """
    Main function called by ingestor server to create new collections in vector-DB
    """
    # Apply defaults from config if not provided
    if vdb_endpoint is None:
        vdb_endpoint = config.vector_store.url
    if embedding_dimension is None:
        embedding_dimension = config.embeddings.dimensions

    vdb_op = _get_vdb_op(
        vdb_endpoint=vdb_endpoint,
        collection_name="",
        custom_metadata=None,
        all_file_paths=None,
        metadata_schema=None,
        config=config,
        vdb_auth_token=vdb_auth_token,
    )
    try:
        if not len(collection_names):
            return {
                "message": "No collections to create. Please provide a list of collection names.",
                "successful": [],
                "failed": [],
                "total_success": 0,
                "total_failed": 0,
            }

        created_collections = []
        failed_collections = []

        for collection_name in collection_names:
            try:
                vdb_op.create_collection(
                    collection_name=collection_name,
                    dimension=embedding_dimension,
                    collection_type=collection_type,
                )
                created_collections.append(collection_name)
                logger.info(f"Collection '{collection_name}' created successfully.")

            except Exception as e:
                failed_collections.append(
                    {"collection_name": collection_name, "error_message": str(e)}
                )
                logger.error(
                    f"Failed to create collection {collection_name}: {str(e)}"
                )

        return {
            "message": "Collection creation process completed.",
            "successful": created_collections,
            "failed": failed_collections,
            "total_success": len(created_collections),
            "total_failed": len(failed_collections),
        }

    except Exception as e:
        logger.error(f"Failed to create collections due to error: {str(e)}")
        failed_collections = [
            {"collection_name": collection, "error_message": str(e)}
            for collection in collection_names
        ]
        return {
            "message": f"Failed to create collections due to error: {str(e)}",
            "successful": [],
            "failed": failed_collections,
            "total_success": 0,
            "total_failed": len(collection_names),
        }


@trace_function("ingestor.collection_manager.delete_collections", tracer=TRACER)
def delete_collections(
    vdb_op: VDBRag,
    config: NvidiaRAGConfig,
    collection_names: list[str],
    vdb_endpoint: str | None = None,
    vdb_auth_token: str = "",
    minio_operator=None,
) -> dict[str, Any]:
    """
    Main function called by ingestor server to delete collections in vector-DB
    """
    # Apply default from config if not provided
    if vdb_endpoint is None:
        vdb_endpoint = config.vector_store.url

    logger.info(f"Deleting collections {collection_names}")

    try:
        vdb_op = _get_vdb_op(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
            custom_metadata=None,
            all_file_paths=None,
            metadata_schema=None,
            config=config,
            vdb_auth_token=vdb_auth_token,
        )

        response = vdb_op.delete_collections(collection_names)
        # Delete citation metadata from Minio (skip if MinIO unavailable)
        if minio_operator is not None:
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(
                    collection
                )
                try:
                    delete_object_names = minio_operator.list_payloads(
                        collection_prefix
                    )
                    minio_operator.delete_payloads(delete_object_names)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete MinIO objects for collection {collection}: {e}"
                    )

            # Delete document summary from Minio
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(
                    f"summary_{collection}"
                )
                try:
                    delete_object_names = minio_operator.list_payloads(
                        collection_prefix
                    )
                    if len(delete_object_names):
                        minio_operator.delete_payloads(delete_object_names)
                        logger.info(
                            f"Deleted all document summaries from Minio for collection: {collection}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete MinIO summaries for collection {collection}: {e}"
                    )
        else:
            logger.warning("MinIO unavailable - skipping metadata deletion")

        return response
    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        logger.error(f"Failed to delete collections in milvus: {e}")
        from traceback import print_exc

        logger.error(print_exc())
        return {
            "message": f"Failed to delete collections due to error: {str(e)}",
            "collections": [],
            "total_collections": 0,
        }


@trace_function("ingestor.collection_manager.get_collections", tracer=TRACER)
def get_collections(
    vdb_op: VDBRag,
    config: NvidiaRAGConfig,
    vdb_endpoint: str | None = None,
    vdb_auth_token: str = "",
) -> dict[str, Any]:
    """
    Main function called by ingestor server to get all collections in vector-DB.

    Args:
        vdb_op (VDBRag): Vector database operator
        config (NvidiaRAGConfig): Configuration object
        vdb_endpoint (str): The endpoint of the vector database.
        vdb_auth_token (str): Authentication token for vector database

    Returns:
        Dict[str, Any]: A dictionary containing the collection list, message, and total count.
    """
    # Apply default from config if not provided
    if vdb_endpoint is None:
        vdb_endpoint = config.vector_store.url

    try:
        vdb_op = _get_vdb_op(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
            custom_metadata=None,
            all_file_paths=None,
            metadata_schema=None,
            config=config,
            vdb_auth_token=vdb_auth_token,
        )
        # Fetch collections from vector store
        collection_info = vdb_op.get_collection()

        # Filter metadata schemas to only show user-defined fields in UI
        # Also remove internal implementation keys that users don't need to see
        for collection in collection_info:
            if "metadata_schema" in collection:
                collection["metadata_schema"] = [
                    {
                        k: v
                        for k, v in field.items()
                        if k not in ("user_defined", "support_dynamic_filtering")
                    }
                    for field in collection["metadata_schema"]
                    if field.get("user_defined", True)
                ]

        return {
            "message": "Collections listed successfully.",
            "collections": collection_info,
            "total_collections": len(collection_info),
        }

    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            # Let APIError propagate so global exception handler can return proper status code
            raise

        logger.error(f"Failed to retrieve collections: {e}")
        return {
            "message": f"Failed to retrieve collections due to error: {str(e)}",
            "collections": [],
            "total_collections": 0,
        }

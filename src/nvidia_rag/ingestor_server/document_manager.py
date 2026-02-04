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
Document management utilities for the RAG ingestion pipeline.

This module provides functions for managing documents in the vector store:
1. update_document_metadata: Update document catalog metadata at runtime
2. get_documents: Retrieve filenames stored in the vector store
3. delete_documents: Delete documents from the vector index
"""

import logging
import os
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from nvidia_rag.rag_server.main import APIError
from nvidia_rag.utils.common import (
    derive_boolean_flags,
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.minio_operator import (
    get_unique_thumbnail_id_collection_prefix,
    get_unique_thumbnail_id_file_name_prefix,
)
from nvidia_rag.utils.observability.tracing import trace_function, get_tracer
from nvidia_rag.utils.vdb import DEFAULT_DOCUMENT_INFO_COLLECTION

# Initialize logger
logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.document_manager")


@trace_function("ingestor.document_manager.update_document_metadata", tracer=TRACER)
def update_document_metadata(
    vdb_op,
    collection_name: str,
    document_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Update document catalog metadata at runtime.

    Args:
        vdb_op: Vector database operator instance
        collection_name (str): Name of the collection
        document_name (str): Name of the document
        description (str, optional): Updated description
        tags (list[str], optional): Updated tags list
    """
    if not vdb_op.check_collection_exists(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")

    # Verify document exists in the collection
    documents_list = vdb_op.get_documents(collection_name)
    document_names = [
        os.path.basename(doc.get("document_name", "")) for doc in documents_list
    ]
    if document_name not in document_names:
        raise ValueError(
            f"Document '{document_name}' does not exist in collection '{collection_name}'"
        )

    updates = {}
    if description is not None:
        updates["description"] = description
    if tags is not None:
        updates["tags"] = tags

    if not updates:
        return {
            "message": "No fields to update.",
            "document_name": document_name,
        }

    try:
        # Ensure document-info collection exists
        vdb_op.create_document_info_collection()
        vdb_op.update_document_catalog_metadata(
            collection_name, document_name, updates
        )

        return {
            "message": f"Document {document_name} metadata updated successfully.",
            "collection_name": collection_name,
        }
    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        logger.exception(f"Failed to update document metadata: {e}")
        raise Exception(f"Failed to update document metadata: {e}") from e


@trace_function("ingestor.document_manager.get_documents", tracer=TRACER)
def get_documents(
    vdb_op,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """
    Retrieves filenames stored in the vector store.
    It's called when the GET endpoint of `/documents` API is invoked.

    Args:
        vdb_op: Vector database operator instance
        collection_name: Name of the collection

    Returns:
        Dict[str, Any]: Response containing a list of documents with metadata.
    """
    try:
        documents_list = vdb_op.get_documents(collection_name)

        # Get metadata schema to filter out chunk-level auto-extracted fields
        metadata_schema = vdb_op.get_metadata_schema(collection_name)
        user_defined_fields = {
            field["name"]
            for field in metadata_schema
            if field.get("user_defined", True)
        }

        # Generate response format
        documents = [
            {
                "document_id": "",  # TODO - Use actual document_id
                "document_name": os.path.basename(
                    doc_item.get("document_name")
                ),  # Extract file name
                "timestamp": "",  # TODO - Use actual timestamp
                "size_bytes": 0,  # TODO - Use actual size
                "metadata": {
                    k: v
                    for k, v in doc_item.get("metadata", {}).items()
                    if k in user_defined_fields
                },
                "document_info": doc_item.get("document_info", {}),
            }
            for doc_item in documents_list
        ]

        return {
            "documents": documents,
            "total_documents": len(documents),
            "message": "Document listing successfully completed.",
        }

    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        logger.exception(f"Failed to retrieve documents due to error {e}.")
        return {
            "documents": [],
            "total_documents": 0,
            "message": f"Document listing failed due to error {e}.",
        }


@trace_function("ingestor.document_manager.delete_documents", tracer=TRACER)
def delete_documents(
    vdb_op,
    config,
    minio_operator,
    document_names: list[str],
    collection_name: str | None = None,
    include_upload_path: bool = False,
) -> dict[str, Any]:
    """Delete documents from the vector index.
    It's called when the DELETE endpoint of `/documents` API is invoked.

    Args:
        vdb_op: Vector database operator instance
        config: Configuration object
        minio_operator: MinIO operator instance (can be None)
        document_names (List[str]): List of filenames to be deleted from vectorstore.
        collection_name (str): Name of the collection to delete documents from.
        include_upload_path (bool): Whether to include upload path in document names.

    Returns:
        Dict[str, Any]: Response containing a list of deleted documents with metadata.
    """
    try:
        logger.info(
            f"Deleting documents {document_names} from collection {collection_name}"
        )

        # Prepare source values for deletion
        if include_upload_path:
            upload_folder = str(
                Path(
                    os.path.join(
                        config.temp_dir, f"uploaded_files/{collection_name}"
                    )
                )
            )
        else:
            upload_folder = ""
        source_values = [
            os.path.join(upload_folder, filename) for filename in document_names
        ]

        # Fetch document info before deletion so we can return it and update collection stats
        documents_list = vdb_op.get_documents(collection_name)
        documents_map = {
            os.path.basename(doc.get("document_name", "")): doc
            for doc in documents_list
        }

        # Get metadata schema to filter out chunk-level auto-extracted fields
        metadata_schema = vdb_op.get_metadata_schema(collection_name)
        user_defined_fields = {
            field["name"]
            for field in metadata_schema
            if field.get("user_defined", True)
        }

        # Process all documents (idempotent - always returns True)
        # Pass result_dict to get detailed deletion results
        # Milvus populates it based on delete_count, Elasticsearch populates it by checking existing documents
        deletion_result = {}
        vdb_op.delete_documents(
            collection_name, source_values, result_dict=deletion_result
        )

        deleted_docs = deletion_result.get("deleted", [])
        not_found_docs = deletion_result.get("not_found", [])

        # If result_dict wasn't populated (fallback for older VDB implementations),
        # assume all documents were deleted successfully
        if not deleted_docs and not not_found_docs:
            deleted_docs = document_names

        # Helper function to delete MinIO metadata for documents
        def delete_minio_metadata(docs_to_delete: list[str]) -> None:
            if minio_operator is None:
                logger.warning("MinIO unavailable - skipping metadata deletion")
                return

            for doc in docs_to_delete:
                # Delete citation metadata
                filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                    collection_name, doc
                )
                try:
                    delete_object_names = minio_operator.list_payloads(
                        filename_prefix
                    )
                    minio_operator.delete_payloads(delete_object_names)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete MinIO objects for doc {doc}: {e}"
                    )

                # Delete document summary
                filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                    f"summary_{collection_name}", doc
                )
                try:
                    delete_object_names = minio_operator.list_payloads(
                        filename_prefix
                    )
                    if len(delete_object_names):
                        minio_operator.delete_payloads(delete_object_names)
                        logger.info(f"Deleted summary for doc: {doc} from Minio")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete MinIO summary for doc {doc}: {e}"
                    )

        # Recalculate collection info from remaining documents after deletion
        # This is more reliable than subtracting, and avoids double-aggregation issues
        if deleted_docs:
            # Get all remaining documents after deletion (fetch again after deletion)
            remaining_documents_list = vdb_op.get_documents(collection_name)

            # Aggregate collection info from all remaining documents
            aggregated_collection_info = {}
            for doc_item in remaining_documents_list:
                doc_info = doc_item.get("document_info", {})
                if doc_info:
                    aggregated_collection_info = perform_document_info_aggregation(
                        aggregated_collection_info, doc_info
                    )

            # Catalog metadata should NOT be stored in collection entry - it's stored separately in catalog entry
            # Only collection metrics need to be recalculated from remaining documents
            # Aggregated info contains: has_images, has_tables, has_charts, total_elements, etc.
            # Always update collection info when documents are deleted, even if all documents are removed
            # Re-derive boolean flags from doc_type_counts to ensure they're proper booleans
            doc_type_counts = aggregated_collection_info.get("doc_type_counts", {})
            boolean_flags = derive_boolean_flags(doc_type_counts)

            # Update only metrics (not catalog metadata) from remaining documents
            updated_collection_info = {
                **aggregated_collection_info,  # Update metrics from remaining documents
                **boolean_flags,  # Override boolean flags to ensure they're proper booleans
                "number_of_files": len(
                    remaining_documents_list
                ),  # Explicitly set file count
                "last_updated": get_current_timestamp(),  # Update timestamp
            }

            # Recalculate collection info by aggregating from remaining documents
            # Need to bypass add_document_info's aggregation which happens before deletion
            # So we manually delete and insert the recalculated value
            if hasattr(vdb_op, "vdb_endpoint") and hasattr(
                vdb_op, "_delete_entities"
            ):
                # Milvus: Delete existing collection info, then insert recalculated value
                vdb_op._delete_entities(
                    collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION,
                    filter=f"info_type == 'collection' and collection_name == '{collection_name}' and document_name == 'NA'",
                )
                # Add new collection info directly without aggregation
                password = (
                    vdb_op.config.vector_store.password.get_secret_value()
                    if vdb_op.config.vector_store.password is not None
                    else ""
                )
                auth_token = getattr(vdb_op, "_auth_token", None)
                client = MilvusClient(
                    vdb_op.vdb_endpoint,
                    token=auth_token
                    if auth_token
                    else f"{vdb_op.config.vector_store.username}:{password}",
                )
                data = {
                    "info_type": "collection",
                    "collection_name": collection_name,
                    "document_name": "NA",
                    "info_value": updated_collection_info,
                    "vector": [0.0] * 2,
                }
                client.insert(
                    collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION, data=data
                )
                logger.info(
                    f"Recalculated collection info for {collection_name} after document deletion"
                )
            elif hasattr(vdb_op, "_es_connection"):
                # Elasticsearch: Delete first, then add without aggregation
                # Lazy import to avoid requiring elasticsearch when not used
                from nvidia_rag.utils.vdb.elasticsearch.es_queries import (
                    get_delete_document_info_query,
                )

                vdb_op._es_connection.delete_by_query(
                    index=DEFAULT_DOCUMENT_INFO_COLLECTION,
                    body=get_delete_document_info_query(
                        collection_name=collection_name,
                        document_name="NA",
                        info_type="collection",
                    ),
                )
                # Insert new collection info directly
                data = {
                    "collection_name": collection_name,
                    "info_type": "collection",
                    "document_name": "NA",
                    "info_value": updated_collection_info,
                }
                vdb_op._es_connection.index(
                    index=DEFAULT_DOCUMENT_INFO_COLLECTION, body=data
                )
                vdb_op._es_connection.indices.refresh(
                    index=DEFAULT_DOCUMENT_INFO_COLLECTION
                )
                logger.info(
                    f"Recalculated collection info for {collection_name} after document deletion"
                )
            else:
                # Fallback: Use add_document_info (may cause double-aggregation, but better than nothing)
                logger.warning(
                    f"Could not directly update collection info for {collection_name}, using add_document_info (may cause aggregation issues)"
                )
                vdb_op.add_document_info(
                    info_type="collection",
                    collection_name=collection_name,
                    document_name="NA",
                    info_value=updated_collection_info,
                )

        # Build response based on what was actually deleted vs not found
        if not_found_docs and not deleted_docs:
            # All documents don't exist
            return {
                "message": f"The following document(s) do not exist in the vectorstore: {', '.join(not_found_docs)}",
                "total_documents": 0,
                "documents": [],
            }

        # Delete MinIO metadata for successfully deleted documents
        delete_minio_metadata(deleted_docs)

        # Build documents response with metadata and document_info from fetched data
        documents = []
        for doc_name in deleted_docs:
            doc_item = documents_map.get(doc_name, {})
            documents.append(
                {
                    "document_id": "",  # TODO - Use actual document_id
                    "document_name": doc_name,
                    "size_bytes": 0,  # TODO - Use actual size
                    "metadata": {
                        k: v
                        for k, v in doc_item.get("metadata", {}).items()
                        if k in user_defined_fields
                    },
                    "document_info": doc_item.get("document_info", {}),
                }
            )

        if not_found_docs:
            # Some documents don't exist, but some were deleted
            return {
                "message": f"Some documents deleted successfully. The following document(s) do not exist in the vectorstore: {', '.join(not_found_docs)}",
                "total_documents": len(documents),
                "documents": documents,
            }

        # All documents were deleted successfully
        return {
            "message": "Files deleted successfully",
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception as e:
        # Re-raise APIError to propagate proper HTTP status code via global exception handler
        if isinstance(e, APIError):
            raise
        return {
            "message": f"Failed to delete files due to error: {e}",
            "total_documents": 0,
            "documents": [],
        }

    return {
        "message": "Failed to delete files due to error. Check logs for details.",
        "total_documents": 0,
        "documents": [],
    }

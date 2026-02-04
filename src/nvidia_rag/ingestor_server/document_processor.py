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
Document Processing Utilities Module.

This module provides utility functions for processing documents during ingestion,
including:
1. Document type counting and classification
2. Result information logging and metrics
3. Failed document tracking and reporting
4. Ingestion response building
5. Document catalog metadata application

These functions support the main ingestion pipeline by handling document metadata,
tracking ingestion status, and preparing response data.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any
from uuid import uuid4

from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE

from nvidia_rag.utils.common import (
    create_document_metadata,
    derive_boolean_flags,
    get_current_timestamp,
)
from nvidia_rag.utils.observability.tracing import get_tracer, trace_function
from nvidia_rag.utils.vdb.vdb_base import VDBRag

# Initialize logger
logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.document_processor")

# Supported file types constant
SUPPORTED_FILE_TYPES = set(EXTENSION_TO_DOCUMENT_TYPE.keys()) - set({"svg"})


def get_document_type_counts(
    results: list[list[dict[str, str | dict]]]
) -> tuple[dict[str, int], int, int, int]:
    """
    Get document type counts from the results.

    Note: Document types are normalized to standard keys (table, chart, image, text)
    to ensure consistency with frontend expectations and derive_boolean_flags().

    Args:
        results: List of lists containing document extraction results

    Returns:
        Tuple containing:
        - doc_type_counts: Dict mapping document types to counts
        - total_documents: Total number of documents processed
        - total_elements: Total number of elements extracted
        - raw_text_elements_size: Total size of raw text elements in bytes
    """
    # Mapping from nv-ingest types/subtypes to normalized keys
    type_normalization = {
        "structured": "table",  # Structured data defaults to table
    }

    doc_type_counts = defaultdict(int)
    total_documents = 0
    total_elements = 0
    raw_text_elements_size = 0  # in bytes

    for result in results:
        total_documents += 1
        for result_element in result:
            total_elements += 1
            document_type = result_element.get("document_type", "unknown")
            document_subtype = (
                result_element.get("metadata", {})
                .get("content_metadata", {})
                .get("subtype", "")
            )
            # Use subtype if available, otherwise use document_type
            if document_subtype:
                doc_type_key = document_subtype
            else:
                doc_type_key = document_type

            # Normalize the key to standard names (table, chart, image, text)
            doc_type_key = type_normalization.get(doc_type_key, doc_type_key)

            doc_type_counts[doc_type_key] += 1
            if document_type == "text":
                content = result_element.get("metadata", {}).get("content", "")
                if isinstance(content, str):
                    raw_text_elements_size += len(content)
                elif content:
                    raw_text_elements_size += len(str(content))
    return doc_type_counts, total_documents, total_elements, raw_text_elements_size


@trace_function("ingestor.document_processor.log_result_info", tracer=TRACER)
def log_result_info(
    batch_number: int,
    results: list[list[dict[str, str | dict]]],
    failures: list[dict[str, Any]],
    total_ingestion_time: float,
    additional_summary: str = "",
) -> dict[str, Any]:
    """Log the results info with document type counts.

    Args:
        batch_number: The batch number being processed
        results: List of lists containing document extraction results
        failures: List of failures during ingestion
        total_ingestion_time: Total time taken for ingestion in seconds
        additional_summary: Additional summary information to append

    Returns:
        dict[str, Any]: Document info with metrics
    """
    doc_type_counts, total_documents, total_elements, raw_text_elements_size = (
        get_document_type_counts(results)
    )

    document_info = {
        "doc_type_counts": doc_type_counts,
        "total_elements": total_elements,
        "raw_text_elements_size": raw_text_elements_size,
        "number_of_files": total_documents,
        **derive_boolean_flags(doc_type_counts),
        "last_indexed": get_current_timestamp(),
        "ingestion_status": "Success"
        if not failures
        else ("Partial" if len(results) > 0 else "Failed"),
        "last_ingestion_error": (
            str(failures[0][1]) if failures and len(failures[0]) > 1 else ""
        ),
    }

    summary_parts = []
    for doc_type in doc_type_counts.keys():
        count = doc_type_counts.get(doc_type, 0)
        if count > 0:
            summary_parts.append(f"{doc_type}:{count}")
    if raw_text_elements_size > 0:
        summary_parts.append(
            f"Raw text elements size: {raw_text_elements_size} bytes"
        )

    summary = (
        f"Successfully processed {total_documents} document(s) with {total_elements} element(s) • "
        + " • ".join(summary_parts)
    )
    if failures:
        summary += f", {len(failures)} files failed ingestion"

    if additional_summary:
        summary += f" • {additional_summary}"

    logger.info(
        f"== Batch {batch_number} Ingestion completed in {total_ingestion_time:.2f} seconds • Summary: {summary} =="
    )
    return document_info


async def get_non_supported_files(filepaths: list[str]) -> list[str]:
    """Get filepaths of non-supported file extensions.

    Args:
        filepaths: List of file paths to check

    Returns:
        List of file paths with unsupported extensions
    """
    non_supported_files = []
    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [
            "." + supported_ext for supported_ext in SUPPORTED_FILE_TYPES
        ]:
            non_supported_files.append(filepath)
    return non_supported_files


@trace_function("ingestor.document_processor.get_failed_documents", tracer=TRACER)
async def get_failed_documents(
    failures: list[dict[str, Any]],
    filepaths: list[str] | None = None,
    collection_name: str | None = None,
    is_final_batch: bool = True,
    get_documents_func: callable = None,
) -> list[dict[str, Any]]:
    """
    Get failed documents.

    Arguments:
        failures: List of failures from ingestion process
        filepaths: List of filepaths that were attempted
        collection_name: Name of the collection
        is_final_batch: Whether this is the final batch
        get_documents_func: Function to retrieve documents from collection

    Returns:
        List of failed documents with error messages
    """
    failed_documents = []
    failed_documents_filenames = set()
    for failure in failures:
        error_message = str(failure[1])
        failed_filename = os.path.basename(str(failure[0]))
        failed_documents.append(
            {"document_name": failed_filename, "error_message": error_message}
        )
        failed_documents_filenames.add(failed_filename)
    if not is_final_batch:
        # For non-final batches, we don't need to add non-supported files
        # and document to failed documents if it is not in the Milvus
        # because we will continue to ingest the next batch
        return failed_documents

    # Add non-supported files to failed documents
    for filepath in await get_non_supported_files(filepaths):
        filename = os.path.basename(filepath)
        if filename not in failed_documents_filenames:
            failed_documents.append(
                {
                    "document_name": filename,
                    "error_message": "Unsupported file type, supported file types are: "
                    + ", ".join(SUPPORTED_FILE_TYPES),
                }
            )
            failed_documents_filenames.add(filename)

    # Add document to failed documents if it is not in the Milvus
    filenames_in_vdb = set()
    if get_documents_func:
        for document in get_documents_func(collection_name, bypass_validation=True).get(
            "documents"
        ):
            filenames_in_vdb.add(document.get("document_name"))
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if (
            filename not in filenames_in_vdb
            and filename not in failed_documents_filenames
        ):
            failed_documents.append(
                {
                    "document_name": filename,
                    "error_message": "Ingestion did not complete successfully",
                }
            )
            failed_documents_filenames.add(filename)

    if failed_documents:
        logger.error("Ingestion failed for %d document(s)", len(failed_documents))
        logger.error(
            "Failed documents details: %s", json.dumps(failed_documents, indent=4)
        )

    return failed_documents


@trace_function("ingestor.document_processor.build_ingestion_response", tracer=TRACER)
async def build_ingestion_response(
    results: list[list[dict[str, str | dict]]],
    failures: list[dict[str, Any]],
    filepaths: list[str] | None = None,
    is_final_batch: bool = True,
    collection_name: str = None,
    custom_metadata: list[dict[str, Any]] = None,
    failed_validation_documents: list[dict[str, Any]] = None,
    validation_errors: list[dict[str, Any]] = None,
    get_documents_func: callable = None,
    vdb_op: VDBRag = None,
) -> dict[str, Any]:
    """
    Builds the ingestion response dictionary.

    Args:
        results: List of results from the ingestion process
        failures: List of failures from the ingestion process
        filepaths: List of file paths that were processed
        is_final_batch: Whether the batch is the final batch
        collection_name: Name of the collection
        custom_metadata: Custom metadata provided for documents
        failed_validation_documents: Documents that failed validation
        validation_errors: Validation errors encountered
        get_documents_func: Function to retrieve documents from collection
        vdb_op: Vector database operations instance

    Returns:
        Dictionary containing ingestion response with documents and failures
    """
    # Get failed documents
    failed_documents = await get_failed_documents(
        failures=failures,
        filepaths=filepaths,
        collection_name=collection_name,
        is_final_batch=is_final_batch,
        get_documents_func=get_documents_func,
    )
    failures_filepaths = [
        failed_document.get("document_name") for failed_document in failed_documents
    ]

    filename_to_metadata_map = {
        custom_metadata_item.get("filename"): custom_metadata_item.get("metadata")
        for custom_metadata_item in (custom_metadata or [])
    }
    filename_to_result_map = {}
    for result in results:
        if len(result) > 0:
            metadata = result[0].get("metadata", {})
            source_metadata = metadata.get("source_metadata", {})
            source_id = source_metadata.get("source_id", "")
            if source_id:
                filename_to_result_map[os.path.basename(source_id)] = result

    # Generate response dictionary
    uploaded_documents = []
    for filepath in filepaths:
        if os.path.basename(filepath) not in failures_filepaths:
            doc_type_counts, _, total_elements, raw_text_elements_size = (
                get_document_type_counts(
                    [filename_to_result_map.get(os.path.basename(filepath), [])]
                )
            )

            document_info = create_document_metadata(
                filepath=filepath,
                doc_type_counts=doc_type_counts,
                total_elements=total_elements,
                raw_text_elements_size=raw_text_elements_size,
            )

            # Always add document info for each document
            if not is_final_batch and vdb_op:
                vdb_op.add_document_info(
                    info_type="document",
                    collection_name=collection_name,
                    document_name=os.path.basename(filepath),
                    info_value=document_info,
                )
            uploaded_document = {
                "document_id": str(uuid4()),
                "document_name": os.path.basename(filepath),
                "size_bytes": os.path.getsize(filepath),
                "metadata": {
                    **filename_to_metadata_map.get(os.path.basename(filepath), {}),
                    "filename": filename_to_metadata_map.get(
                        os.path.basename(filepath), {}
                    ).get("filename")
                    or os.path.basename(filepath),
                },
                "document_info": document_info,
            }
            uploaded_documents.append(uploaded_document)

    # Get current timestamp in ISO format
    # TODO: Store document_id, timestamp and document size as metadata
    if is_final_batch:
        message = "Document upload job successfully completed."
    else:
        message = "Document upload job is in progress."
    response_data = {
        "message": message,
        "total_documents": len(filepaths),
        "documents": uploaded_documents,
        "failed_documents": failed_documents + (failed_validation_documents or []),
        "validation_errors": validation_errors or [],
    }
    return response_data


@trace_function("ingestor.document_processor.apply_documents_catalog_metadata", tracer=TRACER)
async def apply_documents_catalog_metadata(
    results: list[list[dict[str, Any]]],
    vdb_op: VDBRag,
    collection_name: str,
    documents_catalog_metadata: list[dict[str, Any]],
    filepaths: list[str],
) -> None:
    """Apply catalog metadata to successfully ingested documents.

    Args:
        results: List of ingestion results
        vdb_op: Vector database operations instance
        collection_name: Name of the collection
        documents_catalog_metadata: List of dicts with 'filename', 'description', 'tags'
        filepaths: List of file paths that were ingested
    """
    # Build a mapping from filename to catalog metadata
    catalog_map = {
        os.path.basename(meta["filename"]): meta
        for meta in documents_catalog_metadata
    }

    # Extract document names from filepaths (these are the successfully ingested documents)
    ingested_docs = set()
    for filepath in filepaths:
        doc_name = os.path.basename(filepath)
        ingested_docs.add(doc_name)

    # Apply catalog metadata to each successfully ingested document
    for doc_name in ingested_docs:
        if doc_name in catalog_map:
            metadata = catalog_map[doc_name]
            updates = {}
            if metadata.get("description"):
                updates["description"] = metadata["description"]
            if metadata.get("tags"):
                updates["tags"] = metadata["tags"]

            if updates:
                try:
                    vdb_op.update_document_catalog_metadata(
                        collection_name,
                        doc_name,
                        updates,
                    )
                    logger.info(
                        f"Applied catalog metadata to document '{doc_name}': {updates}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply catalog metadata to document '{doc_name}': {e}"
                    )

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
Helper class to build the response of the ingestion server upload document request.
"""
import os
import json
import logging
from collections import defaultdict
from typing import Any
from uuid import uuid4

from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.ingestor_server.pipelines.nv_ingest.constants import SUPPORTED_FILE_TYPES
from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    process_nv_ingest_traces,
    trace_function,
)
from nvidia_rag.utils.common import (
    create_catalog_metadata,
    create_document_metadata,
    derive_boolean_flags,
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.minio_operator import (
    MinioOperator,
    get_unique_thumbnail_id_from_result,
)


logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.response_builder")


class ResponseBuilder:
    """Helper class to build the response of the ingestion server upload document request."""

    def __init__(
        self,
        vdb_op: VDBRag,
        config: dict,
        minio_operator: MinioOperator,
    ):
        self.vdb_op = vdb_op
        self.config = config
        self.minio_operator = minio_operator

    @trace_function("ingestor.response_builder.get_failed_documents", tracer=TRACER)
    async def __get_failed_documents(
        self,
        failures: list[dict[str, Any]],
        filepaths: list[str] | None = None,
        collection_name: str | None = None,
        is_final_batch: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get failed documents

        Arguments:
            - failures: List[Dict[str, Any]] - List of failures
            - filepaths: List[str] - List of filepaths
            - results: List[List[Dict[str, Union[str, dict]]]] - List of results

        Returns:
            - List[Dict[str, Any]] - List of failed documents
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
        for filepath in await self.__get_non_supported_files(filepaths):
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
        for document in self.vdb_op.get_documents(collection_name):
            filenames_in_vdb.add(os.path.basename(document.get("document_name")))
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

    @trace_function("ingestor.response_builder.build_ingestion_response", tracer=TRACER)
    async def build_ingestion_response(
        self,
        results: list[list[dict[str, str | dict]]],
        failures: list[dict[str, Any]],
        filepaths: list[str] | None = None,
        is_final_batch: bool = True,
        state_manager: IngestionStateManager = None,
        vdb_op: VDBRag = None,
    ) -> dict[str, Any]:
        """
        Builds the ingestion response dictionary.

        Args:
            results: List[list[dict[str, str | dict]]] - List of results from the ingestion process
            failures: List[dict[str, Any]] - List of failures from the ingestion process
            is_final_batch: bool - Whether the batch is the final batch
            state_manager: IngestionStateManager - State manager for the ingestion process
        """
        # Get failed documents
        failed_documents = await self.get_failed_documents(
            failures=failures,
            filepaths=filepaths,
            collection_name=state_manager.collection_name,
            is_final_batch=is_final_batch,
        )
        failures_filepaths = [
            failed_document.get("document_name") for failed_document in failed_documents
        ]

        filename_to_metadata_map = {
            custom_metadata_item.get("filename"): custom_metadata_item.get("metadata")
            for custom_metadata_item in (state_manager.custom_metadata or [])
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
                    self.get_document_type_counts(
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
                if not is_final_batch:
                    vdb_op.add_document_info(
                        info_type="document",
                        collection_name=state_manager.collection_name,
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
            "total_documents": len(state_manager.filepaths),
            "documents": uploaded_documents,
            "failed_documents": failed_documents
            + state_manager.failed_validation_documents,
            "validation_errors": state_manager.validation_errors,
        }
        return response_data
    
    @trace_function("ingestor.response_builder.remove_unsupported_files", tracer=TRACER)
    async def remove_unsupported_files(self, filepaths: list[str]) -> list[str]:
        """Remove unsupported files from the list of filepaths"""
        non_supported_files = await self.__get_non_supported_files(filepaths)
        return [filepath for filepath in filepaths if filepath not in non_supported_files]
    
    @trace_function("ingestor.response_builder.get_non_supported_files", tracer=TRACER)
    async def __get_non_supported_files(self, filepaths: list[str]) -> list[str]:
        """Get filepaths of non-supported file extensions"""
        non_supported_files = []
        for filepath in filepaths:
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in [
                "." + supported_ext for supported_ext in SUPPORTED_FILE_TYPES
            ]:
                non_supported_files.append(filepath)
        return non_supported_files
    
    @trace_function("ingestor.response_builder.remove_unsupported_files", tracer=TRACER)
    async def remove_unsupported_files(
        self,
        filepaths: list[str],
    ) -> list[str]:
        """Remove unsupported files from the list of filepaths"""
        non_supported_files = await self.__get_non_supported_files(filepaths)
        return [
            filepath for filepath in filepaths if filepath not in non_supported_files
        ]
    
    @trace_function("ingestor.response_builder.get_document_type_counts", tracer=TRACER)
    def get_document_type_counts(
        self, results: list[list[dict[str, str | dict]]]
    ) -> dict[str, int]:
        """
        Get document type counts from the results.
        
        Note: Document types are normalized to standard keys (table, chart, image, text)
        to ensure consistency with frontend expectations and derive_boolean_flags().
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

    @trace_function("ingestor.response_builder.put_content_to_minio", tracer=TRACER)
    def put_content_to_minio(
        self,
        results: list[list[dict[str, str | dict]]],
        collection_name: str,
    ) -> None:
        """
        Put nv-ingest image/table/chart content to minio
        """
        if not self.config.enable_citations:
            logger.info(f"Skipping minio insertion for collection: {collection_name}")
            return  # Don't perform minio insertion if captioning is disabled

        payloads = []
        object_names = []

        for result in results:
            for result_element in result:
                if result_element.get("document_type") in ["image", "structured"]:
                    # Extract required fields
                    metadata = result_element.get("metadata", {})
                    content = result_element.get("metadata").get("content")

                    file_name = os.path.basename(
                        result_element.get("metadata")
                        .get("source_metadata")
                        .get("source_id")
                    )
                    page_number = (
                        result_element.get("metadata")
                        .get("content_metadata")
                        .get("page_number")
                    )
                    location = (
                        result_element.get("metadata")
                        .get("content_metadata")
                        .get("location")
                    )

                    # Get unique_thumbnail_id using the centralized function
                    # Try with extracted location first, fallback to content_metadata if None
                    unique_thumbnail_id = get_unique_thumbnail_id_from_result(
                        collection_name=collection_name,
                        file_name=file_name,
                        page_number=page_number,
                        location=location,
                        metadata=metadata,
                    )

                    if unique_thumbnail_id is not None:
                        # Pull content from result_element
                        payloads.append({"content": content})
                        object_names.append(unique_thumbnail_id)
                    # If unique_thumbnail_id is None, the item is skipped
                    # (warning already logged in get_unique_thumbnail_id_from_result)

        if self.minio_operator is not None:
            if os.getenv("ENABLE_MINIO_BULK_UPLOAD", "True") in ["True", "true"]:
                logger.info(f"Bulk uploading {len(payloads)} payloads to MinIO")
                try:
                    self.minio_operator.put_payloads_bulk(
                        payloads=payloads, object_names=object_names
                    )
                except Exception as e:
                    logger.warning(f"Failed to bulk upload to MinIO: {e}")
            else:
                logger.info(f"Sequentially uploading {len(payloads)} payloads to MinIO")
                for payload, object_name in zip(payloads, object_names, strict=False):
                    try:
                        self.minio_operator.put_payload(
                            payload=payload, object_name=object_name
                        )
                    except Exception as e:
                        logger.warning(f"Failed to upload {object_name} to MinIO: {e}")
        else:
            logger.warning(
                f"MinIO unavailable - skipping upload of {len(payloads)} payloads"
            )
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
This is the Main module for RAG ingestion pipeline.
1. Upload documents: Upload documents to the vector store. Method name: upload_documents
2. Update documents: Update documents in the vector store. Method name: update_documents
3. Status: Get the status of an ingestion task. Method name: status
4. Create collection: Create a new collection in the vector store. Method name: create_collection
5. Create collections: Create new collections in the vector store. Method name: create_collections
6. Delete collections: Delete collections in the vector store. Method name: delete_collections
7. Get collections: Get all collections in the vector store. Method name: get_collections
8. Get documents: Get documents in the vector store. Method name: get_documents
9. Delete documents: Delete documents in the vector store. Method name: delete_documents

Private methods:
1. __prepare_vdb_op_and_collection_name: Prepare vector database operation and collection name.
2. __run_background_ingest_task: Ingest documents to the vector store.
3. __build_ingestion_response: Build the ingestion response from results and failures.
4. __ingest_document_summary: Drives summary generation and ingestion if enabled.
5. __put_content_to_minio: Put NV-Ingest image/table/chart content to MinIO.
6. __perform_shallow_extraction_workflow: Perform shallow extraction workflow for fast summary generation.
7. __run_nvingest_batched_ingestion: Upload documents to the vector store using NV-Ingest.
8. __nv_ingest_ingestion_pipeline: Run the NV-Ingest ingestion pipeline.
9. __get_failed_documents: Get failed documents from the vector store.
10. __get_non_supported_files: Get non-supported files from the vector store.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from nv_ingest_client.primitives.tasks.extract import _DEFAULT_EXTRACTOR_MAP
from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE
from nv_ingest_client.util.vdb.adt_vdb import VDB
from pymilvus import MilvusClient

from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
)
from nvidia_rag.ingestor_server.task_handler import INGESTION_TASK_HANDLER
from nvidia_rag.rag_server.main import APIError
from nvidia_rag.utils.batch_utils import calculate_dynamic_batch_parameters
from nvidia_rag.utils.common import (
    create_catalog_metadata,
    create_document_metadata,
    derive_boolean_flags,
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import IngestorHealthResponse
from nvidia_rag.utils.llm import get_prompts
from nvidia_rag.utils.metadata_validation import (
    SYSTEM_MANAGED_FIELDS,
    MetadataField,
    MetadataSchema,
    MetadataValidator,
)
from nvidia_rag.utils.minio_operator import (
    get_minio_operator,
    get_unique_thumbnail_id_collection_prefix,
    get_unique_thumbnail_id_file_name_prefix,
    get_unique_thumbnail_id_from_result,
)
from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    process_nv_ingest_traces,
    trace_function,
)
from nvidia_rag.utils.summarization import generate_document_summaries
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER
from nvidia_rag.utils.vdb import DEFAULT_DOCUMENT_INFO_COLLECTION, _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.ingestor_server.collection_manager import (
    create_collection as _create_collection,
    update_collection_metadata as _update_collection_metadata,
    create_collections as _create_collections,
    delete_collections as _delete_collections,
    get_collections as _get_collections,
)
from nvidia_rag.ingestor_server.document_manager import (
    update_document_metadata as _update_document_metadata,
    get_documents as _get_documents,
    delete_documents as _delete_documents,
)
from nvidia_rag.ingestor_server.validation import (
    validate_directory_traversal_attack,
    get_non_supported_files,
    remove_unsupported_files,
    split_pdf_and_non_pdf_files,
    validate_custom_metadata,
)
from nvidia_rag.ingestor_server.document_processor import (
    get_document_type_counts,
    log_result_info,
    get_failed_documents,
    build_ingestion_response,
    apply_documents_catalog_metadata,
)
from nvidia_rag.ingestor_server.minio_handler import put_content_to_minio
from nvidia_rag.ingestor_server.summarization_pipeline import (
    ingest_document_summary,
    process_shallow_batch,
    perform_shallow_extraction_workflow,
    perform_shallow_extraction,
)
from nvidia_rag.ingestor_server.nv_ingest_pipeline import (
    run_nvingest_batched_ingestion,
    nv_ingest_ingestion_pipeline,
    perform_async_nv_ingest_ingestion,
    perform_file_ext_based_nv_ingest_ingestion,
)

# Initialize logger
logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.main")


class Mode(str, Enum):
    """Supported application modes for NvidiaRAGIngestor"""

    LIBRARY = "library"
    SERVER = "server"
    LITE = "lite"


SUPPORTED_FILE_TYPES = set(EXTENSION_TO_DOCUMENT_TYPE.keys()) - set({"svg"})


class NvidiaRAGIngestor:
    """
    Main Class for RAG ingestion pipeline integration for NV-Ingest
    """

    _vdb_upload_bulk_size = 500

    def __init__(
        self,
        vdb_op: VDBRag = None,
        mode: Mode | str = Mode.LIBRARY,
        config: NvidiaRAGConfig | None = None,
        prompts: str | dict | None = None,
    ):
        """Initialize NvidiaRAGIngestor with configuration.

        Args:
            vdb_op: Optional vector database operator
            mode: Operating mode (library or server)
            config: Configuration object. If None, uses default config.
            prompts: Optional prompt configuration. Can be:
                - A path to a YAML/JSON file containing prompts
                - A dictionary with prompt configurations
                - None to use defaults (or PROMPT_CONFIG_FILE env var)
        """
        # Convert string to Mode enum if necessary
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError as err:
                raise ValueError(
                    f"Invalid mode: {mode}. Supported modes are: {[m.value for m in Mode]}"
                ) from err
        self.mode = mode
        self.vdb_op = vdb_op

        # Track background summary tasks to prevent garbage collection
        self._background_tasks = set()
        self.config = config or NvidiaRAGConfig()
        self.prompts = get_prompts(prompts)

        # Initialize instance-based clients
        self.nv_ingest_client = get_nv_ingest_client(
            config=self.config, get_lite_client=self.mode == Mode.LITE
        )

        # Initialize MinIO operator - handle failures gracefully
        try:
            if self.mode == Mode.LITE:
                raise ValueError("MinIO operations are not supported in RAG Lite mode")
            self.minio_operator = get_minio_operator(config=self.config)
            # Ensure default bucket exists (idempotent operation)
            try:
                self.minio_operator._make_bucket(bucket_name="a-bucket")
                logger.debug("Ensured 'a-bucket' exists in MinIO")
            except Exception as bucket_err:
                # Log specific exception for debugging bucket creation issues
                logger.debug("Could not ensure bucket exists: %s", bucket_err)
        except Exception as e:
            self.minio_operator = None
            # Error already logged in MinioOperator.__init__, just note it here
            logger.debug(
                "MinIO operator set to None due to initialization failure, reason: %s",
                e,
            )

        if self.vdb_op is not None:
            if not (isinstance(self.vdb_op, VDBRag) or isinstance(self.vdb_op, VDB)):
                raise ValueError(
                    "vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag. "
                    "or nv_ingest_client.util.vdb.adt_vdb.VDB. "
                    "Please make sure all the required methods are implemented."
                )

    async def health(self, check_dependencies: bool = False) -> IngestorHealthResponse:
        """Check the health of the Ingestion server."""
        if check_dependencies:
            from nvidia_rag.ingestor_server.health import check_all_services_health

            vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
                bypass_validation=True
            )
            return await check_all_services_health(vdb_op, self.config)

        return IngestorHealthResponse(message="Service is up.")

    @trace_function("ingestor.main.validate_directory_traversal_attack", tracer=TRACER)
    @trace_function("ingestor.main.prepare_vdb_op_and_collection_name", tracer=TRACER)
    def __prepare_vdb_op_and_collection_name(
        self,
        vdb_endpoint: str | None = None,
        collection_name: str | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        filepaths: list[str] | None = None,
        bypass_validation: bool = False,
        metadata_schema: list[dict[str, Any]] | None = None,
        vdb_auth_token: str = "",
    ) -> VDBRag:
        """
        Prepare the VDBRag object for ingestion.
        Also, validate the arguments.
        """
        if self.vdb_op is None:
            if not bypass_validation and collection_name is None:
                raise ValueError(
                    "`collection_name` argument is required when `vdb_op` is not "
                    "provided during initialization."
                )
            vdb_op = _get_vdb_op(
                vdb_endpoint=vdb_endpoint or self.config.vector_store.url,
                collection_name=collection_name,
                custom_metadata=custom_metadata,
                all_file_paths=filepaths,
                metadata_schema=metadata_schema,
                config=self.config,
                vdb_auth_token=vdb_auth_token,
            )
            return vdb_op, collection_name

        if not bypass_validation and (collection_name or custom_metadata):
            raise ValueError(
                "`collection_name` and `custom_metadata` arguments are not "
                "supported when `vdb_op` is provided during initialization."
            )

        return self.vdb_op, self.vdb_op.collection_name

    @trace_function("ingestor.main.upload_documents", tracer=TRACER)
    async def upload_documents(
        self,
        filepaths: list[str],
        blocking: bool = False,
        collection_name: str | None = None,
        vdb_endpoint: str | None = None,
        split_options: dict[str, Any] | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
        additional_validation_errors: list[dict[str, Any]] | None = None,
        documents_catalog_metadata: list[dict[str, Any]] | None = None,
        vdb_auth_token: str = "",
        enable_pdf_split_processing: bool = False,
        pdf_split_processing_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload documents to the vector store.

        Args:
            filepaths (List[str]): List of absolute filepaths to upload
            blocking (bool, optional): Whether to block until ingestion completes. Defaults to False.
            collection_name (str, optional): Name of collection in vector database. Defaults to "multimodal_data".
            split_options (Dict[str, Any], optional): Options for splitting documents. Defaults to chunk_size and chunk_overlap from self.config.
            custom_metadata (List[Dict[str, Any]], optional): Custom metadata to add to documents. Defaults to empty list.
            generate_summary (bool, optional): Whether to generate summaries. Defaults to False.
            summary_options (Dict[str, Any] | None, optional): Advanced options for summary (e.g., page_filter). Only used when generate_summary=True. Defaults to None.
            additional_validation_errors (List[Dict[str, Any]] | None, optional): Additional validation errors to include in response. Defaults to None.
            documents_catalog_metadata (List[Dict[str, Any]] | None, optional): Per-document catalog metadata (description, tags) to add during upload. Defaults to None.
        """
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        # Calculate dynamic batch parameters
        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths=filepaths,
            config=self.config,
        )

        state_manager = IngestionStateManager(
            filepaths=filepaths,
            collection_name=collection_name,
            custom_metadata=custom_metadata,
            documents_catalog_metadata=documents_catalog_metadata,
            enable_pdf_split_processing=enable_pdf_split_processing,
            pdf_split_processing_options=pdf_split_processing_options,
            concurrent_batches=concurrent_batches,
            files_per_batch=files_per_batch,
        )
        task_id = state_manager.get_task_id()

        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            filepaths=filepaths,
            vdb_auth_token=vdb_auth_token,
        )

        state_manager.collection_name = collection_name

        vdb_op.create_document_info_collection()

        # Set default values for mutable arguments
        if split_options is None:
            split_options = {
                "chunk_size": self.config.nv_ingest.chunk_size,
                "chunk_overlap": self.config.nv_ingest.chunk_overlap,
            }
        if custom_metadata is None:
            custom_metadata = []
        if additional_validation_errors is None:
            additional_validation_errors = []
        if documents_catalog_metadata is None:
            documents_catalog_metadata = []

        # Validate summary_options using Pydantic model (same validation as API mode)
        if summary_options:
            try:
                # Local import to avoid circular dependency
                from nvidia_rag.ingestor_server.server import SummaryOptions

                validated_options = SummaryOptions(**summary_options)
                # Convert back to dict for internal use
                summary_options = validated_options.model_dump()
            except Exception as e:
                raise ValueError(f"Invalid summary_options: {e}") from e

        if not vdb_op.check_collection_exists(collection_name):
            raise ValueError(
                f"Collection {collection_name} does not exist. Ensure a collection is created using POST /collection endpoint first."
            )

        # Initialize document-wise status
        nv_ingest_status = await state_manager.initialize_nv_ingest_status(filepaths)

        try:
            if not blocking:
                state_manager.is_background = True

                def _task():
                    return self.__run_background_ingest_task(
                        filepaths=filepaths,
                        collection_name=collection_name,
                        vdb_endpoint=vdb_endpoint,
                        vdb_op=vdb_op,
                        split_options=split_options,
                        custom_metadata=custom_metadata,
                        generate_summary=generate_summary,
                        summary_options=summary_options,
                        additional_validation_errors=additional_validation_errors,
                        state_manager=state_manager,
                        documents_catalog_metadata=documents_catalog_metadata,
                        vdb_auth_token=vdb_auth_token,
                    )

                task_id = await INGESTION_TASK_HANDLER.submit_task(
                    _task, task_id=task_id
                )

                # Set initial document-wise status in IngestionTaskHandler
                await INGESTION_TASK_HANDLER.set_task_state_dict(
                    state_manager.get_task_id(),
                    {"nv_ingest_status": nv_ingest_status},
                )

                # Update initial batch progress response to indicate that the ingestion has started
                batch_progress_response = await build_ingestion_response(
                    results=[],
                    failures=[],
                    filepaths=[],
                    is_final_batch=False,
                    collection_name=state_manager.collection_name,
                    custom_metadata=state_manager.custom_metadata,
                    failed_validation_documents=state_manager.failed_validation_documents,
                    validation_errors=state_manager.validation_errors,
                    get_documents_func=self.get_documents,
                    vdb_op=vdb_op,
                )
                ingestion_state = await state_manager.update_batch_progress(
                    batch_progress_response=batch_progress_response,
                    is_batch_zero=True,
                )
                await INGESTION_TASK_HANDLER.set_task_status_and_result(
                    task_id=state_manager.get_task_id(),
                    status="PENDING",
                    result=ingestion_state,
                )
                return {
                    "message": "Ingestion started in background",
                    "task_id": task_id,
                }
            else:
                response_dict = await self.__run_background_ingest_task(
                    filepaths=filepaths,
                    collection_name=collection_name,
                    vdb_endpoint=vdb_endpoint,
                    vdb_op=vdb_op,
                    split_options=split_options,
                    custom_metadata=custom_metadata,
                    generate_summary=generate_summary,
                    summary_options=summary_options,
                    additional_validation_errors=additional_validation_errors,
                    state_manager=state_manager,
                    documents_catalog_metadata=documents_catalog_metadata,
                    vdb_auth_token=vdb_auth_token,
                )
            return response_dict

        except Exception as e:
            logger.exception(f"Failed to upload documents: {e}")
            return {
                "message": f"Failed to upload documents due to error: {str(e)}",
                "total_documents": len(filepaths),
                "documents": [],
                "failed_documents": [],
            }

    @trace_function("ingestor.main.run_background_ingest_task", tracer=TRACER)
    async def __run_background_ingest_task(
        self,
        filepaths: list[str],
        collection_name: str | None = None,
        vdb_endpoint: str | None = None,
        vdb_op: VDBRag | None = None,
        split_options: dict[str, Any] | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
        additional_validation_errors: list[dict[str, Any]] | None = None,
        state_manager: IngestionStateManager | None = None,
        documents_catalog_metadata: list[dict[str, Any]] | None = None,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to ingest
        the documents to vector-DB

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_endpoint: str - URL of the vector database endpoint
            - vdb_op: VDBRag - VDB operator instance
            - split_options: Dict[str, Any] - Options for splitting documents
            - custom_metadata: List[Dict[str, Any]] - Custom metadata to be added to documents
            - generate_summary: bool - Whether to generate summaries
            - summary_options : SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
            - additional_validation_errors: List[Dict[str, Any]] | None - Additional validation errors to include in response (defaults to None)
            - documents_catalog_metadata: List[Dict[str, Any]] | None - Per-document catalog metadata (description, tags) to add after upload (defaults to None)
            - state_manager: IngestionStateManager - State manager for the ingestion process
        """
        logger.info("Performing ingestion in collection_name: %s", collection_name)
        logger.debug("Filepaths for ingestion: %s", filepaths)

        failed_validation_documents = []
        validation_errors = (
            []
            if additional_validation_errors is None
            else list(additional_validation_errors)
        )
        original_file_count = len(filepaths)

        state_manager.validation_errors = validation_errors
        state_manager.failed_validation_documents = failed_validation_documents
        state_manager.documents_catalog_metadata = documents_catalog_metadata or []

        try:
            # Get metadata schema once for validation and CSV preparation
            metadata_schema = vdb_op.get_metadata_schema(collection_name)

            # Always run validation if there's a schema, even without custom_metadata
            (
                validation_status,
                metadata_validation_errors,
            ) = await validate_custom_metadata(
                custom_metadata, collection_name, metadata_schema, filepaths, self.config
            )
            # Merge metadata validation errors with additional validation errors
            validation_errors.extend(metadata_validation_errors)

            # Re-initialize vdb_op if custom_metadata is provided
            # This is needed since custom_metadata is normalized in the _validate_custom_metadata method
            if custom_metadata:
                vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
                    vdb_endpoint=vdb_endpoint,
                    collection_name=collection_name,
                    custom_metadata=custom_metadata,
                    filepaths=filepaths,
                    metadata_schema=metadata_schema,
                    vdb_auth_token=vdb_auth_token,
                )

            if not validation_status:
                failed_filenames = set()
                for error in validation_errors:
                    metadata_item = error.get("metadata", {})
                    filename = metadata_item.get("filename", "")
                    if filename:
                        failed_filenames.add(filename)

                # Add failed documents to the list
                for filename in failed_filenames:
                    failed_validation_documents.append(
                        {
                            "document_name": filename,
                            "error_message": f"Metadata validation failed for {filename}",
                        }
                    )

                filepaths = [
                    file
                    for file in filepaths
                    if os.path.basename(file) not in failed_filenames
                ]
                custom_metadata = [
                    item
                    for item in custom_metadata
                    if item.get("filename") not in failed_filenames
                ]

            # Get all documents in the collection (only if we have files to process)
            existing_documents = set()
            if filepaths:
                get_docs_response = self.get_documents(
                    collection_name, bypass_validation=True
                )
                existing_documents = {
                    doc.get("document_name") for doc in get_docs_response["documents"]
                }

            for file in filepaths:
                await validate_directory_traversal_attack(file)
                filename = os.path.basename(file)
                # Check if the provided filepaths are valid
                if not os.path.exists(file):
                    logger.error(f"File {file} does not exist. Ingestion failed.")
                    failed_validation_documents.append(
                        {
                            "document_name": filename,
                            "error_message": f"File {filename} does not exist at path {file}. Ingestion failed.",
                        }
                    )

                if not os.path.isfile(file):
                    failed_validation_documents.append(
                        {
                            "document_name": filename,
                            "error_message": f"File {filename} is not a file. Ingestion failed.",
                        }
                    )

                # Check if the provided filepaths are already in vector-DB
                if filename in existing_documents:
                    logger.error(
                        f"Document {file} already exists. Upload failed. Please call PATCH /documents endpoint to delete and replace this file."
                    )
                    failed_validation_documents.append(
                        {
                            "document_name": filename,
                            "error_message": f"Document {filename} already exists. Use update document API instead.",
                        }
                    )

                # Check for unsupported file formats (.rst, .rtf, etc.)
                not_supported_formats = (".rst", ".rtf", ".org")
                if filename.endswith(not_supported_formats):
                    logger.info(
                        "Detected a .rst or .rtf file, you need to install Pandoc manually in Docker."
                    )
                    # Provide instructions to install Pandoc in Dockerfile
                    dockerfile_instructions = """
                    # Install pandoc from the tarball to support ingestion .rst, .rtf & .org files
                    RUN curl -L https://github.com/jgm/pandoc/releases/download/3.6/pandoc-3.6-linux-amd64.tar.gz -o /tmp/pandoc.tar.gz && \
                    tar -xzf /tmp/pandoc.tar.gz -C /tmp && \
                    mv /tmp/pandoc-3.6/bin/pandoc /usr/local/bin/ && \
                    rm -rf /tmp/pandoc.tar.gz /tmp/pandoc-3.6
                    """
                    logger.info(dockerfile_instructions)
                    failed_validation_documents.append(
                        {
                            "document_name": filename,
                            "error_message": f"Document {filename} is not a supported format. Check logs for details.",
                        }
                    )

            # Check if all provided files have failed (consolidated check)
            if len(failed_validation_documents) == original_file_count:
                return {
                    "message": "Document upload job failed. All files failed to validate. Check logs for details.",
                    "total_documents": original_file_count,
                    "documents": [],
                    "failed_documents": failed_validation_documents,
                    "validation_errors": validation_errors,
                    "state": "FAILED",
                }

            # Remove the failed validation documents from the filepaths
            failed_filenames_set = {
                failed_document.get("document_name")
                for failed_document in failed_validation_documents
            }
            filepaths = [
                file
                for file in filepaths
                if os.path.basename(file) not in failed_filenames_set
            ]

            if len(failed_validation_documents):
                logger.error(f"Validation errors: {failed_validation_documents}")

            logger.info("Number of filepaths for ingestion after validation: %s", len(filepaths))
            logger.debug("Filepaths for ingestion after validation: %s", filepaths)

            # Peform ingestion using nvingest for all files that have not failed
            # Check if the provided collection_name exists in vector-DB

            start_time = time.time()

            # Create callback wrapper for shallow extraction workflow
            async def shallow_workflow_callback(filepaths, collection_name, split_options, summary_options, state_manager):
                return await perform_shallow_extraction_workflow(
                    filepaths=filepaths,
                    collection_name=collection_name,
                    split_options=split_options,
                    config=self.config,
                    prompts=self.prompts,
                    state_manager=state_manager,
                    perform_async_nv_ingest_ingestion_func=perform_async_nv_ingest_ingestion,
                    background_tasks=self._background_tasks,
                    summary_options=summary_options,
                )

            # Create callback wrapper for put_content_to_minio
            def put_content_to_minio_callback(results, collection_name):
                return put_content_to_minio(
                    results=results,
                    collection_name=collection_name,
                    minio_operator=self.minio_operator,
                    enable_citations=self.config.enable_citations,
                )

            # Create callback wrapper for ingest_document_summary
            async def ingest_document_summary_callback(results, collection_name, page_filter=None, summarization_strategy=None, is_shallow=False):
                return await ingest_document_summary(
                    results=results,
                    collection_name=collection_name,
                    config=self.config,
                    prompts=self.prompts,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                    is_shallow=is_shallow,
                )

            results, failures = await run_nvingest_batched_ingestion(
                filepaths=filepaths,
                collection_name=collection_name,
                config=self.config,
                nv_ingest_client=self.nv_ingest_client,
                prompts=self.prompts,
                get_documents_func=self.get_documents,
                put_content_to_minio_callback=put_content_to_minio_callback,
                ingest_document_summary_callback=ingest_document_summary_callback,
                background_tasks=self._background_tasks,
                mode=self.mode,
                vdb_op=vdb_op,
                split_options=split_options,
                generate_summary=generate_summary,
                summary_options=summary_options,
                state_manager=state_manager,
                perform_shallow_extraction_workflow_callback=shallow_workflow_callback,
            )

            build_ingestion_response_start_time = time.time()
            response_data = await build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                is_final_batch=True,
                collection_name=state_manager.collection_name,
                custom_metadata=state_manager.custom_metadata,
                failed_validation_documents=state_manager.failed_validation_documents,
                validation_errors=state_manager.validation_errors,
                get_documents_func=self.get_documents,
                vdb_op=vdb_op,
            )
            logger.info(
                f"== Final build ingestion response and adding document info is complete! Time taken: {time.time() - build_ingestion_response_start_time} seconds =="
            )

            # Apply catalog metadata for successfully ingested documents
            apply_documents_catalog_metadata_start_time = time.time()
            if state_manager.documents_catalog_metadata:
                await apply_documents_catalog_metadata(
                    results=results,
                    vdb_op=vdb_op,
                    collection_name=collection_name,
                    documents_catalog_metadata=state_manager.documents_catalog_metadata,
                    filepaths=filepaths,
                )
            logger.info(
                f"== Apply documents catalog metadata is complete! Time taken: {time.time() - apply_documents_catalog_metadata_start_time} seconds =="
            )
            ingestion_state = await state_manager.update_total_progress(
                total_progress_response=response_data,
            )
            await INGESTION_TASK_HANDLER.set_task_status_and_result(
                task_id=state_manager.get_task_id(),
                status="FINISHED",
                result=ingestion_state,
            )

            # Optional: Clean up provided files after ingestion, needed for
            # docker workflow
            clean_up_files_start_time = time.time()
            if self.mode == Mode.SERVER:
                logger.info(f"Cleaning up files count: {len(filepaths)}")
                for file in filepaths:
                    try:
                        os.remove(file)
                        logger.debug(f"Deleted temporary file: {file}")
                    except FileNotFoundError:
                        logger.warning(f"File not found: {file}")
                    except Exception as e:
                        logger.error(f"Error deleting {file}: {e}")
            logger.info(
                f"== Clean up files is complete! Time taken: {time.time() - clean_up_files_start_time} seconds =="
            )
            
            logger.info(
                "== Overall Ingestion completed successfully in %s seconds ==",
                time.time() - start_time,
            )

            return ingestion_state

        except Exception as e:
            logger.exception(
                "Ingestion failed due to error: %s",
                e,
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            raise e

    @trace_function("ingestor.main.update_documents", tracer=TRACER)
    async def update_documents(
        self,
        filepaths: list[str],
        blocking: bool = False,
        collection_name: str | None = None,
        vdb_endpoint: str | None = None,
        split_options: dict[str, Any] | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
        additional_validation_errors: list[dict[str, Any]] | None = None,
        documents_catalog_metadata: list[dict[str, Any]] | None = None,
        vdb_auth_token: str = "",
        enable_pdf_split_processing: bool = False,
        pdf_split_processing_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a document to the vector store. If the document already exists, it will be replaced.

        Args:
            filepaths: List of absolute filepaths to upload
            blocking: Whether to block until ingestion completes
            collection_name: Name of collection in vector database
            vdb_endpoint: URL of the vector database endpoint
            split_options: Options for splitting documents
            custom_metadata: Custom metadata to add to documents
            generate_summary: Whether to generate summaries
            summary_options: Advanced options for summary (e.g., page_filter). Only used when generate_summary=True.
            additional_validation_errors: Additional validation errors to include in response
        """

        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        # Set default values for mutable arguments
        if split_options is None:
            split_options = {
                "chunk_size": self.config.nv_ingest.chunk_size,
                "chunk_overlap": self.config.nv_ingest.chunk_overlap,
            }
        if custom_metadata is None:
            custom_metadata = []

        for file in filepaths:
            file_name = os.path.basename(file)

            # Delete the existing document

            if self.mode == Mode.SERVER:
                response = self.delete_documents(
                    [file_name],
                    collection_name=collection_name,
                    include_upload_path=True,
                    vdb_auth_token=vdb_auth_token,
                )
            else:
                response = self.delete_documents(
                    [file],
                    collection_name=collection_name,
                    vdb_auth_token=vdb_auth_token,
                )

            if response["total_documents"] == 0:
                logger.info(
                    "Unable to remove %s from collection. Either the document does not exist or there is an error while removing. Proceeding with ingestion.",
                    file_name,
                )
            else:
                logger.info(
                    "Successfully removed %s from collection %s.",
                    file_name,
                    collection_name,
                )

        response = await self.upload_documents(
            filepaths=filepaths,
            blocking=blocking,
            collection_name=collection_name,
            vdb_endpoint=vdb_endpoint,
            split_options=split_options,
            custom_metadata=custom_metadata,
            generate_summary=generate_summary,
            summary_options=summary_options,
            additional_validation_errors=additional_validation_errors,
            documents_catalog_metadata=documents_catalog_metadata,
            vdb_auth_token=vdb_auth_token,
            enable_pdf_split_processing=enable_pdf_split_processing,
            pdf_split_processing_options=pdf_split_processing_options,
        )
        return response

    @staticmethod
    @trace_function("ingestor.main.status", tracer=TRACER)
    async def status(task_id: str) -> dict[str, Any]:
        """Get the status of an ingestion task."""

        logger.info(f"Getting status of task {task_id}")
        try:
            status_and_result = INGESTION_TASK_HANDLER.get_task_status_and_result(
                task_id
            )
            nv_ingest_status = INGESTION_TASK_HANDLER.get_task_state_dict(task_id).get(
                "nv_ingest_status"
            )
            if status_and_result.get("state") == "PENDING":
                logger.info(f"Task {task_id} is pending")
                return {
                    "state": "PENDING",
                    "result": status_and_result.get("result"),
                    "nv_ingest_status": nv_ingest_status,
                }
            elif status_and_result.get("state") == "FINISHED":
                try:
                    result = status_and_result.get("result")
                    if isinstance(result, dict) and result.get("state") == "FAILED":
                        logger.error(
                            f"Task {task_id} failed with error: {result.get('message')}"
                        )
                        result.pop("state")
                        return {
                            "state": "FAILED",
                            "result": result,
                            "nv_ingest_status": nv_ingest_status,
                        }
                    logger.info(f"Task {task_id} is finished")
                    return {
                        "state": "FINISHED",
                        "result": result,
                        "nv_ingest_status": nv_ingest_status,
                    }
                except Exception as e:
                    logger.exception("Task %s failed with error: %s", task_id, e)
                    return {
                        "state": "FAILED",
                        "result": {"message": str(e)},
                        "nv_ingest_status": nv_ingest_status,
                    }
            elif status_and_result.get("state") == "FAILED":
                logger.error(
                    f"Task {task_id} failed with error: {status_and_result.get('result').get('message')}"
                )
                return {
                    "state": "FAILED",
                    "result": status_and_result.get("result"),
                    "nv_ingest_status": nv_ingest_status,
                }
            else:
                task_state = INGESTION_TASK_HANDLER.get_task_status(task_id)
                logger.error(f"Unknown task state: {task_state}")
                return {
                    "state": "UNKNOWN",
                    "result": {"message": "Unknown task state"},
                    "nv_ingest_status": nv_ingest_status,
                }
        except KeyError as e:
            logger.error(f"Task {task_id} not found with error: {e}")
            return {
                "state": "UNKNOWN",
                "result": {"message": "Unknown task state"},
                "nv_ingest_status": nv_ingest_status,
            }

    @trace_function("ingestor.main.create_collection", tracer=TRACER)
    def create_collection(
        self,
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
        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            vdb_auth_token=vdb_auth_token,
        )
        return _create_collection(
            vdb_op=vdb_op,
            config=self.config,
            collection_name=collection_name,
            vdb_endpoint=vdb_endpoint,
            metadata_schema=metadata_schema,
            description=description,
            tags=tags,
            owner=owner,
            created_by=created_by,
            business_domain=business_domain,
            status=status,
            vdb_auth_token=vdb_auth_token,
        )

    @trace_function("ingestor.main.update_collection_metadata", tracer=TRACER)
    def update_collection_metadata(
        self,
        collection_name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        business_domain: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Update collection catalog metadata at runtime.

        Args:
            collection_name (str): Name of the collection
            description (str, optional): Updated description
            tags (list[str], optional): Updated tags list
            owner (str, optional): Updated owner
            business_domain (str, optional): Updated business domain
            status (str, optional): Updated status
        """
        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=None,
            collection_name=collection_name,
        )
        return _update_collection_metadata(
            vdb_op=vdb_op,
            config=self.config,
            collection_name=collection_name,
            description=description,
            tags=tags,
            owner=owner,
            business_domain=business_domain,
            status=status,
        )

    @trace_function("ingestor.main.update_document_metadata", tracer=TRACER)
    def update_document_metadata(
        self,
        collection_name: str,
        document_name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Update document catalog metadata at runtime.

        Args:
            collection_name (str): Name of the collection
            document_name (str): Name of the document
            description (str, optional): Updated description
            tags (list[str], optional): Updated tags list
        """
        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=None,
            collection_name=collection_name,
        )
        return _update_document_metadata(
            vdb_op=vdb_op,
            collection_name=collection_name,
            document_name=document_name,
            description=description,
            tags=tags,
        )

    @trace_function("ingestor.main.create_collections", tracer=TRACER)
    def create_collections(
        self,
        collection_names: list[str],
        vdb_endpoint: str | None = None,
        embedding_dimension: int | None = None,
        collection_type: str = "text",
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to create new collections in vector-DB
        """
        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
            vdb_auth_token=vdb_auth_token,
        )
        return _create_collections(
            vdb_op=vdb_op,
            config=self.config,
            collection_names=collection_names,
            vdb_endpoint=vdb_endpoint,
            embedding_dimension=embedding_dimension,
            collection_type=collection_type,
            vdb_auth_token=vdb_auth_token,
        )

    @trace_function("ingestor.main.delete_collections", tracer=TRACER)
    def delete_collections(
        self,
        collection_names: list[str],
        vdb_endpoint: str | None = None,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to delete collections in vector-DB
        """
        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
            vdb_auth_token=vdb_auth_token,
        )
        return _delete_collections(
            vdb_op=vdb_op,
            config=self.config,
            collection_names=collection_names,
            vdb_endpoint=vdb_endpoint,
            vdb_auth_token=vdb_auth_token,
            minio_operator=self.minio_operator,
        )

    @trace_function("ingestor.main.get_collections", tracer=TRACER)
    def get_collections(
        self,
        vdb_endpoint: str | None = None,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to get all collections in vector-DB.

        Args:
            vdb_endpoint (str): The endpoint of the vector database.

        Returns:
            Dict[str, Any]: A dictionary containing the collection list, message, and total count.
        """
        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
            vdb_auth_token=vdb_auth_token,
        )
        return _get_collections(
            vdb_op=vdb_op,
            config=self.config,
            vdb_endpoint=vdb_endpoint,
            vdb_auth_token=vdb_auth_token,
        )

    @trace_function("ingestor.main.get_documents", tracer=TRACER)
    def get_documents(
        self,
        collection_name: str | None = None,
        vdb_endpoint: str | None = None,
        bypass_validation: bool = False,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        Retrieves filenames stored in the vector store.
        It's called when the GET endpoint of `/documents` API is invoked.

        Returns:
            Dict[str, Any]: Response containing a list of documents with metadata.
        """
        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            bypass_validation=bypass_validation,
            vdb_auth_token=vdb_auth_token,
        )
        return _get_documents(
            vdb_op=vdb_op,
            collection_name=collection_name,
        )

    @trace_function("ingestor.main.delete_documents", tracer=TRACER)
    def delete_documents(
        self,
        document_names: list[str],
        collection_name: str | None = None,
        vdb_endpoint: str | None = None,
        include_upload_path: bool = False,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """Delete documents from the vector index.
        It's called when the DELETE endpoint of `/documents` API is invoked.

        Args:
            document_names (List[str]): List of filenames to be deleted from vectorstore.
            collection_name (str): Name of the collection to delete documents from.
            vdb_endpoint (str): Vector database endpoint.

        Returns:
            Dict[str, Any]: Response containing a list of deleted documents with metadata.
        """
        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            vdb_auth_token=vdb_auth_token,
        )
        return _delete_documents(
            vdb_op=vdb_op,
            config=self.config,
            minio_operator=self.minio_operator,
            document_names=document_names,
            collection_name=collection_name,
            include_upload_path=include_upload_path,
        )

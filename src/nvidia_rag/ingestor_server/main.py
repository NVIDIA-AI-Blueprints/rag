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
    async def validate_directory_traversal_attack(self, file) -> None:
        try:
            # Path.resolve(strict=True) is a method used to
            # obtain the absolute and normalized path, with
            # the added condition that the path must physically
            # exist on the filesystem. If a directory traversal
            # attack is tried, resulting path after the resolve
            # will be invalid.
            if file:
                _ = Path(file).resolve(strict=True)
        except Exception as e:
            raise ValueError(
                f"File not found or a directory traversal attack detected! Filepath: {file}"
            ) from e

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
                batch_progress_response = await self.__build_ingestion_response(
                    results=[],
                    failures=[],
                    filepaths=[],
                    state_manager=state_manager,
                    is_final_batch=False,
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
            ) = await self._validate_custom_metadata(
                custom_metadata, collection_name, metadata_schema, filepaths
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
                await self.validate_directory_traversal_attack(file)
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

            logger.info(
                "Number of filepaths for ingestion after validation: %s", len(filepaths)
            )
            logger.debug("Filepaths for ingestion after validation: %s", filepaths)

            # Peform ingestion using nvingest for all files that have not failed
            # Check if the provided collection_name exists in vector-DB

            start_time = time.time()
            results, failures = await self.__run_nvingest_batched_ingestion(
                filepaths=filepaths,
                collection_name=collection_name,
                vdb_op=vdb_op,
                split_options=split_options,
                generate_summary=generate_summary,
                summary_options=summary_options,
                state_manager=state_manager,
            )

            build_ingestion_response_start_time = time.time()
            response_data = await self.__build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                state_manager=state_manager,
                is_final_batch=True,
                vdb_op=vdb_op,
            )
            logger.info(
                f"== Final build ingestion response and adding document info is complete! Time taken: {time.time() - build_ingestion_response_start_time} seconds =="
            )

            # Apply catalog metadata for successfully ingested documents
            apply_documents_catalog_metadata_start_time = time.time()
            if state_manager.documents_catalog_metadata:
                await self.__apply_documents_catalog_metadata(
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

    @trace_function("ingestor.main.build_ingestion_response", tracer=TRACER)
    async def __build_ingestion_response(
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
        failed_documents = await self.__get_failed_documents(
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
                    self._get_document_type_counts(
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

    @trace_function("ingestor.main.ingest_document_summary", tracer=TRACER)
    async def __ingest_document_summary(
        self,
        results: list[list[dict[str, str | dict]]],
        collection_name: str,
        page_filter: list[list[int]] | str | None = None,
        summarization_strategy: str | None = None,
        is_shallow: bool = False,
        vdb_op: VDBRag | None = None,
    ) -> None:
        """
        Trigger parallel summary generation for documents with optional page filtering.

        For RAPTOR strategy, this builds hierarchical trees.
        For other strategies, this creates document-level summaries.

        Args:
            results: List of document extraction results from nv-ingest
            collection_name: Name of the collection
            page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
            summarization_strategy: Strategy for summarization ('single', 'iterative', 'hierarchical', 'raptor') or None for default
            is_shallow: Whether this is shallow extraction (text-only, uses simplified prompt)
            vdb_op: VDB operator for RAPTOR tree building (optional, only needed for RAPTOR)
        """
        try:
            stats = await generate_document_summaries(
                results=results,
                collection_name=collection_name,
                page_filter=page_filter,
                summarization_strategy=summarization_strategy,
                config=self.config,
                is_shallow=is_shallow,
                prompts=self.prompts,
                vdb_op=vdb_op,  # Pass VDB operator for RAPTOR
            )

            if stats["failed"] > 0:
                logger.warning(f"Failed summaries for {collection_name}:")
                for file_name, file_stats in stats["files"].items():
                    if file_stats["status"] == "FAILED":
                        logger.warning(
                            f"  - {file_name}: {file_stats.get('error', 'unknown error')}"
                        )

        except Exception as e:
            logger.error(
                f"Summary batch failed for {collection_name}: {e}", exc_info=True
            )

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

    @trace_function("ingestor.main.apply_documents_catalog_metadata", tracer=TRACER)
    async def __apply_documents_catalog_metadata(
        self,
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
    ) -> str:
        """
        Main function called by ingestor server to create a new collection in vector-DB
        """
        # Apply defaults from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url
        embedding_dimension = self.config.embeddings.dimensions

        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name
        )

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

    @trace_function("ingestor.main.create_collections", tracer=TRACER)
    def create_collections(
        self,
        collection_names: list[str],
        vdb_endpoint: str | None = None,
        embedding_dimension: int | None = None,
        collection_type: str = "text",
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to create new collections in vector-DB
        """
        # Apply defaults from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url
        if embedding_dimension is None:
            embedding_dimension = self.config.embeddings.dimensions

        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name="",
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
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        logger.info(f"Deleting collections {collection_names}")

        try:
            vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
                vdb_endpoint=vdb_endpoint,
                collection_name="",
                vdb_auth_token=vdb_auth_token,
            )

            response = vdb_op.delete_collections(collection_names)
            # Delete citation metadata from Minio (skip if MinIO unavailable)
            if self.minio_operator is not None:
                for collection in collection_names:
                    collection_prefix = get_unique_thumbnail_id_collection_prefix(
                        collection
                    )
                    try:
                        delete_object_names = self.minio_operator.list_payloads(
                            collection_prefix
                        )
                        self.minio_operator.delete_payloads(delete_object_names)
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
                        delete_object_names = self.minio_operator.list_payloads(
                            collection_prefix
                        )
                        if len(delete_object_names):
                            self.minio_operator.delete_payloads(delete_object_names)
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
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        try:
            vdb_op, _ = self.__prepare_vdb_op_and_collection_name(
                vdb_endpoint=vdb_endpoint,
                collection_name="",
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
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        try:
            vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
                vdb_endpoint=vdb_endpoint,
                collection_name=collection_name,
                bypass_validation=bypass_validation,
                vdb_auth_token=vdb_auth_token,
            )
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
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        try:
            vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
                vdb_endpoint=vdb_endpoint,
                collection_name=collection_name,
                vdb_auth_token=vdb_auth_token,
            )

            logger.info(
                f"Deleting documents {document_names} from collection {collection_name}"
            )

            # Prepare source values for deletion
            if include_upload_path:
                upload_folder = str(
                    Path(
                        os.path.join(
                            self.config.temp_dir, f"uploaded_files/{collection_name}"
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
                if self.minio_operator is None:
                    logger.warning("MinIO unavailable - skipping metadata deletion")
                    return

                for doc in docs_to_delete:
                    # Delete citation metadata
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                        collection_name, doc
                    )
                    try:
                        delete_object_names = self.minio_operator.list_payloads(
                            filename_prefix
                        )
                        self.minio_operator.delete_payloads(delete_object_names)
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete MinIO objects for doc {doc}: {e}"
                        )

                    # Delete document summary
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                        f"summary_{collection_name}", doc
                    )
                    try:
                        delete_object_names = self.minio_operator.list_payloads(
                            filename_prefix
                        )
                        if len(delete_object_names):
                            self.minio_operator.delete_payloads(delete_object_names)
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

    @trace_function("ingestor.main.put_content_to_minio", tracer=TRACER)
    def __put_content_to_minio(
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

    @trace_function("ingestor.main.process_shallow_batch", tracer=TRACER)
    async def __process_shallow_batch(
        self,
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any],
        page_filter: list[list[int]] | str | None,
        summarization_strategy: str | None,
        batch_num: int,
        state_manager: IngestionStateManager,
    ) -> set[str]:
        """
        Process shallow extraction for a batch of files and start summary task.

        Args:
            filepaths: List of file paths to process
            collection_name: Name of the collection
            split_options: Options for splitting documents
            page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
            summarization_strategy: Strategy for summarization
            batch_num: Batch number for logging

        Returns:
            Set of filenames that failed during shallow extraction
        """
        shallow_failed_files: set[str] = set()

        shallow_results, shallow_failures = await self._perform_shallow_extraction(
            filepaths,
            split_options,
            batch_num,
            state_manager=state_manager,
        )

        # Mark per-file shallow extraction failures immediately
        if shallow_failures:
            for failed_path, error in shallow_failures:
                file_name = os.path.basename(str(failed_path))
                shallow_failed_files.add(file_name)
                SUMMARY_STATUS_HANDLER.update_progress(
                    collection_name=collection_name,
                    file_name=file_name,
                    status="FAILED",
                    error=str(error),
                )

        if shallow_results:
            vdb_op = _get_vdb_op(
                vdb_endpoint=self.config.vector_store.url,
                collection_name=collection_name,
                config=self.config,
            )
            
            task = asyncio.create_task(
                self.__ingest_document_summary(
                    shallow_results,
                    collection_name=collection_name,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                    is_shallow=True,
                    vdb_op=vdb_op,  # Pass VDB operator for RAPTOR
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # No shallow results at all: mark every file in the batch as failed (if not already marked)
            for filepath in filepaths:
                file_name = os.path.basename(filepath)
                if file_name in shallow_failed_files:
                    continue
                shallow_failed_files.add(file_name)
                SUMMARY_STATUS_HANDLER.update_progress(
                    collection_name=collection_name,
                    file_name=file_name,
                    status="FAILED",
                    error="Shallow extraction failed - no text-only results returned",
                )

        return shallow_failed_files

    @trace_function("ingestor.main.perform_shallow_extraction_workflow", tracer=TRACER)
    async def __perform_shallow_extraction_workflow(
        self,
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any],
        summary_options: dict[str, Any] | None,
        state_manager: IngestionStateManager,
    ) -> None:
        """
        Perform shallow extraction workflow for fast summary generation.
        Runs shallow extraction and starts summary tasks.
        Handles both single batch and multi-batch modes.
        Respects ENABLE_NV_INGEST_BATCH_MODE and ENABLE_NV_INGEST_PARALLEL_BATCH_MODE.

        Args:
            filepaths: List of file paths to process
            collection_name: Name of the collection
            split_options: Options for splitting documents
            summary_options: Advanced options for summary
        """
        # Extract options (summary_options is guaranteed to be non-None when this is called)
        page_filter = summary_options.get("page_filter") if summary_options else None
        summarization_strategy = (
            summary_options.get("summarization_strategy") if summary_options else None
        )

        # Determine processing mode
        if not self.config.nv_ingest.enable_batch_mode:
            # Single batch mode
            logger.info("Starting shallow extraction for %d files", len(filepaths))
            failed_files = await self.__process_shallow_batch(
                filepaths=filepaths,
                collection_name=collection_name,
                split_options=split_options,
                page_filter=page_filter,
                summarization_strategy=summarization_strategy,
                batch_num=0,
                state_manager=state_manager,
            )
            if failed_files:
                logger.warning(
                    "Shallow extraction failed for %d files", len(failed_files)
                )
            logger.info("Shallow extraction complete, starting deep ingestion")
        else:
            # Batch mode
            num_batches = (
                len(filepaths) + state_manager.files_per_batch - 1
            ) // state_manager.files_per_batch

            logger.info(
                "Starting shallow extraction for %d files across %d batches",
                len(filepaths),
                num_batches,
            )

            if not self.config.nv_ingest.enable_parallel_batch_mode:
                # Sequential batch processing
                total_failed = 0
                for i in range(
                    0, len(filepaths), state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + state_manager.files_per_batch
                    ]
                    batch_num = i // state_manager.files_per_batch + 1

                    failed_files = await self.__process_shallow_batch(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        split_options=split_options,
                        page_filter=page_filter,
                        summarization_strategy=summarization_strategy,
                        batch_num=batch_num,
                        state_manager=state_manager,
                    )
                    total_failed += len(failed_files)

                if total_failed > 0:
                    logger.warning(
                        "Shallow extraction failed for %d files across all batches",
                        total_failed,
                    )
            else:
                # Parallel batch processing with worker pool
                tasks = []
                semaphore = asyncio.Semaphore(state_manager.concurrent_batches)

                async def process_shallow_batch_parallel(sub_filepaths, batch_num):
                    async with semaphore:
                        return await self.__process_shallow_batch(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            split_options=split_options,
                            page_filter=page_filter,
                            summarization_strategy=summarization_strategy,
                            batch_num=batch_num,
                            state_manager=state_manager,
                        )

                for i in range(
                    0, len(filepaths), state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + state_manager.files_per_batch
                    ]
                    batch_num = i // state_manager.files_per_batch + 1
                    task = process_shallow_batch_parallel(sub_filepaths, batch_num)
                    tasks.append(task)

                # Wait for all shallow extraction tasks to complete
                batch_results = await asyncio.gather(*tasks)

                # Count total failed files from all batches
                total_failed = sum(len(failed_files) for failed_files in batch_results)
                if total_failed > 0:
                    logger.warning(
                        "Shallow extraction failed for %d files across all batches",
                        total_failed,
                    )

            logger.info("Shallow extraction complete, starting deep ingestion")

    @trace_function("ingestor.main.run_nvingest_batched_ingestion", tracer=TRACER)
    async def __run_nvingest_batched_ingestion(
        self,
        filepaths: list[str],
        collection_name: str,
        vdb_op: VDBRag | None = None,
        split_options: dict[str, Any] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
        state_manager: IngestionStateManager | None = None,
    ) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
        """
        Wrapper function to ingest documents in chunks using NV-ingest

        Args:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_op: VDBRag - VDB operator instance
            - split_options: SplitOptions - Options for splitting documents
            - generate_summary: bool - Whether to generate summaries
            - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
            - state_manager: IngestionStateManager - State manager for the ingestion process
        """
        # Extract summary options
        shallow_summary = False
        if summary_options:
            shallow_summary = summary_options.get("shallow_summary", False)

        # Set PENDING status for all files if summary generation is enabled
        if generate_summary:
            logger.debug("Setting PENDING status for %d files", len(filepaths))
            for filepath in filepaths:
                file_name = os.path.basename(filepath)
                SUMMARY_STATUS_HANDLER.set_status(
                    collection_name=collection_name,
                    file_name=file_name,
                    status_data={
                        "status": "PENDING",
                        "queued_at": datetime.now(UTC).isoformat(),
                        "file_name": file_name,
                        "collection_name": collection_name,
                    },
                )

            # Perform shallow extraction workflow if enabled
            if shallow_summary:
                await self.__perform_shallow_extraction_workflow(
                    filepaths=filepaths,
                    collection_name=collection_name,
                    split_options=split_options,
                    summary_options=summary_options,
                    state_manager=state_manager,
                )

        if not self.config.nv_ingest.enable_batch_mode:
            # Single batch mode
            logger.info(
                "== Performing ingestion in SINGLE batch for collection_name: %s with %d files ==",
                collection_name,
                len(filepaths),
            )
            results, failures = await self.__nv_ingest_ingestion_pipeline(
                filepaths=filepaths,
                collection_name=collection_name,
                vdb_op=vdb_op,
                split_options=split_options,
                generate_summary=generate_summary,
                summary_options=summary_options,
                state_manager=state_manager,
            )
            return results, failures

        else:
            # BATCH_MODE
            logger.info(
                f"== Performing ingestion in BATCH_MODE for collection_name: {collection_name} "
                f"with {len(filepaths)} files =="
            )

            # Process batches sequentially
            if not self.config.nv_ingest.enable_parallel_batch_mode:
                logger.info("Processing batches sequentially")
                all_results = []
                all_failures = []
                for i in range(
                    0, len(filepaths), state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + state_manager.files_per_batch
                    ]
                    batch_num = i // state_manager.files_per_batch + 1
                    total_batches = (
                        len(filepaths) + state_manager.files_per_batch - 1
                    ) // state_manager.files_per_batch
                    logger.info(
                        f"=== Batch Processing Status - Collection: {collection_name} - "
                        f"Processing batch {batch_num} of {total_batches} - "
                        f"Documents in current batch: {len(sub_filepaths)} ==="
                    )
                    results, failures = await self.__nv_ingest_ingestion_pipeline(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        vdb_op=vdb_op,
                        batch_number=batch_num,
                        split_options=split_options,
                        generate_summary=generate_summary,
                        summary_options=summary_options,
                        state_manager=state_manager,
                    )
                    all_results.extend(results)
                    all_failures.extend(failures)

                if (
                    hasattr(vdb_op, "csv_file_path")
                    and vdb_op.csv_file_path is not None
                ):
                    os.remove(vdb_op.csv_file_path)
                    logger.debug(
                        f"Deleted temporary custom metadata csv file: {vdb_op.csv_file_path} "
                        f"for collection: {collection_name}"
                    )

                return all_results, all_failures

            else:
                # Process batches in parallel with worker pool
                logger.info(
                    f"Processing batches in parallel with concurrency: {state_manager.concurrent_batches}"
                )
                all_results = []
                all_failures = []
                tasks = []
                semaphore = asyncio.Semaphore(
                    state_manager.concurrent_batches
                )  # Limit concurrent tasks

                async def process_batch(sub_filepaths, batch_num):
                    async with semaphore:
                        if len(filepaths) % state_manager.files_per_batch == 0:
                            total_batches = len(filepaths) // state_manager.files_per_batch
                        else:
                            total_batches = len(filepaths) // state_manager.files_per_batch + 1
                        logger.info(
                            f"=== Processing Batch - Collection: {collection_name} - "
                            f"Batch {batch_num} of {total_batches} - "
                            f"Documents in batch: {len(sub_filepaths)} ==="
                        )
                        return await self.__nv_ingest_ingestion_pipeline(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            vdb_op=vdb_op,
                            batch_number=batch_num,
                            split_options=split_options,
                            generate_summary=generate_summary,
                            summary_options=summary_options,
                            state_manager=state_manager,
                        )

                for i in range(
                    0, len(filepaths), state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + state_manager.files_per_batch
                    ]
                    batch_num = i // state_manager.files_per_batch + 1
                    task = process_batch(sub_filepaths, batch_num)
                    tasks.append(task)

                # Wait for all tasks to complete
                batch_results = await asyncio.gather(*tasks)

                # Combine results from all batches
                for results, failures in batch_results:
                    all_results.extend(results)
                    all_failures.extend(failures)

                if (
                    hasattr(vdb_op, "csv_file_path")
                    and vdb_op.csv_file_path is not None
                ):
                    os.remove(vdb_op.csv_file_path)
                    logger.debug(
                        f"Deleted temporary custom metadata csv file: {vdb_op.csv_file_path} "
                        f"for collection: {collection_name}"
                    )

                return all_results, all_failures

    @trace_function("ingestor.main.run_nv_ingest_ingestion_pipeline", tracer=TRACER)
    async def __nv_ingest_ingestion_pipeline(
        self,
        filepaths: list[str],
        collection_name: str,
        vdb_op: VDBRag | None = None,
        batch_number: int = 0,
        split_options: dict[str, Any] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
        state_manager: IngestionStateManager | None = None,
    ) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
        """
        This methods performs following steps:
        - Perform extraction and splitting using NV-ingest ingestor (NV-Ingest)
        - Embeds and add documents to Vectorstore collection (NV-Ingest)
        - Put content to MinIO (Ingestor Server)
        - Update batch progress with the ingestion response (Ingestor Server)

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_op: VDBRag - VDB operator instance
            - batch_number: int - Batch number for the ingestion process
            - split_options: SplitOptions - Options for splitting documents
            - generate_summary: bool - Whether to generate summaries
            - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
            - state_manager: IngestionStateManager - State manager for the ingestion process
        """
        if split_options is None:
            split_options = {
                "chunk_size": self.config.nv_ingest.chunk_size,
                "chunk_overlap": self.config.nv_ingest.chunk_overlap,
            }

        # Extract summary options
        page_filter = None
        shallow_summary = False
        summarization_strategy = None
        if summary_options:
            page_filter = summary_options.get("page_filter")
            shallow_summary = summary_options.get("shallow_summary", False)
            summarization_strategy = summary_options.get("summarization_strategy")

        filtered_filepaths = await self.__remove_unsupported_files(filepaths)

        if len(filtered_filepaths) == 0:
            logger.error("No files to ingest after filtering.")
            results, failures = [], []
            return results, failures

        results, failures = await self._perform_file_ext_based_nv_ingest_ingestion(
            batch_number=batch_number,
            filtered_filepaths=filtered_filepaths,
            split_options=split_options,
            vdb_op=vdb_op,
            state_manager=state_manager,
        )

        # Start summary task only if not shallow_summary (already started in batch wrapper)
        if generate_summary and not shallow_summary:
            task = asyncio.create_task(
                self.__ingest_document_summary(
                    results,
                    collection_name=collection_name,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                    vdb_op=vdb_op,  # Pass VDB operator for RAPTOR tree building
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            logger.info(
                "Started summary generation (strategy: %s) for batch %d",
                summarization_strategy or "iterative",
                batch_number,
            )

        if not results:
            error_message = "NV-Ingest ingestion failed with no results."
            logger.error(error_message)

            # Update FAILED status only if not shallow_summary
            if generate_summary and not shallow_summary:
                for filepath in filtered_filepaths:
                    file_name = os.path.basename(filepath)
                    SUMMARY_STATUS_HANDLER.update_progress(
                        collection_name=collection_name,
                        file_name=file_name,
                        status="FAILED",
                        error="Ingestion failed - no results returned from NV-Ingest",
                    )
                logger.warning(
                    "Marked %d files as FAILED for batch %d due to ingestion failure",
                    len(filtered_filepaths),
                    batch_number,
                )

            if len(failures) > 0:
                return results, failures
            raise Exception(error_message)

        try:
            if self.mode != Mode.LITE:
                start_time = time.time()
                self.__put_content_to_minio(
                    results=results, collection_name=collection_name
                )
                end_time = time.time()
                logger.info(
                    f"== MinIO upload for collection_name: {collection_name} "
                    f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
                )
            start_time = time.time()
            batch_progress_response = await self.__build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                state_manager=state_manager,
                is_final_batch=False,
                vdb_op=vdb_op,
            )
            end_time = time.time()
            logger.info(
                f"== Build ingestion response and adding document info for collection_name: {collection_name} "
                f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
            )
            ingestion_state = await state_manager.update_batch_progress(
                batch_progress_response=batch_progress_response,
            )
            await INGESTION_TASK_HANDLER.set_task_status_and_result(
                task_id=state_manager.get_task_id(),
                status="PENDING",
                result=ingestion_state,
            )
        except Exception as e:
            logger.error(
                "Failed to put content to minio: %s, citations would be disabled for collection: %s",
                str(e),
                collection_name,
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )

        return results, failures

    @staticmethod
    @trace_function("ingestor.main.perform_async_nv_ingest_ingestion", tracer=TRACER)
    async def __perform_async_nv_ingest_ingestion(
        nv_ingest_ingestor,
        state_manager,
        nv_ingest_traces: bool = False,
        trace_context: dict[str, Any] | None = None,
    ):
        """
        Perform NV-Ingest ingestion asynchronously using .ingest_async() method
        Also, poll the ingestion status until it is complete and update the ingestion status using state_manager

        Arguments:
            - nv_ingest_ingestor: Ingestor - NV-Ingest ingestor instance

        Returns:
            - tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]] - Results and failures
        """
        ingest_start_ns = time.time_ns()
        future = nv_ingest_ingestor.ingest_async(
            return_failures=True,
            show_progress=logger.getEffectiveLevel() <= logging.DEBUG,
            return_traces=nv_ingest_traces,
        )
        # Convert concurrent.futures.Future to asyncio.Future
        async_future = asyncio.wrap_future(future)

        while True:
            status_dict = await asyncio.to_thread(nv_ingest_ingestor.get_status)
            filename_status_map = {}
            # Normalize the status to a dictionary of filename to status
            for filepath, file_status in status_dict.items():
                filename = os.path.basename(filepath)
                filename_status_map[filename] = file_status
            nv_ingest_status = await state_manager.update_nv_ingest_status(
                filename_status_map
            )
            await INGESTION_TASK_HANDLER.set_task_state_dict(
                state_manager.get_task_id(),
                {"nv_ingest_status": nv_ingest_status},
            )

            await asyncio.sleep(1)

            if future.done():
                break

        if nv_ingest_traces:
            results, failures, traces = await async_future

            if trace_context is not None:
                process_nv_ingest_traces(
                    traces,
                    tracer=TRACER,
                    span_namespace=trace_context.get("span_namespace", "nv_ingest"),
                    collection_name=trace_context.get("collection_name"),
                    batch_number=trace_context.get("batch_number"),
                    reference_time_ns=trace_context.get(
                        "reference_time_ns", ingest_start_ns
                    ),
                )

            return results, failures

        results, failures = await async_future
        return results, failures

    @trace_function("ingestor.main.perform_shallow_extraction", tracer=TRACER)
    async def _perform_shallow_extraction(
        self,
        filepaths: list[str],
        split_options: dict[str, Any],
        batch_number: int,
        state_manager: IngestionStateManager,
    ) -> tuple[list[list[dict[str, str | dict]]], list[tuple[str, Exception]]]:
        """
        Perform text-only extraction using NV-Ingest for fast summary generation.

        Extracts only text content without multimodal elements (tables, images, charts).
        Does not generate embeddings or upload to VDB.
        Does not perform text splitting - summarization will handle its own splitting.

        Args:
            filepaths: List of file paths to extract
            split_options: Options for splitting documents (unused in shallow extraction)
            batch_number: Batch number for logging

        Returns:
            Tuple of (results, failures) where failures is list of (filepath, exception) tuples
        """
        extract_override = {
            "extract_text": True,
            "extract_infographics": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_images": False,
            "extract_method": self.config.nv_ingest.pdf_extract_method,
            "text_depth": self.config.nv_ingest.text_depth,
            "table_output_format": "pseudo_markdown",
            "extract_audio_params": {
                "segment_audio": self.config.nv_ingest.segment_audio
            },
            "extract_page_as_image": False,
        }

        try:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=self.nv_ingest_client,
                filepaths=filepaths,
                split_options=None,  # Skip splitting for shallow extraction
                vdb_op=None,
                extract_override=extract_override,
                config=self.config,
                enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                pdf_split_processing_options=state_manager.pdf_split_processing_options,
                prompts=self.prompts,
            )

            start_time = time.time()
            results, failures = await self.__perform_async_nv_ingest_ingestion(
                nv_ingest_ingestor=nv_ingest_ingestor,
                state_manager=state_manager,
                nv_ingest_traces=True,
                trace_context=create_nv_ingest_trace_context(
                    span_namespace=f"nv_ingest.shallow_batch_{batch_number}",
                    batch_number=batch_number,
                ),
            )
            total_time = time.time() - start_time

            logger.debug(
                "Shallow extraction batch %d: %.2fs, %d results, %d failures",
                batch_number,
                total_time,
                len(results) if results else 0,
                len(failures) if failures else 0,
            )

            if failures:
                logger.debug(
                    "Shallow extraction: %d failures in batch %d",
                    len(failures),
                    batch_number,
                )

            # Normalize return values to empty lists instead of None
            return results or [], failures or []

        except Exception as e:
            logger.error(
                "Shallow extraction failed for batch %d: %s",
                batch_number,
                str(e),
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            # Treat every file in this batch as a failure
            failure_records = [(filepath, e) for filepath in filepaths]
            return [], failure_records

    @trace_function(
        "ingestor.main.perform_file_ext_based_nv_ingest_ingestion", tracer=TRACER
    )
    async def _perform_file_ext_based_nv_ingest_ingestion(
        self,
        batch_number: int,
        filtered_filepaths: list[str],
        split_options: dict[str, Any],
        vdb_op: VDBRag,
        state_manager: IngestionStateManager,
    ):
        """
        Perform ingestion using NV-Ingest ingestor based on file extension
        - If pdf extract method is None, perform ingestion for all files
        - If pdf extract method is not None, split the files into PDF and non-PDF files and perform ingestion for PDF files
            - Perform ingestion for non-PDF files with remove_extract_method=True

        Arguments:
            - batch_number: int - Batch number for the ingestion process
            - filtered_filepaths: list[str] - List of filtered filepaths
            - split_options: dict[str, Any] - Options for splitting documents
            - vdb_op: VDBRag - Vector database operation instance

        Returns:
            - tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]] - Results and failures
        """
        if self.config.nv_ingest.pdf_extract_method is None:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=self.nv_ingest_client,
                filepaths=filtered_filepaths,
                split_options=split_options,
                vdb_op=vdb_op,
                config=self.config,
                enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                pdf_split_processing_options=state_manager.pdf_split_processing_options,
                prompts=self.prompts,
            )
            start_time = time.time()
            logger.info(
                f"Performing ingestion for batch {batch_number} with parameters: {split_options}"
            )
            results, failures = await self.__perform_async_nv_ingest_ingestion(
                nv_ingest_ingestor=nv_ingest_ingestor,
                state_manager=state_manager,
                nv_ingest_traces=True,
                trace_context=create_nv_ingest_trace_context(
                    span_namespace=f"nv_ingest.batch_{batch_number}",
                    collection_name=vdb_op.collection_name,
                    batch_number=batch_number,
                ),
            )
            total_ingestion_time = time.time() - start_time
            document_info = self._log_result_info(
                batch_number, results, failures, total_ingestion_time
            )
            vdb_op.add_document_info(
                info_type="collection",
                collection_name=vdb_op.collection_name,
                document_name="NA",
                info_value=document_info,
            )
            return results, failures
        else:
            pdf_filepaths, non_pdf_filepaths = await self.__split_pdf_and_non_pdf_files(
                filtered_filepaths
            )
            logger.info(
                f"Split PDF and non-PDF files for batch {batch_number}: "
                f"Count of PDF files: {len(pdf_filepaths)}, Count of non-PDF files: {len(non_pdf_filepaths)}"
            )

            results, failures = [], []
            # Perform ingestion for PDF files
            if len(pdf_filepaths) > 0:
                nv_ingest_ingestor = get_nv_ingest_ingestor(
                    nv_ingest_client_instance=self.nv_ingest_client,
                    filepaths=pdf_filepaths,
                    split_options=split_options,
                    vdb_op=vdb_op,
                    config=self.config,
                    enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                    pdf_split_processing_options=state_manager.pdf_split_processing_options,
                    prompts=self.prompts,
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for PDF files for batch {batch_number} with parameters: {split_options}"
                )
                (
                    results_pdf,
                    failures_pdf,
                ) = await self.__perform_async_nv_ingest_ingestion(
                    nv_ingest_ingestor=nv_ingest_ingestor,
                    state_manager=state_manager,
                    nv_ingest_traces=True,
                    trace_context=create_nv_ingest_trace_context(
                        span_namespace=f"nv_ingest.batch_{batch_number}.pdf",
                        collection_name=vdb_op.collection_name,
                        batch_number=batch_number,
                    ),
                )
                total_ingestion_time = time.time() - start_time
                document_info = self._log_result_info(
                    batch_number,
                    results,
                    failures,
                    total_ingestion_time,
                    additional_summary="PDF files ingestion completed",
                )
                results.extend(results_pdf)
                failures.extend(failures_pdf)

            # Perform ingestion for non-PDF files
            if len(non_pdf_filepaths) > 0:
                nv_ingest_ingestor = get_nv_ingest_ingestor(
                    nv_ingest_client_instance=self.nv_ingest_client,
                    filepaths=non_pdf_filepaths,
                    split_options=split_options,
                    vdb_op=vdb_op,
                    remove_extract_method=True,
                    config=self.config,
                    enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                    pdf_split_processing_options=state_manager.pdf_split_processing_options,
                    prompts=self.prompts,
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for non-PDF files for batch {batch_number} with parameters: {split_options}"
                )
                (
                    results_non_pdf,
                    failures_non_pdf,
                ) = await self.__perform_async_nv_ingest_ingestion(
                    nv_ingest_ingestor=nv_ingest_ingestor,
                    state_manager=state_manager,
                    nv_ingest_traces=True,
                    trace_context=create_nv_ingest_trace_context(
                        span_namespace=f"nv_ingest.batch_{batch_number}.non_pdf",
                        collection_name=vdb_op.collection_name,
                        batch_number=batch_number,
                    ),
                )
                total_ingestion_time = time.time() - start_time
                document_info = self._log_result_info(
                    batch_number,
                    results_non_pdf,
                    failures_non_pdf,
                    total_ingestion_time,
                    additional_summary="Non-PDF files ingestion completed",
                )
                results.extend(results_non_pdf)
                failures.extend(failures_non_pdf)

            vdb_op.add_document_info(
                info_type="collection",
                collection_name=vdb_op.collection_name,
                document_name="NA",
                info_value=document_info,
            )

            return results, failures

    @trace_function("ingestor.main.get_document_type_counts", tracer=TRACER)
    def _get_document_type_counts(
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

    @trace_function("ingestor.main.log_result_info", tracer=TRACER)
    def _log_result_info(
        self,
        batch_number: int,
        results: list[list[dict[str, str | dict]]],
        failures: list[dict[str, Any]],
        total_ingestion_time: float,
        additional_summary: str = "",
    ) -> dict[str, Any]:
        """Log the results info with document type counts.

        Returns:
            dict[str, Any]: Document info with metrics
        """
        doc_type_counts, total_documents, total_elements, raw_text_elements_size = (
            self._get_document_type_counts(results)
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

    @trace_function("ingestor.main.get_failed_documents", tracer=TRACER)
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
        for document in self.get_documents(collection_name, bypass_validation=True).get(
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

    @trace_function("ingestor.main.remove_unsupported_files", tracer=TRACER)
    async def __remove_unsupported_files(
        self,
        filepaths: list[str],
    ) -> list[str]:
        """Remove unsupported files from the list of filepaths"""
        non_supported_files = await self.__get_non_supported_files(filepaths)
        return [
            filepath for filepath in filepaths if filepath not in non_supported_files
        ]

    @trace_function("ingestor.main.split_pdf_and_non_pdf_files", tracer=TRACER)
    async def __split_pdf_and_non_pdf_files(
        self, filepaths: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split PDF and non-PDF files from the list of filepaths"""
        pdf_filepaths = []
        non_pdf_filepaths = []
        for filepath in filepaths:
            if os.path.splitext(filepath)[1].lower() == ".pdf":
                pdf_filepaths.append(filepath)
            else:
                non_pdf_filepaths.append(filepath)
        return pdf_filepaths, non_pdf_filepaths

    @trace_function("ingestor.main.get_non_supported_files", tracer=TRACER)
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

    @trace_function("ingestor.main.validate_custom_metadata", tracer=TRACER)
    async def _validate_custom_metadata(
        self,
        custom_metadata: list[dict[str, Any]],
        collection_name: str,
        metadata_schema_data: list[dict[str, Any]],
        filepaths: list[str],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Validate custom metadata against schema and return validation status and errors.

        Args:
            custom_metadata: User-provided metadata
            collection_name: Name of the collection
            metadata_schema_data: Metadata schema from VDB
            filepaths: List of file paths

        Returns:
            Tuple[bool, List[Dict[str, Any]]]: (validation_status, validation_errors)
            validation_errors is a list of error dictionaries in the original format
        """
        logger.info(
            f"Metadata schema for collection {collection_name}: {metadata_schema_data}"
        )
        # Validate that metadata filenames match the files being ingested
        filenames = {os.path.basename(filepath) for filepath in filepaths}

        # Setup validation if schema exists
        validator = None
        metadata_schema = None
        if metadata_schema_data:
            logger.debug(
                f"Using metadata schema for collection '{collection_name}' with {len(metadata_schema_data)} fields"
            )
            validator = MetadataValidator(self.config)
            metadata_schema = MetadataSchema(schema=metadata_schema_data)
        else:
            logger.info(
                f"No metadata schema found for collection {collection_name}. Skipping schema validation."
            )

        filename_to_metadata = {
            item.get("filename"): item.get("metadata", {}) for item in custom_metadata
        }

        validation_errors = []
        validation_status = True

        # Process all metadata items and validate them
        for custom_metadata_item in custom_metadata:
            filename = custom_metadata_item.get("filename", "")
            metadata = custom_metadata_item.get("metadata", {})

            # Check if the filename is provided in the ingestion request
            if filename not in filenames:
                validation_errors.append(
                    {
                        "error": f"Filename: {filename} is not provided in the ingestion request",
                        "metadata": {"filename": filename, "file_metadata": metadata},
                    }
                )
                validation_status = False
                continue

            if validator and metadata_schema:
                (
                    is_valid,
                    field_errors,
                    normalized_metadata,
                ) = validator.validate_and_normalize_metadata_values(
                    metadata, metadata_schema
                )
                logger.debug(
                    f"Metadata validation for '{filename}': {'PASSED' if is_valid else 'FAILED'}"
                )
                if not is_valid:
                    validation_status = False
                    # Convert new validator format to original format for backward compatibility
                    for error in field_errors:
                        error_message = error.get("error", "Validation error")
                        validation_errors.append(
                            {
                                "error": f"File '{filename}': {error_message}",
                                "metadata": {
                                    "filename": filename,
                                    "file_metadata": metadata,
                                },
                            }
                        )
                else:
                    # Update the metadata with normalized datetime values
                    custom_metadata_item["metadata"] = normalized_metadata
                    logger.debug(
                        f"Updated metadata for file '{filename}' with normalized datetime values"
                    )
            else:
                # No schema - just do basic validation (ensure it's a dict)
                if not isinstance(metadata, dict):
                    validation_errors.append(
                        {
                            "error": f"Metadata for file '{filename}' must be a dictionary",
                            "metadata": {
                                "filename": filename,
                                "file_metadata": metadata,
                            },
                        }
                    )
                    validation_status = False

        # Check for files without metadata that require it
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            if filename not in filename_to_metadata:
                if validator and metadata_schema:
                    required_fields = metadata_schema.required_fields
                    if required_fields:
                        validation_errors.append(
                            {
                                "error": f"File '{filename}': No metadata provided but schema requires fields: {required_fields}",
                                "metadata": {"filename": filename, "file_metadata": {}},
                            }
                        )
                        validation_status = False
                else:
                    logger.debug(
                        f"File '{filename}': No metadata provided, but no required fields in schema"
                    )

        if not validation_status:
            logger.error(
                f"Custom metadata validation failed: {len(validation_errors)} errors"
            )
        else:
            logger.debug("Custom metadata validated and normalized successfully.")

        return validation_status, validation_errors

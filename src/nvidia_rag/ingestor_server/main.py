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
1. __run_background_ingest_task: Ingest documents to the vector store.
2. __run_nvingest_batched_ingestion: Upload documents to the vector store using nvingest.
3. __nv_ingest_ingestion_pipeline: Run the NV-Ingest ingestion pipeline.
4. __perform_file_ext_based_nv_ingest_ingestion: Perform file extraction and splitting using NV-Ingest.
5. __get_failed_documents: Get failed documents from the vector store.
6. __get_non_supported_files: Get non-supported files from the vector store.
7. __ingest_document_summary: Drives summary generation and ingestion if enabled.
8. __prepare_summary_documents: Prepare summary documents for ingestion.
9. __generate_summary_for_documents: Generate summary for documents.
10. __put_document_summary_to_minio: Put document summaries to minio.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nv_ingest_client.primitives.tasks.extract import _DEFAULT_EXTRACTOR_MAP
from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE
from nv_ingest_client.util.vdb.adt_vdb import VDB

from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
)
from nvidia_rag.ingestor_server.task_handler import INGESTION_TASK_HANDLER
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.metadata_validation import (
    SYSTEM_MANAGED_FIELDS,
    MetadataField,
    MetadataSchema,
    MetadataValidator,
)
from nvidia_rag.utils.minio_operator import (
    get_minio_operator,
    get_unique_thumbnail_id,
    get_unique_thumbnail_id_collection_prefix,
    get_unique_thumbnail_id_file_name_prefix,
    get_unique_thumbnail_id_from_result,
)
from nvidia_rag.utils.vdb import _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag

# Initialize logger
logger = logging.getLogger(__name__)

# Constants
LIBRARY_MODE = "library"
SERVER_MODE = "server"
SUPPORTED_MODES = [LIBRARY_MODE, SERVER_MODE]

SUPPORTED_FILE_TYPES = set(_DEFAULT_EXTRACTOR_MAP.keys()) & set(
    EXTENSION_TO_DOCUMENT_TYPE.keys()
)


class NvidiaRAGIngestor:
    """
    Main Class for RAG ingestion pipeline integration for NV-Ingest
    """

    _vdb_upload_bulk_size = 500

    def __init__(
        self,
        vdb_op: VDBRag = None,
        mode: str = LIBRARY_MODE,
        config: NvidiaRAGConfig | None = None,
    ):
        """Initialize NvidiaRAGIngestor with configuration.

        Args:
            vdb_op: Optional vector database operator
            mode: Operating mode (library or server)
            config: Configuration object. If None, uses default config.
        """
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are: {SUPPORTED_MODES}"
            )
        self.mode = mode
        self.vdb_op = vdb_op
        self.config = config or NvidiaRAGConfig()

        # Initialize instance-based clients
        self.nv_ingest_client = get_nv_ingest_client(self.config)
        self.minio_operator = get_minio_operator(config=self.config)

        # Ensure default bucket exists (idempotent operation)
        try:
            self.minio_operator._make_bucket(bucket_name="a-bucket")
            logger.debug("Ensured 'a-bucket' exists in MinIO")
        except Exception as e:
            logger.warning(f"Failed to create/verify bucket 'a-bucket': {e}")

        if self.vdb_op is not None:
            if not (isinstance(self.vdb_op, VDBRag) or isinstance(self.vdb_op, VDB)):
                raise ValueError(
                    "vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag. "
                    "or nv_ingest_client.util.vdb.adt_vdb.VDB. "
                    "Please make sure all the required methods are implemented."
                )

    async def health(self, check_dependencies: bool = False) -> dict[str, Any]:
        """Check the health of the Ingestion server."""
        response_message = "Service is up."
        health_results = {}
        health_results["message"] = response_message

        vdb_op, _ = self.__prepare_vdb_op_and_collection_name(bypass_validation=True)

        if check_dependencies:
            from nvidia_rag.ingestor_server.health import check_all_services_health

            dependencies_results = await check_all_services_health(vdb_op, self.config)
            health_results.update(dependencies_results)
        return health_results

    async def validate_directory_traversal_attack(self, file):
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

    def __prepare_vdb_op_and_collection_name(
        self,
        vdb_endpoint: str | None = None,
        collection_name: str | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        filepaths: list[str] | None = None,
        bypass_validation: bool = False,
        metadata_schema: list[dict[str, Any]] | None = None,
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
            )
            return vdb_op, collection_name

        if not bypass_validation and (collection_name or custom_metadata):
            raise ValueError(
                "`collection_name` and `custom_metadata` arguments are not "
                "supported when `vdb_op` is provided during initialization."
            )

        return self.vdb_op, self.vdb_op.collection_name

    async def upload_documents(
        self,
        filepaths: list[str],
        blocking: bool = False,
        collection_name: str = None,
        vdb_endpoint: str = None,
        split_options: dict[str, Any] = None,
        custom_metadata: list[dict[str, Any]] = None,
        generate_summary: bool = False,
        additional_validation_errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Upload documents to the vector store.

        Args:
            filepaths (List[str]): List of absolute filepaths to upload
            blocking (bool, optional): Whether to block until ingestion completes. Defaults to False.
            collection_name (str, optional): Name of collection in vector database. Defaults to "multimodal_data".
            split_options (Dict[str, Any], optional): Options for splitting documents. Defaults to chunk_size and chunk_overlap from self.config.
            custom_metadata (List[Dict[str, Any]], optional): Custom metadata to add to documents. Defaults to empty list.
            additional_validation_errors (List[Dict[str, Any]] | None, optional): Additional validation errors to include in response. Defaults to None.
        """
        # Apply default from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url

        state_manager = IngestionStateManager(
            filepaths=filepaths,
            collection_name=collection_name,
            custom_metadata=custom_metadata,
        )
        task_id = state_manager.get_task_id()

        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            filepaths=filepaths,
        )
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

        if not vdb_op.check_collection_exists(collection_name):
            raise ValueError(
                f"Collection {collection_name} does not exist. Ensure a collection is created using POST /collection endpoint first."
            )

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
                        additional_validation_errors=additional_validation_errors,
                        state_manager=state_manager,
                    )

                task_id = await INGESTION_TASK_HANDLER.submit_task(
                    _task, task_id=task_id
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
                    batch_progress_response=batch_progress_response, is_batch_zero=True,
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
                    additional_validation_errors=additional_validation_errors,
                    state_manager=state_manager,
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

    async def __run_background_ingest_task(
        self,
        filepaths: list[str],
        collection_name: str = None,
        vdb_endpoint: str = None,
        vdb_op: VDBRag = None,
        split_options: dict[str, Any] = None,
        custom_metadata: list[dict[str, Any]] = None,
        generate_summary: bool = False,
        additional_validation_errors: list[dict[str, Any]] | None = None,
        state_manager: IngestionStateManager = None,
    ) -> dict[str, Any]:
        """
        Main function called by ingestor server to ingest
        the documents to vector-DB

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - split_options: Dict[str, Any] - Options for splitting documents
            - custom_metadata: List[Dict[str, Any]] - Custom metadata to be added to documents
            - additional_validation_errors: List[Dict[str, Any]] | None - Additional validation errors to include in response (defaults to None)
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

            logger.info("Filepaths for ingestion after validation: %s", filepaths)

            # Peform ingestion using nvingest for all files that have not failed
            # Check if the provided collection_name exists in vector-DB

            start_time = time.time()
            results, failures = await self.__run_nvingest_batched_ingestion(
                filepaths=filepaths,
                collection_name=collection_name,
                vdb_op=vdb_op,
                split_options=split_options,
                generate_summary=generate_summary,
                state_manager=state_manager,
            )

            logger.info(
                "== Overall Ingestion completed successfully in %s seconds ==",
                time.time() - start_time,
            )

            response_data = await self.__build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                state_manager=state_manager,
                is_final_batch=True,
                vdb_op=vdb_op,
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
            if self.mode == SERVER_MODE:
                logger.info(f"Cleaning up files in {filepaths}")
                for file in filepaths:
                    try:
                        os.remove(file)
                        logger.debug(f"Deleted temporary file: {file}")
                    except FileNotFoundError:
                        logger.warning(f"File not found: {file}")
                    except Exception as e:
                        logger.error(f"Error deleting {file}: {e}")

            return ingestion_state

        except Exception as e:
            logger.exception(
                "Ingestion failed due to error: %s",
                e,
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            raise e

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
        filename_to_result_map = {
            os.path.basename(result[0].get("metadata").get("source_metadata").get("source_id")): result for result in results
        }
        # Generate response dictionary
        uploaded_documents = list()
        for filepath in filepaths:
            if os.path.basename(filepath) not in failures_filepaths:
                doc_type_counts, _, total_elements, raw_text_elements_size = \
                    self._get_document_type_counts([filename_to_result_map.get(os.path.basename(filepath))])
                document_info = {
                    "doc_type_counts": doc_type_counts,
                    "total_elements": total_elements,
                    "raw_text_elements_size": raw_text_elements_size,
                }
                if not is_final_batch:
                    vdb_op.add_document_info(
                        info_type="document",
                        collection_name=state_manager.collection_name,
                        document_name=os.path.basename(filepath),
                        info_value=document_info,
                    )
                uploaded_document = {
                    # Generate a document_id from filename
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

    async def __ingest_document_summary(
        self, results: list[list[dict[str, str | dict]]], collection_name: str
    ) -> None:
        """
        Generates and ingests document summaries for a list of files.

        Args:
            filepaths (List[str]): List of paths to documents to generate summaries for
        """

        logger.info("Document summary ingestion started")
        start_time = time.time()
        # Prepare summary documents
        documents = await self.__prepare_summary_documents(results, collection_name)
        # Generate summary for each document
        documents = await self.__generate_summary_for_documents(documents)
        # # Add document summary to minio
        await self.__put_document_summary_to_minio(documents)
        end_time = time.time()
        logger.info(
            f"Document summary ingestion completed! Time taken: {end_time - start_time} seconds"
        )

    async def update_documents(
        self,
        filepaths: list[str],
        blocking: bool = False,
        collection_name: str = None,
        vdb_endpoint: str = None,
        split_options: dict[str, Any] = None,
        custom_metadata: list[dict[str, Any]] = None,
        generate_summary: bool = False,
        additional_validation_errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Upload a document to the vector store. If the document already exists, it will be replaced."""

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

            if self.mode == SERVER_MODE:
                response = self.delete_documents(
                    [file_name],
                    collection_name=collection_name,
                    include_upload_path=True,
                )
            else:
                response = self.delete_documents(
                    [file], collection_name=collection_name
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
            additional_validation_errors=additional_validation_errors,
        )
        return response

    @staticmethod
    async def status(task_id: str) -> dict[str, Any]:
        """Get the status of an ingestion task."""

        logger.info(f"Getting status of task {task_id}")
        try:
            status_and_result = INGESTION_TASK_HANDLER.get_task_status_and_result(
                task_id
            )
            if status_and_result.get("state") == "PENDING":
                logger.info(f"Task {task_id} is pending")
                return {"state": "PENDING", "result": status_and_result.get("result")}
            elif status_and_result.get("state") == "FINISHED":
                try:
                    result = status_and_result.get("result")
                    if isinstance(result, dict) and result.get("state") == "FAILED":
                        logger.error(
                            f"Task {task_id} failed with error: {result.get('message')}"
                        )
                        result.pop("state")
                        return {"state": "FAILED", "result": result}
                    logger.info(f"Task {task_id} is finished")
                    return {"state": "FINISHED", "result": result}
                except Exception as e:
                    logger.error(f"Task {task_id} failed with error: {e}")
                    return {"state": "FAILED", "result": {"message": str(e)}}
            elif status_and_result.get("state") == "FAILED":
                logger.error(
                    f"Task {task_id} failed with error: {status_and_result.get('result').get('message')}"
                )
                return {"state": "FAILED", "result": status_and_result.get("result")}
            else:
                task_state = INGESTION_TASK_HANDLER.get_task_status(task_id)
                logger.error(f"Unknown task state: {task_state}")
                return {"state": "UNKNOWN", "result": {"message": "Unknown task state"}}
        except KeyError as e:
            logger.error(f"Task {task_id} not found with error: {e}")
            return {"state": "UNKNOWN", "result": {"message": "Unknown task state"}}

    def create_collection(
        self,
        collection_name: str = None,
        vdb_endpoint: str = None,
        embedding_dimension: int = None,
        metadata_schema: list[dict[str, str]] = None,
    ) -> str:
        """
        Main function called by ingestor server to create a new collection in vector-DB
        """
        # Apply defaults from config if not provided
        if vdb_endpoint is None:
            vdb_endpoint = self.config.vector_store.url
        if embedding_dimension is None:
            embedding_dimension = self.config.embeddings.dimensions

        vdb_op, collection_name = self.__prepare_vdb_op_and_collection_name(
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
        )

        if metadata_schema is None:
            metadata_schema = []

        existing_field_names = {field.get("name") for field in metadata_schema}

        # Add system-managed fields (RAG-managed and nv-ingest auto-extracted)
        for field_name, field_def in SYSTEM_MANAGED_FIELDS.items():
            if field_name not in existing_field_names:
                metadata_schema.append(
                    {
                        "name": field_name,
                        "type": field_def["type"],
                        "description": field_def["description"],
                        "required": False,  # System fields are never required
                        "user_defined": field_def[
                            "rag_managed"
                        ],  # rag_managed=True means user_defined=True
                        "support_dynamic_filtering": field_def[
                            "support_dynamic_filtering"
                        ],
                    }
                )

        try:
            # Create the metadata schema collection
            vdb_op.create_metadata_schema_collection()
            # Check if the collection already exists
            existing_collections = vdb_op.get_collection()
            if collection_name in [f["collection_name"] for f in existing_collections]:
                return {
                    "message": f"Collection {collection_name} already exists.",
                    "collection_name": collection_name,
                }
            logger.info(f"Creating collection {collection_name}")
            vdb_op.create_collection(collection_name, embedding_dimension)

            # Add metadata schema with validation
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
                logger.info(
                    f"Metadata schema validated and added to collection {collection_name}"
                )

            return {
                "message": f"Collection {collection_name} created successfully.",
                "collection_name": collection_name,
            }
        except Exception as e:
            logger.exception(f"Failed to create collection: {e}")
            raise Exception(f"Failed to create collection: {e}") from e

    def create_collections(
        self,
        collection_names: list[str],
        vdb_endpoint: str = None,
        embedding_dimension: int = None,
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

    def delete_collections(
        self,
        collection_names: list[str],
        vdb_endpoint: str = None,
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
            )

            response = vdb_op.delete_collections(collection_names)
            # Delete citation metadata from Minio
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(
                    collection
                )
                delete_object_names = self.minio_operator.list_payloads(
                    collection_prefix
                )
                self.minio_operator.delete_payloads(delete_object_names)

            # Delete document summary from Minio
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(
                    f"summary_{collection}"
                )
                delete_object_names = self.minio_operator.list_payloads(
                    collection_prefix
                )
                if len(delete_object_names):
                    self.minio_operator.delete_payloads(delete_object_names)
                    logger.info(
                        f"Deleted all document summaries from Minio for collection: {collection}"
                    )

            return response
        except Exception as e:
            logger.error(f"Failed to delete collections in milvus: {e}")
            from traceback import print_exc

            logger.error(print_exc())
            return {
                "message": f"Failed to delete collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0,
            }

    def get_collections(
        self,
        vdb_endpoint: str = None,
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
            logger.error(f"Failed to retrieve collections: {e}")
            return {
                "message": f"Failed to retrieve collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0,
            }

    def get_documents(
        self,
        collection_name: str = None,
        vdb_endpoint: str = None,
        bypass_validation: bool = False,
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
            logger.exception(f"Failed to retrieve documents due to error {e}.")
            return {
                "documents": [],
                "total_documents": 0,
                "message": f"Document listing failed due to error {e}.",
            }

    def delete_documents(
        self,
        document_names: list[str],
        collection_name: str = None,
        vdb_endpoint: str = None,
        include_upload_path: bool = False,
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

            if vdb_op.delete_documents(collection_name, source_values):
                # Generate response dictionary
                documents = [
                    {
                        "document_id": "",  # TODO - Use actual document_id
                        "document_name": doc,
                        "size_bytes": 0,  # TODO - Use actual size
                    }
                    for doc in document_names
                ]
                # Delete citation metadata from Minio
                for doc in document_names:
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                        collection_name, doc
                    )
                    delete_object_names = self.minio_operator.list_payloads(
                        filename_prefix
                    )
                    self.minio_operator.delete_payloads(delete_object_names)

                # Delete document summary from Minio
                for doc in document_names:
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(
                        f"summary_{collection_name}", doc
                    )
                    delete_object_names = self.minio_operator.list_payloads(
                        filename_prefix
                    )
                    if len(delete_object_names):
                        self.minio_operator.delete_payloads(delete_object_names)
                        logger.info(f"Deleted summary for doc: {doc} from Minio")
                return {
                    "message": "Files deleted successfully",
                    "total_documents": len(documents),
                    "documents": documents,
                }

        except Exception as e:
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

        if os.getenv("ENABLE_MINIO_BULK_UPLOAD", "True") in ["True", "true"]:
            logger.info(f"Bulk uploading {len(payloads)} payloads to MinIO")
            self.minio_operator.put_payloads_bulk(
                payloads=payloads, object_names=object_names
            )
        else:
            logger.info(f"Sequentially uploading {len(payloads)} payloads to MinIO")
            for payload, object_name in zip(payloads, object_names, strict=False):
                self.minio_operator.put_payload(
                    payload=payload, object_name=object_name
                )

    async def __run_nvingest_batched_ingestion(
        self,
        filepaths: list[str],
        collection_name: str,
        vdb_op: VDBRag = None,
        split_options: dict[str, Any] = None,
        generate_summary: bool = False,
        state_manager: IngestionStateManager = None,
    ) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
        """
        Wrapper function to ingest documents in chunks using NV-ingest

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_endpoint: str - URL of the vector database endpoint
            - split_options: SplitOptions - Options for splitting documents
            - custom_metadata: List[CustomMetadata] - Custom metadata to be added to documents
        """
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
                    0, len(filepaths), self.config.nv_ingest.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + self.config.nv_ingest.files_per_batch
                    ]
                    batch_num = i // self.config.nv_ingest.files_per_batch + 1
                    logger.info(
                        f"=== Batch Processing Status - Collection: {collection_name} - "
                        f"Processing batch {batch_num} of {len(filepaths) // self.config.nv_ingest.files_per_batch + 1} - "
                        f"Documents in current batch: {len(sub_filepaths)} ==="
                    )
                    results, failures = await self.__nv_ingest_ingestion_pipeline(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        vdb_op=vdb_op,
                        batch_number=batch_num,
                        split_options=split_options,
                        generate_summary=generate_summary,
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
                    f"Processing batches in parallel with concurrency: {self.config.nv_ingest.concurrent_batches}"
                )
                all_results = []
                all_failures = []
                tasks = []
                semaphore = asyncio.Semaphore(
                    self.config.nv_ingest.concurrent_batches
                )  # Limit concurrent tasks

                async def process_batch(sub_filepaths, batch_num):
                    async with semaphore:
                        logger.info(
                            f"=== Processing Batch - Collection: {collection_name} - "
                            f"Batch {batch_num} of {len(filepaths) // self.config.nv_ingest.files_per_batch + 1} - "
                            f"Documents in batch: {len(sub_filepaths)} ==="
                        )
                        return await self.__nv_ingest_ingestion_pipeline(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            vdb_op=vdb_op,
                            batch_number=batch_num,
                            split_options=split_options,
                            generate_summary=generate_summary,
                            state_manager=state_manager,
                        )

                for i in range(
                    0, len(filepaths), self.config.nv_ingest.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + self.config.nv_ingest.files_per_batch
                    ]
                    batch_num = i // self.config.nv_ingest.files_per_batch + 1
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

    async def __nv_ingest_ingestion_pipeline(
        self,
        filepaths: list[str],
        collection_name: str,
        vdb_op: VDBRag = None,
        batch_number: int = 0,
        split_options: dict[str, Any] = None,
        generate_summary: bool = False,
        state_manager: IngestionStateManager = None,
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
            - vdb_endpoint: str - URL of the vector database endpoint
            - batch_number: int - Batch number for the ingestion process
            - split_options: SplitOptions - Options for splitting documents
            - custom_metadata: List[CustomMetadata] - Custom metadata to be added to documents
        """
        if split_options is None:
            split_options = {
                "chunk_size": self.config.nv_ingest.chunk_size,
                "chunk_overlap": self.config.nv_ingest.chunk_overlap,
            }

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
        )

        if generate_summary:
            logger.info(
                f"Document summary generation starting in background for batch {batch_number}.."
            )
            asyncio.create_task(
                self.__ingest_document_summary(results, collection_name=collection_name)
            )

        if not results:
            error_message = "NV-Ingest ingestion failed with no results."
            logger.error(error_message)
            if len(failures) > 0:
                return results, failures
            raise Exception(error_message)

        try:
            start_time = time.time()
            self.__put_content_to_minio(
                results=results, collection_name=collection_name
            )
            end_time = time.time()
            logger.info(
                f"== MinIO upload for collection_name: {collection_name} "
                f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
            )
            batch_progress_response = await self.__build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                state_manager=state_manager,
                is_final_batch=False,
                vdb_op=vdb_op,
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

    async def _perform_file_ext_based_nv_ingest_ingestion(
        self,
        batch_number: int,
        filtered_filepaths: list[str],
        split_options: dict[str, Any],
        vdb_op: VDBRag,
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
        if self.config.nv_ingest.pdf_extract_method in ["None", "none"]:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=self.nv_ingest_client,
                filepaths=filtered_filepaths,
                split_options=split_options,
                vdb_op=vdb_op,
                config=self.config,
            )
            start_time = time.time()
            logger.info(
                f"Performing ingestion for batch {batch_number} with parameters: {split_options}"
            )
            results, failures = await asyncio.to_thread(
                lambda: nv_ingest_ingestor.ingest(
                    return_failures=True,
                    show_progress=logger.getEffectiveLevel() <= logging.DEBUG,
                )
            )
            total_ingestion_time = time.time() - start_time
            document_info = self._log_result_info(batch_number, results, failures, total_ingestion_time)
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
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for PDF files for batch {batch_number} with parameters: {split_options}"
                )
                results_pdf, failures_pdf = await asyncio.to_thread(
                    lambda: nv_ingest_ingestor.ingest(
                        return_failures=True,
                        show_progress=logger.getEffectiveLevel() <= logging.DEBUG,
                    )
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
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for non-PDF files for batch {batch_number} with parameters: {split_options}"
                )
                results_non_pdf, failures_non_pdf = await asyncio.to_thread(
                    lambda: nv_ingest_ingestor.ingest(
                        return_failures=True,
                        show_progress=logger.getEffectiveLevel() <= logging.DEBUG,
                    )
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
    
    def _get_document_type_counts(
        self,
        results: list[list[dict[str, str | dict]]]
    ) -> dict[str, int]:
        """
        Get document type counts from the results
        """
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
                if document_subtype:
                    document_type_subtype = document_subtype
                else:
                    document_type_subtype = document_type
                doc_type_counts[document_type_subtype] += 1
                if document_type == "text":
                    raw_text_elements_size += len(
                        result_element.get("metadata", {}).get("content", "")
                    )
        return doc_type_counts, total_documents, total_elements, raw_text_elements_size

    def _log_result_info(
        self,
        batch_number: int,
        results: list[list[dict[str, str | dict]]],
        failures: list[dict[str, Any]],
        total_ingestion_time: float,
        additional_summary: str = "",
    ) -> dict[str, Any]:
        """
        Log the results info with document type counts

        Returns:
            - dict[str, Any] - Document info
        """
        doc_type_counts, total_documents, total_elements, raw_text_elements_size = \
            self._get_document_type_counts(results)
        document_info = {
            "doc_type_counts": doc_type_counts,
            "total_elements": total_elements,
            "raw_text_elements_size": raw_text_elements_size,
        }
        # Create summary string
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

    async def __remove_unsupported_files(
        self,
        filepaths: list[str],
    ) -> list[str]:
        """Remove unsupported files from the list of filepaths"""
        non_supported_files = await self.__get_non_supported_files(filepaths)
        return [
            filepath for filepath in filepaths if filepath not in non_supported_files
        ]

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

    async def __prepare_summary_documents(
        self, results: list[list[dict[str, str | dict]]], collection_name: str
    ) -> list[Document]:
        """
        Prepare summary documents from the results to gather content for each file
        """
        summary_documents = []

        for result in results:
            documents = self.__parse_documents([result])
            if documents:
                full_content = " ".join([doc.page_content for doc in documents])
                metadata = {
                    "filename": documents[0].metadata["source_name"],
                    "collection_name": collection_name,
                }
                summary_documents.append(
                    Document(page_content=full_content, metadata=metadata)
                )
        return summary_documents

    def __parse_documents(
        self, results: list[list[dict[str, str | dict]]]
    ) -> list[Document]:
        """
        Extract document page content from the results obtained from nv-ingest

        Arguments:
            - results: List[List[Dict[str, Union[str, dict]]]] - Results obtained from nv-ingest

        Returns
            - List[Document] - List of documents with page content
        """
        documents = []
        for result in results:
            for result_element in result:
                # Prepare metadata
                metadata = self.__prepare_metadata(result_element=result_element)

                # Extract documents page_content and prepare docs
                page_content = None
                # For textual data
                if result_element.get("document_type") == "text":
                    page_content = result_element.get("metadata").get("content")

                # For both tables and charts
                elif result_element.get("document_type") == "structured":
                    structured_page_content = (
                        result_element.get("metadata")
                        .get("table_metadata")
                        .get("table_content")
                    )
                    subtype = (
                        result_element.get("metadata")
                        .get("content_metadata")
                        .get("subtype")
                    )
                    # Check for tables
                    if subtype == "table" and self.config.nv_ingest.extract_tables:
                        page_content = structured_page_content
                    # Check for charts
                    elif subtype == "chart" and self.config.nv_ingest.extract_charts:
                        page_content = structured_page_content

                # For image captions
                elif (
                    result_element.get("document_type") == "image"
                    and self.config.nv_ingest.extract_images
                ):
                    page_content = (
                        result_element.get("metadata")
                        .get("image_metadata")
                        .get("caption")
                    )

                # For audio transcripts
                elif result_element.get("document_type") == "audio":
                    page_content = (
                        result_element.get("metadata")
                        .get("audio_metadata")
                        .get("audio_transcript")
                    )

                # Add doc to list
                if page_content:
                    documents.append(
                        Document(page_content=page_content, metadata=metadata)
                    )
        return documents

    def __prepare_metadata(
        self, result_element: dict[str, str | dict]
    ) -> dict[str, str]:
        """
        Prepare metadata object w.r.t. to a single chunk

        Arguments:
            - result_element: Dict[str, Union[str, dict]]] - Result element for single chunk

        Returns:
            - metadata: Dict[str, str] - Dict of metadata for s single chunk
            {
                "source": "<filepath>",
                "chunk_type": "<chunk_type>", # ["text", "image", "table", "chart"]
                "source_name": "<filename>",
                "content": "<base64_str encoded content>" # Only for ["image", "table", "chart"]
            }
        """
        source_id = (
            result_element.get("metadata").get("source_metadata").get("source_id")
        )

        # Get chunk_type
        if result_element.get("document_type") == "structured":
            chunk_type = (
                result_element.get("metadata").get("content_metadata").get("subtype")
            )
        else:
            chunk_type = result_element.get("document_type")

        # Get base64_str encoded content, empty str in case of text
        # content = (
        #     result_element.get("metadata").get("content")
        #     if chunk_type != "text"
        #     else ""
        # )

        metadata = {
            # Add filepath (Key-name same for backward compatibility)
            "source": source_id,
            "chunk_type": chunk_type,  # ["text", "image", "table", "chart"]
            "source_name": os.path.basename(source_id),  # Add filename
            # "content": content # content encoded in base64_str format [Must not exceed 64KB]
        }
        return metadata

    async def __generate_summary_for_documents(
        self, documents: list[Document]
    ) -> list[Document]:
        """
        Generate summaries for documents using iterative chunk-wise approach
        """
        # Generate document summary
        summary_llm_name = self.config.summarizer.model_name
        summary_llm_endpoint = self.config.summarizer.server_url
        prompts = get_prompts()

        llm_params = {
            "model": summary_llm_name,
            "temperature": self.config.summarizer.temperature,
            "top_p": self.config.summarizer.top_p,
        }

        if summary_llm_endpoint:
            llm_params["llm_endpoint"] = summary_llm_endpoint

        summary_llm = get_llm(config=self.config, **llm_params)

        document_summary_prompt = prompts.get("document_summary_prompt")
        logger.debug(f"Document summary prompt: {document_summary_prompt}")

        # Initial summary prompt for first chunk
        initial_summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", document_summary_prompt["system"]),
                ("human", document_summary_prompt["human"]),
            ]
        )

        # Iterative summary prompt for subsequent chunks
        iterative_summary_prompt_config = prompts.get("iterative_summary_prompt")
        iterative_summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", iterative_summary_prompt_config["system"]),
                ("human", iterative_summary_prompt_config["human"]),
            ]
        )

        initial_chain = initial_summary_prompt | summary_llm | StrOutputParser()
        iterative_chain = iterative_summary_prompt | summary_llm | StrOutputParser()

        # Use configured chunk size
        max_chunk_chars = self.config.summarizer.max_chunk_length
        chunk_overlap = self.config.summarizer.chunk_overlap
        logger.info(f"Using chunk size: {max_chunk_chars} characters")

        if not len(documents):
            logger.error(
                "No content returned from nv-ingest to summarize. Skipping summary generation."
            )
            return []

        for document in documents:
            document_text = document.page_content

            # Check if document fits in single request
            if len(document_text) <= max_chunk_chars:
                # Process as single chunk
                logger.info(
                    f"Processing document {document.metadata['filename']} as single chunk"
                )
                summary = await initial_chain.ainvoke(
                    {"document_text": document_text},
                    config={"run_name": "document-summary"},
                )
            else:
                # Process in chunks iteratively using LangChain's text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_chars,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                )
                text_chunks = text_splitter.split_text(document_text)
                logger.info(
                    f"Processing document {document.metadata['filename']} in {len(text_chunks)} chunks"
                )

                # Generate initial summary from first chunk
                summary = await initial_chain.ainvoke(
                    {"document_text": text_chunks[0]},
                    config={"run_name": "document-summary-initial"},
                )

                # Iteratively update summary with remaining chunks
                for i, chunk in enumerate(text_chunks[1:], 1):
                    logger.info(
                        f"Processing chunk {i + 1}/{len(text_chunks)} for {document.metadata['filename']}"
                    )
                    summary = await iterative_chain.ainvoke(
                        {"previous_summary": summary, "new_chunk": chunk},
                        config={"run_name": f"document-summary-chunk-{i + 1}"},
                    )
                    logger.debug(
                        f"Summary for chunk {i + 1}/{len(text_chunks)} for {document.metadata['filename']}: {summary}"
                    )

            document.metadata["summary"] = summary
            logger.debug(
                f"Document summary for {document.metadata['filename']}: {summary}"
            )

        logger.info("Document summary generation complete!")
        return documents

    async def __put_document_summary_to_minio(self, documents: list[Document]) -> None:
        """
        Put document summary to minio
        """
        if not len(documents):
            logger.error("No documents to put to minio")
            return

        for document in documents:
            summary = document.metadata["summary"]
            file_name = document.metadata["filename"]
            collection_name = document.metadata["collection_name"]
            page_number = 0
            location = []

            unique_thumbnail_id = get_unique_thumbnail_id(
                collection_name=f"summary_{collection_name}",
                file_name=file_name,
                page_number=page_number,
                location=location,
            )

            self.minio_operator.put_payload(
                payload={
                    "summary": summary,
                    "file_name": file_name,
                    "collection_name": collection_name,
                },
                object_name=unique_thumbnail_id,
            )
            logger.debug(f"Document summary for {file_name} ingested to minio")

        logger.info("Document summary insertion completed to minio!")

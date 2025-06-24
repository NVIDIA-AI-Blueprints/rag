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
1. __ingest_docs: Ingest documents to the vector store.
2. __nvingest_upload_doc: Upload documents to the vector store using nvingest.
3. __get_failed_documents: Get failed documents from the vector store.
4. __get_non_supported_files: Get non-supported files from the vector store.
5. __ingest_document_summary: Drives summary generation and ingestion if enabled.
6. __prepare_summary_documents: Prepare summary documents for ingestion.
7. __generate_summary_for_documents: Generate summary for documents.
8. __put_document_summary_to_minio: Put document summaries to minio.
"""

import os
import time
import asyncio
import json
from typing import (
    List,
    Dict,
    Union,
    Any,
    Tuple
)
import logging
from uuid import uuid4
from datetime import datetime
from pymilvus import utility, connections
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.ingestor_server.nvingest import get_nv_ingest_client, get_nv_ingest_ingestor
from nvidia_rag.utils.common import get_config
from nvidia_rag.ingestor_server.task_handler import INGESTION_TASK_HANDLER
from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE
from nvidia_rag.utils.minio_operator import (get_minio_operator,
                                      get_unique_thumbnail_id_collection_prefix,
                                      get_unique_thumbnail_id_file_name_prefix,
                                      get_unique_thumbnail_id)
from nvidia_rag.utils.vectorstore import (
    get_vectorstore,
    get_docs_vectorstore_langchain,
    del_docs_vectorstore_langchain,
    create_collection,
    create_collections,
    get_collection,
    delete_collections,
    create_metadata_schema_collection,
    add_metadata_schema,
    get_metadata_schema,
)

# Initialize global objects
logger = logging.getLogger(__name__)

CONFIG = get_config()
DOCUMENT_EMBEDDER = document_embedder = get_embedding_model(model=CONFIG.embeddings.model_name, url=CONFIG.embeddings.server_url)
NV_INGEST_CLIENT_INSTANCE = get_nv_ingest_client()
MINIO_OPERATOR = get_minio_operator()

# NV-Ingest Batch Mode Configuration
ENABLE_NV_INGEST_BATCH_MODE = os.getenv("ENABLE_NV_INGEST_BATCH_MODE", "true").lower() == "true"
NV_INGEST_FILES_PER_BATCH = int(os.getenv("NV_INGEST_FILES_PER_BATCH", 16))
ENABLE_NV_INGEST_PARALLEL_BATCH_MODE = os.getenv("ENABLE_NV_INGEST_PARALLEL_BATCH_MODE", "true").lower() == "true"
NV_INGEST_CONCURRENT_BATCHES = int(os.getenv("NV_INGEST_CONCURRENT_BATCHES", 4))

class NvidiaRAGIngestor():
    """
    Main Class for RAG ingestion pipeline integration for NV-Ingest
    """

    _config = get_config()
    _vdb_upload_bulk_size = 500


    async def upload_documents(
        self,
        filepaths: List[str],
        delete_files_after_ingestion: bool = False,
        blocking: bool = False,
        vdb_endpoint: str = CONFIG.vector_store.url,
        collection_name: str = "multimodal_data",
        split_options: Dict[str, Any] = {"chunk_size": CONFIG.nv_ingest.chunk_size, "chunk_overlap": CONFIG.nv_ingest.chunk_overlap},
        custom_metadata: List[Dict[str, Any]] = [],
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """Upload documents to the vector store.

        Args:
            filepaths (List[str]): List of absolute filepaths to upload
            delete_files_after_ingestion (bool, optional): Whether to delete files after ingestion. Defaults to False.
            blocking (bool, optional): Whether to block until ingestion completes. Defaults to False.
            vdb_endpoint (str, optional): URL of vector database endpoint. Defaults to VECTOR_STORE_URL env var or "http://localhost:19530".
            collection_name (str, optional): Name of collection in vector database. Defaults to "multimodal_data".
            split_options (Dict[str, Any], optional): Options for splitting documents. Defaults to chunk_size and chunk_overlap from settings.
            custom_metadata (List[Dict[str, Any]], optional): Custom metadata to add to documents. Defaults to empty list.
        """

        try:

            if not blocking:
                _task = lambda: self.__ingest_docs(
                    filepaths=filepaths,
                    delete_files_after_ingestion=delete_files_after_ingestion,
                    vdb_endpoint=vdb_endpoint,
                    collection_name=collection_name,
                    split_options=split_options,
                    custom_metadata=custom_metadata,
                    generate_summary=generate_summary
                )
                task_id = INGESTION_TASK_HANDLER.submit_task(_task)
                return {"message": "Ingestion started in background", "task_id": task_id}
            else:
                response_dict = await self.__ingest_docs(
                    filepaths=filepaths,
                    delete_files_after_ingestion=delete_files_after_ingestion,
                    vdb_endpoint=vdb_endpoint,
                    collection_name=collection_name,
                    split_options=split_options,
                    custom_metadata=custom_metadata,
                    generate_summary=generate_summary
                )
            return response_dict

        except Exception as e:
            logger.exception(f"Failed to upload documents: {e}")
            return {
                "message": f"Failed to upload documents due to error: {str(e)}",
                "total_documents": len(filepaths),
                "documents": [],
                "failed_documents": []
            }

    async def __ingest_docs(
        self,
        filepaths: List[str],
        delete_files_after_ingestion: bool = False,
        vdb_endpoint: str = None,
        collection_name: str = "multimodal_data",
        split_options: Dict[str, Any] = {"chunk_size": CONFIG.nv_ingest.chunk_size, "chunk_overlap": CONFIG.nv_ingest.chunk_overlap},
        custom_metadata: List[Dict[str, Any]] = [],
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Main function called by ingestor server to ingest
        the documents to vector-DB

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - delete_files_after_ingestion: bool - Whether to delete files after ingestion
            - vdb_endpoint: str - URL of the vector database endpoint
            - collection_name: str - Name of the collection in the vector database
            - split_options: Dict[str, Any] - Options for splitting documents
            - custom_metadata: List[Dict[str, Any]] - Custom metadata to be added to documents
        """

        vdb_endpoint = vdb_endpoint or CONFIG.vector_store.url
        logger.info("Performing ingestion in collection_name: %s", collection_name)
        logger.debug("Filepaths for ingestion: %s", filepaths)

        try:
            # Verify the metadata
            if custom_metadata:
                validation_status, validation_errors = await self.__verify_metadata(custom_metadata, collection_name, vdb_endpoint, filepaths)
                if not validation_status:
                    return {
                        "message": f"Custom metadata validation failed: {validation_errors}",
                        "total_documents": len(filepaths),
                        "documents": [],
                        "failed_documents": [],
                        "validation_errors": validation_errors,
                        "state": "FAILED",
                    }

            failed_validation_documents = []

            # Get all documents in the collection
            get_docs_response = self.get_documents(collection_name, vdb_endpoint)

            for file in filepaths:
                filename = os.path.basename(file)
                # Check if the provided filepaths are valid
                if not os.path.exists(file):
                    logger.error(f"File {file} does not exist. Ingestion failed.")
                    failed_validation_documents.append({
                        "document_name": filename,
                        "error_message": f"File {filename} does not exist at path {file}. Ingestion failed."
                    })

                if not os.path.isfile(file):
                    failed_validation_documents.append({
                        "document_name": filename,
                        "error_message": f"File {filename} is not a file. Ingestion failed."
                    })

                # Check if the provided filepaths are already in vector-DB
                if filename in [doc.get("document_name") for doc in get_docs_response['documents']]:
                    logger.error(f"Document {file} already exists. Upload failed. Please call PATCH /documents endpoint to delete and replace this file.")
                    failed_validation_documents.append({
                        "document_name": filename,
                        "error_message": f"Document {filename} already exists. Use update document API instead."
                    })

                # Check for unsupported file formats (.rst, .rtf, etc.)
                not_supported_formats = ('.rst', '.rtf', '.org')
                if filename.endswith(not_supported_formats):
                    logger.info("Detected a .rst or .rtf file, you need to install Pandoc manually in Docker.")
                    # Provide instructions to install Pandoc in Dockerfile
                    dockerfile_instructions = """
                    # Install pandoc from the tarball to support ingestion .rst, .rtf & .org files
                    RUN curl -L https://github.com/jgm/pandoc/releases/download/3.6/pandoc-3.6-linux-amd64.tar.gz -o /tmp/pandoc.tar.gz && \
                    tar -xzf /tmp/pandoc.tar.gz -C /tmp && \
                    mv /tmp/pandoc-3.6/bin/pandoc /usr/local/bin/ && \
                    rm -rf /tmp/pandoc.tar.gz /tmp/pandoc-3.6
                    """
                    logger.info(dockerfile_instructions)
                    failed_validation_documents.append({
                        "document_name": filename,
                        "error_message": f"Document {filename} is not a supported format. Check logs for details."
                    })

            # Check if all provided files have failed
            if len(failed_validation_documents) == len(filepaths):
                return {
                    "message": f"Document upload job failed. All files failed to validate. Check logs for details.",
                        "total_documents": len(filepaths),
                        "documents": [],
                        "failed_documents": failed_validation_documents,
                        "validation_errors": [],
                        "state": "FAILED"
                }

            # Remove the failed validation documents from the filepaths
            filepaths = [file for file in filepaths if os.path.basename(file) not in [failed_document.get("document_name") for failed_document in failed_validation_documents]]

            if len(failed_validation_documents):
                logger.error(f"Validation errors: {failed_validation_documents}")

            logger.info("Filepaths for ingestion after validation: %s", filepaths)

            # Peform ingestion using nvingest for all files that have not failed
            # Check if the provided collection_name exists in vector-DB
            # Connect to Milvus to check for collection availability
            url = urlparse(vdb_endpoint)
            connection_alias = f"milvus_{url.hostname}_{url.port}"
            connections.connect(connection_alias, host=url.hostname, port=url.port)

            try:
                if not utility.has_collection(collection_name, using=connection_alias):
                    raise ValueError(f"Collection {collection_name} does not exist in {vdb_endpoint}. Ensure a collection is created using POST /collections endpoint first.")
            finally:
                connections.disconnect(connection_alias)

            start_time = time.time()
            results, failures = await self.__nvingest_upload_doc(
                filepaths=filepaths,
                collection_name=collection_name,
                vdb_endpoint=vdb_endpoint,
                split_options=split_options,
                custom_metadata=custom_metadata,
                generate_summary=generate_summary
            )

            logger.info("== Overall Ingestion completed successfully in %s seconds ==", time.time() - start_time)

            # Get failed documents
            failed_documents = await self.__get_failed_documents(failures, filepaths, collection_name, vdb_endpoint)
            failures_filepaths = [failed_document.get("document_name") for failed_document in failed_documents]

            filename_to_metadata_map = {custom_metadata_item.get("filename"): custom_metadata_item.get("metadata") for custom_metadata_item in custom_metadata}
            # Generate response dictionary
            uploaded_documents = [
                {
                    "document_id": str(uuid4()),  # Generate a document_id from filename
                    "document_name": os.path.basename(filepath),
                    "size_bytes": os.path.getsize(filepath),
                    "metadata": filename_to_metadata_map.get(os.path.basename(filepath), {})
                }
                for filepath in filepaths if os.path.basename(filepath) not in failures_filepaths
            ]

             # Get current timestamp in ISO format
            timestamp = datetime.utcnow().isoformat()
            # TODO: Store document_id, timestamp and document size as metadata

            response_data = {
                "message": "Document upload job successfully completed.",
                "total_documents": len(uploaded_documents) + len(failed_validation_documents) + len(failed_documents),
                "documents": uploaded_documents,
                "failed_documents": failed_documents + failed_validation_documents
            }

            # Optional: Clean up provided files after ingestion, needed for docker workflow
            if delete_files_after_ingestion:
                logger.info(f"Cleaning up files in {filepaths}")
                for file in filepaths:
                    try:
                        os.remove(file)
                        logger.debug(f"Deleted temporary file: {file}")
                    except FileNotFoundError:
                        logger.warning(f"File not found: {file}")
                    except Exception as e:
                        logger.error(f"Error deleting {file}: {e}")

            return response_data

        except Exception as e:
            logger.exception("Ingestion failed due to error: %s", e, exc_info=logger.getEffectiveLevel() <= logging.DEBUG)
            raise e

    async def __ingest_document_summary(
        self,
        results: List[List[Dict[str, Union[str, dict]]]],
        collection_name: str
    )-> None:
        """
        Generates and ingests document summaries for a list of files.

        Args:
            filepaths (List[str]): List of paths to documents to generate summaries for
        """

        logger.info(f"Document summary ingestion started")
        start_time = time.time()
        # Prepare summary documents
        documents = await self.__prepare_summary_documents(results, collection_name)
        # Generate summary for each document
        documents = await self.__generate_summary_for_documents(documents)
        # # Add document summary to minio
        await self.__put_document_summary_to_minio(documents)
        end_time = time.time()
        logger.info(f"Document summary ingestion completed! Time taken: {end_time - start_time} seconds")

    async def update_documents(
        self,
        filepaths: List[str],
        delete_files_after_ingestion: bool = False,
        blocking: bool = False,
        vdb_endpoint: str = None,
        collection_name: str = "multimodal_data",
        split_options: Dict[str, Any] = {"chunk_size": CONFIG.nv_ingest.chunk_size, "chunk_overlap": CONFIG.nv_ingest.chunk_overlap},
        custom_metadata: List[Dict[str, Any]] = [],
        generate_summary: bool = False
    ) -> Dict[str, Any]:

        """Upload a document to the vector store. If the document already exists, it will be replaced."""

        vdb_endpoint = vdb_endpoint or CONFIG.vector_store.url

        for file in filepaths:
            file_name = os.path.basename(file)

            # Delete the existing document

            if delete_files_after_ingestion:
                response = self.delete_documents([file_name],
                                                    collection_name=collection_name,
                                                    vdb_endpoint=vdb_endpoint,
                                                    include_upload_path=True)
            else:
                 response = self.delete_documents([file],
                                                    collection_name=collection_name,
                                                    vdb_endpoint=vdb_endpoint)

            if response["total_documents"] == 0:
                logger.info("Unable to remove %s from collection. Either the document does not exist or there is an error while removing. Proceeding with ingestion.", file_name)
            else:
                logger.info("Successfully removed %s from collection %s.", file_name, collection_name)

        response = await self.upload_documents(
            filepaths=filepaths,
            delete_files_after_ingestion=delete_files_after_ingestion,
            blocking=blocking,
            vdb_endpoint=vdb_endpoint,
            collection_name=collection_name,
            split_options=split_options,
            custom_metadata=custom_metadata,
            generate_summary=generate_summary
        )
        return response


    async def status(self, task_id: str)-> Dict[str, Any]:
        """Get the status of an ingestion task."""

        logger.info(f"Getting status of task {task_id}")
        try:
            if INGESTION_TASK_HANDLER.get_task_status(task_id) == "PENDING":
                logger.info(f"Task {task_id} is pending")
                return {
                    "state": "PENDING",
                    "result": {"message": "Task is pending"}
                }
            elif INGESTION_TASK_HANDLER.get_task_status(task_id) == "FINISHED":
                try:
                    result = INGESTION_TASK_HANDLER.get_task_result(task_id)
                    if isinstance(result, dict) and result.get("state") == "FAILED":
                        logger.error(f"Task {task_id} failed with error: {result.get('message')}")
                        result.pop("state")
                        return {
                            "state": "FAILED",
                            "result" : result
                        }
                    logger.info(f"Task {task_id} is finished")
                    return {
                        "state": "FINISHED",
                        "result": result
                    }
                except Exception as e:
                    logger.error(f"Task {task_id} failed with error: {e}")
                    return {
                        "state": "FAILED",
                        "result" : {"message": str(e)}
                    }
            else:
                logger.error(f"Unknown task state: {INGESTION_TASK_HANDLER.get_task_status(task_id)}")
                return {
                    "state": "UNKNOWN",
                    "result": {"message": "Unknown task state"}
                }
        except KeyError as e:
            logger.error(f"Task {task_id} not found with error: {e}")
            return {
                "state": "UNKNOWN",
                "result": {"message": "Unknown task state"}
            }


    def create_collections(
        self, collection_names: List[str], vdb_endpoint: str, embedding_dimension: int = 2048
    ) -> str:
        """
        Main function called by ingestor server to create new collections in vector-DB
        """
        try:
            logger.info(f"Creating collections {collection_names} at {vdb_endpoint}")
            return create_collections(collection_names, vdb_endpoint, embedding_dimension)
        except Exception as e:
            logger.error(f"Failed to create collections: {e}")
            return {
                "message": f"Failed to create collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0
            }

    def create_collection(
        self, collection_name: str, vdb_endpoint: str, embedding_dimension: int = 2048, metadata_schema: List[Dict[str, str]] = []
    ) -> str:
        """
        Main function called by ingestor server to create a new collection in vector-DB
        """
        try:
            # Create the metadata schema collection
            create_metadata_schema_collection(vdb_endpoint)
            # Check if the collection already exists
            existing_collections = get_collection(vdb_endpoint)
            if collection_name in [f["collection_name"] for f in existing_collections]:
                return {
                    "message": f"Collection {collection_name} already exists.",
                    "collection_name": collection_name
                }
            logger.info(f"Creating collection {collection_name} at {vdb_endpoint}")
            create_collection(collection_name, vdb_endpoint, embedding_dimension)

            # Add metadata schema
            if metadata_schema:
                add_metadata_schema(collection_name, vdb_endpoint, metadata_schema)

            return {
                "message": f"Collection {collection_name} created successfully.",
                "collection_name": collection_name
            }
        except Exception as e:
            logger.exception(f"Failed to create collection: {e}")
            raise Exception(f"Failed to create collection: {e}")

    def delete_collections(
        self, vdb_endpoint: str, collection_names: List[str],
    ) -> Dict[str, Any]:
        """
        Main function called by ingestor server to delete collections in vector-DB
        """
        logger.info(f"Deleting collections {collection_names} at {vdb_endpoint}")

        try:
            response = delete_collections(vdb_endpoint, collection_names)
            # Delete citation metadata from Minio
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(collection)
                delete_object_names = MINIO_OPERATOR.list_payloads(collection_prefix)
                MINIO_OPERATOR.delete_payloads(delete_object_names)

            # Delete document summary from Minio
            for collection in collection_names:
                collection_prefix = get_unique_thumbnail_id_collection_prefix(f"summary_{collection}")
                delete_object_names = MINIO_OPERATOR.list_payloads(collection_prefix)
                if len(delete_object_names):
                    MINIO_OPERATOR.delete_payloads(delete_object_names)
                    logger.info(f"Deleted all document summaries from Minio for collection: {collection}")

            return response
        except Exception as e:
            logger.error(f"Failed to delete collections in milvus: {e}")
            from traceback import print_exc
            logger.error(print_exc())
            return {
                "message": f"Failed to delete collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0
            }


    def get_collections(self, vdb_endpoint: str) -> Dict[str, Any]:
        """
        Main function called by ingestor server to get all collections in vector-DB.

        Args:
            vdb_endpoint (str): The endpoint of the vector database.

        Returns:
            Dict[str, Any]: A dictionary containing the collection list, message, and total count.
        """
        try:
            logger.info(f"Getting collection list from {vdb_endpoint}")

            # Fetch collections from vector store
            collection_info = get_collection(vdb_endpoint)

            return {
                "message": "Collections listed successfully.",
                "collections": collection_info,
                "total_collections": len(collection_info)
            }

        except Exception as e:
            logger.error(f"Failed to retrieve collections: {e}")
            return {
                "message": f"Failed to retrieve collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0
            }


    def get_documents(self, collection_name: str, vdb_endpoint: str) -> Dict[str, Any]:
        """
        Retrieves filenames stored in the vector store.
        It's called when the GET endpoint of `/documents` API is invoked.

        Returns:
            Dict[str, Any]: Response containing a list of documents with metadata.
        """
        try:
            vs = get_vectorstore(DOCUMENT_EMBEDDER, collection_name, vdb_endpoint)
            if not vs:
                raise ValueError(f"Failed to get vectorstore instance for collection: {collection_name}. Please check if the collection exists in {vdb_endpoint}.")

            documents_list = get_docs_vectorstore_langchain(vs, collection_name, vdb_endpoint)

            # Generate response format
            documents = [
                {
                    "document_id": "",  # TODO - Use actual document_id
                    "document_name": os.path.basename(doc_item.get("document_name")),  # Extract file name
                    "timestamp": "",  # TODO - Use actual timestamp
                    "size_bytes": 0,  # TODO - Use actual size
                    "metadata": doc_item.get("metadata", {})
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
            return {"documents": [], "total_documents": 0, "message": f"Document listing failed due to error {e}."}


    def delete_documents(self, document_names: List[str], collection_name: str, vdb_endpoint: str, include_upload_path: bool = False) -> Dict[str, Any]:
        """Delete documents from the vector index.
        It's called when the DELETE endpoint of `/documents` API is invoked.

        Args:
            document_names (List[str]): List of filenames to be deleted from vectorstore.
            collection_name (str): Name of the collection to delete documents from.
            vdb_endpoint (str): Vector database endpoint.

        Returns:
            Dict[str, Any]: Response containing a list of deleted documents with metadata.
        """

        try:
            logger.info(f"Deleting documents {document_names} from collection {collection_name} at {vdb_endpoint}")

            # Get vectorstore instance
            vs = get_vectorstore(DOCUMENT_EMBEDDER, collection_name, vdb_endpoint)
            if not vs:
                raise ValueError(f"Failed to get vectorstore instance for collection: {collection_name}. Please check if the collection exists in {vdb_endpoint}.")

            if not len(document_names):
                raise ValueError("No document names provided for deletion. Please provide document names to delete.")

            # TODO: Delete based on document_ids if provided
            if del_docs_vectorstore_langchain(vs, document_names, collection_name, include_upload_path):
                # Generate response dictionary
                documents = [
                    {
                        "document_id": "",  # TODO - Use actual document_id
                        "document_name": doc,
                        "size_bytes": 0 # TODO - Use actual size
                    }
                    for doc in document_names
                ]
                # Delete citation metadata from Minio
                for doc in document_names:
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(collection_name, doc)
                    delete_object_names = MINIO_OPERATOR.list_payloads(filename_prefix)
                    MINIO_OPERATOR.delete_payloads(delete_object_names)

                # Delete document summary from Minio
                for doc in document_names:
                    filename_prefix = get_unique_thumbnail_id_file_name_prefix(f"summary_{collection_name}", doc)
                    delete_object_names = MINIO_OPERATOR.list_payloads(filename_prefix)
                    if len(delete_object_names):
                        MINIO_OPERATOR.delete_payloads(delete_object_names)
                        logger.info(f"Deleted summary for doc: {doc} from Minio")
                return {f"message": "Files deleted successfully", "total_documents": len(documents), "documents": documents}

        except Exception as e:
            return {f"message": f"Failed to delete files due to error: {e}", "total_documents": 0, "documents": []}

        return {f"message": "Failed to delete files due to error. Check logs for details.", "total_documents": 0, "documents": []}


    def __put_content_to_minio(
        self, results: List[List[Dict[str, Union[str, dict]]]],
        collection_name: str,
    ) -> None:
        """
        Put nv-ingest image/table/chart content to minio
        """
        if not CONFIG.enable_citations:
            logger.info(f"Skipping minio insertion for collection: {collection_name}")
            return # Don't perform minio insertion if captioning is disabled

        payloads = []
        object_names = []

        for result in results:
            for result_element in result:
                if result_element.get("document_type") in ["image", "structured"]:
                    # Pull content from result_element
                    content = result_element.get("metadata").get("content")
                    file_name = os.path.basename(result_element.get("metadata").get("source_metadata").get("source_id"))
                    page_number = result_element.get("metadata").get("content_metadata").get("page_number")
                    location = result_element.get("metadata").get("content_metadata").get("location")

                    if location is not None:
                        # Get unique_thumbnail_id
                        unique_thumbnail_id = get_unique_thumbnail_id(
                            collection_name=collection_name,
                            file_name=file_name,
                            page_number=page_number,
                            location=location
                        )

                        payloads.append({"content": content})
                        object_names.append(unique_thumbnail_id)

        if os.getenv("ENABLE_MINIO_BULK_UPLOAD", "True") in ["True", "true"]:
            logger.info(f"Bulk uploading {len(payloads)} payloads to MinIO")
            MINIO_OPERATOR.put_payloads_bulk(
                payloads=payloads,
                object_names=object_names
            )
        else:
            logger.info(f"Sequentially uploading {len(payloads)} payloads to MinIO")
            for payload, object_name in zip(payloads, object_names):
                MINIO_OPERATOR.put_payload(
                    payload=payload,
                    object_name=object_name
                )


    async def __nvingest_upload_doc(
        self,
        filepaths: List[str],
        collection_name: str,
        vdb_endpoint: str,
        split_options: Dict[str, Any] = {"chunk_size": CONFIG.nv_ingest.chunk_size, "chunk_overlap": CONFIG.nv_ingest.chunk_overlap},
        custom_metadata: List[Dict[str, Any]] = [],
        generate_summary: bool = False
    ) -> Tuple[List[List[Dict[str, Union[str, dict]]]], List[Dict[str, Any]]]:
        """
        Wrapper function to ingest documents in chunks using NV-ingest

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_endpoint: str - URL of the vector database endpoint
            - split_options: SplitOptions - Options for splitting documents
            - custom_metadata: List[CustomMetadata] - Custom metadata to be added to documents
        """
        if not ENABLE_NV_INGEST_BATCH_MODE:
            # Single batch mode
            logger.info(
                "== Performing ingestion in SINGLE batch for collection_name: %s with %d files ==",
                collection_name, len(filepaths)
            )
            results, failures = await self.__nv_ingest_ingestion(
                filepaths=filepaths,
                collection_name=collection_name,
                vdb_endpoint=vdb_endpoint,
                split_options=split_options,
                custom_metadata=custom_metadata
            )
            return results, failures

        else:
            # BATCH_MODE
            logger.info(f"== Performing ingestion in BATCH_MODE for collection_name: {collection_name} "
                        f"with {len(filepaths)} files ==")

            # Process batches sequentially
            if not ENABLE_NV_INGEST_PARALLEL_BATCH_MODE:
                logger.info(f"Processing batches sequentially")
                all_results = []
                all_failures = []
                for i in range(0, len(filepaths), NV_INGEST_FILES_PER_BATCH):
                    sub_filepaths = filepaths[i:i+NV_INGEST_FILES_PER_BATCH]
                    batch_num = i//NV_INGEST_FILES_PER_BATCH + 1
                    logger.info(
                        f"=== Batch Processing Status - Collection: {collection_name} - "
                        f"Processing batch {batch_num} of {len(filepaths)//NV_INGEST_FILES_PER_BATCH + 1} - "
                        f"Documents in current batch: {len(sub_filepaths)} ==="
                    )
                    results, failures = await self.__nv_ingest_ingestion(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        vdb_endpoint=vdb_endpoint,
                        batch_number=batch_num,
                        split_options=split_options,
                        custom_metadata=custom_metadata
                    )
                    all_results.extend(results)
                    all_failures.extend(failures)
                return all_results, all_failures

            else:
                # Process batches in parallel with worker pool of 4
                logger.info(f"Processing batches in parallel with concurrency: {NV_INGEST_CONCURRENT_BATCHES}")
                all_results = []
                all_failures = []
                tasks = []
                semaphore = asyncio.Semaphore(NV_INGEST_CONCURRENT_BATCHES)  # Limit concurrent tasks

                async def process_batch(sub_filepaths, batch_num):
                    async with semaphore:
                        logger.info(
                            f"=== Processing Batch - Collection: {collection_name} - "
                            f"Batch {batch_num} of {len(filepaths)//NV_INGEST_FILES_PER_BATCH + 1} - "
                            f"Documents in batch: {len(sub_filepaths)} ==="
                        )
                        return await self.__nv_ingest_ingestion(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            vdb_endpoint=vdb_endpoint,
                            batch_number=batch_num,
                            split_options=split_options,
                            custom_metadata=custom_metadata,
                            generate_summary=generate_summary
                        )

                for i in range(0, len(filepaths), NV_INGEST_FILES_PER_BATCH):
                    sub_filepaths = filepaths[i:i+NV_INGEST_FILES_PER_BATCH]
                    batch_num = i//NV_INGEST_FILES_PER_BATCH + 1
                    task = process_batch(sub_filepaths, batch_num)
                    tasks.append(task)

                # Wait for all tasks to complete
                batch_results = await asyncio.gather(*tasks)

                # Combine results from all batches
                for results, failures in batch_results:
                    all_results.extend(results)
                    all_failures.extend(failures)

                return all_results, all_failures

    async def __nv_ingest_ingestion(
        self,
        filepaths: List[str],
        collection_name: str,
        vdb_endpoint: str,
        batch_number: int=0,
        split_options: Dict[str, Any] = {"chunk_size": CONFIG.nv_ingest.chunk_size, "chunk_overlap": CONFIG.nv_ingest.chunk_overlap},
        custom_metadata: List[Dict[str, Any]] = [],
        generate_summary: bool = False
    ) -> Tuple[List[List[Dict[str, Union[str, dict]]]], List[Dict[str, Any]]]:
        """
        This methods performs following steps:
        - Perform extraction and splitting using NV-ingest ingestor
        - Prepare langchain documents from the nv-ingest results
        - Embeds and add documents to Vectorstore collection

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - vdb_endpoint: str - URL of the vector database endpoint
            - batch_number: int - Batch number for the ingestion process
            - split_options: SplitOptions - Options for splitting documents
            - custom_metadata: List[CustomMetadata] - Custom metadata to be added to documents
        """
        # Create a temporary directory for custom metadata csv file
        if len(custom_metadata) > 0:
            csv_file_path = os.path.join(CONFIG.temp_dir,
                                        f"custom-metadata/{collection_name}_{batch_number}_{str(uuid4())[:8]}/custom_metadata.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        else:
            csv_file_path = None

        nv_ingest_ingestor = get_nv_ingest_ingestor(
            nv_ingest_client_instance=NV_INGEST_CLIENT_INSTANCE,
            filepaths=filepaths,
            csv_file_path=csv_file_path,
            collection_name=collection_name,
            vdb_endpoint=vdb_endpoint,
            split_options=split_options,
            custom_metadata=custom_metadata
        )
        start_time = time.time()
        logger.info(f"Performing ingestion for batch {batch_number} with parameters: {split_options}")
        results, failures = await asyncio.to_thread(
            lambda: nv_ingest_ingestor.ingest(return_failures=True, show_progress=logger.getEffectiveLevel() <= logging.DEBUG)
        )
        end_time = time.time()

        if generate_summary:
            logger.info(f"Document summary generation starting in background for batch {batch_number}..")
            asyncio.create_task(self.__ingest_document_summary(results, collection_name=collection_name))

        logger.info(f"== NV-ingest Job for collection_name: {collection_name} "
                    f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds ==")

        # Delete the csv file
        if csv_file_path is not None:
            os.remove(csv_file_path)
            logger.debug(f"Deleted temporary custom metadata csv file: {csv_file_path}")

        if not results:
            error_message = "NV-Ingest ingestion failed with no results. Please check the ingestor-server microservice logs for more details."
            logger.error(error_message)
            raise Exception(error_message)

        try:
            start_time = time.time()
            self.__put_content_to_minio(
                results=results,
                collection_name=collection_name
            )
            end_time = time.time()
            logger.info(f"== MinIO upload for collection_name: {collection_name} "
                        f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds ==")
        except Exception as e:
            logger.error("Failed to put content to minio: %s, citations would be disabled for collection: %s", str(e),
                         collection_name, exc_info=logger.getEffectiveLevel() <= logging.DEBUG)

        return results, failures


    async def __get_failed_documents(
        self,
        failures: List[Dict[str, Any]],
        filepaths: List[str],
        collection_name: str,
        vdb_endpoint: str
    ) -> List[Dict[str, Any]]:
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
                {
                    "document_name": failed_filename,
                    "error_message": error_message
                }
            )
            failed_documents_filenames.add(failed_filename)

        # Add non-supported files to failed documents
        for filepath in await self.__get_non_supported_files(filepaths):
            filename = os.path.basename(filepath)
            if filename not in failed_documents_filenames:
                failed_documents.append(
                    {
                        "document_name": filename,
                        "error_message": "Unsupported file type"
                    }
                )
                failed_documents_filenames.add(filename)
        
        # Add document to failed documents if it is not in the Milvus
        filenames_in_vdb = set()
        for document in self.get_documents(collection_name, vdb_endpoint).get("documents"):
            filenames_in_vdb.add(document.get("document_name"))
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            if filename not in filenames_in_vdb and filename not in failed_documents_filenames:
                failed_documents.append(
                    {
                        "document_name": filename,
                        "error_message": "Ingestion did not complete successfully"
                    }
                )
                failed_documents_filenames.add(filename)

        if failed_documents:
            logger.error("Ingestion failed for %d document(s)", len(failed_documents))
            logger.error("Failed documents details: %s", json.dumps(failed_documents, indent=4))

        return failed_documents


    async def __get_non_supported_files(self, filepaths: List[str]) -> List[str]:
        """Get filepaths of non-supported file extensions"""
        non_supported_files = []
        for filepath in filepaths:
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in ["." + supported_ext for supported_ext in EXTENSION_TO_DOCUMENT_TYPE.keys()]:
                non_supported_files.append(filepath)
        return non_supported_files

    async def __verify_metadata(
            self,
            custom_metadata: List[Dict[str, Any]],
            collection_name: str,
            vdb_endpoint: str,
            filepaths: List[str]
        ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify the metadata for schema and other validations

        Arguments:
            - custom_metadata: List[Dict[str, Any]] - List of custom metadata
            - collection_name: str - Name of the collection
            - vdb_endpoint: str - URL of the vector database endpoint
            - filepaths: List[str] - List of filepaths
        """
        # Get the metadata schema from the collection
        metadata_schema = get_metadata_schema(collection_name, vdb_endpoint)
        logger.info(f"Metadata schema for collection {collection_name}: {metadata_schema}")

        metadata_schema_key_type_map = {
            metadata_schema_item.get("name"): metadata_schema_item.get("type")
            for metadata_schema_item in metadata_schema
        }
        filenames = set([os.path.basename(filepath) for filepath in filepaths])

        # Verify the metadata schema
        validation_errors = []
        validation_status = True
        for custom_metadata_item in custom_metadata:
            # Check if the filename is provided in the ingestion request
            filename = custom_metadata_item.get("filename")
            if filename not in filenames:
                validation_errors.append({
                    "error": f"Filename: {filename} is not provided in the ingestion request",
                    "metadata": custom_metadata_item
                })
                validation_status = False

            # Check if the metadata item is a valid metadata schema
            metadata = custom_metadata_item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Validate keys that are present in metadata
            for key, value in metadata.items():
                expected_type = metadata_schema_key_type_map.get(key)
                if expected_type is None:
                    validation_errors.append({
                        "error": f"Custom metadata key {key} is not a valid metadata schema",
                        "metadata": custom_metadata_item
                    })
                    validation_status = False
                    continue

                if expected_type == "string" and not isinstance(value, str):
                    validation_errors.append({
                        "error": f"Custom metadata key {key} is not a valid metadata 'string' type",
                        "metadata": custom_metadata_item
                    })
                    validation_status = False

                elif expected_type == "datetime":
                    if value not in ("", None):
                        try:
                            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
                        except Exception as e:
                            validation_errors.append({
                                "error": f"Custom metadata key {key} is not a valid metadata 'datetime' type: {e}",
                                "metadata": custom_metadata_item
                            })
                            validation_status = False

        if not validation_status:
            logger.error(f"Custom metadata validation failed for collection {collection_name}: {validation_errors}")

        return validation_status, validation_errors


    async def __prepare_summary_documents(
        self,
        results: List[List[Dict[str, Union[str, dict]]]],
        collection_name: str
    ) -> List[Document]:
        """
        Prepare summary documents from the results to gather content for each file
        """
        summary_documents = []

        for result in results:
            documents = self.__parse_documents([result])
            if documents:
                full_content = ' '.join([doc.page_content for doc in documents])
                metadata = {
                    "filename": documents[0].metadata["source_name"],
                    "collection_name": collection_name
                }
                summary_documents.append(
                    Document(
                        page_content=full_content,
                        metadata=metadata
                    )
                )
        return summary_documents


    def __parse_documents(
        self,
        results: List[List[Dict[str, Union[str, dict]]]]
    ) -> List[Document]:
        """
        Extract document page content from the results obtained from nv-ingest

        Arguments:
            - results: List[List[Dict[str, Union[str, dict]]]] - Results obtained from nv-ingest

        Returns
            - List[Document] - List of documents with page content
        """
        documents = list()
        for result in results:
            for result_element in result:

                # Prepare metadata
                metadata = self.__prepare_metadata(result_element=result_element)

                # Extract documents page_content and prepare docs
                page_content = None
                # For textual data
                if result_element.get("document_type") == "text":
                    page_content = result_element.get("metadata")\
                                                 .get("content")

                # For both tables and charts
                elif result_element.get("document_type") == "structured":
                    structured_page_content = result_element.get("metadata")\
                                                 .get("table_metadata")\
                                                 .get("table_content")
                    subtype = result_element.get("metadata").get("content_metadata").get("subtype")
                    # Check for tables
                    if subtype == "table" and self._config.nv_ingest.extract_tables:
                        page_content = structured_page_content
                    # Check for charts
                    elif subtype == "chart" and self._config.nv_ingest.extract_charts:
                        page_content = structured_page_content

                # For image captions
                elif result_element.get("document_type") == "image" and self._config.nv_ingest.extract_images:
                    page_content = result_element.get("metadata")\
                                                 .get("image_metadata")\
                                                 .get("caption")

                # For audio transcripts
                elif result_element.get("document_type") == "audio":
                    page_content = result_element.get("metadata")\
                                                 .get("audio_metadata")\
                                                 .get("audio_transcript")

                # Add doc to list
                if page_content:
                    documents.append(
                        Document(
                            page_content=page_content,
                            metadata=metadata
                        )
                    )
        return documents


    def __prepare_metadata(
        self, result_element: Dict[str, Union[str, dict]]
    ) -> Dict[str, str]:
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
        source_id = result_element.get("metadata").get("source_metadata").get("source_id")

        # Get chunk_type
        if result_element.get("document_type") == "structured":
            chunk_type = result_element.get("metadata").get("content_metadata").get("subtype")
        else:
            chunk_type = result_element.get("document_type")

        # Get base64_str encoded content, empty str in case of text
        content = result_element.get("metadata").get("content") if chunk_type != "text" else ""

        metadata = {
            "source": source_id, # Add filepath (Key-name same for backward compatibility)
            "chunk_type": chunk_type, # ["text", "image", "table", "chart"]
            "source_name": os.path.basename(source_id), # Add filename
            # "content": content # content encoded in base64_str format [Must not exceed 64KB]
        }
        return metadata


    async def __generate_summary_for_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Generate summaries for documents using iterative chunk-wise approach
        """
        # Generate document summary
        summary_llm_name = CONFIG.summarizer.model_name
        summary_llm_endpoint = CONFIG.summarizer.server_url
        prompts = get_prompts()

        # TODO: Make these parameters configurable
        llm_params = {
            "model": summary_llm_name,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 2048
        }

        if summary_llm_endpoint:
            llm_params["llm_endpoint"] = summary_llm_endpoint

        summary_llm = get_llm(**llm_params)

        document_summary_prompt = prompts.get("document_summary_prompt")
        logger.debug(f"Document summary prompt: {document_summary_prompt}")

        # Initial summary prompt for first chunk
        initial_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", document_summary_prompt["system"]),
            ("human", document_summary_prompt["human"])
        ])

        # Iterative summary prompt for subsequent chunks
        iterative_summary_prompt_config = prompts.get("iterative_summary_prompt")
        iterative_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", iterative_summary_prompt_config["system"]),
            ("human", iterative_summary_prompt_config["human"])
        ])

        initial_chain = initial_summary_prompt | summary_llm | StrOutputParser()
        iterative_chain = iterative_summary_prompt | summary_llm | StrOutputParser()

        # Use configured chunk size
        max_chunk_chars = CONFIG.summarizer.max_chunk_length
        chunk_overlap = CONFIG.summarizer.chunk_overlap
        logger.info(f"Using chunk size: {max_chunk_chars} characters")

        if not len(documents):
            logger.error(f"No content returned from nv-ingest to summarize. Skipping summary generation.")
            return []

        for document in documents:
            document_text = document.page_content

            # Check if document fits in single request
            if len(document_text) <= max_chunk_chars:
                # Process as single chunk
                logger.info(f"Processing document {document.metadata['filename']} as single chunk")
                summary = await initial_chain.ainvoke({"document_text": document_text}, config={'run_name':'document-summary'})
            else:
                # Process in chunks iteratively using LangChain's text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_chars,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
                )
                text_chunks = text_splitter.split_text(document_text)
                logger.info(f"Processing document {document.metadata['filename']} in {len(text_chunks)} chunks")

                # Generate initial summary from first chunk
                summary = await initial_chain.ainvoke({"document_text": text_chunks[0]}, config={'run_name':'document-summary-initial'})

                # Iteratively update summary with remaining chunks
                for i, chunk in enumerate(text_chunks[1:], 1):
                    logger.info(f"Processing chunk {i+1}/{len(text_chunks)} for {document.metadata['filename']}")
                    summary = await iterative_chain.ainvoke({
                        "previous_summary": summary,
                        "new_chunk": chunk
                    }, config={'run_name': f'document-summary-chunk-{i+1}'})
                    logger.debug(f"Summary for chunk {i+1}/{len(text_chunks)} for {document.metadata['filename']}: {summary}")

            document.metadata["summary"] = summary
            logger.debug(f"Document summary for {document.metadata['filename']}: {summary}")

        logger.info(f"Document summary generation complete!")
        return documents


    async def __put_document_summary_to_minio(
        self,
        documents: List[Document]
    ) -> None:
        """
        Put document summary to minio
        """
        if not len(documents):
            logger.error(f"No documents to put to minio")
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
                location=location
            )

            MINIO_OPERATOR.put_payload(
                payload={
                    "summary": summary,
                    "file_name": file_name,
                    "collection_name": collection_name
                },
                object_name=unique_thumbnail_id
            )
            logger.debug(f"Document summary for {file_name} ingested to minio")

        logger.info(f"Document summary insertion completed to minio!")
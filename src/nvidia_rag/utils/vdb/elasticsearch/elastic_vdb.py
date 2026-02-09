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
This module contains the implementation of the ElasticVDB class,
which provides Elasticsearch vector database operations for RAG applications.
Extends both VDB class for nv-ingest operations and VDBRag for RAG-specific functionality.

NV-Ingest Client VDB Operations:
1. _check_index_exists: Check if the index exists in Elasticsearch
2. create_index: Create an index in Elasticsearch
3. write_to_index: Write records to the Elasticsearch index
4. retrieval: Retrieve documents from Elasticsearch based on queries
5. reindex: Reindex documents in Elasticsearch
6. run: Run the process of ingestion of records to the Elasticsearch index

Collection Management:
7. create_collection: Create a new collection with specified dimensions and type
8. check_collection_exists: Check if the specified collection exists
9. get_collection: Retrieve all collections with their metadata schemas
10. delete_collections: Delete multiple collections and their associated metadata

Document Management:
11. get_documents: Retrieve all unique documents from the specified collection
12. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
13. create_metadata_schema_collection: Initialize the metadata schema storage collection
14. add_metadata_schema: Store metadata schema configuration for the collection
15. get_metadata_schema: Retrieve the metadata schema for the specified collection

Document Info Management:
16. create_document_info_collection: Initialize the document info storage collection
17. add_document_info: Store document info for a collection or document
18. get_document_info: Retrieve document info for a specified collection/document

Retrieval Operations:
19. retrieval_langchain: Perform semantic search and return top-k relevant documents
20. _get_langchain_vectorstore: Get the vectorstore for a collection
21. _add_collection_name_to_retreived_docs: Add the collection name to the retrieved documents
"""

import logging
import os
import time
from concurrent.futures import Future
from typing import Any

import pandas as pd
import requests
from elastic_transport import ConnectionError as ESConnectionError
from elasticsearch import Elasticsearch, ConflictError
from elasticsearch.helpers.vectorstore import DenseVectorStrategy, VectorStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from langchain_elasticsearch import ElasticsearchStore
from opentelemetry import context as otel_context

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.common import (
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig, SearchType, RankerType
from nvidia_rag.utils.health_models import ServiceStatus
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
    SYSTEM_COLLECTIONS,
)
from nvidia_rag.utils.vdb.elasticsearch.es_queries import (
    create_document_info_collection_mapping,
    create_metadata_collection_mapping,
    get_collection_document_info_query,
    get_delete_docs_query,
    get_delete_document_info_query,
    get_delete_document_info_query_by_collection_name,
    get_delete_metadata_schema_query,
    get_document_info_query,
    get_metadata_schema_query,
    get_unique_sources_query,
    get_weighted_hybrid_custom_query,
)
from nvidia_rag.utils.vdb.vdb_ingest_base import VDBRagIngest

logger = logging.getLogger(__name__)


class ElasticVDB(VDBRagIngest):
    """
    ElasticVDB is a subclass of the VDB class in the nv_ingest_client.util.vdb module.
    It is used to store and retrieve documents in Elasticsearch.
    """

    def __init__(
        self,
        index_name: str,
        es_url: str,
        hybrid: bool = False,
        meta_dataframe: pd.DataFrame | None = None,
        meta_source_field: str | None = None,
        meta_fields: list[str] | None = None,
        embedding_model: str | None = None,
        csv_file_path: str | None = None,
        config: NvidiaRAGConfig | None = None,
        auth_token: str | None = None,
    ):
        self.config = config or NvidiaRAGConfig()
        self.index_name = index_name
        self.es_url = es_url
        # Prefer Bearer token when provided; then API key; otherwise fall back to basic auth.
        resolved_api_key: str | tuple[str, str] | None = None
        resolved_basic_auth: tuple[str, str] | None = None
        resolved_bearer_auth: str | None = None

        if auth_token:
            resolved_bearer_auth = auth_token
        elif self.config.vector_store.api_key:
            resolved_api_key = self.config.vector_store.api_key.get_secret_value()
        elif (
            self.config.vector_store.api_key_id
            and self.config.vector_store.api_key_secret
        ):
            resolved_api_key = (
                self.config.vector_store.api_key_id,
                self.config.vector_store.api_key_secret.get_secret_value(),
            )
        # Resolve basic auth from config
        elif self.config.vector_store.username and self.config.vector_store.password:
            resolved_basic_auth = (
                self.config.vector_store.username,
                self.config.vector_store.password.get_secret_value(),
            )

        # Keep on instance for reuse (e.g., langchain vectorstore)
        self._bearer_auth = resolved_bearer_auth
        self._api_key = resolved_api_key
        self._basic_auth = resolved_basic_auth
        self._username = self.config.vector_store.username
        self._password = (
            self.config.vector_store.password.get_secret_value()
            if self.config.vector_store.password is not None
            else ""
        )

        es_conn_params = {
            "hosts": [self.es_url],
        }

        if self._bearer_auth:
            es_conn_params["bearer_auth"] = self._bearer_auth
        elif self._api_key:
            es_conn_params["api_key"] = self._api_key
        elif self._basic_auth:
            es_conn_params["basic_auth"] = self._basic_auth

        self._es_connection = Elasticsearch(**es_conn_params).options(
            request_timeout=int(os.environ.get("ES_REQUEST_TIMEOUT", 600))
        )

        try:
            self._es_connection.info()
            logger.debug(f"Connected to Elasticsearch at {self.es_url}")
        except (ESConnectionError, ConnectionError, OSError) as e:
            logger.exception(
                "Failed to connect to Elasticsearch at %s: %s", self.es_url, e
            )
            raise APIError(
                f"Vector database (Elasticsearch) is unavailable at {self.es_url}. "
                "Please verify Elasticsearch is running and accessible.",
                ErrorCodeMapping.SERVICE_UNAVAILABLE,
            ) from e

        # Track if system collections have been initialized
        self._metadata_schema_collection_initialized = False
        self._document_info_collection_initialized = False
        self._embedding_model = embedding_model
        self.hybrid = hybrid

        # Metadata fields specific to NV-Ingest Client
        self.meta_dataframe = meta_dataframe
        self.meta_source_field = meta_source_field
        self.meta_fields = meta_fields
        self.csv_file_path = csv_file_path

        # Initialize the Elasticsearch vector store
        self.es_store = self._get_es_store(
            index_name=self.index_name,
            dimensions=self.config.embeddings.dimensions,
            hybrid=self.hybrid,
        )

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self.index_name

    @collection_name.setter
    def collection_name(self, collection_name: str) -> None:
        """Set the collection name."""
        self.index_name = collection_name

    def _get_es_store(
        self,
        index_name: str,
        dimensions: int,
        hybrid: bool = False,
    ):
        """Get the Elasticsearch vector store."""
        return VectorStore(
            client=self._es_connection,
            index=index_name,
            num_dimensions=dimensions,
            text_field="text",
            vector_field="vector",
            retrieval_strategy=DenseVectorStrategy(hybrid=hybrid),
        )

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods of the NV-Ingest Client VDB class
    def _check_index_exists(self, index_name: str) -> bool:
        """
        Check if the index exists in Elasticsearch.
        """
        return self._es_connection.indices.exists(index=index_name)

    def create_index(self):
        """
        Create an index in Elasticsearch.
        """
        logger.info(f"Creating Elasticsearch index if not exists: {self.index_name}")
        self.es_store._create_index_if_not_exists()

    def write_to_index(self, records: list, **kwargs) -> None:
        """
        Write records to the Elasticsearch index in batches.

        Requires nv_ingest_client to be installed. Install with: pip install nvidia-rag[ingest]
        """
        # Lazy import - only needed for ingestion operations
        try:
            from nv_ingest_client.util.milvus import cleanup_records, pandas_file_reader
        except ImportError as e:
            raise ImportError(
                "nv_ingest_client is required for write_to_index operation. "
                "Install with: pip install nvidia-rag[ingest]"
            ) from e

        # Load meta_dataframe lazily if not already loaded
        meta_dataframe = self.meta_dataframe
        if meta_dataframe is None and self.csv_file_path is not None:
            meta_dataframe = pandas_file_reader(self.csv_file_path)

        # Clean up and flatten records to pull appropriate fields from the records
        cleaned_records = cleanup_records(
            records=records,
            meta_dataframe=meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
        )

        # Prepare texts, embeddings, and metadatas from cleaned records
        texts, embeddings, metadatas = [], [], []
        for cleaned_record in cleaned_records:
            texts.append(cleaned_record.get("text"))
            embeddings.append(cleaned_record.get("vector"))
            metadatas.append(
                {
                    "source": cleaned_record.get("source"),
                    "content_metadata": cleaned_record.get("content_metadata"),
                }
            )

        total_records = len(texts)
        batch_size = 200
        uploaded_count = 0

        logger.info(
            f"Commencing Elasticsearch ingestion process for {total_records} records..."
        )

        # Process records in batches of batch_size
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]

            # Upload current batch to Elasticsearch
            self.es_store.add_texts(
                texts=batch_texts,
                vectors=batch_embeddings,
                metadatas=batch_metadatas,
            )

            uploaded_count += len(batch_texts)

            # Log progress every 5 batches (5000 records)
            if (
                uploaded_count % (5 * batch_size) == 0
                or uploaded_count == total_records
            ):
                logger.info(
                    f"Successfully ingested {uploaded_count} records into Elasticsearch index {self.index_name}"
                )

        logger.info(
            f"Elasticsearch ingestion completed. Total records processed: {uploaded_count}"
        )
        self._es_connection.indices.refresh(index=self.index_name)

    def retrieval(self, queries: list, **kwargs) -> list[dict[str, Any]]:
        """
        Retrieve documents from Elasticsearch based on queries.
        """
        # Placeholder: implement actual retrieval logic
        raise NotImplementedError("retrieval must be implemented for ElasticVDB")

    def reindex(self, records: list, **kwargs) -> None:
        """
        Reindex documents in Elasticsearch.
        """
        # Placeholder: implement actual reindex logic
        raise NotImplementedError("reindex must be implemented for ElasticVDB")

    def run(
        self,
        records: list,
    ) -> None:
        """
        Run the process of ingestion of records to the Elasticsearch index.
        """
        self.create_index()
        self.write_to_index(records)

    def run_async(
        self,
        records: list | Future,
    ) -> list:
        """Run ingestion from either a list of records or a Future producing records."""
        logger.info(f"creating index - {self.index_name}")
        self.create_index()

        if isinstance(records, Future):
            records = records.result()

        logger.info(f"writing to index, for collection - {self.index_name}")
        self.write_to_index(records)

        return records

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for ingestion
    async def check_health(self) -> dict[str, Any]:
        """Check Elasticsearch database health"""
        status = {
            "service": "Elasticsearch",
            "url": self.es_url,
            "status": ServiceStatus.UNKNOWN.value,
            "error": None,
        }

        if not self.es_url:
            status["status"] = ServiceStatus.SKIPPED.value
            status["error"] = "No URL provided"
            return status

        try:
            start_time = time.time()

            cluster_health = self._es_connection.cluster.health()
            indices = self._es_connection.cat.indices(format="json")

            status["status"] = ServiceStatus.HEALTHY.value
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["indices"] = len(indices)
            status["cluster_status"] = cluster_health.get(
                "status", ServiceStatus.UNKNOWN.value
            )

        except ImportError:
            status["status"] = ServiceStatus.ERROR.value
            status["error"] = (
                "Elasticsearch client not available (elasticsearch library not installed)"
            )
        except Exception as e:
            status["status"] = ServiceStatus.ERROR.value
            status["error"] = str(e)

        return status

    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """
        Create a new collection in the Elasticsearch index.
        """
        es_store = self._get_es_store(
            index_name=collection_name,
            dimensions=dimension,
            hybrid=self.hybrid,
        )
        es_store._create_index_if_not_exists()

        # Wait for the index to be ready
        self._es_connection.cluster.health(
            index=collection_name, wait_for_status="yellow", timeout="5s"
        )

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the Elasticsearch index.
        """
        return self._check_index_exists(collection_name)

    def get_collection(self):
        """Get the list of collections in the Elasticsearch index."""
        self.create_metadata_schema_collection()
        self.create_document_info_collection()
        indices = self._es_connection.cat.indices(format="json")
        collection_info = []
        for index in indices:
            index_name = index["index"]
            if index_name not in SYSTEM_COLLECTIONS:
                if not index_name.startswith("."):
                    metadata_schema = self.get_metadata_schema(index_name)

                    catalog_data = self.get_document_info(
                        info_type="catalog",
                        collection_name=index_name,
                        document_name="NA",
                    )

                    metrics_data = self.get_document_info(
                        info_type="collection",
                        collection_name=index_name,
                        document_name="NA",
                    )

                    collection_info.append(
                        {
                            "collection_name": index_name,
                            "num_entities": index["docs.count"],
                            "metadata_schema": metadata_schema,
                            "collection_info": {**metrics_data, **catalog_data},
                        }
                    )
        return collection_info

    def delete_collections(
        self,
        collection_names: list[str],
    ) -> dict[str, Any]:
        """
        Delete a collection from the Elasticsearch index.
        """
        deleted_collections = []
        failed_collections = []

        for collection_name in collection_names:
            try:
                # Check if collection exists before attempting deletion
                if self._check_index_exists(collection_name):
                    # Delete the collection
                    self._es_connection.indices.delete(
                        index=collection_name, ignore_unavailable=False
                    )
                    deleted_collections.append(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
                else:
                    # Collection doesn't exist - add to failed list
                    failed_collections.append(
                        {
                            "collection_name": collection_name,
                            "error_message": f"Collection {collection_name} not found.",
                        }
                    )
                    logger.warning(f"Collection {collection_name} not found.")
            except Exception as e:
                # Error during deletion - add to failed list
                failed_collections.append(
                    {"collection_name": collection_name, "error_message": str(e)}
                )
                logger.exception("Failed to delete collection %s", collection_name)

        logger.info(f"Collections deleted: {deleted_collections}")

        # Delete the metadata schema and document info for successfully deleted collections
        for collection_name in deleted_collections:
            try:
                _ = self._es_connection.delete_by_query(
                    index=DEFAULT_METADATA_SCHEMA_COLLECTION,
                    body=get_delete_metadata_schema_query(collection_name),
                )
            except Exception as e:
                logger.exception(
                    "Error deleting metadata schema for collection %s: %s",
                    collection_name,
                    e,
                )
            try:
                _ = self._es_connection.delete_by_query(
                    index=DEFAULT_DOCUMENT_INFO_COLLECTION,
                    body=get_delete_document_info_query_by_collection_name(
                        collection_name
                    ),
                )
            except Exception as e:
                logger.exception(
                    "Error deleting document info for collection %s: %s",
                    collection_name,
                    e,
                )
        return {
            "message": "Collection deletion process completed.",
            "successful": deleted_collections,
            "failed": failed_collections,
            "total_success": len(deleted_collections),
            "total_failed": len(failed_collections),
        }

    def get_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """
        Get the list of documents in a collection.
        """
        metadata_schema = self.get_metadata_schema(collection_name)
        response = self._es_connection.search(
            index=collection_name, body=get_unique_sources_query()
        )
        documents_list = []
        for hit in response["aggregations"]["unique_sources"]["buckets"]:
            source_name = hit["key"]["source_name"]
            metadata = (
                hit["top_hit"]["hits"]["hits"][0]["_source"]
                .get("metadata", {})
                .get("content_metadata", {})
            )
            metadata_dict = {}
            for metadata_item in metadata_schema:
                metadata_name = metadata_item.get("name")
                metadata_value = metadata.get(metadata_name, None)
                metadata_dict[metadata_name] = metadata_value
            documents_list.append(
                {
                    "document_name": os.path.basename(source_name),
                    "metadata": metadata_dict,
                    "document_info": self.get_document_info(
                        info_type="document",
                        collection_name=collection_name,
                        document_name=os.path.basename(source_name),
                    ),
                }
            )
        return documents_list

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """
        Delete documents from a collection by source values.
        """
        if result_dict is not None:
            result_dict["deleted"] = []
            result_dict["not_found"] = []

        source_to_basename = {
            source: os.path.basename(source) if "/" in source else source
            for source in source_values
        }
        existing_doc_basenames = set()
        if result_dict is not None:
            try:
                all_docs = self.get_documents(collection_name)
                existing_doc_basenames = {
                    os.path.basename(doc.get("document_name", "")) for doc in all_docs
                }
            except Exception as e:
                logger.warning(
                    f"Failed to check existing documents before deletion: {e}"
                )
                existing_doc_basenames = set(source_to_basename.values())

        for source_value in source_values:
            doc_basename = source_to_basename[source_value]
            try:
                response = self._es_connection.delete_by_query(
                    index=collection_name, body=get_delete_docs_query(source_value)
                )
                deleted_count = response.get("deleted", 0)

                if result_dict is not None:
                    if deleted_count > 0 or doc_basename in existing_doc_basenames:
                        result_dict["deleted"].append(doc_basename)
                    else:
                        result_dict["not_found"].append(doc_basename)
            except Exception as e:
                logger.warning(f"Failed to delete document {source_value}: {e}")
                if result_dict is not None:
                    if doc_basename in existing_doc_basenames:
                        result_dict["deleted"].append(doc_basename)
                    else:
                        result_dict["not_found"].append(doc_basename)

        self._es_connection.indices.refresh(index=collection_name)
        return True

    def create_metadata_schema_collection(
        self,
    ) -> None:
        """
        Create a metadata schema collection.
        """
        if self._metadata_schema_collection_initialized:
            return

        mapping = create_metadata_collection_mapping()
        if not self._check_index_exists(index_name=DEFAULT_METADATA_SCHEMA_COLLECTION):
            self._es_connection.indices.create(
                index=DEFAULT_METADATA_SCHEMA_COLLECTION, body=mapping
            )
            logging_message = (
                f"Collection {DEFAULT_METADATA_SCHEMA_COLLECTION} created "
                + f"at {self.es_url} with mapping {mapping}"
            )
            logger.info(logging_message)
        else:
            logging_message = f"Collection {DEFAULT_METADATA_SCHEMA_COLLECTION} already exists at {self.es_url}"
            logger.info(logging_message)

        self._metadata_schema_collection_initialized = True

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """
        Add metadata schema to a elasticsearch index.
        """
        # Delete the metadata schema from the index
        _ = self._es_connection.delete_by_query(
            index=DEFAULT_METADATA_SCHEMA_COLLECTION,
            body=get_delete_metadata_schema_query(collection_name),
        )
        # Add the metadata schema to the index
        data = {
            "collection_name": collection_name,
            "metadata_schema": metadata_schema,
        }
        self._es_connection.index(index=DEFAULT_METADATA_SCHEMA_COLLECTION, body=data)
        logger.info(
            f"Metadata schema added to the ES index {collection_name}. Metadata schema: {metadata_schema}"
        )

    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """
        Get the metadata schema for a collection in the Elasticsearch index.
        """
        query = get_metadata_schema_query(collection_name)
        response = self._es_connection.search(
            index=DEFAULT_METADATA_SCHEMA_COLLECTION, body=query
        )
        if len(response["hits"]["hits"]) > 0:
            return response["hits"]["hits"][0]["_source"]["metadata_schema"]
        else:
            logging_message = (
                f"No metadata schema found for the collection: {collection_name}."
                + " Possible reason: The collection is not created with metadata schema."
            )
            logger.info(logging_message)
            return []

    # ----------------------------------------------------------------------------------------------
    # Document Info Management
    def create_document_info_collection(self) -> None:
        """
        Create a document info Index in Elasticsearch.
        """
        if self._document_info_collection_initialized:
            return

        mapping = create_document_info_collection_mapping()
        if not self._check_index_exists(index_name=DEFAULT_DOCUMENT_INFO_COLLECTION):
            self._es_connection.indices.create(
                index=DEFAULT_DOCUMENT_INFO_COLLECTION, body=mapping
            )
            logging_message = (
                f"Collection {DEFAULT_DOCUMENT_INFO_COLLECTION} created "
                + f"at {self.es_url} with mapping {mapping}"
            )
            logger.info(logging_message)
        else:
            logging_message = f"Collection {DEFAULT_DOCUMENT_INFO_COLLECTION} already exists at {self.es_url}"
            logger.info(logging_message)

        self._document_info_collection_initialized = True

    def _get_aggregated_document_info(
        self, collection_name: str, info_value: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Internal function to get the aggregated document info for a collection.
        """
        try:
            # Get the aggregated document info for the collection
            response = self._es_connection.search(
                index=DEFAULT_DOCUMENT_INFO_COLLECTION,
                body=get_collection_document_info_query(
                    info_type="collection",
                    collection_name=collection_name,
                ),
            )
            existing_info_value = response["hits"]["hits"][0]["_source"]["info_value"]
        except IndexError:
            existing_info_value = {}
        except Exception as e:
            logger.error(
                f"Error getting aggregated document info for collection {collection_name}: {e}"
            )
            return info_value
        return perform_document_info_aggregation(existing_info_value, info_value)

    def add_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """
        Add document info to a collection.
        """
        # Since collection may have pre-ingested documents, we need to get the aggregated document info
        if info_type == "collection":
            info_value = self._get_aggregated_document_info(
                collection_name=collection_name,
                info_value=info_value,
            )

        # Delete the document info from the index
        try:
            _ = self._es_connection.delete_by_query(
                index=DEFAULT_DOCUMENT_INFO_COLLECTION,
                body=get_delete_document_info_query(
                    collection_name=collection_name,
                    document_name=document_name,
                    info_type=info_type,
                ),
            )
        except ConflictError as e:
            logger.info(f"Document info not found for collection: {collection_name}, document: {document_name}, info type: {info_type}")
        # Add the document info to the index
        data = {
            "collection_name": collection_name,
            "info_type": info_type,
            "document_name": document_name,
            "info_value": info_value,
        }
        self._es_connection.index(index=DEFAULT_DOCUMENT_INFO_COLLECTION, body=data)
        logger.info(
            f"Document info added to the ES index {DEFAULT_DOCUMENT_INFO_COLLECTION}. \
            Document info: {info_type}, {document_name}, {info_value}."
        )

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get document info from a Elasticsearch index."""
        query = get_document_info_query(collection_name, document_name, info_type)
        response = self._es_connection.search(
            index=DEFAULT_DOCUMENT_INFO_COLLECTION, body=query
        )
        if len(response["hits"]["hits"]) > 0:
            return response["hits"]["hits"][0]["_source"]["info_value"]
        else:
            logger.info(
                f"No document info found for collection: {collection_name}, document: {document_name}, info type: {info_type}"
            )
            return {}

    def get_catalog_metadata(self, collection_name: str) -> dict[str, Any]:
        """Get catalog metadata for a collection."""
        return self.get_document_info(
            info_type="catalog", collection_name=collection_name, document_name="NA"
        )

    def update_catalog_metadata(
        self,
        collection_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a collection."""
        existing = self.get_catalog_metadata(collection_name)
        merged = {**existing, **updates}
        merged["last_updated"] = get_current_timestamp()

        self.add_document_info(
            info_type="catalog",
            collection_name=collection_name,
            document_name="NA",
            info_value=merged,
        )

    def get_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get catalog metadata for a document."""
        doc_info = self.get_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
        )
        return {
            "description": doc_info.get("description", ""),
            "tags": doc_info.get("tags", []),
        }

    def update_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a document."""
        existing = self.get_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
        )

        for key in ["description", "tags"]:
            if key in updates:
                existing[key] = updates[key]

        self.add_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
            info_value=existing,
        )

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for retrieval
    def retrieval_langchain(
        self,
        query: str | None = None,
        collection_name: str = "",
        vectorstore: ElasticsearchStore | None = None,
        top_k: int = 10,
        filter_expr: list[dict[str, Any]] | None = None,
        otel_ctx: Any | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents from a collection using langchain."""
        if query_embedding is not None:
            raise NotImplementedError(
                "HyDE averaged-embedding retrieval (num_hypothetical_docs > 1) is not implemented "
                "for Elasticsearch. Use num_hypothetical_docs=1 or Milvus for multi-doc HyDE."
            )
        if not query:
            raise ValueError("query is required for Elasticsearch retrieval")
        logger.info(
            "Elasticsearch Retrieval: Retrieving documents from index: %s, search type: '%s'",
            collection_name,
            self.config.vector_store.search_type,
        )
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        # Attach OTel context only if provided
        token = otel_context.attach(otel_ctx) if otel_ctx is not None else None

        try:
            start_time = time.time()

            logger.info("  [Embedding] Generating query embedding for retrieval...")
            logger.info("  [Embedding] Query: '%s'", query[:100] if query else "")
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k, "fetch_k": top_k}
            )
            logger.info("  [Embedding] Query embedding generated successfully")
            if self.config.vector_store.search_type == SearchType.HYBRID:
                logger.info(
                    "Elasticsearch Retrieval: Using hybrid search with ranker type: '%s'",
                    self.config.vector_store.ranker_type,
                )
            if self.config.vector_store.search_type == SearchType.HYBRID and \
               self.config.vector_store.ranker_type == RankerType.WEIGHTED:
                retriever_lambda = RunnableLambda(
                    lambda x: retriever.invoke(
                        x,
                        filter=filter_expr,
                        custom_query=get_weighted_hybrid_custom_query(
                            embedding_model=self._embedding_model,
                            dense_weight=self.config.vector_store.dense_weight,
                            sparse_weight=self.config.vector_store.sparse_weight,
                            k=top_k,
                        ),
                    )
                )
            else:
                retriever_lambda = RunnableLambda(
                    lambda x: retriever.invoke(x, filter=filter_expr)
                )
            retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                {"context": lambda input: input["context"]}
            )
            logger.info("  [VDB Search] Performing vector similarity search in collection...")
            retriever_docs = retriever_chain.invoke(
                query, config={"run_name": "retriever"}
            )
            docs = retriever_docs.get("context", [])

            end_time = time.time()
            latency = end_time - start_time
            logger.info("  [VDB Search] Retrieved %d documents from collection '%s'", len(docs), collection_name)
            logger.info("  [VDB Search] Total VDB operation latency: %.4f seconds", latency)

            return self._add_collection_name_to_retreived_docs(docs, collection_name)
        except (requests.exceptions.ConnectionError, ConnectionError, OSError) as e:
            embedding_url = (
                self.embedding_model._client.base_url
                if hasattr(self.embedding_model, "_client")
                else "configured endpoint"
            )
            error_msg = f"Embedding NIM unavailable at {embedding_url}. Please verify the service is running and accessible."
            logger.error("Connection error in retrieval_langchain: %s", e)
            raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e
        finally:
            if token is not None:
                otel_context.detach(token)

    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> ElasticsearchStore:
        """
        Get the vectorstore for a collection.
        Uses the same authentication priority: bearer token -> API key -> basic auth
        """
        vectorstore_params: dict[str, Any] = {
            "index_name": collection_name,
            "es_url": self.es_url,
            "embedding": self._embedding_model,
            "strategy": DenseVectorStrategy(
                hybrid=self.config.vector_store.search_type == SearchType.HYBRID
            ),
        }

        if self._bearer_auth:
            vectorstore_params["es_params"] = {"bearer_auth": self._bearer_auth}
        elif self._api_key:
            vectorstore_params["es_api_key"] = self._api_key
        elif self._basic_auth:
            user, pwd = self._basic_auth
            vectorstore_params.update(
                {
                    "es_user": user,
                    "es_password": pwd,
                }
            )

        vectorstore = ElasticsearchStore(**vectorstore_params)

        return vectorstore

    @staticmethod
    def _add_collection_name_to_retreived_docs(
        docs: list[Document], collection_name: str
    ) -> list[Document]:
        """Add the collection name to the retreived documents.
        This is done to ensure the collection name is available in the
        metadata of the documents for preparing citations in case of multi-collection retrieval.
        """
        for doc in docs:
            doc.metadata["collection_name"] = collection_name
        return docs

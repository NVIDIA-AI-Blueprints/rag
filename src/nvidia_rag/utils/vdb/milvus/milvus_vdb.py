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
This module contains the implementation of the MilvusVDB class,
which provides Milvus vector database operations for RAG applications.
Extends both Milvus class for nv-ingest operations and VDBRag for RAG-specific functionality.

Connection Management:
1. milvus_connection_manager: Decorator to manage Milvus database connections

Collection Management:
2. create_collection: Create a new collection with specified dimensions and type
3. check_collection_exists: Check if the specified collection exists
4. get_collection: Retrieve all collections with their metadata schemas
5. delete_collections: Delete multiple collections and their associated metadata
6. _get_collection_info: Get the list of collections in the Milvus index without metadata schema
7. _delete_collections: Delete a collection from the Milvus index

Document Management:
8. get_documents: Retrieve all unique documents from the specified collection
9. delete_documents: Remove documents matching the specified source values
10. _get_documents_list: Get the list of documents in a collection
11. _extract_filename: Extract the filename from the metadata

Metadata Schema Management:
12. create_metadata_schema_collection: Initialize the metadata schema storage collection
13. add_metadata_schema: Store metadata schema configuration for the collection
14. get_metadata_schema: Retrieve the metadata schema for the specified collection
15. _get_milvus_entities: Get entities from Milvus collection with optional filtering
16. _delete_entities: Delete entities from collection by filter

Document Info Management:
17. create_document_info_collection: Initialize the document info storage collection
18. add_document_info: Store document info for a collection or document
19. get_document_info: Retrieve document info for a specified collection/document

Retrieval Operations:
20. retrieval_langchain: Perform semantic search and return top-k relevant documents
21. _get_langchain_vectorstore: Get the vectorstore for a collection
22. _add_collection_name_to_retreived_docs: Add the collection name to the retrieved documents
"""

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from langchain_milvus import BM25BuiltInFunction
from langchain_milvus import Milvus as LangchainMilvus
from opentelemetry import context as otel_context
from pymilvus import (
    Collection,
    DataType,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)
from pymilvus.orm.types import CONSISTENCY_STRONG

from nvidia_rag.utils.common import (
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig, SearchType
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.health_models import ServiceStatus
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
)
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

try:
    from nv_ingest_client.util.milvus import Milvus, create_nvingest_collection
except ImportError:
    logger.warning("Optional nv_ingest_client module not installed.")


class MilvusVDB(Milvus, VDBRag):
    def __init__(
        self,
        collection_name: str,
        milvus_uri: str,
        embedding_model: Any,
        config: NvidiaRAGConfig | None = None,
        # Minio configurations
        minio_endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str = "nv-ingest",
        # Hybrid search configurations
        sparse: bool = False,
        # Additional configurations
        enable_images: bool = False,
        recreate: bool = False,
        dense_dim: int = 2048,
        # GPU configurations
        gpu_index: bool = True,
        gpu_search: bool = True,
        # Authentication for Milvus
        username: str = "",
        password: str = "",
        # Custom metadata configurations (optional)
        meta_dataframe: str | None = None,
        meta_source_field: str | None = None,
        meta_fields: list[str] | None = None,
        auth_token: str | None = None,
    ):
        """
        Initialize MilvusVDB instance.
        Args:
            collection_name: Name of the Milvus collection
            milvus_uri: URI endpoint for Milvus server
            embedding_model: Embedding model instance for retrieval
            config: NvidiaRAGConfig instance (optional, creates default if None)
            minio_endpoint: MinIO endpoint for object storage
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: MinIO bucket name (default: "nv-ingest")
            sparse: Enable sparse/hybrid search
            enable_images: Enable image extraction and storage
            recreate: Whether to recreate the collection if it exists
            dense_dim: Dimension of dense embeddings
            gpu_index: Enable GPU acceleration for index building
            gpu_search: Enable GPU acceleration for search operations
            username: Milvus username for authentication
            password: Milvus password for authentication
            meta_dataframe: Path to CSV file containing custom metadata
            meta_source_field: Field name for source identification in metadata
            meta_fields: List of metadata field names to include
        """
        self.embedding_model = embedding_model
        self.config = config or NvidiaRAGConfig()
        self._auth_token = auth_token

        # Build kwargs for parent Milvus class
        parent_kwargs = {
            "collection_name": collection_name,
            "milvus_uri": milvus_uri,
            "minio_endpoint": minio_endpoint,
            "access_key": access_key,
            "secret_key": secret_key,
            "bucket_name": bucket_name,
            "sparse": sparse,
            "enable_images": enable_images,
            "recreate": recreate,
            "dense_dim": dense_dim,
            "gpu_index": gpu_index,
            "gpu_search": gpu_search,
        }

        # Add optional metadata configurations if provided
        if meta_dataframe is not None:
            parent_kwargs["meta_dataframe"] = meta_dataframe
        if meta_source_field is not None:
            parent_kwargs["meta_source_field"] = meta_source_field
        if meta_fields is not None:
            parent_kwargs["meta_fields"] = meta_fields

        super().__init__(**parent_kwargs)

        self.vdb_endpoint = milvus_uri
        self._collection_name = collection_name
        self.csv_file_path = meta_dataframe
    
        # Get the connection alias from the url
        self.url = urlparse(self.vdb_endpoint)
        self.connection_alias = (
            f"milvus_{self.url.hostname}_{self.url.port}_{str(uuid4())[:8]}"
        )
        
        # Get credentials from parameters or fall back to environment variables
        username = username or os.environ.get("VECTOR_STORE_USERNAME", "")
        password = password or os.environ.get("VECTOR_STORE_PASSWORD", "")

        # Establish a single persistent connection for the lifetime of this instance
        try:
            # Prefer explicit bearer token if provided; fall back to basic auth
            if self._auth_token:
                milvus_token = self._auth_token
            else:
                # Build basic auth token only if both username and password are available
                cfg_user = getattr(self.config.vector_store, "username", "") or ""
                cfg_pwd_val = getattr(self.config.vector_store, "password", None)
                if hasattr(cfg_pwd_val, "get_secret_value"):
                    cfg_pwd = cfg_pwd_val.get_secret_value()
                elif cfg_pwd_val is None:
                    cfg_pwd = ""
                else:
                    cfg_pwd = str(cfg_pwd_val)
                milvus_token = f"{cfg_user}:{cfg_pwd}" if (cfg_user and cfg_pwd) else ""
            connections.connect(
                self.connection_alias,
                uri=self.vdb_endpoint,
                token=milvus_token,
            )
            self._connected = True
            logger.debug(f"Connected to Milvus at {self.vdb_endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {self.vdb_endpoint}: {e}")
            raise

    def close(self):
        """Close the Milvus connection."""
        if self._connected:
            try:
                connections.disconnect(self.connection_alias)
                logger.debug(f"Disconnected from Milvus at {self.vdb_endpoint}")
                self._connected = False
            except Exception as e:
                logger.warning(f"Error disconnecting from Milvus: {e}")

    def __enter__(self):
        """Enter the runtime context (for use as context manager)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context."""
        self.close()

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, collection_name: str) -> None:
        """Set the collection name."""
        self._collection_name = collection_name

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for ingestion
    async def check_health(self) -> dict[str, Any]:
        """Check Milvus database health"""
        status = {
            "service": "Milvus",
            "url": self.vdb_endpoint,
            "status": ServiceStatus.UNKNOWN.value,
            "error": None,
        }

        if not self.vdb_endpoint:
            status["status"] = ServiceStatus.SKIPPED.value
            status["error"] = "No URL provided"
            return status

        try:
            start_time = time.time()

            # Test basic operation - list collections
            collections = utility.list_collections(using=self.connection_alias)

            status["status"] = ServiceStatus.HEALTHY.value
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["collections"] = len(collections)
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
        Create a new collection in the Milvus index.
        """
        create_nvingest_collection(
            collection_name=collection_name,
            milvus_uri=self.vdb_endpoint,
            sparse=(self.config.vector_store.search_type == SearchType.HYBRID),
            recreate=False,
            gpu_index=self.config.vector_store.enable_gpu_index,
            gpu_search=self.config.vector_store.enable_gpu_search,
            dense_dim=dimension,
            username=self.config.vector_store.username,
            password=self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else "",
        )

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the Milvus index.
        """
        if not utility.has_collection(collection_name, using=self.connection_alias):
            return False
        return True

    def _get_milvus_entities(self, collection_name: str, filter: str = ""):
        """
        Get the metadata schema for a collection in the Milvus index.
        """
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )
        entities = client.query(
            collection_name=collection_name, filter=filter, limit=1000
        )

        if len(entities) == 0:
            logger.warning(
                "No entities found in collection %s for filter %s",
                collection_name,
                filter,
            )

        return entities

    def _get_collection_info(self):
        """
        Get the list of collections in the Milvus index without metadata schema.
        """
        # Get list of collections
        collections = utility.list_collections(using=self.connection_alias)

        # Get document count for each collection
        collection_info = []
        for collection in collections:
            collection_obj = Collection(collection, using=self.connection_alias)
            num_entities = collection_obj.num_entities
            collection_info.append(
                {"collection_name": collection, "num_entities": num_entities}
            )

        return collection_info

    def get_collection(self):
        """Get the list of collections in the Milvus index."""
        self.create_metadata_schema_collection()
        self.create_document_info_collection()
        collection_info = self._get_collection_info()

        entities = self._get_milvus_entities(
            DEFAULT_METADATA_SCHEMA_COLLECTION, filter=""
        )
        collection_metadata_schema_map = {}
        for entity in entities:
            collection_metadata_schema_map[entity["collection_name"]] = entity[
                "metadata_schema"
            ]

        # Fetch both catalog and metrics data in ONE query to reduce Milvus load
        info_entities = self._get_milvus_entities(
            DEFAULT_DOCUMENT_INFO_COLLECTION,
            filter="info_type == 'catalog' or info_type == 'collection'",
        )
        collection_catalog_map = {}
        collection_metrics_map = {}
        for entity in info_entities:
            collection_name = entity["collection_name"]
            if entity["info_type"] == "catalog":
                collection_catalog_map[collection_name] = entity["info_value"]
            elif entity["info_type"] == "collection":
                collection_metrics_map[collection_name] = entity["info_value"]

        for collection_info_item in collection_info:
            collection_name = collection_info_item["collection_name"]

            catalog_data = collection_catalog_map.get(collection_name, {})
            metrics_data = collection_metrics_map.get(collection_name, {})

            collection_info_item.update(
                {
                    "metadata_schema": collection_metadata_schema_map.get(
                        collection_name, []
                    ),
                    "collection_info": {**catalog_data, **metrics_data},
                }
            )

        return collection_info

    def _delete_collections(self, collection_names: list[str]):
        """
        Delete a collection from the Milvus index.
        """
        deleted_collections = []
        failed_collections = []

        for collection in collection_names:
            try:
                if utility.has_collection(collection, using=self.connection_alias):
                    utility.drop_collection(collection, using=self.connection_alias)
                    deleted_collections.append(collection)
                    logger.info(f"Deleted collection: {collection}")
                else:
                    failed_collections.append(
                        {
                            "collection_name": collection,
                            "error_message": f"Collection {collection} not found.",
                        }
                    )
                    logger.warning(f"Collection {collection} not found.")
            except Exception as e:
                failed_collections.append(
                    {"collection_name": collection, "error_message": str(e)}
                )
                logger.error(f"Failed to delete collection {collection}: {str(e)}")

        logger.info(f"Collections deleted: {deleted_collections}")
        return deleted_collections, failed_collections

    def _delete_entities(self, collection_name: str, filter: str = ""):
        """
        Delete the metadata schema from the collection.
        """
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )
        if client.has_collection(collection_name):
            client.delete(collection_name=collection_name, filter=filter)
        else:
            logger.warning(
                f"Collection {collection_name} does not exist. Skipping deletion for filter {filter}"
            )

    def delete_collections(
        self,
        collection_names: list[str],
    ) -> dict[str, Any]:
        """
        Delete a collection from the Milvus index.
        """
        deleted_collections, failed_collections = self._delete_collections(
            collection_names
        )

        # Delete the metadata schema and document info from the collection
        for collection_name in deleted_collections:
            self._delete_entities(
                collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
                filter=f"collection_name == '{collection_name}'",
            )
            self._delete_entities(
                collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION,
                filter=f"collection_name == '{collection_name}'",
            )

        return {
            "message": "Collection deletion process completed.",
            "successful": deleted_collections,
            "failed": failed_collections,
            "total_success": len(deleted_collections),
            "total_failed": len(failed_collections),
        }

    @staticmethod
    def _extract_filename(metadata):
        """
        Extract the filename from the metadata.
        """
        if isinstance(metadata["source"], str):
            return os.path.basename(metadata["source"])
        elif (
            isinstance(metadata["source"], dict) and "source_name" in metadata["source"]
        ):
            return os.path.basename(metadata["source"]["source_name"])
        return None

    def _get_documents_list(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
        document_name_to_document_info_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Get the list of documents in a collection.
        """
        collection = Collection(collection_name, using=self.connection_alias)
        if not collection:
            logger.warning(f"Collection {collection_name} not found.")
            return []

        query_iterator = collection.query_iterator(
            batch_size=1000, output_fields=["source", "content_metadata"], expr=""
        )
        filepaths_added = set()
        documents_list = []

        try:
            milvus_data = query_iterator.next()
            while milvus_data:
                for item in milvus_data:
                    filename = self._extract_filename(item)
                    # Skip items with None filename or already processed files
                    if filename and filename not in filepaths_added:
                        metadata_dict = {}
                        for metadata_item in metadata_schema:
                            metadata_name = metadata_item.get("name")
                            metadata_value = item.get("content_metadata", {}).get(
                                metadata_name, None
                            )
                            metadata_dict[metadata_name] = metadata_value
                        documents_list.append(
                            {
                                "document_name": filename,
                                "metadata": metadata_dict,
                                "document_info": document_name_to_document_info_map.get(
                                    filename, {}
                                ),
                            }
                        )
                        filepaths_added.add(filename)

                # Get next batch, handle potential end of iteration
                try:
                    milvus_data = query_iterator.next()
                except (StopIteration, AttributeError):
                    # Handle cases where iterator is exhausted or next() method doesn't exist
                    break

                # Handle case where next() returns None to indicate end
                if milvus_data is None:
                    break

        except Exception as e:
            logger.error(f"Error during Milvus query iteration: {e}")
            return []

        return documents_list

    def get_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """
        Get the list of documents in a collection.
        """
        metadata_schema = self.get_metadata_schema(collection_name)
        # Get document info for each document in the collection
        entities = self._get_milvus_entities(
            DEFAULT_DOCUMENT_INFO_COLLECTION,
            filter=f"info_type == 'document' and collection_name == '{collection_name}'",
        )
        document_name_to_document_info_map = {}
        for entity in entities:
            document_name_to_document_info_map[entity["document_name"]] = entity[
                "info_value"
            ]
        documents_list = self._get_documents_list(
            collection_name=collection_name,
            metadata_schema=metadata_schema,
            document_name_to_document_info_map=document_name_to_document_info_map,
        )
        return documents_list

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
    ) -> bool:
        """
        Delete documents from a collection by source values.
        """
        collection = Collection(collection_name, using=self.connection_alias)
        for source_value in source_values:
            # Delete Milvus Entities
            logger.info(
                f"Deleting document {source_value} from collection "
                f"{collection_name} at {self.vdb_endpoint}"
            )
            try:
                resp = collection.delete(f"source['source_name'] == '{source_value}'")
                self._delete_entities(
                    collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION,
                    filter=f"info_type == 'document' and collection_name == '{collection_name}' and document_name == '{os.path.basename(source_value)}'",
                )
            except MilvusException:
                logger.debug(
                    f"Failed to delete document {source_value}, source name might be "
                    "available in the source field"
                )
                resp = collection.delete(f"source == '{source_value}'")
            deleted = True
            if resp.delete_count == 0:
                logger.info("File does not exist in the vectorstore")
                return False
        if deleted:
            # Force flush the vectorstore after deleting documents to ensure
            # that the changes are reflected in the vectorstore
            collection.flush()
        return True

    def create_metadata_schema_collection(
        self,
    ) -> None:
        """
        Create a metadata schema collection.
        """
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="collection_name", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
        schema.add_field(field_name="metadata_schema", datatype=DataType.JSON)

        # Check if the metadata schema collection exists
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )
        if not client.has_collection(DEFAULT_METADATA_SCHEMA_COLLECTION):
            # Create the metadata schema collection
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="dense_index",
                index_type="FLAT",
                metric_type="L2",
            )
            client.create_collection(
                collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
                schema=schema,
                index_params=index_params,
                consistency_level=CONSISTENCY_STRONG,
            )
            logger.info(f"Metadata schema collection created at {self.vdb_endpoint}")

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """
        Add metadata schema to a collection.
        """
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )

        # Delete the metadata schema from the collection
        client.delete(
            collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
            filter=f"collection_name == '{collection_name}'",
        )

        # Add the metadata schema to the collection
        data = {
            "collection_name": collection_name,
            "vector": [0.0] * 2,
            "metadata_schema": metadata_schema,
        }
        client.insert(collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION, data=data)
        logger.info(
            f"Metadata schema added to the collection {collection_name}. "
            f"Metadata schema: {metadata_schema}"
        )

    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """
        Get the metadata schema for a collection in the Milvus index.
        """
        filter = f"collection_name == '{collection_name}'"
        entities = self._get_milvus_entities(DEFAULT_METADATA_SCHEMA_COLLECTION, filter)
        if len(entities) > 0:
            return entities[0]["metadata_schema"]
        else:
            logging_message = (
                f"No metadata schema found for: {collection_name}."
                + "Possible reason: The collection is not created with metadata schema."
            )
            logger.info(logging_message)
            return []

    # ----------------------------------------------------------------------------------------------
    # Document Info Management
    def create_document_info_collection(self) -> None:
        """
        Create a document info collection.
        """
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="info_type", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="collection_name", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="document_name", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(field_name="info_value", datatype=DataType.JSON)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)

        # Check if the document info collection exists
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )
        if not client.has_collection(DEFAULT_DOCUMENT_INFO_COLLECTION):
            # Create the document info collection
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="dense_index",
                index_type="FLAT",
                metric_type="L2",
            )
            client.create_collection(
                collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION,
                schema=schema,
                index_params=index_params,
                consistency_level=CONSISTENCY_STRONG,
            )
            logger.info(f"Document info collection created at {self.vdb_endpoint}")

    def _get_aggregated_document_info(
        self, collection_name: str, info_value: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Internal function to get the aggregated document info for a collection.
        """
        # Get the aggregated document info for the collection
        entities = self._get_milvus_entities(
            DEFAULT_DOCUMENT_INFO_COLLECTION,
            filter=f"info_type == 'collection' and collection_name == '{collection_name}'",
        )
        try:
            existing_info_value = entities[0]["info_value"]
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
        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        client = MilvusClient(
            self.vdb_endpoint,
            token=self._auth_token
            if self._auth_token
            else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
        )

        # Since collection may have pre-ingested documents, we need to get the aggregated document info
        if info_type == "collection":
            info_value = self._get_aggregated_document_info(
                collection_name=collection_name,
                info_value=info_value,
            )

        # Delete the existing document info from the collection
        client.delete(
            collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION,
            filter=f"info_type == '{info_type}' and collection_name == '{collection_name}' and document_name == '{document_name}'",
        )

        # Add the document info to the collection
        data = {
            "info_type": info_type,
            "collection_name": collection_name,
            "document_name": document_name,
            "info_value": info_value,
            "vector": [0.0] * 2,
        }
        client.insert(collection_name=DEFAULT_DOCUMENT_INFO_COLLECTION, data=data)
        logger.info(
            f"Document info added to the collection {collection_name}. "
            f"Document info: {info_type}, {document_name}, {info_value}"
        )

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get document info from a collection."""
        filter = f"info_type == '{info_type}' and collection_name == '{collection_name}' and document_name == '{document_name}'"
        entities = self._get_milvus_entities(DEFAULT_DOCUMENT_INFO_COLLECTION, filter)
        if len(entities) > 0:
            return entities[0]["info_value"]
        else:
            logger.info(
                f"No document info found for: {info_type}, {collection_name}, {document_name}"
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
        query: str,
        collection_name: str,
        vectorstore: LangchainMilvus | None = None,
        top_k: int = 10,
        filter_expr: str = "",
        otel_ctx: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents from a collection using langchain."""
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        start_time = time.time()

        # Attach OTel context only if provided
        token = otel_context.attach(otel_ctx) if otel_ctx is not None else None

        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

            retriever_lambda = RunnableLambda(
                lambda x: retriever.invoke(
                    x,
                    expr=filter_expr,
                )
            )
            retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                {"context": lambda input: input["context"]}
            )
            retriever_docs = retriever_chain.invoke(
                query, config={"run_name": "retriever"}
            )
            docs = retriever_docs.get("context", [])
            collection_name = retriever.vectorstore.collection_name

            end_time = time.time()
            latency = end_time - start_time
            logger.info(f" Milvus Retrieval latency: {latency:.4f} seconds")

            return self._add_collection_name_to_retreived_docs(docs, collection_name)
        finally:
            # Detach OTel context only if it was attached
            if token is not None:
                otel_context.detach(token)

    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> LangchainMilvus:
        """
        Get the vectorstore for a collection.
        """
        start_time = time.time()
        logger.debug("Trying to connect to milvus collection: %s", collection_name)
        if not collection_name:
            collection_name = os.getenv("COLLECTION_NAME", "vector_db")

        search_params = {}
        if not self.config.vector_store.enable_gpu_search:
            # ef is required for CPU search
            search_params.update({"ef": self.config.vector_store.ef})

        password = self.config.vector_store.password.get_secret_value() if self.config.vector_store.password is not None else ""
        if self.config.vector_store.search_type == SearchType.HYBRID:
            logger.info("Creating Langchain Milvus object for Hybrid search")
            vectorstore = LangchainMilvus(
                self.embedding_model,
                connection_args={
                    "uri": self.vdb_endpoint,
                    "token": self._auth_token
                    if self._auth_token
                    else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
                },
                builtin_function=BM25BuiltInFunction(
                    output_field_names="sparse", enable_match=True
                ),
                collection_name=collection_name,
                vector_field=[
                    "vector",
                    "sparse",
                ],  # Dense and Sparse fields set by NV-Ingest
            )
        elif self.config.vector_store.search_type == SearchType.DENSE:
            search_params.update({"nprobe": self.config.vector_store.nprobe})
            logger.debug(
                "Index type for milvus: %s", self.config.vector_store.index_type
            )
            vectorstore = LangchainMilvus(
                self.embedding_model,
                connection_args={
                    "uri": self.vdb_endpoint,
                    "token": self._auth_token
                    if self._auth_token
                    else f"{self.config.vector_store.username}:{self.config.vector_store.password}",
                },
                collection_name=collection_name,
                index_params={
                    "index_type": self.config.vector_store.index_type,
                    "metric_type": "L2",
                    "nlist": self.config.vector_store.nlist,
                },
                search_params=search_params,
                auto_id=True,
            )
        else:
            logger.error(
                "Invalid search_type: %s. Please select from ['hybrid', 'dense']",
                self.config.vector_store.search_type,
            )
            raise ValueError(
                f"{self.config.vector_store.search_type} search type is not supported. Please select from ['hybrid', 'dense']"
            )
        end_time = time.time()
        logger.info(
            f" Time to get langchain milvus vectorstore: {end_time - start_time:.4f} seconds"
        )
        return vectorstore

    def retrieval_image_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: LangchainMilvus | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Retrieve documents from a collection using langchain for image query.

        Returns LangChain Document objects with metadata and collection name.

        Note: Uses the embedding_model that was provided during initialization.
        """
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        # Use the embedding model provided during initialization
        client = self.embedding_model

        image_input = query

        try:
            embedding = client.embed_documents([image_input])
            results = vectorstore.similarity_search_with_score_by_vector(
                embedding=embedding[0],
                k=top_k,
            )
        except Exception as e:
            logger.error(
                "Error generating embeddings or performing similarity search: %s", e
            )
            return []

        try:
            # ToDo: If no page number is provided, use content of same file (txt file)
            metadata = results[0][0].metadata
            source_name = metadata["source"]["source_name"]
            page_number = metadata["content_metadata"]["page_number"]
        except (KeyError, IndexError) as e:
            logger.error("Error accessing metadata from search results: %s", e)
            return []
        filter_expr_partial = (
            f'content_metadata["page_number"] == {page_number} and '
            f'source["source_name"] like "%{source_name}%"'
        )
        try:
            client = MilvusClient(self.vdb_endpoint)
            entities = client.query(
                collection_name=collection_name, filter=filter_expr_partial, limit=1000
            )
        except Exception as e:
            logger.error("Error querying Milvus collection: %s", e)
            return []
        client = MilvusClient(self.vdb_endpoint)
        entities = client.query(
            collection_name=collection_name, filter=filter_expr_partial, limit=1000
        )

        # Convert Milvus entities to LangChain Document objects
        docs: list[Document] = []
        for item in entities:
            page_content = item.get("text") or item.get("chunk") or ""
            metadata = {
                "source": item.get("source"),
                "content_metadata": item.get("content_metadata", {}),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))

        return self._add_collection_name_to_retreived_docs(docs, collection_name)

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

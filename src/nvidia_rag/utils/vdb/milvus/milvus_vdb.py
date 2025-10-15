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

Retrieval Operations:
17. retrieval_langchain: Perform semantic search and return top-k relevant documents
18. _get_langchain_vectorstore: Get the vectorstore for a collection
19. _add_collection_name_to_retreived_docs: Add the collection name to the retrieved documents
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

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.vdb import DEFAULT_METADATA_SCHEMA_COLLECTION
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

try:
    from nv_ingest_client.util.milvus import Milvus, create_nvingest_collection
except ImportError:
    logger.warning("Optional nv_ingest_client module not installed.")


CONFIG = get_config()


class MilvusVDB(Milvus, VDBRag):
    def __init__(self, **kwargs):
        self.embedding_model = kwargs.pop(
            "embedding_model"
        )  # Needed in case of retrieval
        super().__init__(**kwargs)
        self.vdb_endpoint = kwargs.get("milvus_uri")
        self._collection_name = kwargs.get("collection_name")

        # Get the connection alias from the url
        self.url = urlparse(self.vdb_endpoint)
        self.connection_alias = (
            f"milvus_{self.url.hostname}_{self.url.port}_{str(uuid4())[:8]}"
        )
        self.csv_file_path = kwargs.get("meta_dataframe")

        # Establish a single persistent connection for the lifetime of this instance
        try:
            connections.connect(self.connection_alias, uri=self.vdb_endpoint)
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
            "status": "unknown",
            "error": None,
        }

        if not self.vdb_endpoint:
            status["status"] = "skipped"
            status["error"] = "No URL provided"
            return status

        try:
            start_time = time.time()

            # Test basic operation - list collections
            collections = utility.list_collections(using=self.connection_alias)

            status["status"] = "healthy"
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["collections"] = len(collections)
        except Exception as e:
            status["status"] = "error"
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
            sparse=(CONFIG.vector_store.search_type == "hybrid"),
            recreate=False,
            gpu_index=CONFIG.vector_store.enable_gpu_index,
            gpu_search=CONFIG.vector_store.enable_gpu_search,
            dense_dim=dimension,
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
        client = MilvusClient(self.vdb_endpoint)
        entities = client.query(
            collection_name=collection_name, filter=filter, limit=1000
        )

        if len(entities) == 0:
            logger.warning(f"No metadata schema found for filter {filter}")

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
        """
        Get the list of collections in the Milvus index.
        """
        self.create_metadata_schema_collection()
        collection_info = self._get_collection_info()

        # Get metadata schema for each collection
        entities = self._get_milvus_entities(
            DEFAULT_METADATA_SCHEMA_COLLECTION, filter=""
        )
        collection_metadata_schema_map = {}
        for entity in entities:
            collection_metadata_schema_map[entity["collection_name"]] = entity[
                "metadata_schema"
            ]
        for collection_info_item in collection_info:
            collection_name = collection_info_item["collection_name"]
            collection_info_item.update(
                {
                    "metadata_schema": collection_metadata_schema_map.get(
                        collection_name, []
                    )
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
        client = MilvusClient(self.vdb_endpoint)
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

        # Delete the metadata schema from the collection# Delete the metadata schema from the collection
        for collection_name in deleted_collections:
            self._delete_entities(
                collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
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
        self, collection_name: str, metadata_schema: list[dict[str, Any]]
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
        documents_list = self._get_documents_list(
            collection_name=collection_name, metadata_schema=metadata_schema
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
                f"Deleting document {source_value} from collection {
                    collection_name
                } at {self.vdb_endpoint}"
            )
            try:
                resp = collection.delete(f"source['source_name'] == '{source_value}'")
            except MilvusException:
                logger.debug(
                    f"Failed to delete document {
                        source_value
                    }, source name might be available in the source field"
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
        """Create metadata collection for the collection."""
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
        client = MilvusClient(self.vdb_endpoint)
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
        client = MilvusClient(self.vdb_endpoint)

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
            f"Metadata schema added to the collection {
                collection_name
            }. Metadata schema: {metadata_schema}"
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
    # Implementations of the abstract methods specific to VDBRag class for retrieval
    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: LangchainMilvus = None,
        top_k: int = 10,
        filter_expr: str = "",
        otel_ctx: otel_context = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents from a collection using langchain."""
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        start_time = time.time()

        token = otel_context.attach(otel_ctx)

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
        retriever_docs = retriever_chain.invoke(query, config={"run_name": "retriever"})
        docs = retriever_docs.get("context", [])
        collection_name = retriever.vectorstore.collection_name

        end_time = time.time()
        latency = end_time - start_time
        logger.info(f" Milvus Retrieval latency: {latency:.4f} seconds")

        otel_context.detach(token)
        return self._add_collection_name_to_retreived_docs(docs, collection_name)

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
        if not CONFIG.vector_store.enable_gpu_search:
            # ef is required for CPU search
            search_params.update({"ef": CONFIG.vector_store.ef})

        if CONFIG.vector_store.search_type == "hybrid":
            logger.info("Creating Langchain Milvus object for Hybrid search")
            vectorstore = LangchainMilvus(
                self.embedding_model,
                connection_args={"uri": self.vdb_endpoint},
                builtin_function=BM25BuiltInFunction(
                    output_field_names="sparse", enable_match=True
                ),
                collection_name=collection_name,
                vector_field=[
                    "vector",
                    "sparse",
                ],  # Dense and Sparse fields set by NV-Ingest
            )
        elif CONFIG.vector_store.search_type == "dense":
            search_params.update({"nprobe": CONFIG.vector_store.nprobe})
            logger.debug("Index type for milvus: %s", CONFIG.vector_store.index_type)
            vectorstore = LangchainMilvus(
                self.embedding_model,
                connection_args={"uri": self.vdb_endpoint},
                collection_name=collection_name,
                index_params={
                    "index_type": CONFIG.vector_store.index_type,
                    "metric_type": "L2",
                    "nlist": CONFIG.vector_store.nlist,
                },
                search_params=search_params,
                auto_id=True,
            )
        else:
            logger.error(
                "Invalid search_type: %s. Please select from ['hybrid', 'dense']",
                CONFIG.vector_store.search_type,
            )
            raise ValueError(
                f"{CONFIG.vector_store.search_type} search type is not supported. Please select from ['hybrid', 'dense']"
            )
        end_time = time.time()
        logger.info(
            f" Time to get langchain milvus vectorstore: {end_time - start_time:.4f} seconds"
        )
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

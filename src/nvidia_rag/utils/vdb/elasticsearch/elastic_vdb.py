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

Retrieval Operations:
16. retrieval_langchain: Perform semantic search and return top-k relevant documents
17. _get_langchain_vectorstore: Get the vectorstore for a collection
18. _add_collection_name_to_retreived_docs: Add the collection name to the retrieved documents
"""

import logging
import os
import time
from typing import Any

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers.vectorstore import DenseVectorStrategy, VectorStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from langchain_elasticsearch import ElasticsearchStore
from nv_ingest_client.util.milvus import cleanup_records
from opentelemetry import context as otel_context

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.vdb import DEFAULT_METADATA_SCHEMA_COLLECTION
from nvidia_rag.utils.vdb.elasticsearch.es_queries import (
    create_metadata_collection_mapping,
    get_delete_docs_query,
    get_delete_metadata_schema_query,
    get_metadata_schema_query,
    get_unique_sources_query,
)
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)
CONFIG = get_config()


class ElasticVDB(VDBRag):
    """
    ElasticVDB is a subclass of the VDB class in the nv_ingest_client.util.vdb module.
    It is used to store and retrieve documents in Elasticsearch.
    """

    def __init__(
        self,
        index_name: str,
        es_url: str,
        hybrid: bool = False,
        meta_dataframe: pd.DataFrame = None,
        meta_source_field: str = None,
        meta_fields: list[str] = None,
        embedding_model: str = None,
        csv_file_path: str = None,
    ):
        self.index_name = index_name
        self.es_url = es_url
        self._es_connection = Elasticsearch(hosts=[self.es_url]).options(
            request_timeout=int(os.environ.get("ES_REQUEST_TIMEOUT", 600))
        )
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
            dimensions=CONFIG.embeddings.dimensions,
            hybrid=self.hybrid,
        )

        kwargs = locals().copy()
        kwargs.pop("self", None)
        super().__init__(**kwargs)

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
        """
        # Clean up and flatten records to pull appropriate fields from the records
        cleaned_records = cleanup_records(
            records=records,
            meta_dataframe=self.meta_dataframe,
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

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for ingestion
    async def check_health(self) -> dict[str, Any]:
        """Check Elasticsearch database health"""
        status = {
            "service": "Elasticsearch",
            "url": self.es_url,
            "status": "unknown",
            "error": None,
        }

        if not self.es_url:
            status["status"] = "skipped"
            status["error"] = "No URL provided"
            return status

        try:
            start_time = time.time()

            cluster_health = self._es_connection.cluster.health()
            indices = self._es_connection.cat.indices(format="json")

            status["status"] = "healthy"
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["indices"] = len(indices)
            status["cluster_status"] = cluster_health.get("status", "unknown")

        except ImportError:
            status["status"] = "error"
            status["error"] = (
                "Elasticsearch client not available (elasticsearch library not installed)"
            )
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
        """
        Get the list of collections in the Elasticsearch index.
        """
        self.create_metadata_schema_collection()
        indices = self._es_connection.cat.indices(format="json")
        collection_info = []
        for index in indices:
            index_name = index["index"]
            if not index_name.startswith("."):  # Ignore hidden indices
                metadata_schema = self.get_metadata_schema(index_name)
                collection_info.append(
                    {
                        "collection_name": index_name,
                        "num_entities": index["docs.count"],
                        "metadata_schema": metadata_schema,
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
        _ = self._es_connection.indices.delete(
            index=",".join(collection_names), ignore_unavailable=True
        )
        deleted_collections, failed_collections = collection_names, []
        logger.info(f"Collections deleted: {deleted_collections}")

        # Delete the metadata schema from the collection
        for collection_name in deleted_collections:
            _ = self._es_connection.delete_by_query(
                index=DEFAULT_METADATA_SCHEMA_COLLECTION,
                body=get_delete_metadata_schema_query(collection_name),
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
                }
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
        for source_value in source_values:
            self._es_connection.delete_by_query(
                index=collection_name, body=get_delete_docs_query(source_value)
            )
        self._es_connection.indices.refresh(index=collection_name)
        return True

    def create_metadata_schema_collection(
        self,
    ) -> None:
        """
        Create a metadata schema collection.
        """
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
    # Implementations of the abstract methods specific to VDBRag class for retrieval
    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: ElasticsearchStore = None,
        top_k: int = 10,
        filter_expr: list[dict[str, Any]] = None,
        otel_ctx: otel_context = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents from a collection using langchain."""
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        token = otel_context.attach(otel_ctx)
        start_time = time.time()

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k, "fetch_k": top_k}
        )
        retriever_lambda = RunnableLambda(
            lambda x: retriever.invoke(x, filter=filter_expr)
        )
        retriever_chain = {"context": retriever_lambda} | RunnableAssign(
            {"context": lambda input: input["context"]}
        )
        retriever_docs = retriever_chain.invoke(query, config={"run_name": "retriever"})
        docs = retriever_docs.get("context", [])

        end_time = time.time()
        latency = end_time - start_time
        logger.info(f" Elasticsearch Retrieval latency: {latency:.4f} seconds")

        otel_context.detach(token)
        return self._add_collection_name_to_retreived_docs(docs, collection_name)

    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> ElasticsearchStore:
        """
        Get the vectorstore for a collection.
        """
        vectorstore = ElasticsearchStore(
            index_name=collection_name,
            es_url=self.es_url,
            embedding=self._embedding_model,
            strategy=DenseVectorStrategy(
                hybrid=CONFIG.vector_store.search_type == "hybrid"
            ),
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

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
Oracle 26ai Vector Database implementation for NVIDIA RAG Blueprint.

This module provides the OracleVDB class which implements vector database
operations using Oracle AI Database 26ai's native vector search capabilities.
Supports CPU-based IVF indexes with optional hybrid search using Oracle Text.

Key Features:
- Native VECTOR data type support
- IVF (Inverted File Index) for CPU-optimized search
- Hybrid search combining vector similarity with Oracle Text
- LangChain OracleVS integration for retrieval
"""

import json
import logging
import os
import time
from concurrent.futures import Future
from typing import Any

import oracledb
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from opentelemetry import context as otel_context

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.common import (
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig, SearchType
from nvidia_rag.utils.health_models import ServiceStatus
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
    SYSTEM_COLLECTIONS,
)
from nvidia_rag.utils.vdb.oracle.oracle_queries import (
    check_table_exists_query,
    create_document_info_table_ddl,
    create_metadata_schema_table_ddl,
    create_text_index_ddl,
    create_vector_index_ddl,
    create_vector_table_ddl,
    drop_table_ddl,
    get_all_collections_query,
    get_collection_document_info_query,
    get_count_query,
    get_delete_document_info_by_collection_query,
    get_delete_document_info_query,
    get_delete_docs_query,
    get_delete_metadata_schema_query,
    get_document_info_query,
    get_hybrid_search_query,
    get_metadata_schema_query,
    get_similarity_search_query,
    get_unique_sources_query,
)
from nvidia_rag.utils.vdb.vdb_ingest_base import VDBRagIngest

logger = logging.getLogger(__name__)


class OracleVDB(VDBRagIngest):
    """
    Oracle 26ai Vector Database implementation.

    Provides vector storage and retrieval using Oracle AI Database 26ai's
    native vector capabilities with IVF indexes optimized for CPU-based deployment.
    """

    def __init__(
        self,
        collection_name: str,
        oracle_user: str | None = None,
        oracle_password: str | None = None,
        oracle_dsn: str | None = None,
        embedding_model: Any | None = None,
        config: NvidiaRAGConfig | None = None,
        meta_dataframe: Any | None = None,
        meta_source_field: str | None = None,
        meta_fields: list[str] | None = None,
        csv_file_path: str | None = None,
        index_type: str = "IVF",
        distance_metric: str = "COSINE",
        hybrid: bool = False,
    ):
        """
        Initialize Oracle VDB connection.

        Args:
            collection_name: Name of the vector collection/table
            oracle_user: Database username (or set ORACLE_USER env var)
            oracle_password: Database password (or set ORACLE_PASSWORD env var)
            oracle_dsn: Connection DSN (or set ORACLE_DSN env var)
            embedding_model: Embedding model instance for retrieval
            config: NvidiaRAGConfig instance
            meta_dataframe: Metadata dataframe for custom metadata
            meta_source_field: Source field name in metadata
            meta_fields: List of metadata field names
            csv_file_path: Path to CSV file for metadata
            index_type: Vector index type (IVF or HNSW)
            distance_metric: Distance metric (COSINE, L2, DOT)
            hybrid: Enable hybrid search with Oracle Text
        """
        self.config = config or NvidiaRAGConfig()
        self._collection_name = collection_name.upper() if collection_name else ""
        self._embedding_model = embedding_model

        # Connection parameters from args or environment
        self._oracle_user = oracle_user or os.getenv("ORACLE_USER")
        self._oracle_password = oracle_password or os.getenv("ORACLE_PASSWORD")
        self._oracle_dsn = oracle_dsn or os.getenv("ORACLE_DSN")

        if not all([self._oracle_user, self._oracle_password, self._oracle_dsn]):
            raise ValueError(
                "Oracle connection requires ORACLE_USER, ORACLE_PASSWORD, and ORACLE_DSN. "
                "Set via parameters or environment variables."
            )

        # Vector index configuration
        self._index_type = os.getenv("ORACLE_VECTOR_INDEX_TYPE", index_type).upper()
        self._distance_metric = os.getenv("ORACLE_DISTANCE_METRIC", distance_metric).upper()
        self._hybrid = hybrid or (self.config.vector_store.search_type == SearchType.HYBRID)

        # Metadata fields for NV-Ingest
        self.meta_dataframe = meta_dataframe
        self.meta_source_field = meta_source_field
        self.meta_fields = meta_fields
        self.csv_file_path = csv_file_path

        # System collection initialization flags
        self._metadata_schema_initialized = False
        self._document_info_initialized = False

        # Initialize connection pool
        try:
            self._pool = oracledb.create_pool(
                user=self._oracle_user,
                password=self._oracle_password,
                dsn=self._oracle_dsn,
                min=2,
                max=10,
                increment=1,
            )
            # Test connection
            with self._pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
            logger.info(f"Connected to Oracle at {self._oracle_dsn}")
        except oracledb.Error as e:
            logger.exception("Failed to connect to Oracle at %s: %s", self._oracle_dsn, e)
            raise APIError(
                f"Oracle database is unavailable at {self._oracle_dsn}. "
                "Please verify Oracle is running and credentials are correct.",
                ErrorCodeMapping.SERVICE_UNAVAILABLE,
            ) from e

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        """Set the collection name."""
        self._collection_name = value.upper() if value else ""

    def _get_connection(self):
        """Acquire a connection from the pool."""
        return self._pool.acquire()

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(check_table_exists_query(), {"table_name": table_name})
                result = cursor.fetchone()
                return result[0] > 0

    # -------------------------------------------------------------------------
    # NV-Ingest VDB Operations
    def _check_index_exists(self, index_name: str) -> bool:
        """Check if the collection table exists."""
        return self._table_exists(index_name)

    def create_index(self):
        """Create the vector table and index."""
        logger.info(f"Creating Oracle collection if not exists: {self._collection_name}")
        self.create_collection(
            self._collection_name,
            dimension=self.config.embeddings.dimensions,
        )

    def write_to_index(self, records: list, **kwargs) -> None:
        """
        Write records to the Oracle vector table.

        Requires nv_ingest_client for record cleanup.
        """
        try:
            from nv_ingest_client.util.milvus import cleanup_records, pandas_file_reader
        except ImportError as e:
            raise ImportError(
                "nv_ingest_client is required for write_to_index operation. "
                "Install with: pip install nvidia-rag[ingest]"
            ) from e

        # Load metadata if needed
        meta_dataframe = self.meta_dataframe
        if meta_dataframe is None and self.csv_file_path is not None:
            meta_dataframe = pandas_file_reader(self.csv_file_path)

        # Clean records
        cleaned_records = cleanup_records(
            records=records,
            meta_dataframe=meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
        )

        total_records = len(cleaned_records)
        batch_size = 100
        uploaded_count = 0

        logger.info(f"Starting Oracle ingestion for {total_records} records...")

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                insert_sql = f"""
                INSERT INTO {self._collection_name} (text, vector, source, content_metadata)
                VALUES (:text, :vector, :source, :content_metadata)
                """

                for i in range(0, total_records, batch_size):
                    batch = cleaned_records[i:i + batch_size]
                    batch_data = []

                    for record in batch:
                        vector = record.get("vector", [])
                        # Convert vector to Oracle VECTOR format
                        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

                        batch_data.append({
                            "text": record.get("text", ""),
                            "vector": vector_str,
                            "source": record.get("source", ""),
                            "content_metadata": json.dumps(record.get("content_metadata", {})),
                        })

                    cursor.executemany(insert_sql, batch_data)
                    conn.commit()

                    uploaded_count += len(batch)
                    if uploaded_count % (5 * batch_size) == 0 or uploaded_count == total_records:
                        logger.info(f"Ingested {uploaded_count}/{total_records} records into {self._collection_name}")

        logger.info(f"Oracle ingestion completed. Total: {uploaded_count} records")

    def retrieval(self, queries: list, **kwargs) -> list[dict[str, Any]]:
        """Retrieve documents based on queries."""
        raise NotImplementedError("Use retrieval_langchain for Oracle retrieval")

    def reindex(self, records: list, **kwargs) -> None:
        """Reindex documents."""
        raise NotImplementedError("Reindex not implemented for Oracle")

    def run(self, records: list) -> None:
        """Run ingestion pipeline."""
        self.create_index()
        self.write_to_index(records)

    def run_async(self, records: list | Future) -> list:
        """Run async ingestion."""
        logger.info(f"Creating index - {self._collection_name}")
        self.create_index()

        if isinstance(records, Future):
            records = records.result()

        logger.info(f"Writing to index - {self._collection_name}")
        self.write_to_index(records)
        return records

    # -------------------------------------------------------------------------
    # VDBRag Collection Management
    async def check_health(self) -> dict[str, Any]:
        """Check Oracle database health."""
        status = {
            "service": "Oracle 26ai",
            "url": self._oracle_dsn,
            "status": ServiceStatus.UNKNOWN.value,
            "error": None,
        }

        try:
            start_time = time.time()
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
                    cursor.execute("SELECT COUNT(*) FROM user_tables")
                    table_count = cursor.fetchone()[0]

            status["status"] = ServiceStatus.HEALTHY.value
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["tables"] = table_count
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
        """Create a new vector collection table."""
        table_name = collection_name.upper()

        if self._table_exists(table_name):
            logger.info(f"Collection {table_name} already exists")
            return

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Create table
                cursor.execute(create_vector_table_ddl(table_name, dimension))
                logger.info(f"Created table {table_name}")

                # Create vector index
                try:
                    cursor.execute(create_vector_index_ddl(
                        table_name,
                        index_type=self._index_type,
                        distance_metric=self._distance_metric,
                    ))
                    logger.info(f"Created {self._index_type} vector index on {table_name}")
                except oracledb.Error as e:
                    logger.warning(f"Could not create vector index: {e}")

                # Create text index for hybrid search
                if self._hybrid:
                    try:
                        cursor.execute(create_text_index_ddl(table_name))
                        logger.info(f"Created text index on {table_name}")
                    except oracledb.Error as e:
                        logger.warning(f"Could not create text index: {e}")

                conn.commit()

    def check_collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self._table_exists(collection_name.upper())

    def get_collection(self) -> list[dict[str, Any]]:
        """Get all collections with metadata."""
        self.create_metadata_schema_collection()
        self.create_document_info_collection()

        collections = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_all_collections_query())
                tables = cursor.fetchall()

                for (table_name,) in tables:
                    if table_name.upper() in [s.upper() for s in SYSTEM_COLLECTIONS]:
                        continue

                    # Get document count
                    cursor.execute(get_count_query(table_name))
                    count = cursor.fetchone()[0]

                    # Get metadata schema
                    metadata_schema = self.get_metadata_schema(table_name)

                    # Get catalog and metrics data
                    catalog_data = self.get_document_info(
                        info_type="catalog",
                        collection_name=table_name,
                        document_name="NA",
                    )
                    metrics_data = self.get_document_info(
                        info_type="collection",
                        collection_name=table_name,
                        document_name="NA",
                    )

                    collections.append({
                        "collection_name": table_name,
                        "num_entities": count,
                        "metadata_schema": metadata_schema,
                        "collection_info": {**metrics_data, **catalog_data},
                    })

        return collections

    def delete_collections(self, collection_names: list[str]) -> dict[str, Any]:
        """Delete collections."""
        deleted = []
        failed = []

        for name in collection_names:
            table_name = name.upper()
            try:
                if self._table_exists(table_name):
                    with self._get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(drop_table_ddl(table_name))
                            conn.commit()
                    deleted.append(name)
                    logger.info(f"Deleted collection: {name}")

                    # Clean up metadata
                    self._delete_collection_metadata(table_name)
                else:
                    failed.append({
                        "collection_name": name,
                        "error_message": f"Collection {name} not found.",
                    })
            except Exception as e:
                failed.append({
                    "collection_name": name,
                    "error_message": str(e),
                })
                logger.exception(f"Failed to delete collection {name}")

        return {
            "message": "Collection deletion completed.",
            "successful": deleted,
            "failed": failed,
            "total_success": len(deleted),
            "total_failed": len(failed),
        }

    def _delete_collection_metadata(self, collection_name: str) -> None:
        """Delete metadata and document info for a collection."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        get_delete_metadata_schema_query(),
                        {"collection_name": collection_name},
                    )
                except Exception as e:
                    logger.warning(f"Error deleting metadata schema: {e}")

                try:
                    cursor.execute(
                        get_delete_document_info_by_collection_query(),
                        {"collection_name": collection_name},
                    )
                except Exception as e:
                    logger.warning(f"Error deleting document info: {e}")

                conn.commit()

    # -------------------------------------------------------------------------
    # Document Management
    def get_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """Get all documents in a collection."""
        table_name = collection_name.upper()
        metadata_schema = self.get_metadata_schema(table_name)

        documents = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_unique_sources_query(table_name))
                for row in cursor:
                    source_name = row[0]
                    content_metadata = json.loads(row[1]) if row[1] else {}

                    metadata_dict = {}
                    for item in metadata_schema:
                        field_name = item.get("name")
                        metadata_dict[field_name] = content_metadata.get(field_name)

                    doc_info = self.get_document_info(
                        info_type="document",
                        collection_name=table_name,
                        document_name=os.path.basename(source_name),
                    )

                    documents.append({
                        "document_name": os.path.basename(source_name),
                        "metadata": metadata_dict,
                        "document_info": doc_info,
                    })

        return documents

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """Delete documents by source values."""
        table_name = collection_name.upper()

        if result_dict is not None:
            result_dict["deleted"] = []
            result_dict["not_found"] = []

        existing_docs = set()
        if result_dict is not None:
            try:
                all_docs = self.get_documents(collection_name)
                existing_docs = {doc.get("document_name", "") for doc in all_docs}
            except Exception as e:
                logger.warning(f"Failed to check existing documents: {e}")

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for source_value in source_values:
                    doc_basename = os.path.basename(source_value)
                    try:
                        cursor.execute(
                            get_delete_docs_query(table_name),
                            {"source_value": source_value},
                        )
                        deleted_count = cursor.rowcount

                        if result_dict is not None:
                            if deleted_count > 0 or doc_basename in existing_docs:
                                result_dict["deleted"].append(doc_basename)
                            else:
                                result_dict["not_found"].append(doc_basename)
                    except Exception as e:
                        logger.warning(f"Failed to delete {source_value}: {e}")
                        if result_dict is not None:
                            result_dict["not_found"].append(doc_basename)

                conn.commit()

        return True

    # -------------------------------------------------------------------------
    # Metadata Schema Management
    def create_metadata_schema_collection(self) -> None:
        """Create metadata schema table if not exists."""
        if self._metadata_schema_initialized:
            return

        if not self._table_exists(DEFAULT_METADATA_SCHEMA_COLLECTION):
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_metadata_schema_table_ddl())
                    conn.commit()
            logger.info(f"Created {DEFAULT_METADATA_SCHEMA_COLLECTION} table")

        self._metadata_schema_initialized = True

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """Add or update metadata schema for a collection."""
        table_name = collection_name.upper()

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete existing
                cursor.execute(
                    get_delete_metadata_schema_query(),
                    {"collection_name": table_name},
                )

                # Insert new
                cursor.execute(
                    """
                    INSERT INTO metadata_schema (collection_name, metadata_schema)
                    VALUES (:collection_name, :metadata_schema)
                    """,
                    {
                        "collection_name": table_name,
                        "metadata_schema": json.dumps(metadata_schema),
                    },
                )
                conn.commit()

        logger.info(f"Added metadata schema for {table_name}")

    def get_metadata_schema(self, collection_name: str) -> list[dict[str, Any]]:
        """Get metadata schema for a collection."""
        table_name = collection_name.upper()

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    get_metadata_schema_query(),
                    {"collection_name": table_name},
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])

        logger.info(f"No metadata schema found for {table_name}")
        return []

    # -------------------------------------------------------------------------
    # Document Info Management
    def create_document_info_collection(self) -> None:
        """Create document info table if not exists."""
        if self._document_info_initialized:
            return

        if not self._table_exists(DEFAULT_DOCUMENT_INFO_COLLECTION):
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_document_info_table_ddl())
                    conn.commit()
            logger.info(f"Created {DEFAULT_DOCUMENT_INFO_COLLECTION} table")

        self._document_info_initialized = True

    def add_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """Add document info."""
        table_name = collection_name.upper()

        # Aggregate collection info
        if info_type == "collection":
            info_value = self._get_aggregated_document_info(table_name, info_value)

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete existing
                cursor.execute(
                    get_delete_document_info_query(),
                    {
                        "collection_name": table_name,
                        "document_name": document_name,
                        "info_type": info_type,
                    },
                )

                # Insert new
                cursor.execute(
                    """
                    INSERT INTO document_info (collection_name, info_type, document_name, info_value)
                    VALUES (:collection_name, :info_type, :document_name, :info_value)
                    """,
                    {
                        "collection_name": table_name,
                        "info_type": info_type,
                        "document_name": document_name,
                        "info_value": json.dumps(info_value),
                    },
                )
                conn.commit()

        logger.info(f"Added document info for {table_name}/{document_name}")

    def _get_aggregated_document_info(
        self,
        collection_name: str,
        info_value: dict[str, Any],
    ) -> dict[str, Any]:
        """Get aggregated document info for a collection."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        get_collection_document_info_query(),
                        {"collection_name": collection_name, "info_type": "collection"},
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        existing = json.loads(row[0])
                        return perform_document_info_aggregation(existing, info_value)
        except Exception as e:
            logger.warning(f"Error getting aggregated info: {e}")

        return info_value

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get document info."""
        table_name = collection_name.upper()

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    get_document_info_query(),
                    {
                        "collection_name": table_name,
                        "document_name": document_name,
                        "info_type": info_type,
                    },
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])

        return {}

    def get_catalog_metadata(self, collection_name: str) -> dict[str, Any]:
        """Get catalog metadata for a collection."""
        return self.get_document_info(
            info_type="catalog",
            collection_name=collection_name,
            document_name="NA",
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

    # -------------------------------------------------------------------------
    # Retrieval Operations
    def get_langchain_vectorstore(self, collection_name: str) -> OracleVS:
        """Get LangChain OracleVS vectorstore."""
        table_name = collection_name.upper()

        # Map distance metric
        distance_map = {
            "COSINE": DistanceStrategy.COSINE,
            "L2": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "DOT": DistanceStrategy.DOT_PRODUCT,
        }
        distance_strategy = distance_map.get(self._distance_metric, DistanceStrategy.COSINE)

        # Create connection for OracleVS
        conn = oracledb.connect(
            user=self._oracle_user,
            password=self._oracle_password,
            dsn=self._oracle_dsn,
        )

        return OracleVS(
            client=conn,
            embedding_function=self._embedding_model,
            table_name=table_name,
            distance_strategy=distance_strategy,
        )

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: OracleVS | None = None,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
        otel_ctx: Any | None = None,
    ) -> list[Document]:
        """Perform semantic search and return documents."""
        table_name = collection_name.upper()

        logger.info(
            "Oracle Retrieval: Retrieving from %s, search type: %s",
            table_name,
            "hybrid" if self._hybrid else "vector",
        )

        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        token = otel_context.attach(otel_ctx) if otel_ctx is not None else None

        try:
            start_time = time.time()

            logger.info("  [Embedding] Generating query embedding...")
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

            retriever_lambda = RunnableLambda(lambda x: retriever.invoke(x))
            retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                {"context": lambda inp: inp["context"]}
            )

            logger.info("  [VDB Search] Performing vector similarity search...")
            result = retriever_chain.invoke(query, config={"run_name": "retriever"})
            docs = result.get("context", [])

            latency = time.time() - start_time
            logger.info("  [VDB Search] Retrieved %d documents in %.4fs", len(docs), latency)

            return self._add_collection_name_to_retrieved_docs(docs, collection_name)

        except Exception as e:
            logger.exception("Error in retrieval_langchain: %s", e)
            raise APIError(
                f"Oracle retrieval failed: {e}",
                ErrorCodeMapping.SERVICE_UNAVAILABLE,
            ) from e
        finally:
            if token is not None:
                otel_context.detach(token)

    @staticmethod
    def _add_collection_name_to_retrieved_docs(
        docs: list[Document],
        collection_name: str,
    ) -> list[Document]:
        """Add collection name to document metadata."""
        for doc in docs:
            doc.metadata["collection_name"] = collection_name
        return docs

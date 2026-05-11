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
import re
from array import array

import oracledb
# OracleVS was moved out of langchain-community into Oracle's own
# ``langchain-oracledb`` package (deprecated in lc-community >=0.4, removed
# from repo main). We target the current home.
# ``DistanceStrategy`` is still shipped by langchain-community.
from langchain_oracledb.vectorstores.oraclevs import (
    OracleVS,
    _get_distance_function,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from opentelemetry import context as otel_context

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping


def _wallet_connect_kwargs() -> dict:
    """Build optional oracledb connect kwargs for mTLS wallet auth.

    python-oracledb thin mode reads ``ewallet.pem`` directly (it does NOT use
    ``cwallet.sso``). When the wallet PEM is encrypted, ``wallet_password`` is
    required or the connect call will hang in retry loops until timeout.

    Env vars honoured:
      * ``TNS_ADMIN`` / ``ORACLE_WALLET_DIR`` -> config_dir + wallet_location
      * ``ORACLE_WALLET_PASSWORD`` -> wallet_password (only if set)
    """
    kwargs: dict = {}
    wallet_dir = os.getenv("ORACLE_WALLET_DIR") or os.getenv("TNS_ADMIN")
    if wallet_dir:
        kwargs["config_dir"] = wallet_dir
        kwargs["wallet_location"] = wallet_dir
    wallet_pw = os.getenv("ORACLE_WALLET_PASSWORD")
    if wallet_pw:
        kwargs["wallet_password"] = wallet_pw
    return kwargs
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
    RAG_CANONICAL_COLUMNS,
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
    get_table_columns_query,
    get_unique_sources_query,
    is_view_query,
    upsert_collection_info_merge,
    upsert_metadata_schema_merge,
)
from nvidia_rag.utils.vdb.vdb_ingest_base import VDBRagIngest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OracleVSCompat(OracleVS):
    """Compatibility shim to align LangChain OracleVS with our schema and vector bind needs."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier
        _validate_identifier(self.table_name, "table name")
        self._embedding_dim = self.get_embedding_dimension()

    @staticmethod
    def _get_clob_value(lob_or_str) -> str:
        """Read a CLOB/LOB column value to a plain string."""
        if hasattr(lob_or_str, "read"):
            return lob_or_str.read()
        return lob_or_str or ""

    def _detect_table_columns(self, table_name: str) -> set[str]:
        """Return the set of upper-cased column names for a table/view."""
        try:
            with self.client.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM all_tab_columns "
                    "WHERE table_name = :tn AND owner = SYS_CONTEXT('USERENV','CURRENT_SCHEMA')",
                    {"tn": table_name.upper()},
                )
                return {row[0].upper() for row in cur.fetchall()}
        except Exception:
            return set()

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding,
        k: int = 4,
        filter=None,
        **kwargs,
    ):
        embedding_arr = array("f", embedding)
        distance_fn = _get_distance_function(self.distance_strategy)

        cols = self._detect_table_columns(self.table_name)
        has_source = "SOURCE" in cols
        has_cm = "CONTENT_METADATA" in cols

        select_parts = ["id", "text"]
        select_parts.append("source" if has_source else "NULL AS source")
        select_parts.append("content_metadata" if has_cm else "NULL AS content_metadata")
        select_parts.append(
            f"VECTOR_DISTANCE(vector, TO_VECTOR(:embedding, {self._embedding_dim}, FLOAT32), {distance_fn}) AS distance"
        )

        query = f"""
            SELECT {', '.join(select_parts)}
            FROM {self.table_name}
            ORDER BY distance
            FETCH APPROX FIRST {k} ROWS ONLY
        """

        docs_and_scores = []
        connection = self.client

        with connection.cursor() as cursor:
            try:
                cursor.execute(query, embedding=embedding_arr)
            except oracledb.Error as e:
                err_str = str(e)
                if "ORA-51808" in err_str:
                    logger.warning(
                        "Vector dimension mismatch on %s — BYO data has "
                        "different embedding dimensions than the query model. "
                        "Re-embed the data with the same model to enable search.",
                        self.table_name,
                    )
                    return []
                raise
            results = cursor.fetchall()

            for result in results:
                metadata: dict[str, Any] = {}
                if result[2]:
                    try:
                        parsed_source = json.loads(result[2])
                    except (TypeError, json.JSONDecodeError):
                        parsed_source = {"source_name": str(result[2]), "source_id": str(result[2])}
                    if isinstance(parsed_source, dict) and "source_id" not in parsed_source:
                        parsed_source["source_id"] = parsed_source.get("source_name", self.table_name)
                    metadata["source"] = parsed_source
                else:
                    metadata["source"] = {"source_name": self.table_name, "source_id": self.table_name}
                content_metadata: dict[str, Any] = {}
                try:
                    if isinstance(result[3], oracledb.LOB) and result[3]:
                        content_metadata = json.loads(self._get_clob_value(result[3]))
                    elif isinstance(result[3], dict) and result[3]:
                        content_metadata = result[3]
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Malformed content_metadata JSON in row %s; skipping", result[0])

                metadata['content_metadata'] = content_metadata

                # Apply filter if provided
                if filter:
                    logger.info(f'Filtering on :{filter}')
                    if not all(metadata.get(key) in value for key, value in filter.items()):
                        continue

                doc = Document(
                    page_content=(self._get_clob_value(result[1]) if result[1] is not None else ""),
                    metadata=metadata,
                )
                distance = result[4]
                docs_and_scores.append((doc, distance))

        return docs_and_scores


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
        oracle_cs: str | None = None,
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
        if self._collection_name:
            from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier
            _validate_identifier(self._collection_name, "collection name")
        self._embedding_model = embedding_model

        # Connection parameters from args or environment
        self._oracle_user = oracle_user or os.getenv("ORACLE_USER")
        self._oracle_password = oracle_password or os.getenv("ORACLE_PASSWORD")
        self._oracle_cs = oracle_cs or os.getenv("ORACLE_CS")

        if not all([
            self._oracle_user, 
            self._oracle_password, 
            self._oracle_cs
            ]):
            raise ValueError(
                "Oracle connection requires ORACLE_USER, ORACLE_PASSWORD, ORACLE_CS variables."
                "Set via parameters or environment variables."
            )

        # Vector index configuration — validated against allowlists to prevent
        # SQL injection via env vars (distance_metric and index_type are
        # interpolated into DDL/DML).
        _VALID_INDEX_TYPES = {"IVF", "HNSW"}
        _VALID_DISTANCE_METRICS = {"COSINE", "L2", "DOT", "MANHATTAN", "EUCLIDEAN"}
        self._index_type = os.getenv("ORACLE_VECTOR_INDEX_TYPE", index_type).upper()
        if self._index_type not in _VALID_INDEX_TYPES:
            raise ValueError(
                f"Invalid ORACLE_VECTOR_INDEX_TYPE: {self._index_type!r}. "
                f"Must be one of {sorted(_VALID_INDEX_TYPES)}."
            )
        self._distance_metric = os.getenv("ORACLE_DISTANCE_METRIC", distance_metric).upper()
        if self._distance_metric not in _VALID_DISTANCE_METRICS:
            raise ValueError(
                f"Invalid ORACLE_DISTANCE_METRIC: {self._distance_metric!r}. "
                f"Must be one of {sorted(_VALID_DISTANCE_METRICS)}."
            )
        self._hybrid = hybrid or (self.config.vector_store.search_type == SearchType.HYBRID)

        # cuVS-backed GPU index offload via Oracle Private AI Services.
        # Set ORACLE_PAI_INDEX_URL to opt in (Helm chart stamps this onto
        # oracle-creds when oracle.gpuIndexOffload.enabled=true). Only HNSW
        # supports offload — see Oracle 26ai SQL reference for CREATE VECTOR
        # INDEX, OFFLOAD_URL / OFFLOAD_CREDENTIAL_NAME parameters.
        pai_url = (os.getenv("ORACLE_PAI_INDEX_URL", "") or "").strip()
        if pai_url and not pai_url.endswith("/v1/index"):
            pai_url = pai_url.rstrip("/") + "/v1/index"
        self._pai_offload_url = pai_url or None
        self._pai_offload_credential = (os.getenv("ORACLE_PAI_OFFLOAD_CREDENTIAL", "") or "").strip() or None

        # Metadata fields for NV-Ingest
        self.meta_dataframe = meta_dataframe
        self.meta_source_field = meta_source_field
        self.meta_fields = meta_fields
        self.csv_file_path = csv_file_path

        # System collection initialization flags
        self._metadata_schema_initialized = False
        self._document_info_initialized = False

        # Initialize connection pool. Any oracledb.Error here is the FIRST
        # thing a customer sees on a misconfigured deployment, so we map
        # the raw ORA-/DPY- code to a category + a copy-paste-able
        # next-step (see oracle_errors.diagnose_oracle_error). This keeps
        # the failure mode visible in pod logs in 5s instead of buried
        # inside a generic "service unavailable" 500 minutes later.
        try:
            self._pool = oracledb.create_pool(
                user=self._oracle_user,
                password=self._oracle_password,
                dsn=self._oracle_cs,
                min=2,
                max=10,
                increment=1,
                **_wallet_connect_kwargs(),
            )
            with self._pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
            redacted_cs = re.sub(r"://[^@]+@", "://***@", str(self._oracle_cs)) if "://" in str(self._oracle_cs) else "<redacted DSN>"
            logger.info(f"Connected to Oracle (dsn={redacted_cs})")
        except oracledb.Error as e:
            from nvidia_rag.utils.vdb.oracle.oracle_errors import (
                diagnose_oracle_error,
            )
            diag = diagnose_oracle_error(e, dsn=self._oracle_cs, user=self._oracle_user)
            logger.error(
                "Oracle connect failed [%s/%s]: %s\n  next step: %s",
                diag.code or "no-code", diag.category, diag.summary, diag.next_step,
            )
            raise APIError(
                diag.user_message, ErrorCodeMapping.SERVICE_UNAVAILABLE,
            ) from e

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        """Set the collection name."""
        name = value.upper() if value else ""
        if name:
            from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier
            _validate_identifier(name, "collection name")
        self._collection_name = name

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
        elif isinstance(meta_dataframe, str):
            meta_dataframe = pandas_file_reader(meta_dataframe)

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
                VALUES (:text, TO_VECTOR(:vector, {self.config.embeddings.dimensions}, FLOAT32), :source, :content_metadata)
                """

                for i in range(0, total_records, batch_size):
                    batch = cleaned_records[i:i + batch_size]
                    batch_data = []

                    for record in batch:
                        vector = record.get("vector", [])

                        source_val = record.get("source", "")
                        if isinstance(source_val, dict):
                            source_val = json.dumps(source_val)

                        content_metadata_val = record.get("content_metadata", {})
                        if not isinstance(content_metadata_val, str):
                            content_metadata_val = json.dumps(content_metadata_val)
                        # Bind vector using TO_VECTOR with dense float32 format
                        vector_array = array("f", vector)

                        batch_data.append({
                            "text": record.get("text", ""),
                            "vector": vector_array,
                            "source": source_val,
                            "content_metadata": content_metadata_val,
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
            "url": self._oracle_cs,
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
            from nvidia_rag.utils.vdb.oracle.oracle_errors import (
                diagnose_oracle_error,
            )
            diag = diagnose_oracle_error(e, dsn=self._oracle_cs, user=self._oracle_user)
            status["status"] = ServiceStatus.ERROR.value
            status["error"] = diag.user_message
            status["error_category"] = diag.category
            status["error_code"] = diag.code or None

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

                # Create vector index.
                # When OFFLOAD_URL is configured (cuVS GPU offload), a DDL
                # failure is FATAL — we refuse to silently degrade to an
                # unindexed table because the customer explicitly opted into
                # GPU-accelerated index builds. Without OFFLOAD_URL, a DDL
                # failure is a warning (the table is usable; Oracle's
                # optimizer falls back to a full scan on small collections).
                try:
                    cursor.execute(create_vector_index_ddl(
                        table_name,
                        index_type=self._index_type,
                        distance_metric=self._distance_metric,
                        pai_offload_url=self._pai_offload_url if self._index_type == "HNSW" else None,
                        pai_offload_credential=self._pai_offload_credential if self._index_type == "HNSW" else None,
                    ))
                    if self._index_type == "HNSW" and self._pai_offload_url:
                        logger.info(
                            "Created HNSW vector index on %s with cuVS GPU offload to %s",
                            table_name, self._pai_offload_url,
                        )
                    else:
                        logger.info(f"Created {self._index_type} vector index on {table_name}")
                except oracledb.Error as e:
                    err_str = str(e)
                    if self._pai_offload_url and self._index_type == "HNSW":
                        if "ORA-52346" in err_str:
                            logger.warning(
                                "ADB requires HTTPS for OFFLOAD_URL but PAI "
                                "endpoint is HTTP. Retrying HNSW index build "
                                "without GPU offload (CPU-based). To enable "
                                "GPU offload, configure TLS on the PAI "
                                "LoadBalancer. Error: %s", e,
                            )
                            try:
                                cursor.execute(create_vector_index_ddl(
                                    table_name,
                                    index_type=self._index_type,
                                    distance_metric=self._distance_metric,
                                    pai_offload_url=None,
                                    pai_offload_credential=None,
                                ))
                                logger.info(
                                    "Created HNSW vector index on %s (CPU, "
                                    "no GPU offload)", table_name,
                                )
                            except oracledb.Error as e2:
                                logger.warning(
                                    "CPU HNSW index also failed: %s", e2)
                        else:
                            logger.error(
                                "FATAL: cuVS GPU index build failed on %s "
                                "(OFFLOAD_URL=%s). Refusing to continue with "
                                "a degraded unindexed table. Fix the PAI "
                                "endpoint and retry. Error: %s",
                                table_name, self._pai_offload_url, e,
                            )
                            raise
                    else:
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

    # -------------------------------------------------------------------------
    # BYO-database support
    #
    # When a customer brings an existing Oracle ADB the schema may already
    # contain vector tables (or VIEWs created by the optional
    # oracle.importExistingTables Helm flow). These helpers inspect each
    # candidate, register canonical-shape ones with the tracking tables so
    # the UI shows accurate document counts / metrics, and gate write paths
    # on read-only objects (views, non-canonical tables) so the customer's
    # data is never mutated by the RAG ingestor.
    # -------------------------------------------------------------------------
    def _get_table_columns(self, table_name: str) -> set[str]:
        """Return upper-cased column names for a table/view. Cached per call."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(get_table_columns_query(), {"table_name": table_name.upper()})
                    return {row[0].upper() for row in cur.fetchall()}
        except Exception:
            return set()

    def _inspect_object(self, name: str) -> dict[str, Any]:
        """Look up columns + view-status for a table or view.

        Returns ``{"columns": {<col>: <type>}, "is_view": bool,
                   "is_canonical": bool, "has_vector": bool}``.

        Cheap (two single-row reads); we only call this once per object per
        ``get_collection`` request. Errors are swallowed and a "minimal"
        descriptor is returned so a single bad table cannot break the API.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(get_table_columns_query(), {"table_name": name.upper()})
                    cols = {row[0]: row[1] for row in cursor.fetchall()}
                    cursor.execute(is_view_query(), {"object_name": name.upper()})
                    is_view = bool((cursor.fetchone() or [0])[0])
        except Exception as e:  # pragma: no cover - degraded but listable
            logger.warning(f"Inspection failed for {name}: {e}")
            return {"columns": {}, "is_view": False, "is_canonical": False, "has_vector": False}

        upper_cols = {k.upper(): v for k, v in cols.items()}
        is_canonical = all(c in upper_cols for c in RAG_CANONICAL_COLUMNS)
        has_vector = any(
            (dt or "").upper().startswith("VECTOR") for dt in upper_cols.values()
        )
        return {
            "columns": upper_cols,
            "is_view": is_view,
            "is_canonical": is_canonical,
            "has_vector": has_vector,
        }

    def _register_byo_collection(
        self,
        table_name: str,
        is_view: bool,
        row_count: int,
    ) -> None:
        """Idempotently register a customer-existing collection.

        Stamps an empty metadata_schema row plus an aggregate
        ``collection_info`` row tagged ``ingested_via='byo'``. The MERGE
        statements no-op if the rows already exist, so this is safe to
        re-run on every list-collections call.
        """
        info = {
            "ingested_via": "byo",
            "is_view": bool(is_view),
            "registered_at": get_current_timestamp(),
            "num_entities": int(row_count),
        }
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        upsert_metadata_schema_merge(),
                        {"collection_name": table_name.upper(), "schema_json": "[]"},
                    )
                    cursor.execute(
                        upsert_collection_info_merge(),
                        {"collection_name": table_name.upper(), "info_json": json.dumps(info)},
                    )
                    conn.commit()
            logger.info(
                "Auto-registered BYO collection %s (view=%s, rows=%d)",
                table_name, is_view, row_count,
            )
        except Exception as e:
            # Don't fail the list call if registration hits a locked row,
            # quota, or permission issue. Log + continue; the table still
            # appears in the response, just without the BYO tag.
            logger.warning(f"BYO auto-register skipped for {table_name}: {e}")

    def get_collection(self) -> list[dict[str, Any]]:
        """Get all collections with metadata.

        BYO behaviour:
        * Tables that match the RAG canonical shape but have no
          ``metadata_schema`` row are auto-registered (idempotent MERGE)
          with ``ingested_via='byo'`` so the UI renders them with
          accurate counts / metrics on first view.
        * Tables that have a vector column but a non-canonical shape are
          still listed, marked ``read_only=true`` and ``schema_match=false``
          so the frontend can show them with a "schema mismatch" badge
          rather than crashing on /documents.
        * Tables that have no vector column at all are skipped — they
          aren't searchable so listing them would be misleading.
        """
        self.create_metadata_schema_collection()
        self.create_document_info_collection()

        collections = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_all_collections_query())
                objects = [row[0] for row in cursor.fetchall()]

        for object_name in objects:
            if object_name.upper() in [s.upper() for s in SYSTEM_COLLECTIONS]:
                continue

            shape = self._inspect_object(object_name)
            if not shape["has_vector"]:
                # Plain tables (no vector column) cannot be queried by RAG.
                continue

            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(get_count_query(object_name))
                        count = cursor.fetchone()[0]
            except Exception as e:
                logger.warning(f"Count failed for {object_name}: {e}")
                count = 0

            metadata_schema = self.get_metadata_schema(object_name)
            catalog_data = self.get_document_info(
                info_type="catalog", collection_name=object_name, document_name="NA",
            )
            metrics_data = self.get_document_info(
                info_type="collection", collection_name=object_name, document_name="NA",
            )

            # Lazy auto-registration: canonical-shape tables that are missing
            # tracking rows get a default registration so subsequent calls
            # surface a populated collection_info block.
            already_registered = bool(metrics_data) or bool(metadata_schema)
            if shape["is_canonical"] and not already_registered:
                self._register_byo_collection(
                    table_name=object_name,
                    is_view=shape["is_view"],
                    row_count=count,
                )
                metrics_data = self.get_document_info(
                    info_type="collection", collection_name=object_name,
                    document_name="NA",
                )

            # Read-only when:
            #  - object is a SQL VIEW (Oracle disallows DML), or
            #  - shape is non-canonical (writes would target the wrong cols)
            read_only = shape["is_view"] or not shape["is_canonical"]

            collection_info = {**metrics_data, **catalog_data}
            collection_info.setdefault("schema_match", shape["is_canonical"])
            collection_info.setdefault("read_only", read_only)
            collection_info.setdefault("is_view", shape["is_view"])

            collections.append({
                "collection_name": object_name,
                "num_entities": count,
                "metadata_schema": metadata_schema,
                "collection_info": collection_info,
            })

        return collections

    def delete_collections(self, collection_names: list[str]) -> dict[str, Any]:
        """Delete collections.

        BYO-safe: refuses to drop SQL views (Oracle disallows
        ``DROP TABLE`` on a view, and we don't want to silently
        ``DROP VIEW`` either — the customer's view definition is owned
        by them, not by RAG). Surfaces a clear failure message instead
        so the UI can disable the delete button on read-only items.
        """
        deleted = []
        failed = []

        for name in collection_names:
            table_name = name.upper()
            try:
                shape = self._inspect_object(table_name)
                if shape["is_view"]:
                    failed.append({
                        "collection_name": name,
                        "error_message": (
                            f"Collection {name} is a SQL VIEW (BYO mapping); "
                            "drop it from the database directly, or remove the "
                            "matching entry under oracle.importExistingTables."
                        ),
                    })
                    continue
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
        """Get all documents in a collection.

        BYO-safe: if the collection's underlying table or view doesn't
        expose ``source`` / ``content_metadata`` (a non-canonical BYO
        table), return an empty list with a warning rather than crashing
        the API with ORA-00904. The frontend already handles empty
        document lists; throwing here would 500 the entire UI tab.
        """
        table_name = collection_name.upper()
        shape = self._inspect_object(table_name)
        if not shape["is_canonical"]:
            logger.warning(
                "get_documents: %s lacks canonical RAG columns "
                "(text/vector/source/content_metadata). Returning empty list. "
                "Map it via oracle.importExistingTables to enable per-document browsing.",
                table_name,
            )
            return []
        metadata_schema = self.get_metadata_schema(table_name)

        documents = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_unique_sources_query(table_name))
                for row in cursor:
                    source_name = self._extract_source_name(row[0])
                    raw_cm = row[1]
                    content_metadata = {}
                    try:
                        if raw_cm is None:
                            pass
                        elif hasattr(raw_cm, 'read'):
                            content_metadata = json.loads(raw_cm.read()) or {}
                        elif isinstance(raw_cm, str):
                            content_metadata = json.loads(raw_cm) or {}
                        else:
                            content_metadata = raw_cm or {}
                    except (json.JSONDecodeError, ValueError):
                        logger.warning("Malformed content_metadata JSON in documents row; skipping")

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
                    return row[0]

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

    def set_collection_info(
        self,
        collection_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """Directly replace the collection-level document_info entry without aggregation.

        Unlike add_document_info, this bypasses _get_aggregated_document_info so the
        caller's pre-computed value is stored as-is.  Use this after document deletion
        when the caller has already recalculated the correct aggregated state.
        """
        table_name = collection_name.upper()

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    get_delete_document_info_query(),
                    {
                        "collection_name": table_name,
                        "document_name": "NA",
                        "info_type": "collection",
                    },
                )
                cursor.execute(
                    """
                    INSERT INTO document_info (collection_name, info_type, document_name, info_value)
                    VALUES (:collection_name, :info_type, :document_name, :info_value)
                    """,
                    {
                        "collection_name": table_name,
                        "info_type": "collection",
                        "document_name": "NA",
                        "info_value": json.dumps(info_value),
                    },
                )
                conn.commit()

        logger.info(f"Set collection info for {table_name}")

    @staticmethod
    def _read_clob_json(value) -> dict[str, Any]:
        """Normalise a CLOB column value to a dict.

        oracledb returns CLOB columns as LOB objects; read() converts them to
        str, then json.loads converts to dict.  If the value is already a dict
        (e.g. when Oracle returns native JSON), it is returned as-is.
        """
        if hasattr(value, "read"):
            value = value.read()
        if isinstance(value, str):
            return json.loads(value)
        return value or {}

    @staticmethod
    def _extract_source_name(value: Any) -> str:
        """Pull a usable filename out of the ``source`` column.

        Handles:
        * the canonical RAG shape  ``{"source_name": "/path/to/file.pdf"}``
        * a plain string source    ``"/path/to/file.pdf"`` (BYO views often
          wrap a VARCHAR2 column with ``JSON_OBJECT('source_name' VALUE …)``
          but legacy data may have inserted a bare string)
        * LOB / dict / None        — falls back to a best-effort string
        """
        if value is None:
            return "unknown"
        if hasattr(value, "read"):
            try:
                value = value.read()
            except Exception:
                value = str(value)
        if isinstance(value, dict):
            return value.get("source_name") or value.get("source") or "unknown"
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{"):
                try:
                    obj = json.loads(stripped)
                    return obj.get("source_name") or obj.get("source") or stripped
                except json.JSONDecodeError:
                    return stripped
            return stripped or "unknown"
        return str(value)

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
                        existing = self._read_clob_json(row[0])
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
                    return self._read_clob_json(row[0])

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

        conn = self._pool.acquire()

        return OracleVSCompat(
            client=conn,
            embedding_function=self._embedding_model,
            table_name=table_name,
            distance_strategy=distance_strategy,
        )

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
        vectorstore: OracleVS | None = None,
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

            logger.info(
                "  [Oracle] query=%r | collection=%s | search_type=%s | "
                "index_type=%s | top_k=%d | pai_offload=%s",
                query[:80] + ("..." if len(query) > 80 else ""),
                table_name, "hybrid" if self._hybrid else "dense",
                self._index_type, top_k,
                bool(self._pai_offload_url),
            )

            if self._hybrid:
                logger.info("  [Oracle] Generating query embedding (hybrid path)...")
                query_vec = self._embedding_model.embed_query(query)
                logger.info("  [Oracle] Embedding done (dim=%d), running hybrid SQL...", len(query_vec))
                docs = self._hybrid_search(
                    collection_name=table_name,
                    query_text=query,
                    query_vector=query_vec,
                    top_k=top_k,
                )
            else:
                logger.info("  [Oracle] Generating query embedding (dense path)...")
                retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

                retriever_lambda = RunnableLambda(lambda x: retriever.invoke(x))
                retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                    {"context": lambda inp: inp["context"]}
                )

                logger.info("  [Oracle] Running dense vector similarity search...")
                result = retriever_chain.invoke(query, config={"run_name": "retriever"})
                docs = result.get("context", [])

            latency = time.time() - start_time
            logger.info(
                "  [Oracle] Search complete: %d docs in %.3fs | sources: %s",
                len(docs), latency,
                [d.metadata.get("source", {}).get("source_name", "?")[:40] for d in docs[:5]],
            )

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

    # ----- hybrid search -----------------------------------------------------
    @staticmethod
    def _sanitize_text_query(q: str) -> str:
        """Turn a natural-language query into a safe Oracle Text expression.

        CONTAINS() rejects reserved operators like ``&``, ``|``, ``(``, ``)``,
        ``{``, ``}``, ``=``, etc. Strip them, then OR-join the remaining
        whitespace tokens so ANY-term match is enough to contribute to score.
        Empty input → return None (caller should skip the text cte).
        """
        import re

        cleaned = re.sub(r"[^A-Za-z0-9_\s]", " ", q).strip()
        if not cleaned:
            return ""
        tokens = [t for t in cleaned.split() if t]
        return " OR ".join(tokens) if tokens else ""

    def _hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[Document]:
        """Run the weighted hybrid (vector + CTXSYS.CONTEXT) SQL and return Documents.

        Falls back to dense-only if the sanitized text query is empty.
        """
        text_query = self._sanitize_text_query(query_text)
        cols = self._get_table_columns(collection_name)
        has_source = "SOURCE" in cols
        has_cm = "CONTENT_METADATA" in cols
        sql = get_hybrid_search_query(
            table_name=collection_name,
            distance_metric=self._distance_metric,
            has_source=has_source,
            has_content_metadata=has_cm,
        )
        # The template uses ``FETCH FIRST :top_k * 2`` and ``FETCH FIRST :top_k``.
        # Oracle allows arithmetic on binds only in certain contexts; easiest
        # and most portable: substitute numeric values via string format here
        # since top_k is an int controlled by trusted code.
        sql = sql.replace(":top_k * 2", str(int(top_k) * 2))
        sql = sql.replace(":top_k", str(int(top_k)))

        vec_bind = array("f", query_vector)
        docs: list[Document] = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                if not text_query:
                    logger.info(
                        "  [Hybrid] Empty sanitized text query; falling back to dense-only vector search"
                    )
                    vs = self.get_langchain_vectorstore(collection_name)
                    results = vs.similarity_search_by_vector_with_relevance_scores(
                        query_vector, k=top_k,
                    )
                    return [doc for doc, _score in results]
                try:
                    cursor.execute(
                        sql,
                        query_vector=vec_bind,
                        query_text=text_query,
                    )
                except oracledb.Error as e:
                    err_str = str(e)
                    text_index_missing = (
                        "DRG-10599" in err_str
                        or "ORA-30600" in err_str
                        or "ORA-29855" in err_str
                    )
                    if "ORA-51808" in err_str:
                        logger.warning(
                            "Vector dimension mismatch on %s — BYO data has "
                            "different embedding dimensions than the query model.",
                            collection_name,
                        )
                        return []
                    if text_index_missing:
                        logger.warning(
                            "Text index missing on %s; falling back to "
                            "dense-only vector search.",
                            collection_name,
                        )
                        vs = self.get_langchain_vectorstore(collection_name)
                        results = vs.similarity_search_by_vector_with_relevance_scores(
                            query_vector, k=top_k,
                        )
                        return [doc for doc, _score in results]
                    raise
                for row in cursor.fetchall():
                    # row shape: (id, text, source, content_metadata, combined_score)
                    text = row[1].read() if hasattr(row[1], "read") else (row[1] or "")
                    # ``source`` column holds the upstream-style document
                    # metadata (source_id, source_name, ...).  Nest under the
                    # ``source`` key so prepare_citations() can read
                    # ``doc.metadata['source']['source_id']`` (matches the
                    # dense path shape).
                    source_raw = row[2] or ""
                    metadata: dict[str, Any] = {}
                    if source_raw:
                        try:
                            parsed = json.loads(source_raw)
                        except (TypeError, json.JSONDecodeError):
                            parsed = {"source_name": str(source_raw), "source_id": str(source_raw)}
                        if isinstance(parsed, dict) and "source_id" not in parsed:
                            parsed["source_id"] = parsed.get("source_name", collection_name)
                        metadata["source"] = parsed
                    else:
                        metadata["source"] = {"source_name": collection_name, "source_id": collection_name}
                    # ``content_metadata`` column holds the chunk-level
                    # metadata (type/page_number/etc.).  MUST be nested under
                    # the ``content_metadata`` key, otherwise
                    # prepare_citations() can't classify the chunk type.
                    raw_meta = row[3]
                    if hasattr(raw_meta, "read"):
                        raw_meta = raw_meta.read()
                    cm: dict[str, Any] = {}
                    if isinstance(raw_meta, str) and raw_meta:
                        try:
                            cm = json.loads(raw_meta)
                        except json.JSONDecodeError:
                            cm = {"raw": raw_meta}
                    elif isinstance(raw_meta, dict):
                        cm = raw_meta
                    metadata["content_metadata"] = cm
                    metadata["hybrid_score"] = float(row[4]) if row[4] is not None else 0.0
                    docs.append(Document(page_content=text, metadata=metadata))
        return docs

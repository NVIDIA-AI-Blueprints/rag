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
This module contains the implementation of the LanceDBVDB class,
which provides LanceDB vector database operations for RAG applications.
Extends VDBRagIngest (VDBRag + VDB from nv_ingest_client) for NRL-backed ingestion.

NOTE: LanceDB is not fork-safe. All ``import lancedb`` statements MUST remain
inside methods, never at module level.  Speed/scalability is not a requirement
for this backend — it is intended for local evaluation and NRL integration.

Write path:
    NRL GraphIngestor → IngestSchemaManager.to_raw_records()
        → LanceDBVDB.write_to_index(records)
            → handle_lancedb(records, uri, table_name)   [from NRL]

NV-Ingest Client VDB Operations (NRL format only):
1. create_index: Ensure the LanceDB table exists
2. write_to_index: Write raw NRL DataFrame records via NRL's handle_lancedb
3. run: Orchestrate create_index + write_to_index
4. run_async: Async-compatible version of run (handles Future)

Collection Management:
5. create_collection: Create a LanceDB table with NRL schema
6. check_collection_exists: Check if a LanceDB table exists
7. get_collection: List all user tables with row counts, metadata schemas, and collection info
8. delete_collections: Drop one or more LanceDB tables and their associated metadata

Document Management:
9. get_documents: List unique source documents with metadata and document info
10. delete_documents: Remove all rows belonging to given source paths and their document info

Retrieval Operations:
11. retrieval_langchain: Dense vector search returning LangChain Documents
12. get_langchain_vectorstore: Return a LangChain-compatible VectorStore wrapper
13. retrieve_chunks_by_filter: Filter-only retrieval by source and page numbers
14. retrieval_image_langchain: Image/multimodal retrieval via embedding search

Metadata Schema Management:
15. create_metadata_schema_collection: Create dedicated LanceDB table for metadata schemas
16. add_metadata_schema: Store metadata schema for a collection
17. get_metadata_schema: Retrieve metadata schema for a collection

Document Info Management:
18. create_document_info_collection: Create dedicated LanceDB table for document info
19. add_document_info: Store document info (type: catalog, collection, document)
20. get_document_info: Retrieve document info by type/collection/document

Catalog Metadata:
21. get_catalog_metadata: Get catalog metadata for a collection
22. update_catalog_metadata: Update catalog metadata with timestamp
23. get_document_catalog_metadata: Get description and tags for a document
24. update_document_catalog_metadata: Update document catalog metadata
"""

import ast
import json
import logging
import os
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from langchain_core.vectorstores import VectorStore

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.common import (
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import ServiceStatus
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
    SYSTEM_COLLECTIONS,
)
from nvidia_rag.utils.vdb.vdb_ingest_base import VDBRagIngest

logger = logging.getLogger(__name__)

_LANCEDB_INSTALL_MSG = (
    "lancedb is required for LanceDBVDB. "
    "Install with: uv sync --extra rag (includes lancedb), "
    "pip install 'nvidia-rag[lancedb]', or pip install 'lancedb>=0.26,<0.30'."
)


def _import_lancedb():
    """Import lancedb, raising a clear error if the package is missing."""
    try:
        import lancedb as lancedb_mod  # noqa: PLC0415 — not fork-safe; kept inside function

        return lancedb_mod
    except ImportError as exc:
        raise ImportError(_LANCEDB_INSTALL_MSG) from exc


def _parse_nrl_metadata(val: Any) -> dict:
    """Parse NRL's ``str(dict)`` metadata representation to a Python dict.

    NRL stores the metadata field as ``str(meta)`` — a Python repr string using
    single quotes and Python boolean literals (``True``/``False``/``None``).
    Returns an empty dict for null or unparseable values instead of raising.
    """
    if not val:
        return {}
    if isinstance(val, dict):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.debug("_parse_nrl_metadata: could not parse value: %r", val)
        return {}


class LanceDBVDB(VDBRagIngest):
    """
    LanceDB vector database implementation for RAG applications.

    Designed exclusively for NRL (NemoRetriever Library) result DataFrames.
    Uses NRL's ``handle_lancedb`` for writing so that the schema and row
    transformation logic stays consistent with the NRL evaluation pipeline.

    System collections (``metadata_schema``, ``document_info``) are stored as
    dedicated LanceDB tables inside the same database directory.  This mirrors
    the approach used by MilvusVDB and ensures that metadata schema and document
    info are persisted across restarts.

    For Milvus or Elasticsearch (performance-critical) use the dedicated
    MilvusVDB / ElasticVDB classes instead.

    Parameters
    ----------
    table_name:
        LanceDB table name (equivalent to collection_name in other backends).
    uri:
        Path to the LanceDB database directory, e.g. ``"/data/lancedb"``
        or a cloud URI supported by LanceDB.
    embedding_model:
        LangChain-compatible embedding model for retrieval queries.
    config:
        NvidiaRAGConfig instance. Defaults to a new NvidiaRAGConfig().
    hybrid:
        When True, also create a full-text search (FTS) index for hybrid
        dense+sparse retrieval.
    overwrite:
        When False (default), each ``write_to_index`` call appends to the
        existing table so previously ingested documents are preserved.
        Set to True only when a full table replacement is explicitly desired.
    """

    def __init__(
        self,
        table_name: str,
        uri: str,
        embedding_model: Any = None,
        config: NvidiaRAGConfig | None = None,
        hybrid: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.config = config or NvidiaRAGConfig()
        self._table_name = table_name
        self.uri = uri
        self.embedding_model = embedding_model
        self._embedding_model = embedding_model  # alias for consistency with ElasticVDB
        self.hybrid = hybrid
        self.overwrite = overwrite

        # Track if system collections have been initialized (avoid repeated create calls)
        self._metadata_schema_collection_initialized = False
        self._document_info_collection_initialized = False

    # ------------------------------------------------------------------
    # collection_name property (required by VDBRag ABC)
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        """Get the table name (collection name for this backend)."""
        return self._table_name

    @collection_name.setter
    def collection_name(self, collection_name: str) -> None:
        """Set the table name."""
        self._table_name = collection_name

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def check_health(self) -> dict[str, Any]:
        """Check LanceDB health by attempting to connect and list tables."""
        status: dict[str, Any] = {
            "service": "LanceDB",
            "url": self.uri,
            "status": ServiceStatus.UNKNOWN.value,
            "error": None,
        }
        if not self.uri:
            status["status"] = ServiceStatus.SKIPPED.value
            status["error"] = "No URI provided"
            return status
        try:
            lancedb_mod = _import_lancedb()

            db = lancedb_mod.connect(self.uri)
            tables = db.table_names()
            status["status"] = ServiceStatus.HEALTHY.value
            status["tables"] = len(tables)
        except Exception as exc:
            status["status"] = ServiceStatus.ERROR.value
            status["error"] = str(exc)
        return status

    # ------------------------------------------------------------------
    # NV-Ingest Client VDB interface
    # ------------------------------------------------------------------

    def create_index(self, **kwargs) -> None:
        """Ensure the LanceDB table exists with the NRL schema.

        Creates an empty table using the NRL canonical schema if the table
        does not already exist.  If the table exists, this is a no-op.
        """
        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415
        from nemo_retriever.vector_store.lancedb_utils import (  # noqa: PLC0415
            lancedb_schema,
        )

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        db = lancedb_mod.connect(self.uri)
        try:
            db.open_table(self._table_name)
            logger.debug("LanceDB table '%s' already exists; skipping creation.", self._table_name)
            return
        except Exception:
            pass

        dim = self.config.embeddings.dimensions if self.config else 2048
        schema = lancedb_schema(vector_dim=dim)
        empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        db.create_table(self._table_name, data=empty, schema=schema, mode="create")
        logger.info("Created LanceDB table '%s' at '%s'.", self._table_name, self.uri)

    def write_to_index(self, records: list, **kwargs) -> None:
        """Write raw NRL DataFrame records to the LanceDB table.

        Parameters
        ----------
        records:
            ``list[dict]`` produced by ``IngestSchemaManager.to_raw_records()``
            (i.e. ``result_df.to_dict("records")``).  Each dict is one row from
            the NRL result DataFrame and is transformed by NRL's
            ``_build_lancedb_rows_from_df`` before being written.

        Notes
        -----
        * Uses NRL's ``handle_lancedb`` which applies NRL's canonical row
          transformation (embedding extraction, path/page resolution, etc.).
        * ``handle_lancedb`` internally uses ``LanceDBConfig(overwrite=True)``
          regardless of the ``mode`` argument; use ``overwrite=False`` on this
          class to trigger the append path implemented below.
        * Indices (IVF_HNSW_SQ vector index, optional FTS index) are rebuilt
          after every write.
        """
        if not records:
            logger.warning(
                "write_to_index: no records provided for LanceDB table '%s'; skipping.",
                self._table_name,
            )
            return

        if self.overwrite:
            # Use NRL's handle_lancedb which creates/overwrites and indexes in one shot.
            from nemo_retriever.vector_store.lancedb_store import (  # noqa: PLC0415
                handle_lancedb,
            )

            logger.info(
                "Writing %d raw NRL records to LanceDB table '%s' (overwrite=True).",
                len(records),
                self._table_name,
            )
            handle_lancedb(records, self.uri, self._table_name, hybrid=self.hybrid)
        else:
            # Append path: transform rows via NRL, then add to existing table.
            lancedb_mod = _import_lancedb()
            from nemo_retriever.vector_store.lancedb_store import (  # noqa: PLC0415
                LanceDBConfig,
                _build_lancedb_rows_from_df,
                create_lancedb_index,
            )
            from nemo_retriever.vector_store.lancedb_utils import (  # noqa: PLC0415
                create_or_append_lancedb_table,
                infer_vector_dim,
                lancedb_schema,
            )

            cleaned_rows = _build_lancedb_rows_from_df(records)
            if not cleaned_rows:
                logger.warning(
                    "write_to_index: no embeddable rows after NRL transformation; skipping.",
                )
                return

            dim = infer_vector_dim(cleaned_rows) or (self.config.embeddings.dimensions if self.config else 2048)
            schema = lancedb_schema(vector_dim=dim)

            db = lancedb_mod.connect(self.uri)
            table = create_or_append_lancedb_table(
                db,
                self._table_name,
                cleaned_rows,
                schema,
                overwrite=False,
            )

            cfg = LanceDBConfig(
                uri=self.uri,
                table_name=self._table_name,
                hybrid=self.hybrid,
                overwrite=False,
                num_partitions=16 if len(cleaned_rows) > 16 else max(len(cleaned_rows)-1, 1),
            )
            create_lancedb_index(table, cfg=cfg)
            logger.info(
                "Appended %d rows to LanceDB table '%s'.",
                len(cleaned_rows),
                self._table_name,
            )

    def run(self, records: list) -> None:
        """Orchestrate index creation and NRL record ingestion.

        Parameters
        ----------
        records:
            Raw NRL DataFrame records from ``IngestSchemaManager.to_raw_records()``.
        """
        logger.info("LanceDBVDB.run: creating index for table '%s'.", self._table_name)
        self.create_index()
        logger.info("LanceDBVDB.run: writing records to table '%s'.", self._table_name)
        self.write_to_index(records)

    def run_async(self, records: list | Future) -> list:
        """Async-compatible ingestion entry point.

        Parameters
        ----------
        records:
            Raw NRL DataFrame records, or a ``concurrent.futures.Future`` that
            resolves to such records.

        Returns
        -------
        list
            The records that were written (resolved from the Future if needed).
        """
        logger.info("LanceDBVDB.run_async: creating index for table '%s'.", self._table_name)
        self.create_index()

        if isinstance(records, Future):
            records = records.result()

        logger.info("LanceDBVDB.run_async: writing records to table '%s'.", self._table_name)
        self.write_to_index(records)
        return records

    def retrieval(self, queries: list, **kwargs) -> list:
        """VDB ABC stub — use retrieval_langchain for RAG queries."""
        raise NotImplementedError(
            "retrieval() is not implemented for LanceDBVDB. "
            "Use retrieval_langchain() instead."
        )

    def reindex(self, records: list, **kwargs) -> None:
        """Re-ingest records with overwrite semantics."""
        old_overwrite = self.overwrite
        self.overwrite = True
        try:
            self.run(records)
        finally:
            self.overwrite = old_overwrite

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """Create a LanceDB table with the NRL canonical schema.

        Idempotent — if the table already exists, this is a no-op.
        """
        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415
        from nemo_retriever.vector_store.lancedb_utils import (
            lancedb_schema,  # noqa: PLC0415
        )

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        db = lancedb_mod.connect(self.uri)
        try:
            db.open_table(collection_name)
            logger.debug(
                "LanceDB table '%s' already exists; skipping create_collection.",
                collection_name,
            )
            return
        except Exception:
            pass

        schema = lancedb_schema(vector_dim=dimension)
        empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        db.create_table(collection_name, data=empty, schema=schema, mode="create")
        logger.info(
            "Created LanceDB table '%s' (dim=%d) at '%s'.",
            collection_name,
            dimension,
            self.uri,
        )

    def check_collection_exists(self, collection_name: str) -> bool:
        """Return True if the LanceDB table exists."""
        lancedb_mod = _import_lancedb()

        db = lancedb_mod.connect(self.uri)
        return collection_name in db.table_names()

    def get_collection(self) -> list[dict[str, Any]]:
        """Return metadata for all user-facing LanceDB tables.

        Queries the ``metadata_schema`` and ``document_info`` system tables to
        populate ``metadata_schema`` and ``collection_info`` for each table,
        mirroring the behaviour of MilvusVDB / ElasticVDB.

        Returns
        -------
        list[dict]
            Each entry has keys: ``collection_name``, ``num_entities``,
            ``metadata_schema``, ``collection_info``.
        """
        self.create_metadata_schema_collection()
        self.create_document_info_collection()

        lancedb_mod = _import_lancedb()

        db = lancedb_mod.connect(self.uri)
        table_names = db.table_names()
        collections: list[dict[str, Any]] = []
        for name in table_names:
            if name in SYSTEM_COLLECTIONS:
                continue
            try:
                table = db.open_table(name)
                num_rows = table.count_rows()
            except Exception as exc:
                logger.warning("get_collection: failed to open table '%s': %s", name, exc)
                num_rows = 0

            metadata_schema = self.get_metadata_schema(name)
            catalog_data = self.get_document_info(
                info_type="catalog",
                collection_name=name,
                document_name="NA",
            )
            metrics_data = self.get_document_info(
                info_type="collection",
                collection_name=name,
                document_name="NA",
            )
            collections.append(
                {
                    "collection_name": name,
                    "num_entities": num_rows,
                    "metadata_schema": metadata_schema,
                    "collection_info": {**metrics_data, **catalog_data},
                }
            )
        return collections

    def delete_collections(
        self,
        collection_names: list[str],
    ) -> dict[str, Any]:
        """Drop one or more LanceDB tables and clean up associated metadata.

        Also removes entries from the ``metadata_schema`` and ``document_info``
        system tables for each successfully deleted collection.

        Returns
        -------
        dict
            ``successful`` and ``failed`` lists plus totals.
        """
        lancedb_mod = _import_lancedb()

        db = lancedb_mod.connect(self.uri)
        existing = set(db.table_names())
        deleted: list[str] = []
        failed: list[dict[str, str]] = []

        for name in collection_names:
            try:
                if name in existing:
                    db.drop_table(name)
                    deleted.append(name)
                    logger.info("Deleted LanceDB table '%s'.", name)
                else:
                    failed.append(
                        {
                            "collection_name": name,
                            "error_message": f"Table '{name}' not found.",
                        }
                    )
                    logger.warning("LanceDB table '%s' not found; skipping deletion.", name)
            except Exception as exc:
                failed.append({"collection_name": name, "error_message": str(exc)})
                logger.error("Failed to delete LanceDB table '%s': %s", name, exc)

        # Clean up system table entries for successfully deleted collections
        for name in deleted:
            self._delete_from_system_table(
                system_table=DEFAULT_METADATA_SCHEMA_COLLECTION,
                filter_col="collection_name",
                filter_val=name,
            )
            self._delete_from_system_table(
                system_table=DEFAULT_DOCUMENT_INFO_COLLECTION,
                filter_col="collection_name",
                filter_val=name,
            )

        return {
            "message": "Collection deletion process completed.",
            "successful": deleted,
            "failed": failed,
            "total_success": len(deleted),
            "total_failed": len(failed),
        }

    # ------------------------------------------------------------------
    # Document Management
    # ------------------------------------------------------------------

    def get_documents(
        self,
        collection_name: str,
        *,
        force_get_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a list of unique source documents with metadata and document info.

        Reads the ``path`` column (preferred) or ``source_id`` column to
        identify distinct documents.  Metadata fields are extracted from the
        NRL ``metadata`` column for each document.  Document info is retrieved
        from the ``document_info`` system table.

        Parameters
        ----------
        collection_name:
            Name of the LanceDB table to query.
        force_get_metadata:
            When True, always extract per-document metadata from the table rows
            even for large collections (no-op for the bypass logic that Milvus
            uses — LanceDB always loads via pandas).
        """
        lancedb_mod = _import_lancedb()

        # Get metadata schema and document info map up front
        metadata_schema = self.get_metadata_schema(collection_name)
        doc_info_map = self._get_document_info_map(collection_name)

        try:
            db = lancedb_mod.connect(self.uri)
            table = db.open_table(collection_name)
            df = table.to_pandas()
        except Exception as exc:
            logger.error(
                "get_documents: failed to open table '%s': %s",
                collection_name,
                exc,
            )
            return []

        # Determine source column
        if "path" in df.columns:
            source_col = "path"
        elif "source_id" in df.columns:
            source_col = "source_id"
        else:
            logger.warning(
                "get_documents: table '%s' has neither 'path' nor 'source_id' column.",
                collection_name,
            )
            return []

        documents: list[dict[str, Any]] = []
        seen: set[str] = set()

        for _, row in df.iterrows():
            source = row.get(source_col)
            if not source:
                continue
            source_str = str(source)
            if source_str in seen:
                continue
            seen.add(source_str)

            doc_name = os.path.basename(source_str)

            # Extract metadata fields from the NRL metadata repr string
            metadata_dict: dict[str, Any] = {}
            if metadata_schema:
                raw_meta = row.get("metadata", "")
                parsed_meta = _parse_nrl_metadata(raw_meta)
                for schema_item in metadata_schema:
                    field_name = schema_item.get("name")
                    metadata_dict[field_name] = parsed_meta.get(field_name)

            documents.append(
                {
                    "document_name": doc_name,
                    "metadata": metadata_dict,
                    "document_info": doc_info_map.get(doc_name, {}),
                }
            )
        return documents

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """Delete all rows in the table whose ``path`` or ``source_id`` matches.

        Also removes the corresponding ``document`` entries from the
        ``document_info`` system table.

        Parameters
        ----------
        collection_name:
            Target LanceDB table.
        source_values:
            List of source file paths to remove.
        result_dict:
            Optional dict populated with ``"deleted"`` and ``"not_found"`` lists.
        """
        lancedb_mod = _import_lancedb()

        if result_dict is not None:
            result_dict["deleted"] = []
            result_dict["not_found"] = []

        try:
            db = lancedb_mod.connect(self.uri)
            table = db.open_table(collection_name)
        except Exception as exc:
            logger.error(
                "delete_documents: failed to open table '%s': %s",
                collection_name,
                exc,
            )
            return False

        for source_value in source_values:
            doc_name = os.path.basename(source_value)
            escaped = source_value.replace("'", "\\'")
            try:
                count_before = table.count_rows()
                # LanceDB SQL-style predicate: match on either path or source_id column.
                table.delete(f"path = '{escaped}' OR source_id = '{escaped}'")
                count_after = table.count_rows()

                # Also remove document-level info from the document_info system table
                self._delete_document_info_entry(
                    collection_name=collection_name,
                    document_name=doc_name,
                    info_type="document",
                )

                if result_dict is not None:
                    if count_before > count_after:
                        result_dict["deleted"].append(doc_name)
                    else:
                        result_dict["not_found"].append(doc_name)
            except Exception as exc:
                logger.warning(
                    "delete_documents: failed to delete '%s' from '%s': %s",
                    source_value,
                    collection_name,
                    exc,
                )
                if result_dict is not None:
                    result_dict["not_found"].append(doc_name)

        return True

    # ------------------------------------------------------------------
    # Retrieval Operations
    # ------------------------------------------------------------------

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: VectorStore | None = None,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
        otel_ctx: Any | None = None,
    ) -> list[Document]:
        """Perform dense vector search and return LangChain Documents.

        Embeds ``query`` with the configured ``embedding_model``, then runs an
        ANN search against the LanceDB table.  Results are returned as
        LangChain ``Document`` objects whose metadata mirrors the structure
        used by Milvus / Elasticsearch backends so the rest of the RAG server
        remains backend-agnostic.

        Parameters
        ----------
        query:
            Natural-language search query.
        collection_name:
            LanceDB table to search.
        vectorstore:
            Optional pre-initialised LangChain VectorStore.  When ``None``
            (default), ``get_langchain_vectorstore(collection_name)`` is called
            lazily — matching the pattern used by ElasticVDB / MilvusVDB.
        top_k:
            Maximum number of results to return.
        filter_expr:
            Not currently applied — reserved for future SQL-predicate support.
        otel_ctx:
            OpenTelemetry context token (ignored — no tracing for LanceDB).
        """
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        logger.info(
            "LanceDB Retrieval: querying table '%s', top_k=%d.", collection_name, top_k
        )

        try:
            start_time = time.time()

            logger.info("  [Embedding] Generating query embedding for retrieval...")
            logger.info("  [Embedding] Query: '%s'", query[:100] if query else "")
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            logger.info("  [Embedding] Query embedding generated successfully")

            retriever_lambda = RunnableLambda(lambda x: retriever.invoke(x))
            retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                {"context": lambda input: input["context"]}
            )

            logger.info("  [VDB Search] Performing vector similarity search in collection...")
            retriever_docs = retriever_chain.invoke(
                query, config={"run_name": "retriever"}
            )
            docs = retriever_docs.get("context", [])

            latency = time.time() - start_time
            logger.info(
                "  [VDB Search] Retrieved %d documents from table '%s'",
                len(docs),
                collection_name,
            )
            logger.info("  [VDB Search] Total VDB operation latency: %.4f seconds", latency)

            return self._add_collection_name_to_retreived_docs(docs, collection_name)

        except (requests.exceptions.ConnectionError, ConnectionError, OSError) as e:
            embedding_url = (
                self.embedding_model._client.base_url
                if hasattr(self.embedding_model, "_client")
                else "configured endpoint"
            )
            error_msg = (
                f"Embedding NIM unavailable at {embedding_url}. "
                "Please verify the service is running and accessible."
            )
            logger.error("Connection error in retrieval_langchain: %s", e)
            raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e

    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> VectorStore:
        """Return a LangChain-compatible VectorStore backed by a LanceDB table.

        Returns a ``NRLLanceDB`` instance — a subclass of
        ``langchain_community.vectorstores.LanceDB`` that overrides
        ``results_to_docs`` to correctly parse NRL's non-standard metadata
        storage format (``str(meta)`` Python repr, not JSON).

        See ``nvidia_rag.utils.vdb.lancedb.nrl_lancedb.NRLLanceDB``
        for the full documentation of the metadata handling.
        """
        # NRLLanceDB lives in a sibling module.  Import here (not at module
        # level) because importing langchain_community.vectorstores.LanceDB
        # pulls in the lancedb package, which is not fork-safe.
        from nvidia_rag.utils.vdb.lancedb.nrl_lancedb import NRLLanceDB  # noqa: PLC0415

        lancedb_mod = _import_lancedb()

        try:
            db = lancedb_mod.connect(self.uri)
        except Exception as exc:
            raise RuntimeError(
                f"get_langchain_vectorstore: failed to connect to LanceDB at '{self.uri}': {exc}"
            ) from exc

        # Pass the LanceDBConnection object (not a LanceTable).
        # langchain_community >= 0.3 deprecated accepting a LanceTable directly.
        return NRLLanceDB(
            connection=db,
            embedding=self.embedding_model,
            vector_key="vector",
            id_key="source_id",
            text_key="text",
            table_name=collection_name,
        )

    def retrieve_chunks_by_filter(
        self,
        collection_name: str,
        source_name: str,
        page_numbers: list[int],
        limit: int = 1000,
    ) -> list[Document]:
        """Retrieve ALL chunks matching (source, page_numbers) via filter-only query.

        No semantic search — used for page-context expansion when
        ``fetch_full_page_context`` is enabled.

        Filters rows by matching the ``path`` or ``source_id`` column against
        ``source_name``, then parses the NRL ``metadata`` column to filter by
        ``page_number``.  Both operations run in Python/pandas (no ANN index).

        Parameters
        ----------
        collection_name:
            LanceDB table to query.
        source_name:
            Source document path to filter on.
        page_numbers:
            Page numbers to include (matched against the NRL metadata field).
        limit:
            Maximum number of chunks to return.
        """
        if not page_numbers:
            return []

        lancedb_mod = _import_lancedb()

        try:
            db = lancedb_mod.connect(self.uri)
            table = db.open_table(collection_name)
            df = table.to_pandas()
        except Exception as exc:
            logger.error("retrieve_chunks_by_filter: failed to open table '%s': %s", collection_name, exc)
            return []

        # Filter by source path using path or source_id column
        import pandas as pd  # noqa: PLC0415

        source_mask = pd.Series([False] * len(df), index=df.index)
        if "path" in df.columns:
            source_mask |= df["path"].astype(str) == source_name
        if "source_id" in df.columns:
            source_mask |= df["source_id"].astype(str) == source_name

        filtered_df = df[source_mask]
        if filtered_df.empty:
            logger.debug(
                "retrieve_chunks_by_filter: no rows matching source '%s' in table '%s'.",
                source_name,
                collection_name,
            )
            return []

        # Further filter by page number from NRL metadata
        page_numbers_set = set(page_numbers)
        docs: list[Document] = []

        for _, row in filtered_df.iterrows():
            if len(docs) >= limit:
                break

            raw_meta = row.get("metadata", "")
            parsed_meta = _parse_nrl_metadata(raw_meta)

            # Try several common field names for page number
            page_num = (
                parsed_meta.get("page_number")
                or parsed_meta.get("page_num")
                or parsed_meta.get("page")
            )

            # If no page_number in metadata, include the chunk regardless
            # (avoids dropping all results for schemas that don't record page_number)
            if page_num is not None and page_num not in page_numbers_set:
                continue

            text = str(row.get("text", "")) if row.get("text") is not None else ""
            source_val = row.get("path") or row.get("source_id", source_name)
            metadata = {
                "source": source_val,
                "content_metadata": parsed_meta,
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return self._add_collection_name_to_retreived_docs(docs, collection_name)

    def retrieval_image_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: VectorStore | None = None,
        top_k: int = 10,
        reranker_top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve documents from a collection using an image query.

        Embeds the image query via the configured embedding model, performs a
        vector similarity search to find the most relevant document page, then
        returns all chunks from that page for multimodal context.

        Args:
            query: The image query (base64-encoded string or URL).
            collection_name: Name of the LanceDB table to search.
            vectorstore: Optional pre-initialised VectorStore.
            top_k: Number of results for the initial similarity search.
            reranker_top_k: Final number of documents to return.
                            Defaults to ``top_k`` when ``None``.
        """
        final_limit = reranker_top_k if reranker_top_k is not None else top_k

        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        try:
            embedding = self._embedding_model.embed_documents([query])
            scored = vectorstore.similarity_search_by_vector_with_relevance_scores(
                embedding=embedding[0],
                k=top_k,
            )
            results = [doc for doc, _ in scored]
        except Exception as exc:
            logger.error(
                "retrieval_image_langchain: error generating embeddings or searching: %s",
                exc,
                exc_info=True,
            )
            return []

        if not results:
            return []

        # Extract source and page from the top result
        try:
            top_meta = results[0].metadata

            # NRL metadata is stored under the "metadata" key (parsed dict) or
            # directly on the document metadata depending on NRLLanceDB version.
            nrl_meta = top_meta.get("metadata", {})
            if isinstance(nrl_meta, str):
                nrl_meta = _parse_nrl_metadata(nrl_meta)

            # Source name: prefer path > source_id > "source" key
            source_name = (
                top_meta.get("path")
                or top_meta.get("source_id")
                or nrl_meta.get("source_path")
                or nrl_meta.get("source_name")
                or ""
            )

            # Page number: look in NRL metadata
            page_number = (
                nrl_meta.get("page_number")
                or nrl_meta.get("page_num")
                or nrl_meta.get("page")
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "retrieval_image_langchain: error accessing metadata from search results: %s",
                exc,
            )
            return []

        if not source_name:
            logger.warning(
                "retrieval_image_langchain: could not determine source name from top result metadata."
            )
            return self._add_collection_name_to_retreived_docs(
                results[:final_limit], collection_name
            )

        page_numbers = [page_number] if page_number is not None else []
        return self.retrieve_chunks_by_filter(
            collection_name=collection_name,
            source_name=source_name,
            page_numbers=page_numbers,
            limit=final_limit,
        )

    # ------------------------------------------------------------------
    # Metadata Schema Management
    # ------------------------------------------------------------------

    def create_metadata_schema_collection(self) -> None:
        """Create the ``metadata_schema`` system table if it does not exist.

        The table uses a simple two-column schema:
        - ``collection_name`` (string): the user collection this schema belongs to.
        - ``metadata_schema`` (string): JSON-serialised list of schema field dicts.
        """
        if self._metadata_schema_collection_initialized:
            return

        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        db = lancedb_mod.connect(self.uri)

        if DEFAULT_METADATA_SCHEMA_COLLECTION not in db.table_names():
            schema = pa.schema([
                pa.field("collection_name", pa.string()),
                pa.field("metadata_schema", pa.string()),
            ])
            empty = pa.table(
                {
                    "collection_name": pa.array([], type=pa.string()),
                    "metadata_schema": pa.array([], type=pa.string()),
                },
                schema=schema,
            )
            db.create_table(
                DEFAULT_METADATA_SCHEMA_COLLECTION,
                data=empty,
                schema=schema,
                mode="create",
            )
            logger.info(
                "Created LanceDB metadata schema table '%s' at '%s'.",
                DEFAULT_METADATA_SCHEMA_COLLECTION,
                self.uri,
            )
        else:
            logger.debug(
                "LanceDB metadata schema table '%s' already exists.",
                DEFAULT_METADATA_SCHEMA_COLLECTION,
            )

        self._metadata_schema_collection_initialized = True

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """Store (or replace) the metadata schema for ``collection_name``.

        Deletes any existing schema entry for the collection before inserting
        the new one, so this is effectively an upsert.

        Parameters
        ----------
        collection_name:
            The user collection whose schema is being recorded.
        metadata_schema:
            List of field definition dicts (same format as Milvus / Elasticsearch).
        """
        self.create_metadata_schema_collection()

        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415

        db = lancedb_mod.connect(self.uri)
        table = db.open_table(DEFAULT_METADATA_SCHEMA_COLLECTION)

        # Delete existing entry for this collection
        escaped = collection_name.replace("'", "\\'")
        try:
            table.delete(f"collection_name = '{escaped}'")
        except Exception as exc:
            logger.debug(
                "add_metadata_schema: delete attempt for '%s' raised: %s (may not exist yet).",
                collection_name,
                exc,
            )

        # Insert new schema row
        new_row = pa.table(
            {
                "collection_name": pa.array([collection_name], type=pa.string()),
                "metadata_schema": pa.array([json.dumps(metadata_schema)], type=pa.string()),
            }
        )
        table.add(new_row)
        logger.info(
            "Metadata schema stored for collection '%s': %s",
            collection_name,
            metadata_schema,
        )

    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve the metadata schema for ``collection_name``.

        Returns an empty list if no schema has been registered or if the
        system table does not exist yet.
        """
        lancedb_mod = _import_lancedb()

        try:
            db = lancedb_mod.connect(self.uri)
            if DEFAULT_METADATA_SCHEMA_COLLECTION not in db.table_names():
                return []
            table = db.open_table(DEFAULT_METADATA_SCHEMA_COLLECTION)
            df = table.to_pandas()
        except Exception as exc:
            logger.error(
                "get_metadata_schema: error reading system table for '%s': %s",
                collection_name,
                exc,
            )
            return []

        row = df[df["collection_name"] == collection_name]
        if row.empty:
            logger.info(
                "get_metadata_schema: no schema found for collection '%s'.",
                collection_name,
            )
            return []
        try:
            return json.loads(row.iloc[0]["metadata_schema"])
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error(
                "get_metadata_schema: failed to parse schema for '%s': %s",
                collection_name,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Document Info Management
    # ------------------------------------------------------------------

    def create_document_info_collection(self) -> None:
        """Create the ``document_info`` system table if it does not exist.

        The table schema has four columns:
        - ``info_type`` (string): "catalog", "collection", or "document".
        - ``collection_name`` (string): which user collection the info belongs to.
        - ``document_name`` (string): document filename or "NA" for collection-level info.
        - ``info_value`` (string): JSON-serialised info dict.
        """
        if self._document_info_collection_initialized:
            return

        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        db = lancedb_mod.connect(self.uri)

        if DEFAULT_DOCUMENT_INFO_COLLECTION not in db.table_names():
            schema = pa.schema([
                pa.field("info_type", pa.string()),
                pa.field("collection_name", pa.string()),
                pa.field("document_name", pa.string()),
                pa.field("info_value", pa.string()),
            ])
            empty = pa.table(
                {
                    "info_type": pa.array([], type=pa.string()),
                    "collection_name": pa.array([], type=pa.string()),
                    "document_name": pa.array([], type=pa.string()),
                    "info_value": pa.array([], type=pa.string()),
                },
                schema=schema,
            )
            db.create_table(
                DEFAULT_DOCUMENT_INFO_COLLECTION,
                data=empty,
                schema=schema,
                mode="create",
            )
            logger.info(
                "Created LanceDB document info table '%s' at '%s'.",
                DEFAULT_DOCUMENT_INFO_COLLECTION,
                self.uri,
            )
        else:
            logger.debug(
                "LanceDB document info table '%s' already exists.",
                DEFAULT_DOCUMENT_INFO_COLLECTION,
            )

        self._document_info_collection_initialized = True

    def _get_aggregated_document_info(
        self,
        collection_name: str,
        info_value: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate new collection-level info with existing info.

        Used internally by ``add_document_info`` when ``info_type == "collection"``
        to merge new ingestion statistics with any already-stored values (e.g. to
        accumulate counts across multiple ingestion calls).

        Parameters
        ----------
        collection_name:
            The user collection whose aggregated info is needed.
        info_value:
            The new info dict from the current ingestion.

        Returns
        -------
        dict
            Merged dict produced by ``perform_document_info_aggregation``.
        """
        existing = self.get_document_info(
            info_type="collection",
            collection_name=collection_name,
            document_name="NA",
        )
        try:
            return perform_document_info_aggregation(existing, info_value)
        except Exception as exc:
            logger.error(
                "_get_aggregated_document_info: aggregation failed for '%s': %s",
                collection_name,
                exc,
            )
            return info_value

    def add_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """Store (or replace) document info for a collection or document.

        For ``info_type == "collection"`` the new ``info_value`` is aggregated
        with any existing collection-level info using
        ``perform_document_info_aggregation`` before storage (same semantics as
        Milvus and Elasticsearch).

        Parameters
        ----------
        info_type:
            One of ``"catalog"``, ``"collection"``, or ``"document"``.
        collection_name:
            Target user collection.
        document_name:
            Document filename, or ``"NA"`` for collection/catalog-level entries.
        info_value:
            Info dict to store.
        """
        self.create_document_info_collection()

        # Aggregate collection-level info with existing data before storing
        if info_type == "collection":
            info_value = self._get_aggregated_document_info(collection_name, info_value)

        lancedb_mod = _import_lancedb()
        import pyarrow as pa  # noqa: PLC0415

        db = lancedb_mod.connect(self.uri)
        table = db.open_table(DEFAULT_DOCUMENT_INFO_COLLECTION)

        # Delete existing entry for this (info_type, collection_name, document_name)
        esc_type = info_type.replace("'", "\\'")
        esc_col = collection_name.replace("'", "\\'")
        esc_doc = document_name.replace("'", "\\'")
        try:
            table.delete(
                f"info_type = '{esc_type}' "
                f"AND collection_name = '{esc_col}' "
                f"AND document_name = '{esc_doc}'"
            )
        except Exception as exc:
            logger.debug(
                "add_document_info: delete attempt raised: %s (may not exist yet).", exc
            )

        # Insert new row
        new_row = pa.table(
            {
                "info_type": pa.array([info_type], type=pa.string()),
                "collection_name": pa.array([collection_name], type=pa.string()),
                "document_name": pa.array([document_name], type=pa.string()),
                "info_value": pa.array([json.dumps(info_value)], type=pa.string()),
            }
        )
        table.add(new_row)
        logger.debug(
            "Document info stored: info_type=%s, collection=%s, document=%s.",
            info_type,
            collection_name,
            document_name,
        )

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Retrieve document info from the ``document_info`` system table.

        Returns an empty dict when no matching entry is found or when the
        system table does not exist yet.

        Parameters
        ----------
        info_type:
            One of ``"catalog"``, ``"collection"``, or ``"document"``.
        collection_name:
            Target user collection.
        document_name:
            Document filename, or ``"NA"`` for collection/catalog-level entries.
        """
        lancedb_mod = _import_lancedb()

        try:
            db = lancedb_mod.connect(self.uri)
            if DEFAULT_DOCUMENT_INFO_COLLECTION not in db.table_names():
                return {}
            table = db.open_table(DEFAULT_DOCUMENT_INFO_COLLECTION)
            df = table.to_pandas()
        except Exception as exc:
            logger.error(
                "get_document_info: error reading system table for '%s/%s/%s': %s",
                info_type,
                collection_name,
                document_name,
                exc,
            )
            return {}

        mask = (
            (df["info_type"] == info_type)
            & (df["collection_name"] == collection_name)
            & (df["document_name"] == document_name)
        )
        rows = df[mask]
        if rows.empty:
            logger.debug(
                "get_document_info: no entry for info_type=%s, collection=%s, document=%s.",
                info_type,
                collection_name,
                document_name,
            )
            return {}
        try:
            return json.loads(rows.iloc[0]["info_value"])
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error(
                "get_document_info: failed to parse info_value for '%s/%s/%s': %s",
                info_type,
                collection_name,
                document_name,
                exc,
            )
            return {}

    # ------------------------------------------------------------------
    # Catalog Metadata
    # ------------------------------------------------------------------

    def get_catalog_metadata(self, collection_name: str) -> dict[str, Any]:
        """Get catalog metadata for a collection.

        Wraps ``get_document_info`` with ``info_type="catalog"`` and
        ``document_name="NA"``, consistent with Milvus / Elasticsearch.
        """
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
        """Update catalog metadata for a collection.

        Merges ``updates`` into the existing catalog metadata dict and
        refreshes the ``last_updated`` timestamp before saving.
        """
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
        """Get catalog metadata (description and tags) for a document.

        Returns a dict with keys ``"description"`` (str) and ``"tags"``
        (list), consistent with Milvus / Elasticsearch behaviour.
        """
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
        """Update catalog metadata for a specific document.

        Only ``"description"`` and ``"tags"`` keys from ``updates`` are applied;
        all other existing fields are preserved.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_from_system_table(
        self,
        system_table: str,
        filter_col: str,
        filter_val: str,
    ) -> None:
        """Delete all rows in a system table where ``filter_col == filter_val``.

        Silently ignores errors (e.g. table does not exist).
        """
        lancedb_mod = _import_lancedb()
        try:
            db = lancedb_mod.connect(self.uri)
            if system_table not in db.table_names():
                return
            table = db.open_table(system_table)
            escaped = filter_val.replace("'", "\\'")
            table.delete(f"{filter_col} = '{escaped}'")
        except Exception as exc:
            logger.debug(
                "_delete_from_system_table: error deleting from '%s' where %s='%s': %s",
                system_table,
                filter_col,
                filter_val,
                exc,
            )

    def _delete_document_info_entry(
        self,
        collection_name: str,
        document_name: str,
        info_type: str,
    ) -> None:
        """Delete a specific entry from the document_info system table."""
        lancedb_mod = _import_lancedb()
        try:
            db = lancedb_mod.connect(self.uri)
            if DEFAULT_DOCUMENT_INFO_COLLECTION not in db.table_names():
                return
            table = db.open_table(DEFAULT_DOCUMENT_INFO_COLLECTION)
            esc_type = info_type.replace("'", "\\'")
            esc_col = collection_name.replace("'", "\\'")
            esc_doc = document_name.replace("'", "\\'")
            table.delete(
                f"info_type = '{esc_type}' "
                f"AND collection_name = '{esc_col}' "
                f"AND document_name = '{esc_doc}'"
            )
        except Exception as exc:
            logger.debug(
                "_delete_document_info_entry: error for collection='%s', document='%s': %s",
                collection_name,
                document_name,
                exc,
            )

    def _get_document_info_map(self, collection_name: str) -> dict[str, dict[str, Any]]:
        """Return a ``{document_name: info_value}`` map for ``info_type="document"``.

        Used by ``get_documents`` to attach per-document info without making a
        separate ``get_document_info`` call per document.
        """
        lancedb_mod = _import_lancedb()
        result: dict[str, dict[str, Any]] = {}
        try:
            db = lancedb_mod.connect(self.uri)
            if DEFAULT_DOCUMENT_INFO_COLLECTION not in db.table_names():
                return result
            table = db.open_table(DEFAULT_DOCUMENT_INFO_COLLECTION)
            df = table.to_pandas()
            mask = (df["info_type"] == "document") & (df["collection_name"] == collection_name)
            for _, row in df[mask].iterrows():
                doc_name = row["document_name"]
                try:
                    result[doc_name] = json.loads(row["info_value"])
                except (json.JSONDecodeError, KeyError):
                    result[doc_name] = {}
        except Exception as exc:
            logger.error(
                "_get_document_info_map: error for collection '%s': %s",
                collection_name,
                exc,
            )
        return result

    @staticmethod
    def _add_collection_name_to_retreived_docs(
        docs: list[Document], collection_name: str
    ) -> list[Document]:
        """Attach ``collection_name`` to each Document's metadata."""
        for doc in docs:
            doc.metadata["collection_name"] = collection_name
        return docs

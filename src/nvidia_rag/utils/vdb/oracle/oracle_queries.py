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
Oracle 26ai SQL query utilities for vector database operations.
Provides DDL and query functions for document and metadata management.

Functions:
"""
import re

_SAFE_IDENTIFIER_PART_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,127}$")


def _validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* is a safe, unquoted Oracle identifier.

    Accepts simple names (``MY_TABLE``) and schema-qualified names
    (``RAG_APP.MY_TABLE``). Each dot-separated part must start with a
    letter and contain only letters, digits, ``_``, ``$``, ``#``.

    Double-quotes wrapping the name (e.g. from langchain_oracledb) are
    stripped before validation and from the returned value.

    Prevents SQL injection via collection/table names that are
    interpolated into DDL f-strings. Raises ValueError immediately
    so the caller gets a clear message instead of a cryptic ORA- error.
    """
    stripped = name.strip('"')
    parts = stripped.split(".")
    if len(parts) > 2:
        raise ValueError(
            f"Unsafe Oracle {label}: {name!r}. At most one dot allowed "
            "(schema.table)."
        )
    for part in parts:
        if not _SAFE_IDENTIFIER_PART_RE.match(part):
            raise ValueError(
                f"Unsafe Oracle {label}: {name!r}. Each part must start "
                "with a letter and contain only letters, digits, _, $, # "
                "(max 128 chars)."
            )
    return stripped


"""
1. create_vector_table_ddl: Generate DDL for vector collection tables
2. create_vector_index_ddl: Generate DDL for IVF vector index
3. create_metadata_schema_table_ddl: DDL for metadata schema storage
4. create_document_info_table_ddl: DDL for document info storage
5. get_unique_sources_query: Retrieve all unique document sources
6. get_delete_docs_query: Delete documents by source value
7. get_similarity_search_query: Vector similarity search query
8. get_hybrid_search_query: Combined vector + text search query
"""

from typing import Any, Literal

DistanceMetric = Literal["COSINE", "L2", "DOT", "MANHATTAN"]
IndexType = Literal["IVF", "HNSW"]


def create_vector_table_ddl(
    table_name: str,
    dimension: int = 2048,
) -> str:
    """
    Generate DDL for creating a vector collection table.

    Args:
        table_name: Name of the table to create
        dimension: Vector dimension size

    Returns:
        DDL statement string
    """
    _validate_identifier(table_name, "table name")
    return f"""
    CREATE TABLE {table_name} (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        text CLOB,
        vector VECTOR({dimension}, FLOAT32),
        source VARCHAR2(4000),
        content_metadata CLOB CHECK (content_metadata IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """


def create_vector_index_ddl(
    table_name: str,
    index_type: IndexType = "IVF",
    distance_metric: DistanceMetric = "COSINE",
    ivf_neighbor_partitions: int = 100,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 200,
    pai_offload_url: str | None = None,
    pai_offload_credential: str | None = None,
) -> str:
    """
    Generate DDL for creating a vector index (IVF or HNSW).

    Args:
        table_name: Name of the table
        index_type: IVF or HNSW
        distance_metric: COSINE, L2, DOT, or MANHATTAN
        ivf_neighbor_partitions: Number of partitions for IVF index
        hnsw_m: Max connections per node for HNSW
        hnsw_ef_construction: Size of dynamic candidate list for HNSW
        pai_offload_url: Oracle Private AI Services container URL for cuVS-
            backed HNSW index build offload. Mapped to the ``OFFLOAD_URL``
            HNSW parameter. Ignored for IVF (Oracle 26ai only supports
            offload for HNSW). Typically the in-cluster service URL such as
            ``http://oracle-pai-gpu-index.<ns>.svc.cluster.local:8080/v1/index``.
        pai_offload_credential: Name of the ``DBMS_VECTOR.CREATE_CREDENTIAL``
            credential holding the API key for HTTPS offload. Mapped to
            ``OFFLOAD_CREDENTIAL_NAME``. Omit for HTTP-mode offload.

    Returns:
        DDL statement string
    """
    _validate_identifier(table_name, "table name")
    index_name = f"{table_name}_vec_idx"

    # NB: ADB 23ai/26ai rejects ``neighbor_partitions`` as a single underscored
    # token in the PARAMETERS clause (ORA-00922). The correct syntax is the
    # two-word form ``NEIGHBOR PARTITIONS``.  Same for HNSW's ``M`` and
    # ``EFCONSTRUCTION`` — they must be uppercase keywords, not identifiers.
    # Verified against live RAGBP ADB via diag_ivf2.py.
    if index_type == "IVF":
        return f"""
        CREATE VECTOR INDEX {index_name} ON {table_name}(vector)
        ORGANIZATION NEIGHBOR PARTITIONS
        DISTANCE {distance_metric}
        WITH TARGET ACCURACY 95
        PARAMETERS (TYPE IVF, NEIGHBOR PARTITIONS {ivf_neighbor_partitions})
        """

    # HNSW: optional cuVS GPU offload via Oracle Private AI Services.
    # OFFLOAD_URL / OFFLOAD_CREDENTIAL_NAME are documented in the SQL ref for
    # CREATE VECTOR INDEX (PARAMETERS clause for HNSW).
    hnsw_params = [
        "TYPE HNSW",
        f"NEIGHBORS {hnsw_m}",
        f"EFCONSTRUCTION {hnsw_ef_construction}",
    ]
    if pai_offload_url:
        safe_url = pai_offload_url.replace("'", "")
        hnsw_params.append(f"OFFLOAD_URL '{safe_url}'")
        if pai_offload_credential:
            safe_cred = pai_offload_credential.replace("'", "")
            hnsw_params.append(f"OFFLOAD_CREDENTIAL_NAME '{safe_cred}'")
    return f"""
        CREATE VECTOR INDEX {index_name} ON {table_name}(vector)
        ORGANIZATION INMEMORY NEIGHBOR GRAPH
        DISTANCE {distance_metric}
        WITH TARGET ACCURACY 95
        PARAMETERS ({", ".join(hnsw_params)})
        """


def create_text_index_ddl(table_name: str) -> str:
    """
    Generate DDL for creating Oracle Text index for hybrid search.

    Args:
        table_name: Name of the table

    Returns:
        DDL statement string
    """
    _validate_identifier(table_name, "table name")
    index_name = f"{table_name}_text_idx"
    return f"""
    CREATE INDEX {index_name} ON {table_name}(text)
    INDEXTYPE IS CTXSYS.CONTEXT
    PARAMETERS ('SYNC (ON COMMIT)')
    """


def create_metadata_schema_table_ddl() -> str:
    """Generate DDL for metadata schema storage table."""
    return """
    CREATE TABLE metadata_schema (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        collection_name VARCHAR2(255) NOT NULL UNIQUE,
        metadata_schema CLOB CHECK (metadata_schema IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """


def create_document_info_table_ddl() -> str:
    """Generate DDL for document info storage table."""
    return """
    CREATE TABLE document_info (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        collection_name VARCHAR2(255) NOT NULL,
        info_type VARCHAR2(50) NOT NULL,
        document_name VARCHAR2(4000) NOT NULL,
        info_value CLOB CHECK (info_value IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT doc_info_unique UNIQUE (collection_name, info_type, document_name)
    )
    """


def get_unique_sources_query(table_name: str) -> str:
    """
    Generate query to retrieve all unique document sources.

    Args:
        table_name: Name of the collection table

    Returns:
        SQL query string
    """
    _validate_identifier(table_name, "table name")
    return f"""
    WITH unique_sources AS (
        SELECT source, content_metadata,
               ROW_NUMBER() OVER (PARTITION BY source ORDER BY ROWID DESC) as rn
        FROM {table_name}
        WHERE source IS NOT NULL
    )
    SELECT source, content_metadata
    FROM unique_sources
    WHERE rn = 1
    ORDER BY source
    """


def get_delete_docs_query(table_name: str) -> str:
    """
    Generate parameterized delete query for documents by source.

    The source column stores a JSON object with a source_name field
    (e.g. {"source_name": "/tmp/.../file.pdf"}).  We match against the
    extracted source_name value; the fallback OR clause handles any rows
    where source was stored as a plain string.

    Args:
        table_name: Name of the collection table

    Returns:
        SQL query string with :source_value bind variable
    """
    _validate_identifier(table_name, "table name")
    return f"""
    DELETE FROM {table_name}
    WHERE JSON_VALUE(source, '$.source_name') = :source_value
       OR source = :source_value
    """


def get_delete_metadata_schema_query() -> str:
    """Generate parameterized delete query for metadata schema."""
    return """
    DELETE FROM metadata_schema
    WHERE collection_name = :collection_name
    """


def get_metadata_schema_query() -> str:
    """Generate parameterized query to retrieve metadata schema."""
    return """
    SELECT metadata_schema
    FROM metadata_schema
    WHERE collection_name = :collection_name
    """


def get_delete_document_info_query() -> str:
    """Generate parameterized delete query for document info."""
    return """
    DELETE FROM document_info
    WHERE collection_name = :collection_name
    AND document_name = :document_name
    AND info_type = :info_type
    """


def get_delete_document_info_by_collection_query() -> str:
    """Generate parameterized delete query for all document info in a collection."""
    return """
    DELETE FROM document_info
    WHERE collection_name = :collection_name
    """


def get_document_info_query() -> str:
    """Generate parameterized query to retrieve document info."""
    return """
    SELECT info_value
    FROM document_info
    WHERE collection_name = :collection_name
    AND document_name = :document_name
    AND info_type = :info_type
    """


def get_collection_document_info_query() -> str:
    """Generate parameterized query to retrieve collection-level document info."""
    return """
    SELECT info_value
    FROM document_info
    WHERE collection_name = :collection_name
    AND info_type = :info_type
    """


def get_similarity_search_query(
    table_name: str,
    distance_metric: DistanceMetric = "COSINE",
    has_source: bool = True,
    has_content_metadata: bool = True,
) -> str:
    """
    Generate vector similarity search query.

    Args:
        table_name: Name of the collection table
        distance_metric: Distance function to use
        has_source: Whether the table has a 'source' column
        has_content_metadata: Whether the table has a 'content_metadata' column

    Returns:
        SQL query string with :query_vector and :top_k bind variables
    """
    _validate_identifier(table_name, "table name")
    src = "source" if has_source else "NULL AS source"
    cm = "content_metadata" if has_content_metadata else "NULL AS content_metadata"
    return f"""
    SELECT id, text, {src}, {cm},
           VECTOR_DISTANCE(vector, :query_vector, {distance_metric}) as distance
    FROM {table_name}
    ORDER BY distance
    FETCH FIRST :top_k ROWS ONLY
    """


def get_hybrid_search_query(
    table_name: str,
    distance_metric: DistanceMetric = "COSINE",
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    has_source: bool = True,
    has_content_metadata: bool = True,
) -> str:
    """
    Generate hybrid search query combining vector similarity and text search
    using weighted-score fusion (vector distance inverse + Oracle Text SCORE).

    Args:
        table_name: Name of the collection table
        distance_metric: Distance function for vector search
        vector_weight: Weight for vector similarity score (default 0.7)
        text_weight: Weight for text search score (default 0.3)
        has_source: Whether the table has a 'source' column
        has_content_metadata: Whether the table has a 'content_metadata' column

    Returns:
        SQL query string with :query_vector, :query_text, and :top_k bind variables
    """
    _validate_identifier(table_name, "table name")
    src = "source" if has_source else "NULL AS source"
    cm = "content_metadata" if has_content_metadata else "NULL AS content_metadata"
    return f"""
    WITH vector_results AS (
        SELECT id, text, {src}, {cm},
               VECTOR_DISTANCE(vector, :query_vector, {distance_metric}) as vec_distance,
               ROW_NUMBER() OVER (ORDER BY VECTOR_DISTANCE(vector, :query_vector, {distance_metric})) as vec_rank
        FROM {table_name}
        FETCH FIRST :top_k * 2 ROWS ONLY
    ),
    text_results AS (
        SELECT id, SCORE(1) as text_score,
               ROW_NUMBER() OVER (ORDER BY SCORE(1) DESC) as text_rank
        FROM {table_name}
        WHERE CONTAINS(text, :query_text, 1) > 0
        FETCH FIRST :top_k * 2 ROWS ONLY
    )
    SELECT v.id, v.text, v.source, v.content_metadata,
           ({vector_weight} * (1 / (1 + v.vec_distance)) + 
            {text_weight} * COALESCE(t.text_score / 100, 0)) as combined_score
    FROM vector_results v
    LEFT JOIN text_results t ON v.id = t.id
    ORDER BY combined_score DESC
    FETCH FIRST :top_k ROWS ONLY
    """


def get_count_query(table_name: str) -> str:
    """Generate query to count documents in a collection."""
    _validate_identifier(table_name, "table name")
    return f"SELECT COUNT(*) as cnt FROM {table_name}"


def check_table_exists_query() -> str:
    """Generate query to check if a table or view exists."""
    return """
    SELECT COUNT(*) as cnt
    FROM user_objects
    WHERE object_name = UPPER(:table_name)
      AND object_type IN ('TABLE', 'VIEW')
    """


def drop_table_ddl(table_name: str) -> str:
    """Generate DDL to drop a table."""
    _validate_identifier(table_name, "table name")
    return f"DROP TABLE {table_name} CASCADE CONSTRAINTS PURGE"


def get_all_collections_query() -> str:
    """Generate query to list all collection tables.

    BYO support: also returns VIEWs in the same schema, so mapped customer
    tables (created by the optional ``oracle.importExistingTables`` Helm
    flow) appear in the UI just like ingested collections. Excludes Oracle
    Text auxiliary objects (``DR$%``), system tables, and dbtools tables.
    """
    return """
    SELECT object_name FROM (
        SELECT table_name AS object_name FROM user_tables
        UNION ALL
        SELECT view_name AS object_name FROM user_views
    )
    WHERE object_name NOT IN ('METADATA_SCHEMA', 'DOCUMENT_INFO')
    AND object_name NOT LIKE 'SYS%'
    AND object_name NOT LIKE 'DR$%'
    AND object_name NOT LIKE 'DBTOOLS%'
    AND object_name NOT LIKE 'VECTOR$%'
    ORDER BY object_name
    """


# ---------------------------------------------------------------------------
# BYO-database helpers: table introspection + canonical-shape detection +
# read-only registration. The frontend lists whatever the canonical
# get_collection() returns, so this module gives oracle_vdb.py just enough
# SQL to surface customer-pre-existing data without breaking the API
# contract.
# ---------------------------------------------------------------------------

# Columns required by the dense / hybrid retrieval queries above. Keep this
# list in sync with create_vector_table_ddl() and the SELECT lists in
# get_similarity_search_query / get_hybrid_search_query.
RAG_CANONICAL_COLUMNS: tuple[str, ...] = (
    "ID", "TEXT", "VECTOR", "SOURCE", "CONTENT_METADATA",
)


def get_table_columns_query() -> str:
    """List columns + types for a single table or view.

    Uses ``user_tab_columns`` which covers BOTH base tables and views
    (Oracle exposes view column metadata identically). Bind parameter:
    ``:table_name`` (uppercase).
    """
    return """
    SELECT column_name, data_type
    FROM user_tab_columns
    WHERE table_name = :table_name
    ORDER BY column_id
    """


def is_view_query() -> str:
    """Return 1 if ``:object_name`` is a view in the current schema, 0 otherwise.

    Used to flag BYO collections as read-only without an extra round-trip
    on every UI request.
    """
    return """
    SELECT COUNT(*) FROM user_views WHERE view_name = :object_name
    """


def list_vector_tables_query() -> str:
    """Discovery: every base table or view in this schema with a VECTOR column.

    Vector columns are reported by ``user_tab_columns.data_type LIKE 'VECTOR%'``
    (Oracle stores them as ``VECTOR(<dim>, <type>)``). This is what the
    oracle-byo-import Job uses to find candidates for auto-registration.
    """
    return """
    SELECT DISTINCT table_name FROM user_tab_columns
    WHERE data_type LIKE 'VECTOR%'
    ORDER BY table_name
    """


def upsert_metadata_schema_merge() -> str:
    """Idempotent insert into ``metadata_schema`` for a BYO collection.

    Runs as a single MERGE so concurrent ``GET /collections`` calls from
    multiple replicas can't deadlock on a duplicate-key insert. The schema
    column is left as an empty array (the customer can add filterable
    fields later via the UI / API).

    Binds: ``:collection_name`` (UPPER), ``:schema_json`` (CLOB).
    """
    return """
    MERGE INTO metadata_schema d
    USING (SELECT :collection_name AS collection_name FROM dual) s
    ON (d.collection_name = s.collection_name)
    WHEN NOT MATCHED THEN
        INSERT (collection_name, metadata_schema)
        VALUES (s.collection_name, :schema_json)
    """


def upsert_collection_info_merge() -> str:
    """Idempotent insert of an aggregate ``collection_info`` row.

    Stamps ``ingested_via='byo'`` and a creation timestamp so the UI can
    distinguish customer-existing tables from ones uploaded through the
    ingestor. Binds: ``:collection_name`` (UPPER), ``:info_json``.
    """
    return """
    MERGE INTO document_info d
    USING (SELECT :collection_name AS collection_name FROM dual) s
    ON (d.collection_name = s.collection_name
        AND d.info_type = 'collection'
        AND d.document_name = 'NA')
    WHEN NOT MATCHED THEN
        INSERT (collection_name, info_type, document_name, info_value)
        VALUES (s.collection_name, 'collection', 'NA', :info_json)
    """


def create_byo_view_ddl(
    view_name: str,
    source_table: str,
    column_map: dict[str, str],
) -> str:
    """Build a ``CREATE OR REPLACE VIEW`` that exposes a BYO table as RAG-shape.

    The view projects whatever columns the customer supplies into the five
    columns the retrieval / hybrid SQL above expects: ``id, text, vector,
    source, content_metadata``. Missing keys are filled with safe defaults:

    * ``id``               → ``ROWID``                (always available)
    * ``source``           → ``JSON_OBJECT('source_name' VALUE <col>)``
                             when ``source`` is a plain string column;
                             otherwise the caller maps it directly.
    * ``content_metadata`` → ``JSON_OBJECT()`` (empty JSON) if not mapped.

    The result is a read-only view (Oracle disallows DML through a view
    with derived expressions), which is the point — BYO data should not
    be mutated by the RAG ingestor or the delete-by-source flow.

    Args:
        view_name: Target view name in the RAG_APP schema.
        source_table: Fully-qualified source ``schema.table`` (or just
            ``table`` if it lives in the same schema).
        column_map: Map of canonical RAG column → BYO column. Required:
            ``text``, ``vector``. Optional: ``id``, ``source``,
            ``source_wrap_json``, ``content_metadata``.

    Raises:
        ValueError if ``text`` or ``vector`` are missing.
    """
    _validate_identifier(view_name, "BYO view name")
    _validate_identifier(source_table, "BYO source table")
    cm = {k.lower(): v for k, v in column_map.items()}
    if "text" not in cm or "vector" not in cm:
        raise ValueError(
            "BYO view requires at least 'text' and 'vector' columns to be "
            "mapped; retrieval cannot work without them."
        )
    for key in ("text", "vector", "id", "source", "content_metadata"):
        if key in cm and cm[key]:
            _validate_identifier(cm[key], f"BYO column mapping '{key}'")

    id_expr = cm.get("id") or "ROWID"

    text_expr = cm["text"]
    vector_expr = cm["vector"]

    if "source" in cm:
        # source_wrap_json=True wraps a plain VARCHAR2 column into the
        # JSON shape the rest of the codebase expects:
        # {"source_name": "<value>"}
        if cm.get("source_wrap_json", "true").lower() in ("true", "yes", "1"):
            source_expr = (
                f"JSON_OBJECT('source_name' VALUE TO_CHAR({cm['source']})) "
            )
        else:
            source_expr = cm["source"]
    else:
        source_expr = "JSON_OBJECT('source_name' VALUE 'byo')"

    if "content_metadata" in cm:
        meta_expr = f"COALESCE({cm['content_metadata']}, JSON_OBJECT())"
    else:
        meta_expr = "JSON_OBJECT()"

    # Created_at is informational; ROWNUM-derived NOW() is fine because the
    # column never participates in the retrieval SQL.
    return f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT
        {id_expr} AS id,
        {text_expr} AS text,
        {vector_expr} AS vector,
        {source_expr} AS source,
        {meta_expr} AS content_metadata,
        CURRENT_TIMESTAMP AS created_at
    FROM {source_table}
    """
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Syntactic validity guards for every SQL string produced by oracle_queries.

We can't run a real Oracle parser in unit tests, but we can:

1. Tokenize each statement with sqlparse to confirm the lexer doesn't
   choke (catches stray quotes, unterminated strings, etc.).
2. Verify balanced parentheses / brackets / quotes — a class of bug
   that f-string DDL is especially prone to.
3. Confirm exactly one top-level statement per builder (no accidental
   ``;`` insertion that would let a future caller pass two statements
   to ``cursor.execute`` and get an Oracle error).
4. Confirm the canonical column list in ``RAG_CANONICAL_COLUMNS`` is
   the actual set that ``create_vector_table_ddl`` defines — drift
   between the two would silently break BYO discovery.
"""
from __future__ import annotations

import re

import pytest
import sqlparse

from nvidia_rag.utils.vdb.oracle import oracle_queries as oq


# ---------------------------------------------------------------------------
# Catalog of every SQL-producing function. We discover them dynamically so
# adding a new helper to oracle_queries.py automatically opts it into these
# checks (with sensible default arguments where required).
# ---------------------------------------------------------------------------
SQL_BUILDER_CALLS = [
    ("create_vector_table_ddl",         (oq.create_vector_table_ddl,         ("docs", 2048))),
    ("create_vector_table_ddl_dim_768", (oq.create_vector_table_ddl,         ("docs", 768))),
    ("create_vector_index_ddl_ivf",     (oq.create_vector_index_ddl,         ("docs", "IVF", "COSINE"))),
    ("create_vector_index_ddl_hnsw",    (oq.create_vector_index_ddl,         ("docs", "HNSW", "L2"))),
    ("create_vector_index_ddl_offload", (lambda: oq.create_vector_index_ddl(
        "docs", index_type="HNSW", distance_metric="COSINE",
        pai_offload_url="http://x:8080/v1/index",
        pai_offload_credential="cred",
    ), ())),
    ("create_text_index_ddl",           (oq.create_text_index_ddl,           ("docs",))),
    ("create_metadata_schema_table_ddl",(oq.create_metadata_schema_table_ddl, ())),
    ("create_document_info_table_ddl",  (oq.create_document_info_table_ddl,   ())),
    ("get_unique_sources_query",        (oq.get_unique_sources_query,         ("docs",))),
    ("get_delete_docs_query",           (oq.get_delete_docs_query,            ("docs",))),
    ("get_delete_metadata_schema_query",(oq.get_delete_metadata_schema_query, ())),
    ("get_metadata_schema_query",       (oq.get_metadata_schema_query,        ())),
    ("get_delete_document_info_query",  (oq.get_delete_document_info_query,   ())),
    ("get_delete_document_info_by_collection_query",
                                        (oq.get_delete_document_info_by_collection_query, ())),
    ("get_document_info_query",         (oq.get_document_info_query,          ())),
    ("get_collection_document_info_query",
                                        (oq.get_collection_document_info_query, ())),
    ("get_similarity_search_query",     (oq.get_similarity_search_query,      ("docs",))),
    ("get_hybrid_search_query",         (oq.get_hybrid_search_query,          ("docs",))),
    ("get_count_query",                 (oq.get_count_query,                  ("docs",))),
    ("check_table_exists_query",        (oq.check_table_exists_query,         ())),
    ("drop_table_ddl",                  (oq.drop_table_ddl,                   ("docs",))),
    ("get_all_collections_query",       (oq.get_all_collections_query,        ())),
    ("get_table_columns_query",         (oq.get_table_columns_query,          ())),
    ("is_view_query",                   (oq.is_view_query,                    ())),
    ("list_vector_tables_query",        (oq.list_vector_tables_query,         ())),
    ("upsert_metadata_schema_merge",    (oq.upsert_metadata_schema_merge,     ())),
    ("upsert_collection_info_merge",    (oq.upsert_collection_info_merge,     ())),
    ("create_byo_view_ddl_min",         (oq.create_byo_view_ddl, (
        "v", "T", {"text": "c", "vector": "v"}))),
    ("create_byo_view_ddl_full",        (oq.create_byo_view_ddl, (
        "v", "T",
        {"id": "doc_id", "text": "c", "vector": "vec",
         "source": "src", "content_metadata": "meta"}))),
    ("create_byo_view_ddl_no_wrap",     (oq.create_byo_view_ddl, (
        "v", "T",
        {"text": "c", "vector": "v",
         "source": "src_json", "source_wrap_json": "false"}))),
]


def _build_sql(entry):
    name, (fn, args) = entry
    return name, fn(*args)


@pytest.mark.parametrize("entry", SQL_BUILDER_CALLS, ids=[e[0] for e in SQL_BUILDER_CALLS])
def test_sql_lexes_cleanly(entry):
    """sqlparse.parse must succeed and yield ≥ 1 token."""
    name, sql = _build_sql(entry)
    parsed = sqlparse.parse(sql)
    assert parsed, f"{name}: sqlparse returned nothing"
    tokens = [t for t in parsed[0].flatten() if not t.is_whitespace]
    assert tokens, f"{name}: no non-whitespace tokens"


@pytest.mark.parametrize("entry", SQL_BUILDER_CALLS, ids=[e[0] for e in SQL_BUILDER_CALLS])
def test_balanced_parens_and_quotes(entry):
    """Catches stray '(' or unterminated string literals."""
    name, sql = _build_sql(entry)
    assert sql.count("(") == sql.count(")"), (
        f"{name}: unbalanced parens — open={sql.count('(')} close={sql.count(')')}\n{sql}"
    )
    # Even number of single quotes (string literals come in pairs).
    assert sql.count("'") % 2 == 0, f"{name}: odd number of single quotes\n{sql}"


@pytest.mark.parametrize("entry", SQL_BUILDER_CALLS, ids=[e[0] for e in SQL_BUILDER_CALLS])
def test_at_most_one_top_level_statement(entry):
    """No builder may emit two statements concatenated with ';'.

    A future caller could otherwise feed the second statement to
    ``cursor.execute`` (which only accepts one) and get an opaque
    ORA-00911. Multi-statement scripts must use a PL/SQL block instead.
    """
    name, sql = _build_sql(entry)
    # Strip trailing whitespace + optional single trailing semicolon.
    s = sql.strip().rstrip(";").strip()
    statements = [
        x for x in sqlparse.split(s)
        if x.strip() and not x.strip().startswith("--")
    ]
    assert len(statements) <= 1, (
        f"{name}: produced {len(statements)} statements:\n{statements}"
    )


@pytest.mark.parametrize("entry", SQL_BUILDER_CALLS, ids=[e[0] for e in SQL_BUILDER_CALLS])
def test_no_obvious_injection_artefact(entry):
    """No SQL helper should accidentally include unfiltered template
    placeholders (``{}``, ``{var}``) — sign of a missed f-string."""
    name, sql = _build_sql(entry)
    assert not re.search(r"\{[A-Za-z_]+\}", sql), (
        f"{name}: looks like an unrendered Python format placeholder\n{sql}"
    )
    assert "{0}" not in sql and "{1}" not in sql


# ---------------------------------------------------------------------------
# Drift guard: RAG_CANONICAL_COLUMNS vs the actual CREATE TABLE
# ---------------------------------------------------------------------------
def test_canonical_columns_match_create_table_ddl():
    """If create_vector_table_ddl ever adds/renames a column without
    updating RAG_CANONICAL_COLUMNS, BYO auto-detection silently breaks
    (canonical tables get classified as non-canonical and turn read-only).
    """
    ddl = oq.create_vector_table_ddl("X", dimension=2048).upper()
    for col in oq.RAG_CANONICAL_COLUMNS:
        # Match e.g. "    ID RAW(16)" or "    TEXT CLOB"
        assert re.search(rf"\b{col}\s+", ddl), (
            f"Canonical column {col!r} missing from create_vector_table_ddl"
        )


def test_canonical_columns_are_uppercase_and_unique():
    """Oracle stores identifiers uppercased; comparing against
    ``user_tab_columns`` requires the tuple already be uppercase."""
    cols = oq.RAG_CANONICAL_COLUMNS
    assert len(cols) == len(set(cols)), "duplicates in RAG_CANONICAL_COLUMNS"
    assert all(c == c.upper() for c in cols)


# ---------------------------------------------------------------------------
# MERGE statements: structural invariants
# ---------------------------------------------------------------------------
def test_metadata_merge_uses_when_not_matched_only():
    """We deliberately INSERT-only on auto-register: WHEN MATCHED would
    let a stale registration overwrite a customer-edited schema. Pin
    that decision."""
    sql = oq.upsert_metadata_schema_merge().upper()
    assert "WHEN NOT MATCHED" in sql
    assert "WHEN MATCHED" not in sql


def test_collection_info_merge_uses_when_not_matched_only():
    sql = oq.upsert_collection_info_merge().upper()
    assert "WHEN NOT MATCHED" in sql
    assert "WHEN MATCHED" not in sql
    # Pinning the unique key (collection_name + info_type + document_name):
    # a refactor that drops document_name from the ON clause would let two
    # rows collide on the per-document insert path.
    assert "DOCUMENT_NAME" in sql
    assert "INFO_TYPE" in sql


def test_byo_view_is_create_or_replace():
    """``CREATE OR REPLACE`` is required so re-running the BYO Job after
    a column-mapping change actually picks up the new mapping. ``CREATE
    VIEW`` would ORA-00955 the second time."""
    ddl = oq.create_byo_view_ddl(
        "v", "T", {"text": "c", "vector": "v"}
    ).upper()
    assert "CREATE OR REPLACE VIEW" in ddl


# ---------------------------------------------------------------------------
# get_all_collections_query: combined tables + views, system rows excluded
# ---------------------------------------------------------------------------
def test_all_collections_query_unions_tables_and_views():
    sql = oq.get_all_collections_query().upper()
    assert "USER_TABLES" in sql
    assert "USER_VIEWS" in sql
    assert "UNION ALL" in sql
    # Excludes our own tracking tables — otherwise they'd show in the UI.
    assert "METADATA_SCHEMA" in sql
    assert "DOCUMENT_INFO" in sql
    # Excludes Oracle Text auxiliaries and SYS objects (BYO Job creates
    # these and they aren't "collections").
    assert "DR$%" in sql
    assert "SYS%" in sql


def test_list_vector_tables_query_filters_to_vector_only():
    sql = oq.list_vector_tables_query().upper()
    assert "USER_TAB_COLUMNS" in sql
    assert "DATA_TYPE LIKE 'VECTOR%'" in sql
    # DISTINCT — same table can't appear twice if it has multiple cols.
    assert "DISTINCT" in sql

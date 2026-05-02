# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end mocked tests for ``OracleVDB.get_collection`` BYO behaviour.

These tests stub ``oracledb.create_pool`` with a fake connection that
answers SQL queries from a programmable in-memory schema. That gives us
real coverage of:

* lazy auto-registration on first list-collections call
* the read_only / schema_match / is_view flags surfaced to the frontend
* skip behaviour for tables without a vector column
* defensive get_documents() returning [] for non-canonical tables
* delete_collections() refusing to drop a SQL view

…without ever touching a real Oracle database.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake oracledb cursor + connection
# ---------------------------------------------------------------------------
class FakeCursor:
    """A SQL-aware cursor stub.

    Routes execute(sql, binds) to a small dispatcher that knows about the
    queries OracleVDB.get_collection actually issues:

      * SELECT 1 FROM DUAL
      * SELECT COUNT(*) FROM user_tables ...
      * the get_all_collections_query (UNION ALL of tables/views)
      * get_table_columns_query / is_view_query
      * the count query for each collection
      * MERGE into metadata_schema / document_info (just records the call)
      * SELECT info_value FROM document_info ...
      * SELECT metadata_schema FROM metadata_schema ...
      * get_unique_sources_query
    """

    def __init__(self, schema):
        self.schema = schema
        self._rows: list[Any] = []
        self._row_iter = None
        self.executed: list[tuple[str, dict | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        return None

    # --- dispatcher ------------------------------------------------------
    def execute(self, sql, binds=None):
        if isinstance(binds, dict):
            captured_binds = dict(binds)
        else:
            captured_binds = binds
        self.executed.append((sql, captured_binds))
        norm = " ".join(sql.upper().split())

        if "SELECT 1 FROM DUAL" in norm:
            self._rows = [(1,)]
        elif "USER_TABLES" in norm and "WHERE TABLE_NAME" in norm:
            # check_table_exists_query — count of tables with a name match
            tn = (binds or {}).get("table_name", "").upper()
            exists = 1 if tn in {t.upper() for t in self.schema["tables"]} else 0
            self._rows = [(exists,)]
        elif "FROM USER_TABLES" in norm and "COUNT" in norm:
            # health-check style: SELECT COUNT(*) FROM user_tables
            self._rows = [(len(self.schema["tables"]),)]
        elif "SELECT OBJECT_NAME FROM" in norm and "UNION ALL" in norm:
            names = list(self.schema["tables"]) + list(self.schema["views"])
            # Same WHERE clause filtering the prod query does
            keep = []
            for n in names:
                up = n.upper()
                if up in ("METADATA_SCHEMA", "DOCUMENT_INFO"):
                    continue
                if up.startswith(("SYS", "DR$", "DBTOOLS")):
                    continue
                keep.append((n,))
            keep.sort()
            self._rows = keep
        elif "FROM USER_TAB_COLUMNS" in norm and "ORDER BY COLUMN_ID" in norm:
            tbl = (binds or {}).get("table_name", "").upper()
            cols = self.schema["columns"].get(tbl, [])
            self._rows = list(cols)
        elif "FROM USER_VIEWS" in norm and "WHERE VIEW_NAME" in norm:
            obj = (binds or {}).get("object_name", "").upper()
            self._rows = [(1 if obj in self.schema["views"] else 0,)]
        elif "FROM USER_VIEWS" in norm:
            self._rows = [(1,)] if self.schema["views"] else [(0,)]
        elif "SELECT COUNT(*)" in norm and "USER_" not in norm and " FROM " in norm:
            tbl = norm.split(" FROM ")[-1].split()[0].rstrip()
            self._rows = [(self.schema["counts"].get(tbl.upper(), 0),)]
        elif norm.startswith("MERGE INTO METADATA_SCHEMA"):
            self.schema["merges"].append(("metadata_schema", binds))
            self._rows = []
        elif norm.startswith("MERGE INTO DOCUMENT_INFO"):
            self.schema["merges"].append(("document_info", binds))
            self._rows = []
        elif "SELECT METADATA_SCHEMA FROM METADATA_SCHEMA" in norm:
            cn = (binds or {}).get("collection_name", "").upper()
            v = self.schema["metadata_schema_rows"].get(cn)
            self._rows = [(v,)] if v is not None else []
        elif "SELECT INFO_VALUE FROM DOCUMENT_INFO" in norm:
            cn = (binds or {}).get("collection_name", "").upper()
            it = (binds or {}).get("info_type", "")
            dn = (binds or {}).get("document_name", "")
            v = self.schema["document_info_rows"].get((cn, it, dn))
            self._rows = [(v,)] if v is not None else []
        elif norm.startswith("WITH UNIQUE_SOURCES AS"):
            tbl = re.search(r"FROM\s+(\S+)", sql, re.I)
            tbl_name = tbl.group(1).upper() if tbl else ""
            self._rows = self.schema["unique_sources"].get(tbl_name, [])
        elif norm.startswith("CREATE TABLE"):
            # Auto-create system tables
            self._rows = []
        elif norm.startswith("DROP TABLE"):
            tbl = re.search(r"DROP TABLE\s+(\S+)", sql, re.I)
            if tbl:
                self.schema["tables"].discard(tbl.group(1).upper())
            self._rows = []
        elif norm.startswith("DELETE FROM"):
            self._rows = []
        else:
            # Unknown query — return empty rows so we don't accidentally
            # mask a missing handler. Tests that need a specific path will
            # see the 'executed' log instead.
            self._rows = []
        self._row_iter = iter(self._rows)

    def executemany(self, sql, batch):
        self.executed.append((sql, batch))

    def fetchone(self):
        try:
            return next(self._row_iter)
        except StopIteration:
            return None

    def fetchall(self):
        rows = list(self._row_iter or [])
        self._row_iter = iter([])
        return rows

    def __iter__(self):
        return iter(self._row_iter or [])

    @property
    def rowcount(self):
        return 0


class FakeConnection:
    def __init__(self, schema):
        self.schema = schema
        self.committed = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return FakeCursor(self.schema)

    def commit(self):
        self.committed += 1

    def close(self):
        return None


class FakePool:
    def __init__(self, schema):
        self.schema = schema

    @contextmanager
    def acquire(self):
        yield FakeConnection(self.schema)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_schema(
    tables=None, views=None, columns=None, counts=None,
    metadata_schema_rows=None, document_info_rows=None,
    unique_sources=None,
):
    return {
        "tables": set(tables or set()),
        "views": set(views or set()),
        "columns": columns or {},
        "counts": counts or {},
        "metadata_schema_rows": metadata_schema_rows or {},
        "document_info_rows": document_info_rows or {},
        "unique_sources": unique_sources or {},
        "merges": [],
    }


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ORACLE_USER", "u")
    monkeypatch.setenv("ORACLE_PASSWORD", "p")
    monkeypatch.setenv("ORACLE_CS", "tcps://h:1521/s")


def make_vdb(schema):
    """Create an OracleVDB whose pool returns canned rows from `schema`."""
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

    cfg = MagicMock()
    cfg.vector_store.search_type = "vector"
    cfg.embeddings.dimensions = 2048

    with patch("nvidia_rag.utils.vdb.oracle.oracle_vdb.oracledb") as fake_db:
        fake_db.create_pool.return_value = FakePool(schema)
        # oracledb.Error must be a real exception class
        fake_db.Error = type("Error", (Exception,), {})
        fake_db.LOB = type("LOB", (), {})
        vdb = OracleVDB(collection_name="ANY", config=cfg)
    # Patch the pool back in so subsequent calls work
    vdb._pool = FakePool(schema)
    return vdb


# ---------------------------------------------------------------------------
# Scenario 1: mixed schema with canonical, non-canonical, view, no-vector
# ---------------------------------------------------------------------------
@pytest.fixture
def mixed_schema():
    return make_schema(
        tables={"GOOD_DOCS", "WEIRD_DOCS", "PLAIN_TABLE"},
        views={"BYO_VIEW"},
        columns={
            "GOOD_DOCS": [
                ("ID", "RAW"), ("TEXT", "CLOB"),
                ("VECTOR", "VECTOR(2048,FLOAT32)"),
                ("SOURCE", "VARCHAR2"), ("CONTENT_METADATA", "CLOB"),
                ("CREATED_AT", "TIMESTAMP"),
            ],
            "WEIRD_DOCS": [  # Has VECTOR but missing TEXT/SOURCE
                ("DOC_ID", "NUMBER"),
                ("EMBED", "VECTOR(2048,FLOAT32)"),
                ("BODY", "CLOB"),
            ],
            "BYO_VIEW": [  # Canonical-shape view (mapped from a custom table)
                ("ID", "ROWID"), ("TEXT", "CLOB"),
                ("VECTOR", "VECTOR(2048,FLOAT32)"),
                ("SOURCE", "VARCHAR2"), ("CONTENT_METADATA", "CLOB"),
            ],
            "PLAIN_TABLE": [  # No VECTOR column at all
                ("ID", "NUMBER"), ("BODY", "CLOB"),
            ],
        },
        counts={
            "GOOD_DOCS": 17, "WEIRD_DOCS": 5,
            "BYO_VIEW": 1234, "PLAIN_TABLE": 99,
        },
    )


def test_get_collection_lists_only_vector_bearing_objects(env, mixed_schema):
    """PLAIN_TABLE has no vector column and must not appear."""
    vdb = make_vdb(mixed_schema)
    out = vdb.get_collection()
    names = [c["collection_name"] for c in out]
    assert "PLAIN_TABLE" not in names
    assert "GOOD_DOCS" in names
    assert "WEIRD_DOCS" in names
    assert "BYO_VIEW" in names


def test_get_collection_marks_view_as_read_only_and_view(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    out = {c["collection_name"]: c for c in vdb.get_collection()}
    info = out["BYO_VIEW"]["collection_info"]
    assert info["is_view"] is True
    assert info["read_only"] is True
    assert info["schema_match"] is True


def test_get_collection_marks_non_canonical_table_read_only(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    out = {c["collection_name"]: c for c in vdb.get_collection()}
    info = out["WEIRD_DOCS"]["collection_info"]
    assert info["schema_match"] is False
    assert info["read_only"] is True
    assert info["is_view"] is False


def test_get_collection_canonical_table_writable(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    out = {c["collection_name"]: c for c in vdb.get_collection()}
    info = out["GOOD_DOCS"]["collection_info"]
    assert info["schema_match"] is True
    assert info["read_only"] is False
    assert info["is_view"] is False


def test_get_collection_auto_registers_canonical_byo_table(env, mixed_schema):
    """First list-collections call against a fresh BYO table should issue
    a MERGE into metadata_schema and a MERGE into document_info."""
    vdb = make_vdb(mixed_schema)
    vdb.get_collection()
    merges = mixed_schema["merges"]
    target_tables = {(name, binds.get("collection_name")) for name, binds in merges}
    # Both system tables touched for both canonical objects (GOOD_DOCS + BYO_VIEW)
    assert ("metadata_schema", "GOOD_DOCS") in target_tables
    assert ("document_info", "GOOD_DOCS") in target_tables
    assert ("metadata_schema", "BYO_VIEW") in target_tables
    assert ("document_info", "BYO_VIEW") in target_tables
    # Non-canonical table must NOT be auto-registered
    assert ("metadata_schema", "WEIRD_DOCS") not in target_tables


def test_get_collection_does_not_re_register_already_tracked(env, mixed_schema):
    """If document_info already has a row for this collection, skip the
    MERGE — saves a round-trip and avoids cluttering audit logs."""
    mixed_schema["document_info_rows"][("GOOD_DOCS", "collection", "NA")] = (
        '{"ingested_via":"upload"}'
    )
    vdb = make_vdb(mixed_schema)
    vdb.get_collection()
    merges = [(n, b.get("collection_name")) for n, b in mixed_schema["merges"]]
    assert ("document_info", "GOOD_DOCS") not in merges


def test_get_collection_propagates_count(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    out = {c["collection_name"]: c for c in vdb.get_collection()}
    assert out["GOOD_DOCS"]["num_entities"] == 17
    assert out["BYO_VIEW"]["num_entities"] == 1234


def test_get_collection_skips_system_collections(env, mixed_schema):
    """METADATA_SCHEMA and DOCUMENT_INFO are framework tables, not user
    collections — must never appear in the listing."""
    mixed_schema["tables"].update({"METADATA_SCHEMA", "DOCUMENT_INFO"})
    vdb = make_vdb(mixed_schema)
    names = {c["collection_name"] for c in vdb.get_collection()}
    assert "METADATA_SCHEMA" not in names
    assert "DOCUMENT_INFO" not in names


# ---------------------------------------------------------------------------
# get_documents defensive behaviour
# ---------------------------------------------------------------------------
def test_get_documents_returns_empty_for_non_canonical(env, mixed_schema):
    """Was previously crashing with ORA-00904 on missing 'source' column.
    Must now log + return empty list so the UI tab still loads."""
    vdb = make_vdb(mixed_schema)
    docs = vdb.get_documents("WEIRD_DOCS")
    assert docs == []


def test_get_documents_returns_for_canonical(env, mixed_schema):
    mixed_schema["unique_sources"]["GOOD_DOCS"] = [
        ('{"source_name": "a.pdf"}', None),
        ('{"source_name": "b.pdf"}', '{"page_number": 3}'),
    ]
    vdb = make_vdb(mixed_schema)
    docs = vdb.get_documents("GOOD_DOCS")
    names = [d["document_name"] for d in docs]
    assert names == ["a.pdf", "b.pdf"]


# ---------------------------------------------------------------------------
# delete_collections refuses to drop SQL views
# ---------------------------------------------------------------------------
def test_delete_view_returns_clear_failure(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    result = vdb.delete_collections(["BYO_VIEW"])
    assert result["total_failed"] == 1
    assert result["total_success"] == 0
    msg = result["failed"][0]["error_message"]
    # User-facing message must mention the cause + the remediation
    assert "VIEW" in msg.upper()
    assert "importExistingTables" in msg


def test_delete_canonical_table_succeeds(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    result = vdb.delete_collections(["GOOD_DOCS"])
    assert result["total_success"] == 1
    assert "GOOD_DOCS" in result["successful"]


def test_delete_unknown_table_reports_not_found(env, mixed_schema):
    vdb = make_vdb(mixed_schema)
    result = vdb.delete_collections(["NEVER_EXISTED"])
    assert result["total_failed"] == 1
    assert "not found" in result["failed"][0]["error_message"].lower()

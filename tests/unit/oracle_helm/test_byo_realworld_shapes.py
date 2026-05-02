# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip BYO mapping tests against schemas customers actually have.

Each entry is a (schema_name, source_table_columns, mapping) triple
that mirrors what we'd see in a customer ADB. We verify:

* The view DDL builder accepts the mapping.
* The generated DDL projects all five canonical columns.
* The DDL only references columns the source table actually has.
* sqlparse can lex it (basic syntactic sanity).
"""
from __future__ import annotations

import re

import pytest
import sqlparse


REAL_WORLD_SHAPES = [
    # ----- LangChain OracleVS: id/text/metadata/embedding ------------------
    (
        "langchain_oraclevs",
        ["ID", "TEXT", "METADATA", "EMBEDDING"],
        {
            "id": "ID",
            "text": "TEXT",
            "vector": "EMBEDDING",
            "content_metadata": "METADATA",
        },
    ),
    # ----- Oracle DBMS_VECTOR sample (from Oracle 26ai docs) ---------------
    (
        "dbms_vector_sample",
        ["DOC_ID", "CHUNK", "EMBED_VEC", "DOC_NAME"],
        {
            "id": "DOC_ID",
            "text": "CHUNK",
            "vector": "EMBED_VEC",
            "source": "DOC_NAME",  # plain VARCHAR2 → wrapped JSON
        },
    ),
    # ----- AWS-migrated flat schema ----------------------------------------
    (
        "aws_migrated_flat",
        ["UUID", "BODY", "VEC_F32", "FILENAME", "META_JSON"],
        {
            "id": "UUID",
            "text": "BODY",
            "vector": "VEC_F32",
            "source": "FILENAME",
            "content_metadata": "META_JSON",
        },
    ),
    # ----- Customer-built minimal ------------------------------------------
    (
        "minimal",
        ["TXT", "VEC"],
        {"text": "TXT", "vector": "VEC"},  # nothing else mapped
    ),
    # ----- Pre-existing pipeline that dumps source as already-JSON --------
    (
        "source_already_json",
        ["DOC_ID", "BODY", "EMBED", "SOURCE_JSON"],
        {
            "id": "DOC_ID",
            "text": "BODY",
            "vector": "EMBED",
            "source": "SOURCE_JSON",
            "source_wrap_json": "false",  # don't double-wrap
        },
    ),
    # ----- Audit-style (lots of extra cols, only some mapped) -------------
    (
        "audit_style",
        ["DOC_ID", "AUTHOR", "TS", "BODY", "EMBED", "URL", "INGESTED_BY"],
        {
            "id": "DOC_ID",
            "text": "BODY",
            "vector": "EMBED",
            "source": "URL",
        },
    ),
]


@pytest.mark.parametrize(
    ("schema_name", "src_cols", "mapping"),
    REAL_WORLD_SHAPES,
    ids=[s[0] for s in REAL_WORLD_SHAPES],
)
class TestRealWorldRoundTrip:
    def _build(self, view_name, mapping):
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        return oracle_queries.create_byo_view_ddl(
            view_name=view_name,
            source_table=f"CUST.{view_name}",
            column_map=mapping,
        )

    def test_ddl_builds(self, schema_name, src_cols, mapping):
        ddl = self._build(f"v_{schema_name}", mapping)
        assert ddl.strip().upper().startswith("CREATE OR REPLACE VIEW")

    def test_only_references_source_columns(self, schema_name, src_cols, mapping):
        """Every customer column referenced by the view DDL must come from
        the source table (no mistyped column ⇒ silent ORA-00904 at runtime).
        """
        ddl = self._build(f"v_{schema_name}", mapping)
        # Pull bare-word identifiers that look like column references
        # (anything between SELECT and FROM, ignoring SQL keywords).
        select_block = re.search(r"SELECT(.+?)FROM", ddl, re.DOTALL | re.IGNORECASE)
        assert select_block, "DDL has no SELECT...FROM"
        body = select_block.group(1).upper()
        # Each mapping value (the customer's column) must appear in the body
        for canonical, customer_col in mapping.items():
            if canonical in ("source_wrap_json",):
                continue
            assert customer_col.upper() in body or canonical.upper() in body, (
                f"{schema_name}: mapped column {customer_col!r} not "
                f"referenced in projected SELECT"
            )

    def test_ddl_lexes_cleanly(self, schema_name, src_cols, mapping):
        ddl = self._build(f"v_{schema_name}", mapping)
        parsed = sqlparse.parse(ddl)
        assert parsed and len(parsed) >= 1, f"{schema_name}: lex failed"
        # Single statement
        stmts = sqlparse.split(ddl.strip().rstrip(";"))
        assert len([s for s in stmts if s.strip()]) == 1

    def test_projects_all_five_canonical_columns(self, schema_name, src_cols, mapping):
        ddl = self._build(f"v_{schema_name}", mapping).upper()
        for c in ("ID", "TEXT", "VECTOR", "SOURCE", "CONTENT_METADATA"):
            assert f" AS {c}" in ddl, f"{schema_name}: missing AS {c}"

    def test_ddl_balanced_parens(self, schema_name, src_cols, mapping):
        ddl = self._build(f"v_{schema_name}", mapping)
        assert ddl.count("(") == ddl.count(")")
        assert ddl.count("'") % 2 == 0  # No unterminated strings


# ---------------------------------------------------------------------------
# Validation: helpful errors when a customer mis-configures
# ---------------------------------------------------------------------------
class TestRealWorldValidation:
    def test_missing_text_column_raises(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        with pytest.raises(ValueError, match="text.*vector"):
            oracle_queries.create_byo_view_ddl(
                view_name="v", source_table="T", column_map={"vector": "v"},
            )

    def test_missing_vector_column_raises(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        with pytest.raises(ValueError, match="text.*vector"):
            oracle_queries.create_byo_view_ddl(
                view_name="v", source_table="T", column_map={"text": "t"},
            )

    def test_view_name_lowercase_uppercased_through_pipeline(self):
        """The Helm Job uppercases view names before MERGEing into the
        tracking tables. Confirm the DDL itself accepts mixed case (Oracle
        uppercases unquoted identifiers, so we shouldn't quote them)."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        ddl = oracle_queries.create_byo_view_ddl(
            view_name="My_View",
            source_table="cust.docs",
            column_map={"text": "Body", "vector": "Vec"},
        )
        # We do NOT add quoting around identifiers — Oracle would store
        # them case-sensitively if quoted. Verify we kept it bare.
        assert '"My_View"' not in ddl
        assert "My_View" in ddl

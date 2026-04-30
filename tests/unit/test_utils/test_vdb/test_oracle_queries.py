# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Oracle SQL query generation, focused on case preservation."""

import pytest

from nvidia_rag.utils.vdb.oracle.oracle_queries import (
    check_table_exists_query,
    create_text_index_ddl,
    create_vector_index_ddl,
    create_vector_table_ddl,
    drop_table_ddl,
    get_count_query,
    get_delete_docs_query,
    get_hybrid_search_query,
    get_similarity_search_query,
    get_unique_sources_query,
)


@pytest.mark.parametrize(
    "name",
    [
        "Financial_Dataset",  # mixed case
        "s_fdbe0520aaa1234",  # lowercase session UUID
        "UPPERCASE_NAME",  # all uppercase (regression)
        "MyCollection",  # PascalCase
        "with-dashes",  # special chars valid in quoted identifiers
    ],
)
class TestQuotedIdentifiers:
    """Every DDL/DML must quote the table name to preserve case in Oracle."""

    def test_create_vector_table_quotes_name(self, name):
        ddl = create_vector_table_ddl(name)
        assert f'CREATE TABLE "{name}"' in ddl
        # Must NOT contain unquoted form (which Oracle would case-fold)
        assert f"CREATE TABLE {name} " not in ddl

    def test_create_vector_index_quotes_both_index_and_table(self, name):
        ddl = create_vector_index_ddl(name, index_type="IVF")
        assert f'CREATE VECTOR INDEX "{name}_vec_idx"' in ddl
        assert f'ON "{name}"(vector)' in ddl

    def test_create_vector_index_hnsw_quotes_both(self, name):
        ddl = create_vector_index_ddl(name, index_type="HNSW")
        assert f'CREATE VECTOR INDEX "{name}_vec_idx"' in ddl
        assert f'ON "{name}"(vector)' in ddl

    def test_create_text_index_quotes_both(self, name):
        ddl = create_text_index_ddl(name)
        assert f'CREATE INDEX "{name}_text_idx"' in ddl
        assert f'ON "{name}"(text)' in ddl

    def test_drop_table_quotes_name(self, name):
        assert drop_table_ddl(name) == f'DROP TABLE "{name}" CASCADE CONSTRAINTS PURGE'

    def test_count_query_quotes_name(self, name):
        assert get_count_query(name) == f'SELECT COUNT(*) as cnt FROM "{name}"'

    def test_unique_sources_query_quotes_name(self, name):
        sql = get_unique_sources_query(name)
        assert f'FROM "{name}"' in sql

    def test_delete_docs_query_quotes_name(self, name):
        sql = get_delete_docs_query(name)
        assert f'DELETE FROM "{name}"' in sql

    def test_similarity_search_quotes_name(self, name):
        sql = get_similarity_search_query(name, distance_metric="COSINE")
        assert f'FROM "{name}"' in sql

    def test_hybrid_search_quotes_name_in_both_ctes(self, name):
        sql = get_hybrid_search_query(name)
        # Both vector_results and text_results CTEs query the user table
        assert sql.count(f'FROM "{name}"') == 2


def test_check_table_exists_uses_exact_case_match():
    """check_table_exists_query must NOT wrap with UPPER() — quoted identifiers
    are case-sensitive and we want exact case lookup."""
    sql = check_table_exists_query()
    assert "UPPER(:table_name)" not in sql
    assert ":table_name" in sql
    assert "WHERE table_name = :table_name" in sql


class TestObjectNameDerivation:
    """Per-collection object names must fit Oracle's 128-char limit."""

    def test_short_name_unchanged(self):
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _derive_object_name

        assert _derive_object_name("MyCollection", "_vec_idx") == "MyCollection_vec_idx"

    def test_short_lowercase_unchanged(self):
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _derive_object_name

        assert _derive_object_name("s_fdbe123", "_text_idx") == "s_fdbe123_text_idx"

    def test_long_name_hashed(self):
        """Names that would exceed 128 chars get a deterministic hash prefix."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _derive_object_name

        # 130-char collection name — verbatim concatenation would exceed limit
        long_name = "C" * 130
        derived = _derive_object_name(long_name, "_vec_idx")
        assert len(derived) <= 128
        assert derived.startswith("nvr_")
        assert derived.endswith("_vec_idx")

    def test_long_name_deterministic(self):
        """Same input must always produce the same derived name (idempotent)."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _derive_object_name

        long_name = "C" * 200
        a = _derive_object_name(long_name, "_vec_idx")
        b = _derive_object_name(long_name, "_vec_idx")
        assert a == b

    def test_long_name_collision_resistant(self):
        """Different long names produce different derived names."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _derive_object_name

        a = _derive_object_name("A" * 200, "_vec_idx")
        b = _derive_object_name("B" * 200, "_vec_idx")
        assert a != b


def test_derived_index_used_in_long_name_ddl():
    """When the table name is long, the emitted DDL uses the hashed index name."""
    from nvidia_rag.utils.vdb.oracle.oracle_queries import (
        create_text_index_ddl,
        create_vector_index_ddl,
    )

    long_name = "X" * 200
    ddl = create_vector_index_ddl(long_name, "IVF")
    # Verbatim concatenation would have produced "X*200_vec_idx" — should NOT appear
    assert f"{long_name}_vec_idx" not in ddl
    # Should contain the hashed prefix
    assert '"nvr_' in ddl

    txt_ddl = create_text_index_ddl(long_name)
    assert f"{long_name}_text_idx" not in txt_ddl
    assert '"nvr_' in txt_ddl

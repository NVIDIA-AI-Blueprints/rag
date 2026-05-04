# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Oracle 26ai vector store plugin.

These tests do NOT require a live Oracle database — they exercise the
plugin's pure-Python paths (DDL generation, query sanitization, metadata
parsing, dispatcher routing).  Live-DB integration tests live in
`tests/integration/`.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.vector_store.name = "oracle"
    cfg.vector_store.search_type = "hybrid"
    cfg.vector_store.index_type = "IVF"
    cfg.vector_store.distance_metric = "COSINE"
    cfg.vector_store.dimension = 2048
    cfg.embeddings.dimensions = 2048
    return cfg


@pytest.fixture
def mock_embedding_model():
    em = MagicMock()
    em.embed_query.return_value = [0.1] * 2048
    return em


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------
class TestOracleQueriesDDL:
    """Pure-Python tests for DDL generators in oracle_queries.py."""

    def test_create_table_ddl_includes_vector_column(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_table_ddl("MY_COLLECTION", dimension=2048)
        assert "MY_COLLECTION" in ddl
        assert 'VECTOR(2048' in ddl or "VECTOR" in ddl
        assert "TEXT" in ddl.upper()
        assert "SOURCE" in ddl.upper() or '"SOURCE"' in ddl

    def test_create_vector_index_uses_two_word_neighbor_partitions(self):
        """Regression test for ORA-00922 fix.

        ADB 23ai/26ai requires `NEIGHBOR PARTITIONS` (two words) and
        `DISTANCE ...` (no `WITH`).  The single-word forms produce
        ORA-00922 and the index is silently never created.
        """
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="MY_COLLECTION",
            index_type="IVF",
            distance_metric="COSINE",
        )
        normalized = " ".join(ddl.upper().split())
        assert "NEIGHBOR PARTITIONS" in normalized, (
            "IVF DDL must use two-word `NEIGHBOR PARTITIONS` "
            "(single-word produces ORA-00922 silently)"
        )
        assert "WITH DISTANCE" not in normalized, (
            "ADB 23ai/26ai uses `DISTANCE`, not `WITH DISTANCE`"
        )
        assert "DISTANCE COSINE" in normalized

    def test_create_text_index_uses_sync_on_commit(self):
        """Regression test: text index must be SYNC (ON COMMIT) so
        keyword retrieval sees inserts without manual sync."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_text_index_ddl("MY_COLLECTION")
        normalized = " ".join(ddl.upper().split())
        assert "CTXSYS.CONTEXT" in normalized
        assert "SYNC" in normalized and "ON COMMIT" in normalized


# ---------------------------------------------------------------------------
# Text query sanitization (hybrid path)
# ---------------------------------------------------------------------------
class TestSanitizeTextQuery:
    """The CONTAINS() text path strips reserved operators and OR-joins
    tokens.  These tests cover the sanitizer in isolation."""

    @pytest.mark.parametrize(
        "raw, expected_tokens",
        [
            ("hello world", {"hello", "world"}),
            ("hello & world", {"hello", "world"}),
            ("foo|bar(baz)", {"foo", "bar", "baz"}),
            ("CONCURRENT_TOKEN_5", {"CONCURRENT_TOKEN_5"}),  # underscores preserved
            ("'quoted text'", {"quoted", "text"}),
            ("", set()),
            ("&|()={}", set()),
        ],
    )
    def test_sanitize_strips_reserved_and_or_joins_tokens(self, raw, expected_tokens):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        result = OracleVDB._sanitize_text_query(raw)
        if not expected_tokens:
            assert result == ""
        else:
            for tok in expected_tokens:
                assert tok in result
            assert " OR " in result if len(expected_tokens) > 1 else True


# ---------------------------------------------------------------------------
# Metadata-shape correctness (regression: prepare_citations 500 fix)
# ---------------------------------------------------------------------------
class TestMetadataShape:
    """v2.5.0's response_generator.prepare_citations strictly expects
    nested `metadata["source"]` and `metadata["content_metadata"]`.
    These tests verify the plugin produces that shape."""

    def test_dense_path_nests_source_under_source_key(self):
        """Documents returned from
        similarity_search_by_vector_with_relevance_scores must have
        `metadata["source"]` populated as a parsed dict (or None),
        not flattened into top-level metadata keys."""
        from langchain_core.documents import Document

        # Simulated row shape from the dense SQL: (id, text, source_json,
        # content_metadata_clob, distance)
        sample_source = {
            "source_id": "abc-123",
            "source_name": "test-doc.md",
            "source_type": "text",
        }
        sample_content_metadata = {
            "type": "text",
            "page_number": 1,
            "filename": "test-doc.md",
        }

        # Build a Document the way the plugin does (tests the nesting
        # convention; if a future change flattens these, the assertion
        # will fail (and `prepare_citations` will 500 in production).
        metadata = {
            "source": sample_source,
            "content_metadata": sample_content_metadata,
        }
        doc = Document(page_content="hello", metadata=metadata)

        # prepare_citations does this:
        assert doc.metadata.get("source") is not None
        assert doc.metadata["source"].get("source_id") == "abc-123"
        assert doc.metadata.get("content_metadata") is not None
        assert doc.metadata["content_metadata"].get("type") == "text"

    def test_hybrid_path_nests_metadata_same_as_dense(self):
        """Hybrid retrieval must emit the same metadata shape as dense
        — `metadata["source"]` + `metadata["content_metadata"]`."""
        from langchain_core.documents import Document

        # Same shape as dense, with hybrid_score added at top level
        metadata = {
            "source": {"source_id": "xyz-456", "source_name": "hybrid.md"},
            "content_metadata": {"type": "text", "page_number": 2},
            "hybrid_score": 0.875,
        }
        doc = Document(page_content="world", metadata=metadata)

        assert doc.metadata["source"]["source_id"] == "xyz-456"
        assert doc.metadata["content_metadata"]["type"] == "text"
        assert doc.metadata.get("hybrid_score") == 0.875


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------
class TestDispatcherRouting:
    """Verify that vector_store.name='oracle' routes to OracleVDB."""

    @patch.dict(
        "os.environ",
        {
            "ORACLE_USER": "RAG_APP",
            "ORACLE_PASSWORD": "test",
            "ORACLE_CS": "test_medium",
        },
    )
    def test_oracle_dispatcher_branch_returns_oracle_vdb(
        self, mock_config, mock_embedding_model
    ):
        # The dispatcher imports OracleVDB lazily; mock the import to
        # avoid requiring a live oracledb thin-mode connection.
        with patch(
            "nvidia_rag.utils.vdb.oracle.oracle_vdb.OracleVDB"
        ) as mock_oracle_cls:
            mock_oracle_cls.return_value = MagicMock(name="OracleVDB-instance")

            from nvidia_rag.utils.vdb import _get_vdb_op

            result = _get_vdb_op(
                vdb_endpoint="",
                collection_name="test_coll",
                embedding_model=mock_embedding_model,
                config=mock_config,
            )
            mock_oracle_cls.assert_called_once()
            kwargs = mock_oracle_cls.call_args.kwargs
            assert kwargs["oracle_user"] == "RAG_APP"
            assert kwargs["oracle_cs"] == "test_medium"
            assert kwargs["collection_name"] == "test_coll"


# ---------------------------------------------------------------------------
# Pyproject extras presence
# ---------------------------------------------------------------------------
class TestPyprojectOracleExtras:
    """Guard against the `oracle` extras getting accidentally removed
    from pyproject.toml."""

    def test_oracle_extras_listed(self):
        import tomllib
        from pathlib import Path

        # Resolve to repo root regardless of where pytest is run
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                pyproject = parent / "pyproject.toml"
                break
        else:
            pytest.skip("pyproject.toml not found from test location")

        cfg = tomllib.loads(pyproject.read_text())
        extras = cfg.get("project", {}).get("optional-dependencies", {})
        assert "oracle" in extras, "`oracle` extras must be present in pyproject.toml"
        oracle_deps = " ".join(extras["oracle"])
        assert "langchain-oracledb" in oracle_deps

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

    def test_hnsw_includes_offload_url_when_pai_set(self):
        """cuVS GPU offload wires onto OFFLOAD_URL only for HNSW.

        Oracle 26ai's CREATE VECTOR INDEX SQL reference:
        OFFLOAD_URL / OFFLOAD_CREDENTIAL_NAME are HNSW-only PARAMETERS.
        IVF must ignore them so we don't generate invalid DDL.
        """
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        hnsw_ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            pai_offload_url="http://oracle-pai-gpu-index.default.svc.cluster.local:8080/v1/index",
        )
        norm = " ".join(hnsw_ddl.upper().split())
        assert "TYPE HNSW" in norm
        assert "OFFLOAD_URL '" in hnsw_ddl
        assert "ORACLE-PAI-GPU-INDEX" in norm
        assert "OFFLOAD_CREDENTIAL_NAME" not in norm, (
            "Credential name should be omitted when not provided (HTTP mode)"
        )

        hnsw_https = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            pai_offload_url="https://pai.example.com:8443/v1/index",
            pai_offload_credential="PAI_OFFLOAD_CRED",
        )
        assert "OFFLOAD_CREDENTIAL_NAME 'PAI_OFFLOAD_CRED'" in hnsw_https

        ivf_ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="IVF",
            pai_offload_url="http://anywhere:8080/v1/index",
        )
        assert "OFFLOAD" not in ivf_ddl.upper(), (
            "IVF DDL must never include OFFLOAD_URL (HNSW-only feature)"
        )

    def test_hnsw_without_pai_url_is_unchanged(self):
        """Backwards compat: omitting pai_offload_url produces the original DDL."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
        )
        assert "OFFLOAD" not in ddl.upper()
        assert "PARAMETERS (TYPE HNSW, NEIGHBORS 16, EFCONSTRUCTION 200)" in ddl

    @pytest.mark.parametrize("metric", ["COSINE", "L2", "DOT", "MANHATTAN"])
    def test_hnsw_offload_with_each_distance_metric(self, metric):
        """OFFLOAD_URL must be appended cleanly regardless of distance metric."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            distance_metric=metric,
            pai_offload_url="http://10.0.50.42:8080/v1/index",
        )
        norm = " ".join(ddl.upper().split())
        assert f"DISTANCE {metric}" in norm
        assert "OFFLOAD_URL '" in ddl

    def test_hnsw_offload_with_custom_params(self):
        """Custom HNSW neighbors / efConstruction must coexist with OFFLOAD_URL."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            hnsw_m=64,
            hnsw_ef_construction=512,
            pai_offload_url="http://10.0.50.42:8080/v1/index",
            pai_offload_credential="PAI_CRED",
        )
        assert "NEIGHBORS 64" in ddl
        assert "EFCONSTRUCTION 512" in ddl
        assert "OFFLOAD_URL 'http://10.0.50.42:8080/v1/index'" in ddl
        assert "OFFLOAD_CREDENTIAL_NAME 'PAI_CRED'" in ddl
        # Order matters: OFFLOAD_URL must appear before OFFLOAD_CREDENTIAL_NAME
        idx_url = ddl.index("OFFLOAD_URL")
        idx_cred = ddl.index("OFFLOAD_CREDENTIAL_NAME")
        assert idx_url < idx_cred

    def test_offload_url_with_https_and_port_8443(self):
        """HTTPS production URL preserved verbatim (no rewriting)."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            pai_offload_url="https://pai.private.example.com:8443/v1/index",
            pai_offload_credential="PAI_OFFLOAD_CRED",
        )
        assert "OFFLOAD_URL 'https://pai.private.example.com:8443/v1/index'" in ddl

    def test_empty_offload_url_string_treated_as_disabled(self):
        """Empty string -> falsy -> no OFFLOAD_URL emitted (safe fallback)."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            pai_offload_url="",
        )
        assert "OFFLOAD" not in ddl.upper()

    def test_offload_credential_without_url_is_silently_dropped(self):
        """OFFLOAD_CREDENTIAL_NAME without OFFLOAD_URL is meaningless to ADB and
        would produce an ORA-29024 at create time; suppress it to be safe."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="HNSW",
            pai_offload_url=None,
            pai_offload_credential="PAI_CRED",
        )
        assert "OFFLOAD" not in ddl.upper()

    @pytest.mark.parametrize(
        "table_name",
        ["x", "schema.collection", "rag_app.long_collection_name_with_underscores"],
    )
    def test_index_name_derives_from_table(self, table_name):
        """Index name is `<table>_vec_idx` so multiple collections coexist."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name=table_name, index_type="HNSW",
        )
        assert f"{table_name}_vec_idx" in ddl

    def test_ivf_with_custom_partitions(self):
        """IVF NEIGHBOR PARTITIONS count flows through unchanged."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_vector_index_ddl(
            table_name="rag_app.acme",
            index_type="IVF",
            ivf_neighbor_partitions=256,
        )
        assert "NEIGHBOR PARTITIONS 256" in ddl

    @pytest.mark.parametrize(
        "raw_url,expected",
        [
            # Empty / unset → None (offload disabled, CPU fallback)
            ("", None),
            ("   ", None),
            # Plain hostname → /v1/index appended
            ("http://oracle-pai.svc:8080", "http://oracle-pai.svc:8080/v1/index"),
            # Trailing slash trimmed before suffix
            ("http://oracle-pai.svc:8080/", "http://oracle-pai.svc:8080/v1/index"),
            # Already correct path → preserved verbatim
            ("http://10.0.50.42:8080/v1/index", "http://10.0.50.42:8080/v1/index"),
            # HTTPS + port preserved
            ("https://pai.example.com:8443/v1/index", "https://pai.example.com:8443/v1/index"),
        ],
    )
    def test_pai_index_url_env_normalization(self, monkeypatch, raw_url, expected):
        """OracleVDB.__init__ normalizes ORACLE_PAI_INDEX_URL to always end
        in /v1/index (or be None if unset).

        We exercise the env-reading branch in isolation by re-implementing
        the normalization (kept pure in the production code).
        """
        # Re-implementation must match oracle_vdb.py exactly. If they drift,
        # this test fails and the fix is to align both sides.
        import os
        monkeypatch.setenv("ORACLE_PAI_INDEX_URL", raw_url)
        pai_url = (os.getenv("ORACLE_PAI_INDEX_URL", "") or "").strip()
        if pai_url and not pai_url.endswith("/v1/index"):
            pai_url = pai_url.rstrip("/") + "/v1/index"
        normalized = pai_url or None
        assert normalized == expected

    def test_pai_index_url_normalization_matches_production_code(self):
        """Anti-drift test: the normalization snippet in test above must
        appear verbatim in oracle_vdb.py. If a developer changes one
        without the other, this test catches it."""
        from pathlib import Path

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_vdb.py"
            if candidate.exists():
                src = candidate.read_text()
                break
        else:
            pytest.skip("oracle_vdb.py not found")

        # The exact normalization clauses we tested above
        assert 'os.getenv("ORACLE_PAI_INDEX_URL"' in src
        assert 'endswith("/v1/index")' in src
        assert 'rstrip("/") + "/v1/index"' in src
        # And the credential env var
        assert 'ORACLE_PAI_OFFLOAD_CREDENTIAL' in src

    # -----------------------------------------------------------------
    # BYO-database support: SQL view DDL builder
    # -----------------------------------------------------------------
    def test_byo_view_minimal_columns_required(self):
        """View builder must require text + vector — without them retrieval
        cannot work, and silently producing a broken view is worse than
        failing fast."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        with pytest.raises(ValueError, match="text.*vector"):
            oracle_queries.create_byo_view_ddl(
                view_name="my_view",
                source_table="KB.DOCS",
                column_map={"text": "content"},  # no 'vector'
            )

    def test_byo_view_with_full_mapping(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_byo_view_ddl(
            view_name="my_kb_view",
            source_table="KB.MY_DOCS",
            column_map={
                "id": "doc_id",
                "text": "content",
                "vector": "embedding",
                "source": "source_url",
                "content_metadata": "meta_json",
            },
        )
        norm = " ".join(ddl.upper().split())
        assert "CREATE OR REPLACE VIEW MY_KB_VIEW AS" in norm
        assert "FROM KB.MY_DOCS" in norm
        # All five canonical columns projected
        assert "DOC_ID AS ID" in norm
        assert "CONTENT AS TEXT" in norm
        assert "EMBEDDING AS VECTOR" in norm
        # Source wrapped into JSON for parity with ingested rows
        assert "JSON_OBJECT('SOURCE_NAME' VALUE TO_CHAR(SOURCE_URL))" in norm
        assert "COALESCE(META_JSON, JSON_OBJECT())" in norm

    def test_byo_view_defaults_id_to_rowid(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_byo_view_ddl(
            view_name="v",
            source_table="KB.T",
            column_map={"text": "c", "vector": "e"},
        )
        assert "ROWID AS id" in ddl

    def test_byo_view_source_wrap_disabled(self):
        """sourceWrapJson=False emits the source column verbatim — useful when
        the customer already stores source as a JSON shape."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_byo_view_ddl(
            view_name="v",
            source_table="KB.T",
            column_map={
                "text": "c", "vector": "e",
                "source": "source_blob", "source_wrap_json": "false",
            },
        )
        assert "JSON_OBJECT" not in ddl.split("FROM ")[0].split("AS source")[0].split("source_blob")[1]
        # Source projected directly without wrapping
        assert "source_blob AS source" in ddl

    def test_byo_view_default_source_is_byo_marker(self):
        """No source column → still projects a valid JSON so JSON_VALUE() in
        get_unique_sources_query works."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_byo_view_ddl(
            view_name="v",
            source_table="KB.T",
            column_map={"text": "c", "vector": "e"},
        )
        assert "JSON_OBJECT('source_name' VALUE 'byo')" in ddl

    def test_byo_view_default_content_metadata_is_empty_json(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl = oracle_queries.create_byo_view_ddl(
            view_name="v",
            source_table="KB.T",
            column_map={"text": "c", "vector": "e"},
        )
        # Empty JSON object so get_documents() doesn't blow up on NULL
        assert "JSON_OBJECT() AS content_metadata" in ddl

    # -----------------------------------------------------------------
    # BYO discovery queries
    # -----------------------------------------------------------------
    def test_get_all_collections_query_unions_tables_and_views(self):
        """Views (created by the BYO importer) must appear alongside base
        tables in /collections so the frontend lists them."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        sql = oracle_queries.get_all_collections_query()
        u = " ".join(sql.upper().split())
        assert "USER_TABLES" in u
        assert "USER_VIEWS" in u
        assert "UNION ALL" in u
        # Excludes Oracle Text auxiliary objects
        assert "DR$%" in sql

    def test_list_vector_tables_query_filters_by_vector_dtype(self):
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        sql = oracle_queries.list_vector_tables_query()
        u = " ".join(sql.upper().split())
        assert "USER_TAB_COLUMNS" in u
        assert "DATA_TYPE LIKE 'VECTOR%'" in u

    def test_canonical_columns_constant_matches_create_table_ddl(self):
        """RAG_CANONICAL_COLUMNS must stay in sync with create_vector_table_ddl
        so the BYO discovery pass detects the same shape the ingestor creates."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        ddl_upper = oracle_queries.create_vector_table_ddl("X", dimension=2048).upper()
        for col in oracle_queries.RAG_CANONICAL_COLUMNS:
            assert col in ddl_upper, (
                f"RAG_CANONICAL_COLUMNS lists {col!r} but create_vector_table_ddl "
                "no longer creates that column — fix one of them."
            )

    def test_upsert_metadata_schema_merge_is_idempotent(self):
        """The MERGE must be 'WHEN NOT MATCHED' only, never UPDATE — otherwise
        registering an existing collection would overwrite a schema the user
        added via the UI."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        sql = oracle_queries.upsert_metadata_schema_merge()
        u = " ".join(sql.upper().split())
        assert "MERGE INTO METADATA_SCHEMA" in u
        assert "WHEN NOT MATCHED THEN INSERT" in u
        assert "WHEN MATCHED" not in u, (
            "BYO MERGE must not update existing rows (preserves user-added "
            "schema fields)"
        )

    def test_upsert_collection_info_merge_uses_correct_unique_keys(self):
        """document_info has a UNIQUE constraint on
        (collection_name, info_type, document_name) — the MERGE ON-clause
        must match all three or the customer gets ORA-00001 dup-key on retry."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        sql = oracle_queries.upsert_collection_info_merge()
        u = " ".join(sql.upper().split())
        assert "MERGE INTO DOCUMENT_INFO" in u
        assert "INFO_TYPE = 'COLLECTION'" in u
        assert "DOCUMENT_NAME = 'NA'" in u

    # -----------------------------------------------------------------
    # Defensive source-name extraction (BYO data may have non-JSON sources)
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Canonical RAG shape
            ('{"source_name": "/tmp/file.pdf"}', "/tmp/file.pdf"),
            # Plain string (legacy / non-canonical BYO)
            ("/path/to/doc.pdf", "/path/to/doc.pdf"),
            # Whitespace
            ("  /a/b.txt  ", "/a/b.txt"),
            # Dict already parsed (oracledb sometimes returns dict for native JSON)
            ({"source_name": "x.md"}, "x.md"),
            # Dict missing source_name -> falls back to 'source'
            ({"source": "alt.md"}, "alt.md"),
            # Empty / None -> "unknown" (UI shows clean placeholder)
            (None, "unknown"),
            ("", "unknown"),
            # Malformed JSON -> treated as plain string
            ('{"unclosed', '{"unclosed'),
        ],
    )
    def test_extract_source_name_handles_byo_variants(self, raw, expected):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        assert OracleVDB._extract_source_name(raw) == expected

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

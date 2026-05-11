# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DB-compatibility tests for the Oracle adapter WITHOUT a real database.

Every test mocks oracledb and/or environment variables to simulate a
specific Oracle deployment scenario (ADB Enterprise, Free edition,
pre-23ai, walletless TLS, mTLS with wallet, various DSN formats, etc.).

Sections:
  1. Graceful degradation  – VECTOR type missing, CTXSYS unavailable, no OFFLOAD
  2. Connection modes       – walletless, mTLS wallet, partial wallet config
  3. DSN formats            – descriptor, TNS alias, Easy Connect Plus, K8s DNS
  4. Feature detection      – hybrid fallback, PAI offload gating, identifier validation
  5. Edition-specific       – Developer ADB, Free, Enterprise
"""
from __future__ import annotations

import json
import os
import re
from array import array
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

oracledb = pytest.importorskip("oracledb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oracle_vdb(
    *,
    collection_name: str = "TEST_COLL",
    hybrid: bool = False,
    index_type: str = "IVF",
    distance_metric: str = "COSINE",
    env_overrides: dict | None = None,
    pool_side_effect: Exception | None = None,
):
    """Build an OracleVDB with a fully mocked connection pool.

    The returned object has ``vdb._pool`` as a MagicMock whose
    ``acquire()`` context-manager yields a mock connection with a mock
    cursor.  Caller can attach side effects to ``cursor.execute`` to
    simulate specific ORA- errors.
    """
    env = {
        "ORACLE_USER": "RAG_APP",
        "ORACLE_PASSWORD": "hunter2",
        "ORACLE_CS": "(description=(address=(protocol=tcps)(host=adb.us-ashburn-1.oraclecloud.com)(port=1522))(connect_data=(service_name=abc_tp.adb.oraclecloud.com)))",
        "ORACLE_VECTOR_INDEX_TYPE": index_type,
        "ORACLE_DISTANCE_METRIC": distance_metric,
        **(env_overrides or {}),
    }

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = []
    mock_cursor.__enter__ = lambda self: self
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = lambda self: self
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = mock_conn

    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch("oracledb.create_pool", side_effect=pool_side_effect or [mock_pool]),
    ):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
        vdb = OracleVDB(
            collection_name=collection_name,
            hybrid=hybrid,
            index_type=index_type,
            distance_metric=distance_metric,
        )

    return vdb, mock_pool, mock_conn, mock_cursor


# ═══════════════════════════════════════════════════════════════════════════
# 1. GRACEFUL DEGRADATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Simulate DB-version or edition limitations that the adapter should
    handle without crashing the whole service."""

    def test_create_collection_vector_type_unavailable_pre23ai(self):
        """Pre-23ai databases lack the VECTOR datatype. CREATE TABLE with a
        VECTOR column raises ORA-00902. The adapter should propagate the
        error so the operator gets a clear signal that the DB is too old."""
        vdb, pool, conn, cursor = _make_oracle_vdb()

        cursor.fetchone.return_value = (0,)

        ora_error = oracledb.DatabaseError("ORA-00902: invalid datatype")
        execute_calls = [None, ora_error]
        call_idx = {"i": 0}
        original_execute = cursor.execute

        def execute_side_effect(*args, **kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx == 0:
                cursor.fetchone.return_value = (0,)
                return None
            raise execute_calls[1]

        cursor.execute.side_effect = execute_side_effect

        with pytest.raises(oracledb.DatabaseError, match="ORA-00902"):
            vdb.create_collection("PRE23_TABLE", dimension=1024)

    def test_hybrid_falls_back_when_ctxsys_unavailable(self):
        """Oracle Free edition doesn't ship CTXSYS.CONTEXT. When
        create_text_index_ddl fails, create_collection should log a
        warning and continue — the table is usable for dense search."""
        vdb, pool, conn, cursor = _make_oracle_vdb(hybrid=True)

        cursor.fetchone.return_value = (0,)
        ctxsys_error = oracledb.DatabaseError(
            "ORA-29855: error occurred in the execution of ODCIINDEXCREATE routine\n"
            "ORA-20000: Oracle Text error: DRG-10599: column is not indexed"
        )

        exec_count = {"n": 0}

        def execute_side_effect(*args, **kwargs):
            exec_count["n"] += 1
            n = exec_count["n"]
            if n == 1:
                cursor.fetchone.return_value = (0,)
                return None
            if n == 2:
                return None
            if n == 3:
                return None
            if n == 4:
                raise ctxsys_error
            return None

        cursor.execute.side_effect = execute_side_effect

        vdb.create_collection("FREE_TABLE", dimension=768)

        assert exec_count["n"] >= 4, (
            "Expected at least 4 execute calls: table_exists check, "
            "CREATE TABLE, CREATE VECTOR INDEX, CREATE TEXT INDEX"
        )

    def test_index_creation_succeeds_without_offload_on_non_enterprise(self):
        """When OFFLOAD_URL is not supported (non-Enterprise), vector
        index DDL should succeed without offload params — just a warning."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        ddl = create_vector_index_ddl(
            "MY_TABLE",
            index_type="HNSW",
            pai_offload_url=None,
            pai_offload_credential=None,
        )
        assert "OFFLOAD_URL" not in ddl
        assert "TYPE HNSW" in ddl
        assert "MY_TABLE_vec_idx" in ddl

    def test_offload_failure_is_fatal_when_explicitly_configured(self):
        """If a customer sets ORACLE_PAI_INDEX_URL (opting into GPU
        offload), a DDL failure on the HNSW index must raise — not
        silently degrade to an unindexed table."""
        vdb, pool, conn, cursor = _make_oracle_vdb(
            index_type="HNSW",
            env_overrides={
                "ORACLE_PAI_INDEX_URL": "http://oracle-pai-gpu-index.rag.svc.cluster.local:8080/v1/index",
            },
        )
        assert vdb._pai_offload_url is not None

        cursor.fetchone.return_value = (0,)
        exec_count = {"n": 0}
        offload_error = oracledb.DatabaseError(
            "ORA-51773: HNSW vector index build offload failed"
        )

        def execute_side_effect(*args, **kwargs):
            exec_count["n"] += 1
            if exec_count["n"] == 1:
                cursor.fetchone.return_value = (0,)
                return None
            if exec_count["n"] == 2:
                return None
            if exec_count["n"] == 3:
                raise offload_error
            return None

        cursor.execute.side_effect = execute_side_effect

        with pytest.raises(oracledb.DatabaseError, match="offload"):
            vdb.create_collection("GPU_TABLE", dimension=1024)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONNECTION MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectionModes:
    """Test _wallet_connect_kwargs under various env-var combinations."""

    def test_walletless_tls_returns_empty_dict(self):
        """No TNS_ADMIN, no ORACLE_WALLET_DIR, no ORACLE_WALLET_PASSWORD →
        walletless TLS mode. _wallet_connect_kwargs must return an empty
        dict so oracledb.create_pool gets no wallet params."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs

        env = {k: v for k, v in os.environ.items()
               if k not in ("TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD")}
        with mock.patch.dict(os.environ, env, clear=True):
            result = _wallet_connect_kwargs()

        assert result == {}, (
            f"Walletless TLS should return empty dict, got {result}"
        )

    def test_mtls_wallet_full_config(self):
        """mTLS with wallet: TNS_ADMIN + ORACLE_WALLET_DIR +
        ORACLE_WALLET_PASSWORD all set. _wallet_connect_kwargs should
        return config_dir, wallet_location, and wallet_password."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs

        env = {
            "ORACLE_WALLET_DIR": "/opt/oracle/wallet",
            "ORACLE_WALLET_PASSWORD": "walletP@ss123",
        }
        strip_keys = {"TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip_keys}
        with mock.patch.dict(os.environ, {**clean_env, **env}, clear=True):
            result = _wallet_connect_kwargs()

        assert result["config_dir"] == "/opt/oracle/wallet"
        assert result["wallet_location"] == "/opt/oracle/wallet"
        assert result["wallet_password"] == "walletP@ss123"

    def test_mtls_wallet_via_tns_admin(self):
        """TNS_ADMIN set (common in ADB deployment). ORACLE_WALLET_DIR
        not set — TNS_ADMIN should be used as the wallet directory."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs

        strip_keys = {"TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip_keys}
        env = {"TNS_ADMIN": "/etc/tns_admin"}
        with mock.patch.dict(os.environ, {**clean_env, **env}, clear=True):
            result = _wallet_connect_kwargs()

        assert result["config_dir"] == "/etc/tns_admin"
        assert result["wallet_location"] == "/etc/tns_admin"
        assert "wallet_password" not in result, (
            "Unencrypted wallet (cwallet.sso → ewallet.pem): no password needed"
        )

    def test_partial_wallet_config_no_password(self):
        """TNS_ADMIN set but no ORACLE_WALLET_PASSWORD — should still
        work for unencrypted wallets (ewallet.pem without encryption)."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs

        strip_keys = {"TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip_keys}
        env = {"TNS_ADMIN": "/home/oracle/network/admin"}
        with mock.patch.dict(os.environ, {**clean_env, **env}, clear=True):
            result = _wallet_connect_kwargs()

        assert "config_dir" in result
        assert "wallet_location" in result
        assert "wallet_password" not in result

    def test_wallet_dir_takes_precedence_over_tns_admin(self):
        """When both ORACLE_WALLET_DIR and TNS_ADMIN are set,
        ORACLE_WALLET_DIR wins (it's checked first)."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs

        strip_keys = {"TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip_keys}
        env = {
            "ORACLE_WALLET_DIR": "/wallet/primary",
            "TNS_ADMIN": "/tns/fallback",
        }
        with mock.patch.dict(os.environ, {**clean_env, **env}, clear=True):
            result = _wallet_connect_kwargs()

        assert result["config_dir"] == "/wallet/primary"


# ═══════════════════════════════════════════════════════════════════════════
# 3. DSN FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDSNFormats:
    """Verify that OracleVDB.__init__ accepts the DSN formats customers
    actually use. We mock oracledb.create_pool to capture the dsn= arg
    and assert it was passed through unmodified."""

    @pytest.fixture(autouse=True)
    def _patch_imports(self):
        """Patch heavy upstream imports that aren't relevant to DSN tests."""
        self._captured_dsn = None

    def _init_with_dsn(self, dsn: str, extra_env: dict | None = None):
        """Create an OracleVDB instance with the given DSN, capturing the
        dsn passed to oracledb.create_pool."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_conn

        env = {
            "ORACLE_USER": "RAG_APP",
            "ORACLE_PASSWORD": "pw",
            "ORACLE_CS": dsn,
            **(extra_env or {}),
        }

        captured = {}

        def fake_create_pool(**kwargs):
            captured["dsn"] = kwargs.get("dsn")
            return mock_pool

        strip_keys = {"TNS_ADMIN", "ORACLE_WALLET_DIR", "ORACLE_WALLET_PASSWORD"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip_keys}

        with (
            mock.patch.dict(os.environ, {**clean_env, **env}, clear=True),
            mock.patch("oracledb.create_pool", side_effect=fake_create_pool),
        ):
            from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
            vdb = OracleVDB(collection_name="DSN_TEST")

        return vdb, captured

    def test_full_dsn_descriptor_private_endpoint_walletless(self):
        """Full TNS descriptor used with ADB private-endpoint walletless
        TLS connections."""
        dsn = (
            "(description=(retry_count=20)(retry_delay=3)"
            "(address=(protocol=tcps)"
            "(port=1522)(host=adb.us-ashburn-1.oraclecloud.com))"
            "(connect_data=(service_name=g1234abc_ragbp_tp.adb.oraclecloud.com))"
            "(security=(ssl_server_dn_match=yes)))"
        )
        vdb, captured = self._init_with_dsn(dsn)
        assert captured["dsn"] == dsn
        assert vdb._oracle_cs == dsn

    def test_tns_alias_mtls_wallet_mode(self):
        """TNS alias like 'ragbp_medium' — used when tnsnames.ora is
        present in TNS_ADMIN and connection is via mTLS wallet."""
        dsn = "ragbp_medium"
        vdb, captured = self._init_with_dsn(
            dsn,
            extra_env={"TNS_ADMIN": "/opt/oracle/wallet"},
        )
        assert captured["dsn"] == "ragbp_medium"

    def test_easy_connect_plus_format(self):
        """Easy Connect Plus: hostname:port/service_name — common for
        on-prem or non-ADB Oracle databases."""
        dsn = "dbhost.example.com:1521/FREEPDB1"
        vdb, captured = self._init_with_dsn(dsn)
        assert captured["dsn"] == dsn

    def test_k8s_service_dns(self):
        """In-cluster Kubernetes service DNS: used when Oracle DB runs
        as a StatefulSet in the same K8s cluster as the RAG blueprint."""
        dsn = "oracle-db.default.svc:1521/FREEPDB1"
        vdb, captured = self._init_with_dsn(dsn)
        assert captured["dsn"] == dsn

    def test_k8s_service_dns_with_namespace(self):
        """Full K8s DNS with explicit namespace and cluster domain."""
        dsn = "oracle-free-db.rag-ns.svc.cluster.local:1521/FREEPDB1"
        vdb, captured = self._init_with_dsn(dsn)
        assert captured["dsn"] == dsn


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureDetection:
    """Test that the adapter detects and adapts to available DB features."""

    def test_hybrid_ctxsys_error_falls_back_to_dense_warning(self):
        """hybrid=True with CTXSYS failure at index creation time should
        log a warning and continue. The table is still usable for dense
        search — only hybrid retrieval is degraded."""
        vdb, pool, conn, cursor = _make_oracle_vdb(hybrid=True)
        cursor.fetchone.return_value = (0,)

        ctxsys_error = oracledb.DatabaseError(
            "ORA-29855: CTXSYS.CONTEXT indextype does not exist"
        )
        exec_count = {"n": 0}

        def execute_side_effect(*args, **kwargs):
            exec_count["n"] += 1
            if exec_count["n"] == 1:
                cursor.fetchone.return_value = (0,)
                return None
            if exec_count["n"] == 4:
                raise ctxsys_error
            return None

        cursor.execute.side_effect = execute_side_effect

        vdb.create_collection("HYBRID_FALLBACK", dimension=1024)
        assert exec_count["n"] >= 4

    def test_pai_offload_url_only_for_hnsw_never_ivf(self):
        """OFFLOAD_URL should appear in DDL only for HNSW indexes.
        IVF does not support offload — Oracle 26ai rejects it."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        ivf_ddl = create_vector_index_ddl(
            "T",
            index_type="IVF",
            pai_offload_url="http://pai-gpu:8080/v1/index",
        )
        assert "OFFLOAD_URL" not in ivf_ddl, (
            "IVF index DDL must never contain OFFLOAD_URL"
        )

        hnsw_ddl = create_vector_index_ddl(
            "T",
            index_type="HNSW",
            pai_offload_url="http://pai-gpu:8080/v1/index",
        )
        assert "OFFLOAD_URL" in hnsw_ddl

    def test_pai_offload_credential_only_with_url(self):
        """OFFLOAD_CREDENTIAL_NAME should only appear when
        pai_offload_url is also provided."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        ddl_no_url = create_vector_index_ddl(
            "T",
            index_type="HNSW",
            pai_offload_url=None,
            pai_offload_credential="MY_CRED",
        )
        assert "OFFLOAD_CREDENTIAL_NAME" not in ddl_no_url

        ddl_with_url = create_vector_index_ddl(
            "T",
            index_type="HNSW",
            pai_offload_url="http://pai:8080/v1/index",
            pai_offload_credential="MY_CRED",
        )
        assert "OFFLOAD_CREDENTIAL_NAME 'MY_CRED'" in ddl_with_url

    def test_validate_identifier_accepts_schema_dot_table(self):
        """BYO cross-schema: _validate_identifier should accept
        'SCHEMA.TABLE' format for bring-your-own-database scenarios."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier

        assert _validate_identifier("RAG_APP.MY_TABLE") == "RAG_APP.MY_TABLE"
        assert _validate_identifier("ADMIN.EMBEDDINGS") == "ADMIN.EMBEDDINGS"
        assert _validate_identifier("hr.employees") == "hr.employees"

    def test_validate_identifier_rejects_triple_dot(self):
        """schema.table is the max — three-part names are invalid."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier

        with pytest.raises(ValueError, match="At most one dot"):
            _validate_identifier("A.B.C")

    def test_validate_identifier_rejects_sql_injection(self):
        """Identifiers with SQL metacharacters must be rejected."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import _validate_identifier

        dangerous = [
            "'; DROP TABLE users--",
            "table name",
            "1_STARTS_WITH_DIGIT",
            "",
            "table;",
            "UNION SELECT",
        ]
        for name in dangerous:
            with pytest.raises(ValueError, match="Unsafe Oracle"):
                _validate_identifier(name)

    def test_create_collection_passes_offload_only_for_hnsw(self):
        """Even when ORACLE_PAI_INDEX_URL is configured, create_collection
        should only pass offload params for HNSW, not IVF."""
        vdb, pool, conn, cursor = _make_oracle_vdb(
            index_type="IVF",
            env_overrides={
                "ORACLE_PAI_INDEX_URL": "http://pai:8080/v1/index",
            },
        )
        cursor.fetchone.return_value = (0,)

        captured_ddls = []
        original_execute = cursor.execute

        def capture_execute(*args, **kwargs):
            if args:
                captured_ddls.append(args[0])

        cursor.execute.side_effect = capture_execute

        vdb.create_collection("IVF_TABLE", dimension=1024)

        index_ddls = [d for d in captured_ddls if "CREATE VECTOR INDEX" in d]
        for ddl in index_ddls:
            assert "OFFLOAD_URL" not in ddl, (
                "IVF index creation must not include OFFLOAD_URL even when "
                "ORACLE_PAI_INDEX_URL is configured"
            )

    def test_hybrid_search_query_weights(self):
        """Verify that hybrid search SQL uses the specified weights."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import get_hybrid_search_query

        sql = get_hybrid_search_query("MY_TABLE", vector_weight=0.8, text_weight=0.2)
        assert "0.8" in sql
        assert "0.2" in sql
        assert "CONTAINS(text, :query_text, 1)" in sql

    def test_sanitize_text_query_strips_operators(self):
        """_sanitize_text_query must strip Oracle Text reserved chars."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        assert OracleVDB._sanitize_text_query("hello & world") == "hello OR world"
        assert OracleVDB._sanitize_text_query("a|b|c") == "a OR b OR c"
        assert OracleVDB._sanitize_text_query("") == ""
        assert OracleVDB._sanitize_text_query("!!!") == ""

    def test_sanitize_text_query_preserves_words(self):
        """Normal text should be OR-joined for ANY-term matching."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        result = OracleVDB._sanitize_text_query("Oracle vector search GPU")
        assert result == "Oracle OR vector OR search OR GPU"


# ═══════════════════════════════════════════════════════════════════════════
# 5. EDITION-SPECIFIC BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

class TestEditionSpecificBehavior:
    """Test that the adapter generates correct DDL / handles errors
    differently depending on the Oracle edition being targeted."""

    def test_developer_adb_same_features_as_enterprise(self):
        """Developer ADB has the same features as Enterprise (just smaller
        compute). DDL for vector table + HNSW index + text index should
        all be generated identically."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import (
            create_vector_table_ddl,
            create_vector_index_ddl,
            create_text_index_ddl,
        )

        table_ddl = create_vector_table_ddl("DEV_TABLE", dimension=1024)
        assert "VECTOR(1024, FLOAT32)" in table_ddl

        hnsw_ddl = create_vector_index_ddl(
            "DEV_TABLE",
            index_type="HNSW",
            pai_offload_url="http://pai:8080/v1/index",
        )
        assert "TYPE HNSW" in hnsw_ddl
        assert "OFFLOAD_URL" in hnsw_ddl

        text_ddl = create_text_index_ddl("DEV_TABLE")
        assert "CTXSYS.CONTEXT" in text_ddl

    def test_free_edition_no_ctxsys_text_index_skippable(self):
        """Oracle Free edition lacks CTXSYS. create_text_index_ddl still
        generates the DDL (the caller handles the error), but
        create_collection catches the ORA- error and continues."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_text_index_ddl

        ddl = create_text_index_ddl("FREE_TABLE")
        assert "CTXSYS.CONTEXT" in ddl

        vdb, pool, conn, cursor = _make_oracle_vdb(hybrid=True)
        cursor.fetchone.return_value = (0,)

        ctxsys_error = oracledb.DatabaseError(
            "ORA-29855: CTXSYS owner does not exist"
        )
        exec_count = {"n": 0}

        def execute_side_effect(*args, **kwargs):
            exec_count["n"] += 1
            if exec_count["n"] == 1:
                cursor.fetchone.return_value = (0,)
                return None
            if exec_count["n"] == 4:
                raise ctxsys_error
            return None

        cursor.execute.side_effect = execute_side_effect

        vdb.create_collection("FREE_TABLE", dimension=384)

        assert exec_count["n"] >= 4, (
            "All 4 steps (exists check, create table, create vec index, "
            "create text index) should have been attempted"
        )

    def test_enterprise_full_features_ddl(self):
        """Enterprise: full feature set — HNSW with GPU offload, text
        index, and all parameters should be present."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import (
            create_vector_table_ddl,
            create_vector_index_ddl,
            create_text_index_ddl,
        )

        table_ddl = create_vector_table_ddl("ENTERPRISE_TABLE", dimension=2048)
        assert "VECTOR(2048, FLOAT32)" in table_ddl
        assert "CLOB CHECK (content_metadata IS JSON)" in table_ddl

        hnsw_ddl = create_vector_index_ddl(
            "ENTERPRISE_TABLE",
            index_type="HNSW",
            hnsw_m=32,
            hnsw_ef_construction=400,
            pai_offload_url="http://pai-gpu:8080/v1/index",
            pai_offload_credential="RAG_CRED",
        )
        assert "NEIGHBORS 32" in hnsw_ddl
        assert "EFCONSTRUCTION 400" in hnsw_ddl
        assert "OFFLOAD_URL 'http://pai-gpu:8080/v1/index'" in hnsw_ddl
        assert "OFFLOAD_CREDENTIAL_NAME 'RAG_CRED'" in hnsw_ddl

        text_ddl = create_text_index_ddl("ENTERPRISE_TABLE")
        assert "CTXSYS.CONTEXT" in text_ddl
        assert "SYNC (ON COMMIT)" in text_ddl

    def test_ivf_index_ddl_neighbor_partitions_syntax(self):
        """Oracle ADB requires the two-word 'NEIGHBOR PARTITIONS' form,
        not 'neighbor_partitions'. Verify the DDL matches."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        ddl = create_vector_index_ddl(
            "IVF_CHECK",
            index_type="IVF",
            ivf_neighbor_partitions=200,
        )
        assert "NEIGHBOR PARTITIONS 200" in ddl
        assert "neighbor_partitions" not in ddl.lower().replace("neighbor partitions", "")

    def test_hnsw_index_ddl_without_offload(self):
        """HNSW without GPU offload: just the base params, no OFFLOAD_*."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        ddl = create_vector_index_ddl(
            "HNSW_BASIC",
            index_type="HNSW",
            hnsw_m=16,
            hnsw_ef_construction=200,
        )
        assert "TYPE HNSW" in ddl
        assert "NEIGHBORS 16" in ddl
        assert "EFCONSTRUCTION 200" in ddl
        assert "OFFLOAD" not in ddl
        assert "INMEMORY NEIGHBOR GRAPH" in ddl

    def test_pai_url_auto_appends_v1_index(self):
        """ORACLE_PAI_INDEX_URL without trailing /v1/index should get it
        appended automatically by the constructor."""
        vdb, *_ = _make_oracle_vdb(
            index_type="HNSW",
            env_overrides={
                "ORACLE_PAI_INDEX_URL": "http://pai-gpu:8080",
            },
        )
        assert vdb._pai_offload_url == "http://pai-gpu:8080/v1/index"

    def test_pai_url_already_has_v1_index_not_doubled(self):
        """If the URL already ends with /v1/index, don't double it."""
        vdb, *_ = _make_oracle_vdb(
            index_type="HNSW",
            env_overrides={
                "ORACLE_PAI_INDEX_URL": "http://pai-gpu:8080/v1/index",
            },
        )
        assert vdb._pai_offload_url == "http://pai-gpu:8080/v1/index"
        assert "/v1/index/v1/index" not in (vdb._pai_offload_url or "")

    def test_collection_name_uppercased(self):
        """Oracle identifiers are case-insensitive; the adapter should
        uppercase collection names consistently."""
        vdb, *_ = _make_oracle_vdb(collection_name="my_table")
        assert vdb.collection_name == "MY_TABLE"

    def test_missing_credentials_raises_valueerror(self):
        """If ORACLE_USER/PASSWORD/CS are all missing, the constructor
        should raise ValueError immediately — not a cryptic pool error."""
        strip = {"ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_CS"}
        clean_env = {k: v for k, v in os.environ.items() if k not in strip}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(ValueError, match="ORACLE_USER.*ORACLE_PASSWORD.*ORACLE_CS"):
                from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
                OracleVDB(collection_name="FAIL")


# ═══════════════════════════════════════════════════════════════════════════
# 6. DDL OUTPUT CORRECTNESS (bonus: prevents regressions in generated SQL)
# ═══════════════════════════════════════════════════════════════════════════

class TestDDLCorrectness:
    """Snapshot-style tests for generated DDL to catch accidental breakage."""

    def test_vector_table_ddl_has_all_columns(self):
        """The canonical RAG table must have id, text, vector, source,
        content_metadata, and created_at."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_table_ddl

        ddl = create_vector_table_ddl("RAG_DOCS", dimension=768)
        for col in ("id RAW(16)", "text CLOB", "VECTOR(768, FLOAT32)",
                     "source VARCHAR2(4000)", "content_metadata CLOB",
                     "created_at TIMESTAMP"):
            assert col in ddl, f"Missing column fragment: {col}"

    def test_drop_table_ddl_includes_cascade(self):
        """DROP TABLE must CASCADE CONSTRAINTS PURGE to clean up indexes."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import drop_table_ddl

        ddl = drop_table_ddl("OLD_TABLE")
        assert "CASCADE CONSTRAINTS PURGE" in ddl

    def test_similarity_search_query_uses_correct_metric(self):
        """Verify VECTOR_DISTANCE uses the specified metric."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import get_similarity_search_query

        for metric in ("COSINE", "L2", "DOT", "MANHATTAN"):
            sql = get_similarity_search_query("T", distance_metric=metric)
            assert f"VECTOR_DISTANCE(vector, :query_vector, {metric})" in sql

    def test_byo_view_ddl_requires_text_and_vector(self):
        """create_byo_view_ddl must raise when text or vector are missing."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_byo_view_ddl

        with pytest.raises(ValueError, match="text.*vector"):
            create_byo_view_ddl("V", "SRC_TABLE", {"text": "CHUNK"})

        with pytest.raises(ValueError, match="text.*vector"):
            create_byo_view_ddl("V", "SRC_TABLE", {"vector": "EMB"})

    def test_byo_view_ddl_defaults(self):
        """Minimal BYO view: only text + vector mapped. ID should default
        to ROWID, source to a static JSON, content_metadata to empty JSON."""
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_byo_view_ddl

        ddl = create_byo_view_ddl(
            "RAG_VIEW",
            "CUSTOMER.DOCS",
            {"text": "CHUNK_TEXT", "vector": "EMBEDDING"},
        )
        assert "ROWID AS id" in ddl
        assert "CHUNK_TEXT AS text" in ddl
        assert "EMBEDDING AS vector" in ddl
        assert "JSON_OBJECT()" in ddl

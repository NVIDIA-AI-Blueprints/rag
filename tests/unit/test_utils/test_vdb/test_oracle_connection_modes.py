# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection-mode unit tests for OracleVDB.

Each test class covers a real customer deployment topology:
  - Walletless TLS (private-endpoint ADB on the same VCN)
  - mTLS with wallet  (public-endpoint / cross-region ADB)
  - Unencrypted wallet (ewallet.pem without password)
  - ORACLE_WALLET_DIR alternative env var
  - In-cluster container DB (Oracle Free / DB 23ai, no TLS)
  - PAI GPU-offload URL propagation
  - Collection-name SQL-injection guard

No live database is required.  ``oracledb.create_pool`` is always
mocked so the tests run anywhere pytest can import the source tree.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BASE_ENV = {
    "ORACLE_USER": "RAG_APP",
    "ORACLE_PASSWORD": "s3cret",
}

_WALLET_VARS = (
    "TNS_ADMIN",
    "ORACLE_WALLET_DIR",
    "ORACLE_WALLET_PASSWORD",
    "ORACLE_PAI_INDEX_URL",
    "ORACLE_PAI_OFFLOAD_CREDENTIAL",
    "ORACLE_VECTOR_INDEX_TYPE",
    "ORACLE_DISTANCE_METRIC",
)


@pytest.fixture(autouse=True)
def _clean_wallet_env(monkeypatch):
    """Remove wallet / PAI env vars so each test starts from a known baseline."""
    for var in _WALLET_VARS:
        monkeypatch.delenv(var, raising=False)


def _mock_pool():
    """Return a MagicMock that quacks like an oracledb.ConnectionPool."""
    pool = MagicMock(name="ConnectionPool")
    conn = MagicMock(name="Connection")
    cursor = MagicMock(name="Cursor")
    cursor.execute.return_value = None
    cursor.fetchone.return_value = (1,)
    cursor.__enter__ = lambda s: s
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    pool.acquire.return_value = conn
    return pool


def _build_vdb(monkeypatch, env_overrides: dict | None = None, **ctor_kwargs):
    """Construct an OracleVDB with mocked pool and controlled env vars.

    Returns ``(vdb_instance, create_pool_mock)`` so the caller can inspect
    how ``oracledb.create_pool`` was called.
    """
    env = {**_BASE_ENV, **(env_overrides or {})}
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    pool = _mock_pool()

    with patch("oracledb.create_pool", return_value=pool) as mock_cp:
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        defaults = {
            "collection_name": "TEST",
            "embedding_model": MagicMock(),
        }
        defaults.update(ctor_kwargs)
        vdb = OracleVDB(**defaults)

    return vdb, mock_cp


# ===================================================================
# 1. WALLETLESS TLS  (private-endpoint ADB, same VCN)
# ===================================================================
class TestWalletlessTLS:
    """Customer scenario: ADB configured with a *private endpoint* on
    the same VCN as the OKE cluster.  No wallet is required — the
    connection string is a full DSN descriptor with ``(protocol=tcps)``
    on port 1521.  ``TNS_ADMIN`` / ``ORACLE_WALLET_*`` are unset.
    """

    FULL_DSN = (
        "(description=(retry_count=3)(retry_delay=3)"
        "(address=(protocol=tcps)(port=1521)"
        "(host=adb.us-ashburn-1.oraclecloud.com))"
        "(connect_data=(service_name=xyz_medium.adb.oraclecloud.com)))"
    )

    @pytest.mark.timeout(120)
    def test_wallet_connect_kwargs_empty(self, monkeypatch):
        """No wallet env vars -> _wallet_connect_kwargs returns {}."""
        for v in _WALLET_VARS:
            monkeypatch.delenv(v, raising=False)

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        assert _wallet_connect_kwargs() == {}

    def test_pool_receives_full_dsn_descriptor(self, monkeypatch):
        """create_pool must be called with dsn= set to the raw descriptor
        string so thin-mode resolves the host directly (no tnsnames.ora).
        """
        _, mock_cp = _build_vdb(monkeypatch, {"ORACLE_CS": self.FULL_DSN})

        mock_cp.assert_called_once()
        call_kwargs = mock_cp.call_args
        assert call_kwargs.kwargs["dsn"] == self.FULL_DSN

    def test_no_wallet_kwargs_passed_to_pool(self, monkeypatch):
        """Walletless mode must NOT pass config_dir / wallet_location /
        wallet_password — doing so causes DPY-4011 on thin-mode.
        """
        _, mock_cp = _build_vdb(monkeypatch, {"ORACLE_CS": self.FULL_DSN})

        kw = mock_cp.call_args.kwargs
        assert "config_dir" not in kw
        assert "wallet_location" not in kw
        assert "wallet_password" not in kw


# ===================================================================
# 2. mTLS WITH WALLET  (public endpoint / cross-region)
# ===================================================================
class TestMtlsWithWallet:
    """Customer scenario: ADB with a *public endpoint*.  The connection
    uses an Oracle wallet (ewallet.pem) downloaded from OCI console.
    ``TNS_ADMIN`` points to the wallet directory, ``ORACLE_WALLET_PASSWORD``
    decrypts the PEM.  ``ORACLE_CS`` is a TNS alias resolved from
    ``tnsnames.ora`` inside the wallet directory.
    """

    def test_wallet_kwargs_include_all_three_fields(self, monkeypatch):
        """config_dir, wallet_location, and wallet_password must all
        appear when TNS_ADMIN and ORACLE_WALLET_PASSWORD are set.
        """
        monkeypatch.setenv("TNS_ADMIN", "/app/wallet")
        monkeypatch.setenv("ORACLE_WALLET_PASSWORD", "walletpw")

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        kw = _wallet_connect_kwargs()

        assert kw == {
            "config_dir": "/app/wallet",
            "wallet_location": "/app/wallet",
            "wallet_password": "walletpw",
        }

    def test_pool_receives_tns_alias_and_wallet_kwargs(self, monkeypatch):
        """create_pool must be called with dsn='test1_medium' plus the
        three wallet kwargs so thin-mode reads tnsnames.ora + ewallet.pem.
        """
        _, mock_cp = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test1_medium",
                "TNS_ADMIN": "/app/wallet",
                "ORACLE_WALLET_PASSWORD": "walletpw",
            },
        )

        kw = mock_cp.call_args.kwargs
        assert kw["dsn"] == "test1_medium"
        assert kw["config_dir"] == "/app/wallet"
        assert kw["wallet_location"] == "/app/wallet"
        assert kw["wallet_password"] == "walletpw"


# ===================================================================
# 3. WALLET DIR WITHOUT PASSWORD  (unencrypted ewallet.pem)
# ===================================================================
class TestWalletWithoutPassword:
    """Customer scenario: the DBA generated an *unencrypted* ewallet.pem
    (no password protection).  ``TNS_ADMIN`` is set but
    ``ORACLE_WALLET_PASSWORD`` is deliberately omitted.
    """

    def test_wallet_kwargs_omit_password(self, monkeypatch):
        """config_dir + wallet_location present, wallet_password absent."""
        monkeypatch.setenv("TNS_ADMIN", "/app/wallet")

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        kw = _wallet_connect_kwargs()

        assert kw["config_dir"] == "/app/wallet"
        assert kw["wallet_location"] == "/app/wallet"
        assert "wallet_password" not in kw

    def test_pool_has_no_wallet_password(self, monkeypatch):
        """create_pool must omit wallet_password entirely — passing
        an empty string causes DPY-4011.
        """
        _, mock_cp = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "mydb_high",
                "TNS_ADMIN": "/app/wallet",
            },
        )

        kw = mock_cp.call_args.kwargs
        assert kw["config_dir"] == "/app/wallet"
        assert "wallet_password" not in kw


# ===================================================================
# 4. ORACLE_WALLET_DIR alternative  (instead of TNS_ADMIN)
# ===================================================================
class TestOracleWalletDirAlternative:
    """Customer scenario: the Helm chart sets ``ORACLE_WALLET_DIR``
    instead of ``TNS_ADMIN``.  The code must honour either variable.
    """

    def test_wallet_dir_used_when_tns_admin_unset(self, monkeypatch):
        """ORACLE_WALLET_DIR alone is sufficient for config_dir + wallet_location."""
        monkeypatch.setenv("ORACLE_WALLET_DIR", "/mnt/wallet")

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        kw = _wallet_connect_kwargs()

        assert kw["config_dir"] == "/mnt/wallet"
        assert kw["wallet_location"] == "/mnt/wallet"
        assert "wallet_password" not in kw

    def test_wallet_dir_with_password(self, monkeypatch):
        """ORACLE_WALLET_DIR + ORACLE_WALLET_PASSWORD works the same
        as TNS_ADMIN + ORACLE_WALLET_PASSWORD.
        """
        monkeypatch.setenv("ORACLE_WALLET_DIR", "/mnt/wallet")
        monkeypatch.setenv("ORACLE_WALLET_PASSWORD", "pw123")

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        kw = _wallet_connect_kwargs()

        assert kw == {
            "config_dir": "/mnt/wallet",
            "wallet_location": "/mnt/wallet",
            "wallet_password": "pw123",
        }

    def test_wallet_dir_takes_precedence_over_empty_tns_admin(self, monkeypatch):
        """If TNS_ADMIN is empty-string and ORACLE_WALLET_DIR is set,
        the code should use ORACLE_WALLET_DIR.
        """
        monkeypatch.setenv("TNS_ADMIN", "")
        monkeypatch.setenv("ORACLE_WALLET_DIR", "/opt/wallet")

        from nvidia_rag.utils.vdb.oracle.oracle_vdb import _wallet_connect_kwargs
        kw = _wallet_connect_kwargs()

        # Source code: os.getenv("ORACLE_WALLET_DIR") or os.getenv("TNS_ADMIN")
        # ORACLE_WALLET_DIR is checked first, so it wins when TNS_ADMIN is empty.
        assert kw["config_dir"] == "/opt/wallet"

    def test_pool_receives_wallet_dir_kwargs(self, monkeypatch):
        """End-to-end: ORACLE_WALLET_DIR flows through to create_pool."""
        _, mock_cp = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "dev_tp",
                "ORACLE_WALLET_DIR": "/mnt/wallet",
                "ORACLE_WALLET_PASSWORD": "secret",
            },
        )

        kw = mock_cp.call_args.kwargs
        assert kw["dsn"] == "dev_tp"
        assert kw["config_dir"] == "/mnt/wallet"
        assert kw["wallet_location"] == "/mnt/wallet"
        assert kw["wallet_password"] == "secret"


# ===================================================================
# 5. IN-CLUSTER CONTAINER DB  (no TLS)
# ===================================================================
class TestInClusterContainerDB:
    """Customer scenario: Oracle Database Free 23ai running as a pod
    in the same k8s cluster.  Connection is a plain ``host:port/service``
    EZ-connect string — no TLS, no wallet.
    """

    EZ_CONNECT = "oracle-db.default.svc:1521/FREEPDB1"

    def test_pool_receives_ez_connect_string(self, monkeypatch):
        """create_pool called with a simple host:port/service DSN."""
        _, mock_cp = _build_vdb(monkeypatch, {"ORACLE_CS": self.EZ_CONNECT})

        kw = mock_cp.call_args.kwargs
        assert kw["dsn"] == self.EZ_CONNECT
        assert kw["user"] == "RAG_APP"
        assert kw["password"] == "s3cret"

    def test_no_tls_kwargs(self, monkeypatch):
        """No wallet / TLS env vars -> create_pool has no config_dir etc."""
        _, mock_cp = _build_vdb(monkeypatch, {"ORACLE_CS": self.EZ_CONNECT})

        kw = mock_cp.call_args.kwargs
        assert "config_dir" not in kw
        assert "wallet_location" not in kw
        assert "wallet_password" not in kw

    def test_pool_sizing_defaults(self, monkeypatch):
        """Verify the hard-coded pool sizing from __init__."""
        _, mock_cp = _build_vdb(monkeypatch, {"ORACLE_CS": self.EZ_CONNECT})

        kw = mock_cp.call_args.kwargs
        assert kw["min"] == 2
        assert kw["max"] == 10
        assert kw["increment"] == 1

    def test_smoke_select_dual_executed(self, monkeypatch):
        """__init__ does a SELECT 1 FROM DUAL to verify the pool is usable."""
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": self.EZ_CONNECT})

        pool = vdb._pool
        conn = pool.acquire.return_value
        cursor = conn.cursor.return_value
        cursor.execute.assert_any_call("SELECT 1 FROM DUAL")


# ===================================================================
# 6. PAI URL PROPAGATION  (cuVS GPU-offload)
# ===================================================================
class TestPAIUrlPropagation:
    """Customer scenario: the Helm chart enables GPU-accelerated HNSW
    index builds via Oracle Private AI Services.  The PAI URL is
    stamped into ``ORACLE_PAI_INDEX_URL`` and optionally a credential
    name in ``ORACLE_PAI_OFFLOAD_CREDENTIAL``.
    """

    def test_pai_url_stored_on_instance(self, monkeypatch):
        """ORACLE_PAI_INDEX_URL is normalised (trailing /v1/index) and
        stored on the VDB instance for use in DDL.
        """
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "http://oracle-pai.svc:8080",
            },
        )
        assert vdb._pai_offload_url == "http://oracle-pai.svc:8080/v1/index"

    def test_pai_url_already_has_suffix(self, monkeypatch):
        """If the URL already ends in /v1/index, don't double-append."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "http://pai.svc:8080/v1/index",
            },
        )
        assert vdb._pai_offload_url == "http://pai.svc:8080/v1/index"

    def test_pai_url_trailing_slash_normalised(self, monkeypatch):
        """Trailing slash before /v1/index append."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "http://pai.svc:8080/",
            },
        )
        assert vdb._pai_offload_url == "http://pai.svc:8080/v1/index"

    def test_pai_url_empty_disables_offload(self, monkeypatch):
        """Empty string -> offload disabled (None)."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "",
            },
        )
        assert vdb._pai_offload_url is None

    def test_pai_url_whitespace_only_disables_offload(self, monkeypatch):
        """Whitespace-only string -> offload disabled."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "   ",
            },
        )
        assert vdb._pai_offload_url is None

    def test_pai_url_unset_disables_offload(self, monkeypatch):
        """Env var entirely absent -> offload disabled."""
        monkeypatch.delenv("ORACLE_PAI_INDEX_URL", raising=False)
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "test_medium"})

        assert vdb._pai_offload_url is None

    def test_pai_credential_stored(self, monkeypatch):
        """ORACLE_PAI_OFFLOAD_CREDENTIAL flows to _pai_offload_credential."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "https://pai.example.com:8443/v1/index",
                "ORACLE_PAI_OFFLOAD_CREDENTIAL": "PAI_OFFLOAD_CRED",
            },
        )
        assert vdb._pai_offload_credential == "PAI_OFFLOAD_CRED"

    def test_pai_credential_empty_is_none(self, monkeypatch):
        """Empty credential string -> None (HTTP mode, no OCI auth)."""
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "http://pai.svc:8080",
                "ORACLE_PAI_OFFLOAD_CREDENTIAL": "",
            },
        )
        assert vdb._pai_offload_credential is None

    def test_pai_credential_unset_is_none(self, monkeypatch):
        """Credential env var absent -> None."""
        monkeypatch.delenv("ORACLE_PAI_OFFLOAD_CREDENTIAL", raising=False)
        vdb, _ = _build_vdb(
            monkeypatch,
            {
                "ORACLE_CS": "test_medium",
                "ORACLE_PAI_INDEX_URL": "http://pai.svc:8080",
            },
        )
        assert vdb._pai_offload_credential is None


# ===================================================================
# 7. COLLECTION NAME VALIDATION
# ===================================================================
class TestCollectionNameValidation:
    """Guard against SQL injection through collection_name.  The
    ``_validate_identifier`` function is called inside ``__init__`` and
    again on the ``collection_name`` property setter.
    """

    def test_simple_name_accepted(self, monkeypatch):
        """Plain uppercase name passes validation."""
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "x"}, collection_name="TEST")
        assert vdb.collection_name == "TEST"

    def test_schema_qualified_name_accepted(self, monkeypatch):
        """'RAG_APP.MY_TABLE' is a valid schema-qualified identifier."""
        vdb, _ = _build_vdb(
            monkeypatch, {"ORACLE_CS": "x"},
            collection_name="RAG_APP.MY_TABLE",
        )
        assert vdb.collection_name == "RAG_APP.MY_TABLE"

    def test_sql_injection_raises_valueerror(self, monkeypatch):
        """Semicolons / SQL keywords in the name must be rejected."""
        monkeypatch.setenv("ORACLE_CS", "x")
        monkeypatch.setenv("ORACLE_USER", "u")
        monkeypatch.setenv("ORACLE_PASSWORD", "p")

        with pytest.raises(ValueError, match="[Uu]nsafe"):
            _build_vdb(
                monkeypatch, {"ORACLE_CS": "x"},
                collection_name="X; DROP TABLE",
            )

    def test_empty_collection_name_accepted(self, monkeypatch):
        """Empty string at init is allowed (collection set later)."""
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "x"}, collection_name="")
        assert vdb.collection_name == ""

    def test_setter_validates(self, monkeypatch):
        """Setting collection_name to an unsafe value after __init__
        must also raise ValueError.
        """
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "x"}, collection_name="OK")

        with pytest.raises(ValueError, match="[Uu]nsafe"):
            vdb.collection_name = "X; DROP TABLE"

    def test_setter_accepts_valid_name(self, monkeypatch):
        """Property setter accepts a valid name and upper-cases it."""
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "x"}, collection_name="OLD")

        vdb.collection_name = "new_collection"
        assert vdb.collection_name == "NEW_COLLECTION"

    def test_setter_allows_empty_to_clear(self, monkeypatch):
        """Setting collection_name to '' effectively clears it."""
        vdb, _ = _build_vdb(monkeypatch, {"ORACLE_CS": "x"}, collection_name="INIT")

        vdb.collection_name = ""
        assert vdb.collection_name == ""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "1DIGIT_START",
            "has spaces",
            "semi;colon",
            "back`tick",
            "slash/path",
            "a.b.c.d",
        ],
    )
    def test_various_unsafe_names_rejected(self, monkeypatch, bad_name):
        """Exhaustive: every obviously-unsafe pattern is caught."""
        with pytest.raises(ValueError):
            _build_vdb(
                monkeypatch, {"ORACLE_CS": "x"},
                collection_name=bad_name,
            )


# ===================================================================
# Missing credentials
# ===================================================================
class TestMissingCredentials:
    """If ORACLE_USER / ORACLE_PASSWORD / ORACLE_CS are unset, __init__
    must fail immediately with a clear ValueError (not NPE deep in
    oracledb.connect).
    """

    def test_missing_user_raises(self, monkeypatch):
        """ORACLE_USER absent -> ValueError."""
        monkeypatch.delenv("ORACLE_USER", raising=False)
        monkeypatch.setenv("ORACLE_PASSWORD", "pw")
        monkeypatch.setenv("ORACLE_CS", "dsn")

        with pytest.raises(ValueError, match="ORACLE_USER"):
            with patch("oracledb.create_pool", return_value=_mock_pool()):
                from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
                OracleVDB(collection_name="T", embedding_model=MagicMock())

    def test_missing_password_raises(self, monkeypatch):
        """ORACLE_PASSWORD absent -> ValueError."""
        monkeypatch.setenv("ORACLE_USER", "u")
        monkeypatch.delenv("ORACLE_PASSWORD", raising=False)
        monkeypatch.setenv("ORACLE_CS", "dsn")

        with pytest.raises(ValueError, match="ORACLE_PASSWORD"):
            with patch("oracledb.create_pool", return_value=_mock_pool()):
                from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
                OracleVDB(collection_name="T", embedding_model=MagicMock())

    def test_missing_cs_raises(self, monkeypatch):
        """ORACLE_CS absent -> ValueError."""
        monkeypatch.setenv("ORACLE_USER", "u")
        monkeypatch.setenv("ORACLE_PASSWORD", "pw")
        monkeypatch.delenv("ORACLE_CS", raising=False)

        with pytest.raises(ValueError, match="ORACLE_CS"):
            with patch("oracledb.create_pool", return_value=_mock_pool()):
                from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
                OracleVDB(collection_name="T", embedding_model=MagicMock())

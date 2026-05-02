# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime fail-fast tests for OracleVDB.

When the rag-server / ingestor pod starts, every common DB
misconfiguration must surface in the pod's logs in <5s with a clear
next step. These tests simulate every category by raising the matching
ORA-/DPY- code from a fake oracledb and asserting:

  1. ``OracleVDB.__init__`` raises an ``APIError`` whose ``message``
     contains the operator-facing next step.
  2. The error category and code propagate to ``check_health()`` so the
     /health endpoint can return a structured 503.
  3. Pod logs see the *category* and *next_step* (so a tail of the log
     shows the fix without scrolling).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def good_env(monkeypatch):
    """Set every required env var so OracleVDB.__init__ doesn't bail
    on the pre-connect ValueError check."""
    monkeypatch.setenv("ORACLE_USER", "ragapp")
    monkeypatch.setenv("ORACLE_PASSWORD", "p")
    monkeypatch.setenv("ORACLE_CS", "ragdb_medium")


@pytest.fixture
def fake_oracledb_error_class():
    """oracledb.Error is the parent of every connect/query exception.
    We need a real class so ``except oracledb.Error`` catches our
    simulated failures."""
    class _Error(Exception):
        pass
    return _Error


def _patched_oracledb(error_class, on_create_pool):
    """Build a fake oracledb module and return it ready to monkeypatch.

    on_create_pool: callable(**kwargs) — raise to simulate a connect
    failure, return a MagicMock to simulate success.
    """
    fake = SimpleNamespace()
    fake.Error = error_class
    fake.create_pool = MagicMock(side_effect=on_create_pool)
    return fake


# ===========================================================================
# Each (raw_message, expected_category, expected_log_keyword)
# parametrisation simulates a pod hitting that ORA code on first connect.
# ===========================================================================
SCENARIOS = [
    pytest.param(
        "ORA-01017: invalid username/password; logon denied",
        "auth", "ORACLE_PASSWORD",
        id="bad-password",
    ),
    pytest.param(
        "ORA-12541: TNS:no listener — host db.example.com:1522",
        "network", "VCN routing",
        id="adb-unreachable",
    ),
    pytest.param(
        "ORA-12154: cannot resolve the connect identifier specified",
        "network", "TNS",
        id="bad-connect-string",
    ),
    pytest.param(
        "ORA-28759: failure to open file ewallet.pem",
        "wallet", "ORACLE_WALLET_PASSWORD",
        id="missing-wallet-password",
    ),
    pytest.param(
        "ORA-28000: the account is locked",
        "auth", "UNLOCK",
        id="account-locked",
    ),
    pytest.param(
        "DPY-4011: socket was closed during DNS resolution",
        "network", "DNS",
        id="dns-failure",
    ),
    pytest.param(
        "ORA-12514: TNS:listener does not currently know of service",
        "service", "AVAILABLE",
        id="adb-not-ready",
    ),
]


@pytest.mark.parametrize(("raw", "expected_cat", "log_kw"), SCENARIOS)
def test_init_raises_with_categorised_message(
    good_env, fake_oracledb_error_class, raw, expected_cat, log_kw,
):
    """On a misconfigured connect the operator must see, in the
    APIError message that propagates to pod logs and external probes:

      * the actionable keyword (a Secret name, an env var, an OCI step)
        — so the fix is one tail-of-log scroll away

    We assert against the APIError directly because pytest's stderr
    capture is unreliable when run alongside other tests that
    reconfigure logging.
    """
    from nvidia_rag.rag_server.response_generator import APIError
    from nvidia_rag.utils.vdb.oracle import oracle_vdb as ovdb

    err = fake_oracledb_error_class(raw)
    fake = _patched_oracledb(
        fake_oracledb_error_class,
        on_create_pool=lambda **kw: (_ for _ in ()).throw(err),
    )

    with patch.object(ovdb, "oracledb", fake):
        with pytest.raises(APIError) as exc_info:
            ovdb.OracleVDB(collection_name="docs")

    api_msg = str(exc_info.value)
    assert log_kw in api_msg, (
        f"Expected actionable keyword {log_kw!r} in APIError: {api_msg!r}"
    )
    # Also call the diagnoser directly to pin the category invariant.
    from nvidia_rag.utils.vdb.oracle.oracle_errors import diagnose_oracle_error
    diag = diagnose_oracle_error(fake_oracledb_error_class(raw))
    assert diag.category == expected_cat, (
        f"diagnoser put {raw!r} in category {diag.category!r}, "
        f"expected {expected_cat!r}"
    )


def test_health_check_reports_category_and_code(
    good_env, fake_oracledb_error_class,
):
    """check_health() must return a structured dict with category+code
    so an external probe can surface "wallet password missing" without
    parsing a freeform error string."""
    import asyncio
    from contextlib import contextmanager

    from nvidia_rag.utils.vdb.oracle import oracle_vdb as ovdb

    err = fake_oracledb_error_class("ORA-12541: TNS:no listener")

    class _Cursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **kw):
            raise err
        def fetchone(self): return (0,)

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _Cursor()

    class _GoodCursor(_Cursor):
        def execute(self, *a, **kw):  # __init__ ping must succeed
            return None

    class _GoodConn(_Conn):
        def cursor(self): return _GoodCursor()

    class _Pool:
        def __init__(self): self._first = True
        @contextmanager
        def acquire(self):
            if self._first:
                self._first = False
                yield _GoodConn()  # __init__ ping
            else:
                yield _Conn()      # health check fails

    pool = _Pool()
    fake = _patched_oracledb(
        fake_oracledb_error_class,
        on_create_pool=lambda **kw: pool,
    )

    with patch.object(ovdb, "oracledb", fake):
        vdb = ovdb.OracleVDB(collection_name="docs")
        status = asyncio.get_event_loop().run_until_complete(vdb.check_health())

    from nvidia_rag.utils.health_models import ServiceStatus
    assert status["status"] == ServiceStatus.ERROR.value
    assert status["error_category"] == "network"
    assert status["error_code"] == "ORA-12541"
    assert "VCN" in status["error"] or "TCP 1522" in status["error"]


def test_init_succeeds_when_pool_works(good_env, fake_oracledb_error_class):
    """Sanity: with a working oracledb mock, __init__ doesn't raise."""
    from nvidia_rag.utils.vdb.oracle import oracle_vdb as ovdb

    pool = MagicMock()
    cursor = MagicMock()
    pool.acquire.return_value.__enter__.return_value.cursor.return_value\
        .__enter__.return_value = cursor

    fake = _patched_oracledb(
        fake_oracledb_error_class,
        on_create_pool=lambda **kw: pool,
    )

    with patch.object(ovdb, "oracledb", fake):
        vdb = ovdb.OracleVDB(collection_name="docs")

    assert vdb._collection_name == "DOCS"


def test_init_fails_clearly_when_env_vars_missing(monkeypatch):
    """Pre-connect check: missing creds must give a one-line error
    naming all three required env vars (so a typo'd Secret value is
    immediately obvious)."""
    monkeypatch.delenv("ORACLE_USER", raising=False)
    monkeypatch.delenv("ORACLE_PASSWORD", raising=False)
    monkeypatch.delenv("ORACLE_CS", raising=False)

    from nvidia_rag.utils.vdb.oracle import oracle_vdb as ovdb

    with pytest.raises(ValueError) as e:
        ovdb.OracleVDB(collection_name="docs")
    msg = str(e.value)
    for must in ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_CS"):
        assert must in msg, f"Missing {must} hint in: {msg!r}"

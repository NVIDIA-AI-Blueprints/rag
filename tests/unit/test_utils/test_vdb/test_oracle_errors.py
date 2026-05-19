# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Oracle ORA-/DPY- error translator.

The translator (``nvidia_rag.utils.vdb.oracle.oracle_errors``) maps
raw oracledb exceptions into operator-readable diagnostics with a
single actionable next step. These tests pin every category, confirm
no password ever leaks into the diagnosis, and guard against silent
table drift.

Pure-Python — no external services, no helm, no docker.
"""
from __future__ import annotations

import pytest

from nvidia_rag.utils.vdb.oracle.oracle_errors import (
    diagnose_oci_error,
    diagnose_oracle_error,
)


VALID_CATEGORIES = {
    "auth", "network", "wallet", "service",
    "schema", "config", "conflict", "quota", "other",
}


# ---------------------------------------------------------------------------
# Every code in the table maps to a known category. Synced with
# oracle_errors._TABLE; the drift guard test below cross-checks.
# ---------------------------------------------------------------------------
EXPECTED_CATEGORY = {
    "ORA-01017": "auth",
    "ORA-28000": "auth",
    "ORA-01045": "auth",
    "ORA-12154": "network",
    "ORA-12541": "network",
    "ORA-12170": "network",
    "ORA-12537": "network",
    "DPY-4011":  "network",
    "DPY-4027":  "network",
    "ORA-12514": "service",
    "ORA-28759": "wallet",
    "ORA-28365": "wallet",
    "ORA-28860": "wallet",
    "ORA-00001": "schema",
    "ORA-00942": "schema",
    "ORA-00955": "schema",
    "ORA-65096": "schema",
    "DPI-1047":  "config",
}


@pytest.mark.parametrize("code, expected_cat", sorted(EXPECTED_CATEGORY.items()))
def test_known_codes_map_to_correct_category(code, expected_cat):
    err = RuntimeError(f"{code}: simulated message")
    diag = diagnose_oracle_error(err)
    assert diag.code == code
    assert diag.category == expected_cat
    assert diag.category in VALID_CATEGORIES


@pytest.mark.parametrize("code", sorted(EXPECTED_CATEGORY))
def test_summary_and_next_step_are_actionable(code):
    diag = diagnose_oracle_error(RuntimeError(f"{code}: simulated"))
    assert len(diag.summary) >= 10
    assert len(diag.next_step) >= 20
    actionable_markers = (
        "kubectl", "oci ", "ALTER ", "GRANT ", "Secret", "secret",
        "Console", "ORACLE_", "TNS_", "wallet", "subnet", "policy",
        "compartment", "--set", "Helm", "OCI ", "VCN", "provisioner",
        "helm uninstall",
    )
    assert any(m in diag.next_step for m in actionable_markers), (
        f"{code}: next_step has no actionable artefact: {diag.next_step!r}"
    )


def test_password_never_appears_in_diagnosis():
    err = RuntimeError("ORA-01017: invalid username/password")
    diag = diagnose_oracle_error(err, dsn="myhost:1521/svc", user="ragapp")
    text = diag.user_message + diag.summary + diag.raw_message
    for forbidden in ("supersecret", "P@ssw0rd", "P4ssword!"):
        assert forbidden not in text


def test_dsn_with_inline_credentials_is_redacted():
    diag = diagnose_oracle_error(
        RuntimeError("ORA-12541: TNS:no listener"),
        dsn="tcps://admin:secret123@db.example.com:1522/service",
        user="ragapp",
    )
    assert "secret123" not in diag.user_message
    assert "***" in diag.summary


def test_unknown_error_returns_other_category():
    diag = diagnose_oracle_error(RuntimeError("Something completely random: BANG"))
    assert diag.code == ""
    assert diag.category == "other"
    assert "BANG" in diag.raw_message


def test_user_message_combines_summary_and_step():
    diag = diagnose_oracle_error(RuntimeError("ORA-12541: ADB unreachable"))
    assert diag.summary in diag.user_message
    assert diag.next_step in diag.user_message
    assert "\n" not in diag.user_message


@pytest.mark.parametrize(("hint", "expected_cat"), [
    ("NotAuthorizedOrNotFound for resource",                       "auth"),
    ("InvalidParameter: displayName already exists",               "conflict"),
    ("LimitExceeded: free-tier ADB count",                         "quota"),
    ("SubnetNotFound: subnet must be in same VCN",                 "config"),
    ("compartment ocid1.compartment.oc1..xyz not found",           "config"),
    ("region us-phoenix-1 not subscribed",                         "config"),
])
def test_oci_error_categorisation(hint, expected_cat):
    diag = diagnose_oci_error(RuntimeError(hint))
    assert diag.category == expected_cat
    assert len(diag.next_step) >= 20


def test_oci_unknown_falls_back_safely():
    diag = diagnose_oci_error(RuntimeError("Some random non-OCI message"))
    assert diag.code == "OCI-UNKNOWN"
    assert "policy" in diag.next_step.lower()


def test_no_drift_in_oracle_error_table():
    """Future contributor adds a category typo → this catches it."""
    from nvidia_rag.utils.vdb.oracle import oracle_errors
    for code, (cat, summary, next_step) in oracle_errors._TABLE.items():
        assert cat in VALID_CATEGORIES, f"{code}: unknown category {cat!r}"
        assert summary
        assert next_step

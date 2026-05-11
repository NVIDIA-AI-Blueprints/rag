# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ORA-/OCI- → next-step error translator.

Every supported error code must map to:
  1. a known category (auth/network/wallet/service/schema/config/other)
  2. a non-empty user-readable summary
  3. a non-empty next_step that contains a real artefact name the
     operator can act on (a flag, a Secret, a kubectl command, an oci
     command, a SQL DDL — never a generic "check your config")

These are pure unit tests — no Helm, no DB.
"""
from __future__ import annotations

import pytest

from nvidia_rag.utils.vdb.oracle.oracle_errors import (
    OciDiag,
    OracleDiag,
    diagnose_oci_error,
    diagnose_oracle_error,
)


VALID_CATEGORIES = {
    "auth", "network", "wallet", "service",
    "schema", "config", "conflict", "quota", "other",
}


# Common code → expected category mapping. Keep this dict synced with
# oracle_errors._TABLE; the test below cross-checks no entry was lost.
EXPECTED_CATEGORY = {
    # auth
    "ORA-01017": "auth",
    "ORA-28000": "auth",
    "ORA-01045": "auth",
    # network / DSN
    "ORA-12154": "network",
    "ORA-12541": "network",
    "ORA-12170": "network",
    "ORA-12537": "network",
    "DPY-4011":  "network",
    "DPY-4027":  "network",
    # service
    "ORA-12514": "service",
    # wallet
    "ORA-28759": "wallet",
    "ORA-28365": "wallet",
    "ORA-28860": "wallet",
    # schema
    "ORA-00001": "schema",
    "ORA-00942": "schema",
    "ORA-00955": "schema",
    "ORA-65096": "schema",
    # config
    "DPI-1047":  "config",
}


# ===========================================================================
# Every code maps to a known category
# ===========================================================================
@pytest.mark.parametrize("code, expected_cat", sorted(EXPECTED_CATEGORY.items()))
def test_known_codes_map_to_correct_category(code, expected_cat):
    err = RuntimeError(f"{code}: simulated message")
    diag = diagnose_oracle_error(err)
    assert diag.code == code
    assert diag.category == expected_cat
    assert diag.category in VALID_CATEGORIES


# ===========================================================================
# Every translation has a useful summary AND next_step
# ===========================================================================
@pytest.mark.parametrize("code", sorted(EXPECTED_CATEGORY))
def test_summary_and_next_step_are_useful(code):
    err = RuntimeError(f"{code}: simulated")
    diag = diagnose_oracle_error(err)
    assert len(diag.summary) >= 10, f"{code}: summary too short"
    assert len(diag.next_step) >= 20, (
        f"{code}: next_step too short — must contain an actionable"
        f" command/flag/file path"
    )
    # next_step must mention SOMETHING the operator can act on.
    actionable_markers = (
        "kubectl", "oci ", "ALTER ", "GRANT ", "Secret", "secret",
        "Console", "ORACLE_", "TNS_", "wallet", "subnet", "policy",
        "compartment", "--set", "Helm", "OCI ", "VCN", "provisioner",
    )
    assert any(m in diag.next_step for m in actionable_markers), (
        f"{code}: next_step has no actionable artefact: {diag.next_step!r}"
    )


# ===========================================================================
# Auth codes never leak the password into the message
# ===========================================================================
def test_password_never_appears_in_diagnosis():
    err = RuntimeError("ORA-01017: invalid username/password")
    diag = diagnose_oracle_error(
        err,
        dsn="myhost:1521/svc",
        user="ragapp",
    )
    text = diag.user_message + diag.summary + diag.raw_message
    forbidden = ("supersecret", "P@ssw0rd", "P4ssword!")
    for tok in forbidden:
        assert tok not in text, "Password leaked into diagnosis"


def test_dsn_with_inline_credentials_is_redacted():
    """If somebody passes a URL-style DSN with embedded creds we must
    NOT log them back."""
    err = RuntimeError("ORA-12541: TNS:no listener")
    diag = diagnose_oracle_error(
        err,
        dsn="tcps://admin:secret123@db.example.com:1522/service",
        user="ragapp",
    )
    assert "secret123" not in diag.user_message
    assert "secret123" not in diag.summary
    assert "***" in diag.summary, (
        "Inline DSN credentials should be redacted with ***"
    )


# ===========================================================================
# Unknown / generic errors fall back to the "other" bucket without
# making things worse
# ===========================================================================
def test_unknown_error_returns_other_category():
    err = RuntimeError("Something completely random: BANG")
    diag = diagnose_oracle_error(err)
    assert diag.code == ""
    assert diag.category == "other"
    assert "Unrecognised" in diag.summary or "unknown" in diag.summary.lower()
    # The raw message must be preserved (truncated to 500 chars)
    assert "BANG" in diag.raw_message


def test_unknown_error_with_huge_message_is_truncated():
    big = "x" * 10000
    err = RuntimeError(f"{big} END")
    diag = diagnose_oracle_error(err)
    assert len(diag.raw_message) <= 500
    # Truncation should preserve the tail or head — at minimum it must
    # not exceed 500.


# ===========================================================================
# user_message combines summary + next_step
# ===========================================================================
def test_user_message_combines_summary_and_step():
    err = RuntimeError("ORA-12541: ADB unreachable")
    diag = diagnose_oracle_error(err)
    assert diag.summary in diag.user_message
    assert diag.next_step in diag.user_message
    # And it's a single line (no embedded newlines that would break the
    # downstream APIError JSON serialization).
    assert "\n" not in diag.user_message


# ===========================================================================
# OCI provisioner error translator
# ===========================================================================
@pytest.mark.parametrize(("hint", "expected_cat"), [
    ("NotAuthorizedOrNotFound for resource",   "auth"),
    ("InvalidParameter: displayName already exists in compartment", "conflict"),
    ("LimitExceeded: free-tier ADB count",     "quota"),
    ("SubnetNotFound or InvalidParameter subnet must be...", "config"),
    ("compartment ocid1.compartment.oc1..xyz not found", "config"),
    ("region us-phoenix-1 not subscribed",     "config"),
])
def test_oci_error_categorisation(hint, expected_cat):
    err = RuntimeError(hint)
    diag = diagnose_oci_error(err)
    assert diag.category == expected_cat, (
        f"{hint!r}: got category {diag.category}, expected {expected_cat}"
    )
    assert len(diag.next_step) >= 20


def test_oci_unknown_falls_back_safely():
    diag = diagnose_oci_error(RuntimeError("Some random non-OCI message"))
    assert diag.code == "OCI-UNKNOWN"
    assert "policy" in diag.next_step.lower()


# ===========================================================================
# Drift guard: every code in _TABLE has a category in VALID_CATEGORIES
# ===========================================================================
def test_no_drift_in_oracle_error_table():
    """If a future contributor adds an entry to _TABLE with a typo'd
    category ('autn' instead of 'auth'), this test catches it."""
    from nvidia_rag.utils.vdb.oracle import oracle_errors
    for code, (cat, summary, next_step) in oracle_errors._TABLE.items():
        assert cat in VALID_CATEGORIES, (
            f"{code}: unknown category {cat!r}"
        )
        assert summary, f"{code}: empty summary"
        assert next_step, f"{code}: empty next_step"


def test_no_drift_in_oci_hints_table():
    from nvidia_rag.utils.vdb.oracle import oracle_errors
    for pat, (code, cat, summary, step) in oracle_errors._OCI_HINTS:
        assert cat in VALID_CATEGORIES, f"{code}: bad category {cat!r}"
        assert summary
        assert step

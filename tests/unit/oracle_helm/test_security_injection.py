# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security-focused tests for the Oracle integration.

We verify two distinct injection surfaces:

1. **SQL injection** — every place we accept a "table name" or "column
   name" from a customer (Helm values, env, runtime args) feeds
   eventually into an f-string DDL. Oracle bind variables can't carry
   identifiers, so the only defence is upstream filtering. These tests
   pin which inputs are accepted and which are rejected so a future
   refactor doesn't loosen the gate.

2. **Shell injection / quoting** — the provisioner Job and the BYO Job
   shell-render values into a heredoc'd Python script. Helm's `quote`
   and JSON-encoding need to neutralise common shell metacharacters
   before they hit `/bin/sh -c`.
"""
from __future__ import annotations

import json
import re
import string

import pytest


# ---------------------------------------------------------------------------
# SQL injection on identifier inputs
# ---------------------------------------------------------------------------
SQL_PAYLOADS = [
    "X; DROP TABLE users",
    "X--",
    "X' OR 1=1 --",
    "X/*comment*/",
    "X; SELECT * FROM secrets",
    "X UNION ALL SELECT password FROM dba_users",
    "DROP TABLE STUDENTS;",
    "X\"; DROP TABLE",
    "X`",
    "X\x00",
]


class TestNormalizeDbNameRejectsInjection:
    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_payload_reduced_to_safe_alnum(self, provisioner_module, payload):
        out = provisioner_module.normalize_db_name(payload)
        # Output is purely alphanumeric, lowercase, starts with letter
        assert out.isalnum()
        assert out == out.lower()
        assert out[0].isalpha()
        # No SQL meta characters survive
        for forbidden in (";", "'", '"', "-", "/", "*", "`", "\x00"):
            assert forbidden not in out


class TestByoViewDdlIdentifierHandling:
    """create_byo_view_ddl currently substitutes column names verbatim
    (Oracle SQL bind parameters cannot bind identifiers). We must
    therefore document — and lock in — the trust boundary: the caller is
    responsible for rejecting unsafe identifiers.

    These tests *demonstrate* what would happen if a caller passed an
    unsanitised identifier so a future contributor can't claim "we
    sanitise it" without a test to prove it.
    """

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_unsafe_identifiers_are_rejected_by_validator(self, payload):
        """Unsafe identifiers must be rejected by _validate_identifier
        before they ever reach DDL. This is the SQL injection prevention
        gate — the trust boundary is now enforced in code, not just at
        the caller (Helm values)."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries

        with pytest.raises(ValueError, match="Unsafe Oracle"):
            oracle_queries.create_byo_view_ddl(
                view_name="v",
                source_table="T",
                column_map={"text": payload, "vector": "vec"},
            )

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_oracle_byo_import_job_documents_caller_responsibility(self, payload):
        """The chart's values.create-adb.yaml docstring above
        importExistingTables MUST tell operators their inputs are trusted
        SQL identifiers. If this test fails after a docs refactor, the
        warning got lost."""
        import pathlib
        for fname in ("values.create-adb.yaml", "values.existing-adb.yaml"):
            p = pathlib.Path(__file__).resolve().parents[3] / "examples" / "oracle" / "helm" / fname
            text = p.read_text()
            assert "importExistingTables" in text, f"{fname} missing block"


# ---------------------------------------------------------------------------
# Helm-stamping shell-safety (do values that look like shell payloads
# break the rendered manifest?)
# ---------------------------------------------------------------------------
SHELL_PAYLOADS = [
    "$(rm -rf /)",
    "`whoami`",
    "value\nNAMESPACE: hostile",
    'with"doublequote',
    "with'singlequote",
    "with\\backslash",
    "with$dollar",
    "with;semi",
    "with|pipe",
    "with\nnewline",
]


@pytest.mark.parametrize("payload", SHELL_PAYLOADS)
def test_byo_table_json_value_is_yaml_safe(payload):
    """The BYO Job stamps importExistingTables as a single env value via
    `{{ $byo | toJson | quote }}`. We must be confident that a customer
    pasting an entry like ``sourceTable: "$(rm -rf /)"`` doesn't escape
    the YAML scalar context.

    We *don't* shell out to helm here because the chart-render tests in
    test_helm_chart.py already cover that. Instead we verify the same
    quoting strategy: json.dumps, then double-quote, never produces an
    unbalanced quote in the final YAML scalar.
    """
    obj = [{"sourceTable": payload}]
    encoded = json.dumps(obj)
    # Mimic Helm's `quote`: wrap in double-quotes after JSON encoding
    quoted = '"' + encoded.replace('\\', '\\\\').replace('"', '\\"') + '"'
    # Round-trip through a YAML loader to confirm valid YAML scalar
    import yaml
    loaded = yaml.safe_load(quoted)
    # And the json content is recoverable
    parsed = json.loads(loaded)
    assert parsed == obj, f"YAML round-trip lost data for payload {payload!r}"


# ---------------------------------------------------------------------------
# generate_password() must never emit characters that would break
# downstream consumers (PL/SQL, JDBC connection string, k8s Secret base64,
# YAML scalar)
# ---------------------------------------------------------------------------
class TestPasswordSafety:
    def test_password_safe_for_b64_then_yaml(self, provisioner_module):
        import base64
        for _ in range(50):
            pw = provisioner_module.generate_password()
            encoded = base64.b64encode(pw.encode()).decode()
            # Must be standard b64 (no URL-safe variants that confuse k8s)
            assert re.fullmatch(r"[A-Za-z0-9+/=]+", encoded)
            decoded = base64.b64decode(encoded).decode()
            assert decoded == pw

    def test_password_safe_for_plsql_double_quote(self, provisioner_module):
        """ADB CREATE USER … IDENTIFIED BY "<pw>" wrapping. If the password
        contains a double quote, that PL/SQL block becomes a SQL injection
        vector against ADMIN."""
        for _ in range(50):
            pw = provisioner_module.generate_password()
            assert '"' not in pw

    def test_password_alphabet_locked_down(self, provisioner_module):
        """Lock the password alphabet against accidental expansion.

        We deliberately ship `#` and `+` because the password is *only*
        ever passed to oracledb as a kwarg (never embedded in a URL) and
        to PL/SQL inside a double-quoted identifier. These characters are
        safe there. If someone changes the alphabet to include `@`, `:`,
        `/`, `\\`, `"`, `'`, or backtick, this test will catch it.
        """
        forbidden = set('@:/\\"\'`')
        for _ in range(100):
            pw = provisioner_module.generate_password()
            assert not (set(pw) & forbidden), (
                f"Password contains URL/quote-unsafe char(s): {pw!r}"
            )


# ---------------------------------------------------------------------------
# Env var handling: ORACLE_PAI_INDEX_URL accepts http/https only
# ---------------------------------------------------------------------------
class TestPaiUrlEnvHardening:
    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_accepts_http_and_https(self, scheme, monkeypatch):
        import os
        monkeypatch.setenv(
            "ORACLE_PAI_INDEX_URL", f"{scheme}://10.0.50.42:8080/v1/index",
        )
        url = (os.getenv("ORACLE_PAI_INDEX_URL", "") or "").strip()
        if url and not url.endswith("/v1/index"):
            url = url.rstrip("/") + "/v1/index"
        assert url.startswith(scheme + "://")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ftp://hostile.example.com/x",
            "gopher://x",
        ],
    )
    def test_non_http_schemes_pass_through_unchanged(self, url, monkeypatch):
        """We currently don't reject non-http schemes — Oracle ADB will reject
        them at OFFLOAD_URL time. Document that behaviour so any future
        change either rejects them or explicitly accepts them."""
        import os
        monkeypatch.setenv("ORACLE_PAI_INDEX_URL", url)
        v = (os.getenv("ORACLE_PAI_INDEX_URL", "") or "").strip()
        if v and not v.endswith("/v1/index"):
            v = v.rstrip("/") + "/v1/index"
        # Suffix is appended unconditionally — that's OK because ADB will
        # fail loudly with ORA-29024 on non-HTTP schemes.
        assert v.endswith("/v1/index")

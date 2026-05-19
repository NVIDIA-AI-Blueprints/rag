# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual BYO Job heredoc against a fake oracledb.

The other heredoc tests are *static* — they parse the script and look
for required patterns. This one is *dynamic*: we render the chart,
extract the heredoc, swap in a fake ``oracledb`` module, and execute
the heredoc body in-process. Then we assert the script's behaviour
matches the fail-fast policy customers depend on:

  * a missing source table → ``sys.exit(1)`` with the table name in the log
  * a missing column → ``sys.exit(1)`` with the column name + alternatives
  * a non-VECTOR column → ``sys.exit(1)`` with a hint to use TO_VECTOR
  * a fully valid entry → exit 0, view appears in created list
  * partial failures → ALL bad entries are reported in one log scrape,
    THEN sys.exit(1)

This is the closest thing to "smoke test the install" we can run
without an OKE cluster.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml


HELM = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM is None, reason="helm not on PATH")
REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "examples" / "oracle" / "helm"


# ---------------------------------------------------------------------------
# Fake oracledb
# ---------------------------------------------------------------------------
class FakeError(Exception):
    pass


class FakeCursor:
    """Programmable cursor.

    The schema fixture has:
      - tables: { (owner, name): {col_name: data_type} }
      - existing_views: set of (owner, name)
    The cursor matches the SELECTs the heredoc actually issues
    (``all_tables``, ``all_views``, ``all_tab_columns``,
    ``user_tab_columns``, ``user_views``) and returns canned rows.
    """

    def __init__(self, schema):
        self.schema = schema
        self._rows: list = []
        self.executed: list[tuple[str, dict | None]] = []

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def close(self): pass

    def execute(self, sql, binds=None):
        self.executed.append((sql, binds))
        s = " ".join(sql.split()).upper()
        if "ALL_TABLES" in s and "COUNT(*)" in s:
            owner = (binds or {}).get("o") if binds else None
            name = (binds or {}).get("t") if binds else None
            n = sum(1 for (o, t) in self.schema["tables"]
                    if t == name and (owner is None or o == owner))
            self._rows = [(n,)]
            return
        if ("ALL_VIEWS" in s or "USER_VIEWS" in s) and "COUNT(*)" in s:
            owner = (binds or {}).get("o") if binds else None
            name = (binds or {}).get("t") or (binds or {}).get("n") if binds else None
            n = sum(1 for (o, t) in self.schema["views"]
                    if t == name and (owner is None or o == owner))
            self._rows = [(n,)]
            return
        if "ALL_TAB_COLUMNS" in s and "COLUMN_NAME" in s and "DATA_TYPE" in s:
            owner = (binds or {}).get("o") if binds else None
            name = (binds or {}).get("t")
            cols = {}
            for (o, t), c in self.schema["tables"].items():
                if t == name and (owner is None or o == owner):
                    cols.update(c)
            self._rows = [(k, v) for k, v in cols.items()]
            return
        if "USER_TAB_COLUMNS" in s and "COLUMN_NAME" in s:
            tname = (binds or {}).get("t")
            cols = {}
            for (o, t), c in self.schema["tables"].items():
                if t == tname:
                    cols.update(c)
            self._rows = [(k,) for k in cols]
            return
        if "USER_TAB_COLUMNS" in s and "DISTINCT TABLE_NAME" in s:
            # Discovery: list any table with a VECTOR column
            seen = set()
            for (o, t), c in self.schema["tables"].items():
                if any(v.startswith("VECTOR") for v in c.values()):
                    seen.add(t)
            self._rows = sorted((t,) for t in seen)
            return
        if "CREATE OR REPLACE VIEW" in s:
            if self.schema.get("ddl_failure"):
                raise FakeError("ORA-00904: invalid identifier")
            self.schema["created_ddl"].append(sql)
            return
        if "MERGE" in s:
            return
        if "DUAL" in s:
            self._rows = [(1,)]
            return
        # default: empty
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        rows = list(self._rows)
        self._rows = []
        return rows

    def __iter__(self):
        return iter(self._rows)

    @property
    def rowcount(self): return 0


class FakeConnection:
    def __init__(self, schema): self.schema = schema
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self): return FakeCursor(self.schema)
    def commit(self): pass
    def close(self): pass


def _build_fake_oracledb(schema):
    mod = ModuleType("oracledb")
    mod.Error = FakeError
    mod.connect = lambda **kw: FakeConnection(schema)
    return mod


# ---------------------------------------------------------------------------
# Heredoc extractor (reads the rendered Job manifest)
# ---------------------------------------------------------------------------
def _render(*flags):
    p = subprocess.run(
        [HELM, "template", "rag-test", ".",
         "--set", "ngcApiSecret.password=fake",
         "--set", "imagePullSecret.password=fake",
         "--set", "oracle.containerRegistry.username=u",
         "--set", "oracle.containerRegistry.password=p",
         # existing-adb mode → BYO Job always renders even with empty
         # importExistingTables (discovery pass runs).
         "-f", "values.existing-adb.yaml",
         "--set", "oracle.existing.user=RAG_APP",
         "--set", "oracle.existing.password=secret",
         "--set", "oracle.existing.connectString=ragdb_medium",
         *flags],
        cwd=str(WRAPPER), capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return p.stdout


def _extract_byo_heredoc():
    """Find the Python heredoc in the rendered BYO Job manifest."""
    rendered = _render()
    for d in yaml.safe_load_all(rendered):
        if not d or d.get("kind") != "Job":
            continue
        if d["metadata"]["name"] != "oracle-byo-import":
            continue
        c = d["spec"]["template"]["spec"]["containers"][0]
        args = c["args"][0]
        m = re.search(r"python <<'PY'\n(.*?)\nPY", args, re.DOTALL)
        return m.group(1) if m else None
    return None


# ---------------------------------------------------------------------------
# Run the heredoc with a fake oracledb + custom env
# ---------------------------------------------------------------------------
@contextmanager
def _exec_heredoc(body, schema, env, capsys):
    fake = _build_fake_oracledb(schema)
    saved_env = {}
    for k, v in env.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v

    saved_modules = sys.modules.get("oracledb")
    sys.modules["oracledb"] = fake
    try:
        ns = {"__name__": "__byo_main__"}
        exit_code = 0
        try:
            exec(compile(body, "<byo-import>", "exec"), ns)
        except SystemExit as e:
            exit_code = int(e.code or 0)
        yield exit_code, capsys.readouterr()
    finally:
        if saved_modules is None:
            sys.modules.pop("oracledb", None)
        else:
            sys.modules["oracledb"] = saved_modules
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_schema(tables=None, views=None, ddl_failure=False):
    return {
        "tables": tables or {},
        "views": views or set(),
        "ddl_failure": ddl_failure,
        "created_ddl": [],
    }


# ===========================================================================
# Smoke test: happy path (one valid entry)
# ===========================================================================
@NEEDS_HELM
def test_happy_path_creates_view_and_exits_zero(capsys):
    body = _extract_byo_heredoc()
    assert body, "BYO heredoc not extractable"

    schema = make_schema(
        tables={("KB", "MY_DOCS"): {
            "ID": "NUMBER", "BODY": "CLOB", "EMBED": "VECTOR(*, 768, FLOAT32)",
            "URL": "VARCHAR2",
        }},
    )
    env = {
        "ORACLE_USER": "RAG_APP",
        "ORACLE_PASSWORD": "p",
        "ORACLE_CS": "ragdb_medium",
        "BYO_TABLES_JSON": json.dumps([{
            "sourceTable": "KB.MY_DOCS",
            "collectionName": "kb_view",
            "columns": {"text": "BODY", "vector": "EMBED", "source": "URL"},
        }]),
    }
    with _exec_heredoc(body, schema, env, capsys) as (rc, captured):
        assert rc == 0, captured.out + captured.err
        assert any("CREATE OR REPLACE VIEW" in s for s in schema["created_ddl"])


# ===========================================================================
# Fail-fast: missing source table
# ===========================================================================
@NEEDS_HELM
def test_missing_source_table_fails_fast(capsys):
    body = _extract_byo_heredoc()
    schema = make_schema(tables={})  # nothing exists
    env = {
        "ORACLE_USER": "RAG_APP", "ORACLE_PASSWORD": "p",
        "ORACLE_CS": "ragdb_medium",
        "BYO_TABLES_JSON": json.dumps([{
            "sourceTable": "DOES_NOT_EXIST",
            "collectionName": "v",
            "columns": {"text": "T", "vector": "V"},
        }]),
    }
    with _exec_heredoc(body, schema, env, capsys) as (rc, captured):
        assert rc == 1, "Job must exit 1 when source table is missing"
        out = captured.out + captured.err
        assert "DOES_NOT_EXIST" in out
        assert "does not exist" in out
        assert "GRANT SELECT" in out or "grant" in out.lower()


# ===========================================================================
# Fail-fast: missing required column
# ===========================================================================
@NEEDS_HELM
def test_missing_column_fails_fast(capsys):
    body = _extract_byo_heredoc()
    schema = make_schema(
        tables={("KB", "MY_DOCS"): {
            "BODY": "CLOB", "EMBED": "VECTOR(*, 768, FLOAT32)",
        }},
    )
    env = {
        "ORACLE_USER": "RAG_APP", "ORACLE_PASSWORD": "p",
        "ORACLE_CS": "ragdb_medium",
        "BYO_TABLES_JSON": json.dumps([{
            "sourceTable": "KB.MY_DOCS",
            "collectionName": "v",
            "columns": {"text": "TYPO_BODY", "vector": "EMBED"},
        }]),
    }
    with _exec_heredoc(body, schema, env, capsys) as (rc, captured):
        assert rc == 1
        out = captured.out + captured.err
        assert "TYPO_BODY" in out
        # And the available columns must be listed so the operator can fix
        assert "BODY" in out and "EMBED" in out


# ===========================================================================
# Fail-fast: 'vector' column points at a non-VECTOR type
# ===========================================================================
@NEEDS_HELM
def test_non_vector_column_fails_fast(capsys):
    body = _extract_byo_heredoc()
    schema = make_schema(
        tables={("KB", "MY_DOCS"): {
            "BODY": "CLOB", "FAKE_VEC": "NUMBER",
        }},
    )
    env = {
        "ORACLE_USER": "RAG_APP", "ORACLE_PASSWORD": "p",
        "ORACLE_CS": "ragdb_medium",
        "BYO_TABLES_JSON": json.dumps([{
            "sourceTable": "KB.MY_DOCS",
            "collectionName": "v",
            "columns": {"text": "BODY", "vector": "FAKE_VEC"},
        }]),
    }
    with _exec_heredoc(body, schema, env, capsys) as (rc, captured):
        assert rc == 1
        out = captured.out + captured.err
        assert "FAKE_VEC" in out
        assert "VECTOR" in out
        assert "TO_VECTOR" in out or "convert" in out.lower()


# ===========================================================================
# Fail-fast: ALL bad entries reported, then exit 1
# ===========================================================================
@NEEDS_HELM
def test_multiple_failures_all_reported_then_exits_one(capsys):
    body = _extract_byo_heredoc()
    schema = make_schema(
        tables={("KB", "GOOD"): {"BODY": "CLOB", "EMBED": "VECTOR(*, 768, FLOAT32)"}},
    )
    env = {
        "ORACLE_USER": "RAG_APP", "ORACLE_PASSWORD": "p",
        "ORACLE_CS": "ragdb_medium",
        "BYO_TABLES_JSON": json.dumps([
            {"sourceTable": "KB.GOOD", "collectionName": "ok",
             "columns": {"text": "BODY", "vector": "EMBED"}},
            {"sourceTable": "KB.MISSING", "collectionName": "missing_one",
             "columns": {"text": "T", "vector": "V"}},
            {"sourceTable": "KB.GOOD", "collectionName": "bad_col",
             "columns": {"text": "TYPO", "vector": "EMBED"}},
        ]),
    }
    with _exec_heredoc(body, schema, env, capsys) as (rc, captured):
        out = captured.out + captured.err
        assert rc == 1
        # First entry succeeded
        assert "ok ->" in out or "VIEW ok" in out
        # Both errors must appear in the SAME log scrape
        assert "MISSING" in out
        assert "TYPO" in out
        # And the summary block must appear
        assert "BYO mapping had errors" in out

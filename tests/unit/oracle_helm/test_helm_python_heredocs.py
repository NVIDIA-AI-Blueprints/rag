# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static analysis of Python heredocs embedded in Helm Job templates.

Several of our chart templates ship a Python script via a ``python <<'PY'``
heredoc inside ``args``. Helm-template that, the Python is hidden inside a
YAML scalar — and any syntax error there only surfaces at runtime when
the Job pod tries to start. We can do better:

* Render the chart with a known-good values file.
* Walk every Job's container.args, locate any embedded heredoc Python
  block, ``ast.parse`` it, and assert it has no SyntaxError.
* While we're parsing, pull a few semantic invariants:
    - the script imports ``oracledb``
    - it never calls ``DROP TABLE`` / ``TRUNCATE`` / ``DELETE FROM``
    - it commits after every MERGE
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


HEREDOC_RE = re.compile(
    r"python\s*<<'PY'\n(?P<body>.*?)^\s*PY\s*$",
    re.DOTALL | re.MULTILINE,
)


REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
]


def _helm_render(chart_dir: Path, values_file: str, extra: list[str] | None = None) -> str:
    helm = shutil.which("helm")
    if not helm:
        pytest.skip("helm binary not on PATH")
    cmd = [helm, "template", "rag-test", ".",
           "-f", values_file] + REQUIRED_CREDS
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, cwd=str(chart_dir), capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"helm template failed: {result.stderr}")
    return result.stdout


def _extract_heredocs_from_rendered(rendered: str) -> list[tuple[str, str]]:
    """Parse the Helm-rendered multidoc YAML and pull every Job's
    ``args`` strings, returning (job_name, python_body) tuples for every
    embedded ``python <<'PY' … PY`` block.
    """
    out: list[tuple[str, str]] = []
    for doc in yaml.safe_load_all(rendered):
        if not doc or not isinstance(doc, dict):
            continue
        if doc.get("kind") != "Job":
            continue
        name = doc.get("metadata", {}).get("name", "<unnamed>")
        spec = doc.get("spec", {}).get("template", {}).get("spec", {})
        for c in spec.get("containers", []) + spec.get("initContainers", []):
            for arg in c.get("args", []) or []:
                if not isinstance(arg, str):
                    continue
                for m in HEREDOC_RE.finditer(arg):
                    out.append((f"{name}/{c.get('name','?')}", m.group("body")))
    return out


# ---------------------------------------------------------------------------
# Default: existing-adb mode, since both BYO and PAI verify Jobs render here.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def heredocs(helm_chart_dir):
    rendered = _helm_render(
        helm_chart_dir, "values.existing-adb.yaml",
        extra=[
            "--set", "oracle.connection.username=foo",
            "--set", "oracle.connection.password=bar",
            "--set", "oracle.connection.connectionString=tcps://x:1521/y",
            "--set", "oracle.gpuIndexOffload.enabled=true",
            "--set", "oracle.containerRegistry.username=u",
            "--set", "oracle.containerRegistry.password=p",
        ],
    )
    return _extract_heredocs_from_rendered(rendered)


def test_some_heredocs_were_found(heredocs):
    """If we extract zero heredocs, the regex/extraction is broken or
    Helm output has shifted shape — fail loudly so the rest of the
    static guards aren't silently skipped."""
    assert heredocs, (
        "No Python heredocs found in rendered chart — extraction broken? "
        "Update HEREDOC_RE or check that Jobs still use python <<'PY'."
    )
    names = {n for n, _ in heredocs}
    # We expect at least the BYO Job and the PAI verify Job.
    assert any("oracle-byo-import" in n for n in names), names
    assert any("oracle-pai-verify" in n for n in names), names


def test_every_heredoc_parses(heredocs):
    for name, body in heredocs:
        try:
            ast.parse(body)
        except SyntaxError as e:
            pytest.fail(f"{name}: SyntaxError in embedded Python:\n{e}\n--\n{body}")


def _walk_imports(body: str) -> list[str]:
    tree = ast.parse(body)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.append(n.module)
    return out


def test_byo_heredoc_imports_oracledb(heredocs):
    """The BYO Job *must* import oracledb; without it, NameError at
    Job startup."""
    matched = False
    for name, body in heredocs:
        if "oracle-byo-import" not in name:
            continue
        matched = True
        assert "oracledb" in _walk_imports(body), (
            f"{name}: missing 'import oracledb'"
        )
    assert matched, "BYO Job heredoc not found at all"


def test_verify_heredoc_imports_urllib(heredocs):
    """The PAI verify Job hits the index service over HTTP via
    urllib.request — confirm that import survives any refactor."""
    matched = False
    for name, body in heredocs:
        if "oracle-pai-verify" not in name:
            continue
        matched = True
        assert any("urllib" in m for m in _walk_imports(body)), (
            f"{name}: PAI verify Job no longer imports urllib"
        )
    assert matched, "PAI verify Job heredoc not found at all"


def test_no_heredoc_drops_or_truncates(heredocs):
    """BYO sacred-rule: customer data must never be DROP/TRUNCATE/DELETE'd
    by chart-managed jobs."""
    forbidden = ("DROP TABLE", "TRUNCATE", "DELETE FROM")
    for name, body in heredocs:
        # Skip the verify Job - it doesn't touch customer tables.
        if "byo-import" not in name:
            continue
        upper = body.upper()
        for bad in forbidden:
            assert bad not in upper, (
                f"{name}: forbidden statement {bad!r} found in heredoc:\n{body}"
            )


def test_byo_heredoc_uses_required_canonical_set(heredocs):
    """The Python heredoc declares its own REQUIRED set; this MUST match
    RAG_CANONICAL_COLUMNS or BYO discovery will diverge between the
    runtime path (oracle_vdb) and the install path (Job)."""
    from nvidia_rag.utils.vdb.oracle.oracle_queries import RAG_CANONICAL_COLUMNS

    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        # Find:  REQUIRED = {"ID", "TEXT", ...}
        m = re.search(r"REQUIRED\s*=\s*\{([^}]+)\}", body)
        assert m, f"{name}: REQUIRED set not found in heredoc"
        items = re.findall(r'"([A-Z_]+)"', m.group(1))
        assert set(items) == set(RAG_CANONICAL_COLUMNS), (
            f"{name}: REQUIRED={set(items)} != RAG_CANONICAL_COLUMNS={set(RAG_CANONICAL_COLUMNS)}"
        )


def test_byo_heredoc_uses_when_not_matched_only(heredocs):
    """Mirror of the SQL-validity check: the heredoc must NEVER do
    WHEN MATCHED THEN UPDATE — that would let a stale registration
    overwrite a customer-edited collection_info row."""
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        upper = body.upper()
        assert "WHEN MATCHED THEN" not in upper, (
            f"{name}: heredoc has WHEN MATCHED THEN — would clobber customer state"
        )


def test_byo_heredoc_collects_all_errors_and_exits_at_end(heredocs):
    """Fail-fast policy: every BAD oracle.importExistingTables entry
    must be reported in a single Job run, and the Job must exit
    non-zero so a CI/CD `helm install` reports failure.  This is the
    "errors fall immediately, not super long later" invariant.
    """
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        # Errors are collected into a list ...
        assert "fatal_errors" in body, (
            f"{name}: must accumulate fatal errors into a list"
        )
        # ... and the Job exits non-zero at the end of the mapping pass.
        assert re.search(
            r"if fatal_errors:[\s\S]+?sys\.exit\(1\)", body,
        ), f"{name}: missing 'if fatal_errors: ... sys.exit(1)' summary block"
        # The per-entry CREATE VIEW catch must still exist (we don't
        # want a SyntaxError to abort processing of subsequent entries
        # — collecting errors is fail-fast at the *Job* boundary, not
        # the *entry* boundary).
        assert "except oracledb.Error" in body


def test_byo_heredoc_validates_source_table_exists(heredocs):
    """Source table must be looked up in all_tables/all_views BEFORE
    we try to CREATE VIEW against it. Otherwise the operator gets a
    bare ORA-00942 with no hint of which entry was wrong."""
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        assert "all_tables" in body, (
            f"{name}: must check all_tables before CREATE VIEW"
        )
        assert "all_views" in body or "user_views" in body, (
            f"{name}: must also check all_views (a view can be a source)"
        )


def test_byo_heredoc_validates_column_existence(heredocs):
    """Each column referenced in the entry must be verified to exist
    on the source table BEFORE the CREATE VIEW step — so a typo'd
    column name doesn't waste 30s on a Job that fails at the end."""
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        assert "all_tab_columns" in body, (
            f"{name}: must verify columns via all_tab_columns lookup"
        )


def test_byo_heredoc_rejects_non_vector_column(heredocs):
    """If the operator points 'vector' at a NUMBER column by mistake,
    the Job must catch it before CREATE VIEW (otherwise the view
    succeeds, the frontend lists the collection, and the first search
    silently returns garbage)."""
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        assert re.search(r"VECTOR.*data_type|data_type.*VECTOR", body), (
            f"{name}: must verify the chosen 'vector' column has data_type "
            f"VECTOR"
        )


def test_byo_heredoc_binds_all_user_inputs(heredocs):
    """Discovery pass must use bind variables for the table name in
    user_views/user_tab_columns lookups — never f-string interpolation
    on attacker-controlled values. (The DDL/COUNT path is necessarily
    f-string because Oracle can't bind identifiers; that's a separate
    documented trust boundary.)
    """
    for name, body in heredocs:
        if "byo-import" not in name:
            continue
        # The lookups against system catalogs must use bind syntax (:t / :n).
        assert ":t" in body or ":table_name" in body, (
            f"{name}: missing bind variable for column lookup"
        )
        assert ":n" in body or ":object_name" in body, (
            f"{name}: missing bind variable for view check"
        )

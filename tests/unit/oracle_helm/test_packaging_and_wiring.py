# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaging & integration-wiring tests for the Oracle PR.

These pin every "small" thing the original PR added: pyproject extras,
Dockerfile install lines, dispatcher routing, doc cross-references,
and that no Oracle code accidentally imports from the OCI provisioner
script (because that script is intentionally NOT installed at runtime).

Pure-Python — no external services, no helm, no docker.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

try:  # tomllib is stdlib on 3.11+
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# pyproject.toml — extras
# ---------------------------------------------------------------------------
def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_pyproject_declares_oracle_extra():
    proj = _pyproject()
    extras = proj["project"]["optional-dependencies"]
    assert "oracle" in extras, (
        "pyproject.toml must declare an [oracle] extra so customers / "
        "CI can run `pip install .[oracle]`."
    )


def test_oracle_extra_pulls_oracledb_and_langchain_oracledb():
    extras = _pyproject()["project"]["optional-dependencies"]["oracle"]
    joined = " ".join(extras).lower()
    assert "oracledb" in joined, "[oracle] must depend on `oracledb`"
    assert "langchain-oracledb" in joined, (
        "[oracle] must depend on `langchain-oracledb`"
    )


def test_oracle_packages_pinned_with_lower_bound():
    """Floors must be set so we don't pick up an ancient, broken release."""
    extras = _pyproject()["project"]["optional-dependencies"]["oracle"]
    for spec in extras:
        if "oracledb" in spec or "langchain-oracledb" in spec:
            assert ">=" in spec, (
                f"{spec!r} should pin a lower bound (>=) — otherwise pip "
                "could resolve to an old, broken release."
            )


def test_all_extra_includes_oracle_packages():
    """`pip install .[all]` must pull oracle deps too — that's how
    upstream CI actually exercises our adapter."""
    extras = _pyproject()["project"]["optional-dependencies"]
    all_specs = " ".join(extras["all"]).lower()
    assert "oracledb" in all_specs, (
        "`[all]` must include `oracledb` — CI runs `pip install .[all]` "
        "and the Oracle adapter must be importable."
    )
    assert "langchain-oracledb" in all_specs


# ---------------------------------------------------------------------------
# Dockerfiles — both rag-server and ingestor-server must install [oracle].
# ---------------------------------------------------------------------------
DOCKERFILES = [
    REPO_ROOT / "src" / "nvidia_rag" / "rag_server" / "Dockerfile",
    REPO_ROOT / "src" / "nvidia_rag" / "ingestor_server" / "Dockerfile",
]


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_dockerfile_uv_sync_includes_oracle_extra(dockerfile):
    text = dockerfile.read_text()
    uv_sync_lines = [ln for ln in text.splitlines() if "uv sync" in ln]
    assert uv_sync_lines, f"{dockerfile} has no `uv sync` line"
    assert any("--extra oracle" in ln for ln in uv_sync_lines), (
        f"{dockerfile.relative_to(REPO_ROOT)}: `uv sync` line must include "
        "`--extra oracle`, otherwise the runtime image won't have oracledb. "
        f"Found: {uv_sync_lines!r}"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_dockerfile_does_not_install_oracle_via_pip_at_runtime(dockerfile):
    """oracledb must come from the pyproject extras path, not an ad-hoc
    `pip install oracledb` inside the Dockerfile (avoids version drift)."""
    text = dockerfile.read_text()
    forbidden = re.compile(r"pip\s+install\s+(?:--\S+\s+)*['\"]?oracledb")
    assert not forbidden.search(text), (
        f"{dockerfile.relative_to(REPO_ROOT)} has an ad-hoc "
        "`pip install oracledb`. Use the `[oracle]` extra instead so the "
        "version is pinned in one place (pyproject.toml)."
    )


# ---------------------------------------------------------------------------
# Dispatcher — `vector_store.name == "oracle"` returns OracleVDB.
# ---------------------------------------------------------------------------
def test_dispatcher_has_oracle_branch():
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "__init__.py").read_text()
    assert 'config.vector_store.name == "oracle"' in src, (
        "vdb/__init__.py must dispatch on vector_store.name == 'oracle'"
    )
    assert "from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB" in src


def test_dispatcher_returns_oracle_vdb_for_oracle_name():
    """The dispatcher must contain an `if … == 'oracle'` branch that
    constructs OracleVDB. (We verify via AST instead of instantiating
    because OracleVDB requires a live DB.)"""
    import ast
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "__init__.py").read_text()
    tree = ast.parse(src)
    found_oracle_branch = False
    found_oracle_vdb_construct = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if (isinstance(node.left, ast.Attribute)
                and node.left.attr == "name"
                and any(isinstance(c, ast.Constant) and c.value == "oracle"
                        for c in node.comparators)):
                found_oracle_branch = True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "OracleVDB":
                found_oracle_vdb_construct = True
            elif isinstance(func, ast.Attribute) and func.attr == "OracleVDB":
                found_oracle_vdb_construct = True
    assert found_oracle_branch, (
        "vdb/__init__.py is missing an `if config.vector_store.name == "
        "'oracle'` dispatch branch."
    )
    assert found_oracle_vdb_construct, (
        "vdb/__init__.py's oracle branch must construct an OracleVDB(...) "
        "instance."
    )


def test_dispatcher_does_not_break_milvus_branch():
    """Adding the Oracle branch must not have removed milvus/elastic."""
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "__init__.py").read_text()
    assert 'config.vector_store.name == "milvus"' in src
    assert 'config.vector_store.name == "elasticsearch"' in src or (
        'config.vector_store.name == "elastic"' in src
    )


# ---------------------------------------------------------------------------
# Imports — every Oracle public symbol resolves from a fresh interpreter.
# ---------------------------------------------------------------------------
def test_oracle_module_imports_cleanly():
    """A subprocess `<our-python> -c "import …"` must succeed for each
    Oracle public module."""
    pytest.importorskip("oracledb")  # adapter requires the runtime dep
    import sys
    proc = subprocess.run(
        [sys.executable, "-c",
         "import nvidia_rag.utils.vdb.oracle.oracle_vdb;"
         "import nvidia_rag.utils.vdb.oracle.oracle_queries;"
         "import nvidia_rag.utils.vdb.oracle.oracle_errors;"
         "from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB;"
         "from nvidia_rag.utils.vdb.oracle.oracle_errors import "
         "diagnose_oracle_error, diagnose_oci_error;"
         "print('OK')"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"Oracle module import failed (exit {proc.returncode}).\n"
        f"STDOUT: {proc.stdout!r}\nSTDERR: {proc.stderr!r}"
    )
    assert "OK" in proc.stdout


def test_oracle_adapter_does_not_import_oci_sdk_at_module_level():
    """Importing the Oracle adapter must NOT pull in `oci` (the OCI Python
    SDK). That SDK lives in the accelerator pack only, and is not a
    runtime dependency of the blueprint."""
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_vdb.py").read_text()
    assert not re.search(r"^\s*import\s+oci\b", src, re.MULTILINE)
    assert not re.search(r"^\s*from\s+oci\b", src, re.MULTILINE)


def test_oracle_adapter_does_not_import_kubernetes_at_module_level():
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_vdb.py").read_text()
    assert not re.search(r"^\s*import\s+kubernetes\b", src, re.MULTILINE)
    assert not re.search(r"^\s*from\s+kubernetes\b", src, re.MULTILINE)


# ---------------------------------------------------------------------------
# Documentation cross-references resolve.
# ---------------------------------------------------------------------------
ORACLE_DOC_PATHS = [
    REPO_ROOT / "docs" / "oracle.md",
    REPO_ROOT / "examples" / "oracle" / "helm" / "README.md",
    REPO_ROOT / "examples" / "oracle" / "helm" / "QUICKSTART.md",
    REPO_ROOT / "examples" / "oracle" / "README.md",
]


# Paths that are CREATED at runtime (provisioner Job, docker compose) and
# don't exist in the repo — but customers see them in the docs. Allow-listed.
RUNTIME_GENERATED_PATH_PREFIXES = (
    "examples/oracle/generated",  # docker-compose write target
    "examples/oracle/helm/charts",  # produced by `helm dep update`
)


@pytest.mark.parametrize("doc", [d for d in ORACLE_DOC_PATHS if d.exists()])
def test_doc_internal_path_references_resolve(doc):
    """Every `examples/oracle/...` or `deploy/helm/...` path mentioned in
    a doc must either exist on disk OR be a runtime-generated path."""
    text = doc.read_text()
    pattern = re.compile(
        r"\b(examples/oracle/\S+|deploy/helm/\S+|src/nvidia_rag/\S+)"
    )
    missing = []
    for match in pattern.finditer(text):
        ref = match.group(1).rstrip(".,;:)`'\"]")
        if "*" in ref or "<" in ref or "$" in ref:
            continue
        ref = ref.split("`")[0]
        if any(ref.startswith(prefix) for prefix in RUNTIME_GENERATED_PATH_PREFIXES):
            continue
        target = REPO_ROOT / ref
        if target.exists():
            continue
        if target.parent.exists() and ref.endswith("/"):
            continue
        missing.append(f"{ref} (in {doc.relative_to(REPO_ROOT)})")
    assert not missing, (
        f"{doc.relative_to(REPO_ROOT)} references paths that don't exist:\n  "
        + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# Slim PR boundary — nothing Oracle-specific should leak into the upstream
# chart's templates (we kept the upstream chart strictly generic).
# ---------------------------------------------------------------------------
def test_upstream_chart_templates_have_no_oracle_specific_strings():
    """Inspect every template under deploy/helm/.../templates for any
    Oracle-specific substring like `ORACLE_USER`. The accepted edits in
    the upstream chart are GENERIC (envFrom, extraVolumes)."""
    upstream_templates = (REPO_ROOT / "deploy" / "helm" / "nvidia-blueprint-rag" / "templates")
    if not upstream_templates.is_dir():
        pytest.skip("upstream chart not present")
    forbidden_strings = ("ORACLE_USER", "ORACLE_PASSWORD", "oracle-creds",
                         "RAG_APP", "DBMS_VECTOR")
    leaks = []
    for tpl in upstream_templates.glob("*.yaml"):
        text = tpl.read_text()
        for forbidden in forbidden_strings:
            if forbidden in text:
                leaks.append(f"{tpl.name}: leaks {forbidden!r}")
    assert not leaks, (
        "Oracle-specific strings leaked into the stock chart's templates. "
        "The upstream chart should only have generic edits (envFrom, "
        "extraVolumes). Move Oracle-named bits to examples/oracle/helm.\n  "
        + "\n  ".join(leaks)
    )


def test_upstream_chart_only_oracle_change_is_envfrom_extravolumes():
    """The upstream chart's deployment.yaml + ingestor-server-deployment.yaml
    edits must be limited to envFrom / extraVolumes / extraVolumeMounts."""
    proc = subprocess.run(
        ["git", "diff", "origin/main", "--",
         "deploy/helm/nvidia-blueprint-rag/templates/deployment.yaml",
         "deploy/helm/nvidia-blueprint-rag/templates/ingestor-server-deployment.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.skip("can't diff against origin/main")
    diff = proc.stdout
    if not diff.strip():
        return  # no diff, fine
    added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    blob = "\n".join(added).lower()
    # Every added line must be related to one of these generic knobs.
    allowed_terms = ("envfrom", "extravolume", "extravolumemount", "tomyaml",
                     "nindent", "{{", "}}", "secret", ".values", "if")
    suspicious = []
    for line in added:
        if not line.strip():
            continue
        if any(term in line.lower() for term in allowed_terms):
            continue
        suspicious.append(line)
    # A few empty/structural lines might slip through; allow up to 5
    assert len(suspicious) <= 5, (
        f"More than 5 added lines in the upstream chart's deployment "
        f"templates aren't related to envFrom/extraVolumes:\n  "
        + "\n  ".join(suspicious[:10])
    )

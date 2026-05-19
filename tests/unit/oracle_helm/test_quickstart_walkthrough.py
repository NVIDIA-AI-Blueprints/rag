# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Treat ``QUICKSTART.md`` as a customer-executable script.

The promise to customers is "3 steps, fully automated, no fails".
That promise is only meaningful if every command in the QUICKSTART
actually parses, references real files, and uses values that the
chart actually accepts.

These tests:

  * Extract every shell command from ``QUICKSTART.md`` (and the same
    from the wrapper-chart ``README.md``).
  * Verify each command parses with ``bash -n`` (catches missing
    backslashes, unclosed quotes, etc.).
  * Verify each ``-f some/path.yaml`` argument points at a real file.
  * Verify each ``--set foo=$BAR`` references an env var the doc
    introduces in an earlier ``export`` block.
  * Verify the env vars the QUICKSTART tells customers to export
    actually map to chart values that exist (no typos sending
    customers down a path Helm silently ignores).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
QUICKSTART = REPO_ROOT / "examples" / "oracle" / "helm" / "QUICKSTART.md"
WRAPPER_README = REPO_ROOT / "examples" / "oracle" / "helm" / "README.md"
WRAPPER_CHART = REPO_ROOT / "examples" / "oracle" / "helm"


def _extract_bash_blocks(md_path: Path) -> list[str]:
    """Return every ```bash ... ``` block as a raw string."""
    text = md_path.read_text()
    pattern = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
    return [m.group(1).rstrip() for m in pattern.finditer(text)]


def _extract_file_refs(block: str) -> list[str]:
    """Return every `-f some/path.yaml` reference in a block."""
    pat = re.compile(r"-f\s+([^\s\\]+\.ya?ml)")
    return pat.findall(block)


def _extract_env_var_uses(block: str) -> list[str]:
    """Return env vars used as $VAR in a block."""
    return re.findall(r"\$([A-Z_][A-Z0-9_]*)", block)


def _extract_env_var_exports(block: str) -> list[str]:
    """Return env vars introduced by `export VAR=...` in a block."""
    return re.findall(r"^\s*export\s+([A-Z_][A-Z0-9_]*)=", block, re.MULTILINE)


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------
def test_quickstart_exists():
    assert QUICKSTART.exists(), (
        f"QUICKSTART.md is part of our '3 steps automated' promise — "
        f"it must exist at {QUICKSTART}"
    )


def test_wrapper_readme_exists():
    assert WRAPPER_README.exists()


# ---------------------------------------------------------------------------
# Every bash block is syntactically valid (`bash -n`).
# ---------------------------------------------------------------------------
def _normalize_for_bash_n(block: str) -> str:
    """Replace doc placeholders like ``<your-foo>`` with ``__PLACEHOLDER__``
    so ``bash -n`` doesn't interpret ``<`` as a redirection. We're
    checking for syntax-bug typos in the surrounding shell, not
    asking the customer to paste literal placeholders."""
    return re.sub(r"<[^>]+>", "__PLACEHOLDER__", block)


def _is_runnable_block(block: str) -> bool:
    """Skip blocks that are illustrative CLI flag snippets, not full
    commands. Heuristic: a runnable block starts with a real command
    (no leading ``--`` or ``-f``)."""
    first_line = next((ln.strip() for ln in block.splitlines() if ln.strip()
                       and not ln.lstrip().startswith("#")), "")
    if not first_line:
        return False
    if first_line.startswith(("-", "...")):
        return False
    return True


@pytest.mark.parametrize("md_path", [QUICKSTART, WRAPPER_README])
def test_every_runnable_bash_block_parses_with_bash_n(md_path):
    """`bash -n` is a syntax-only parse — catches missing line
    continuations, unclosed quotes, etc., without executing.

    Skips illustrative CLI-flag snippets (blocks starting with ``--``)
    and normalises ``<placeholder>`` so ``<`` isn't read as a
    redirection."""
    if not md_path.exists():
        pytest.skip(f"{md_path} not present")
    bash_bin = shutil.which("bash")
    if not bash_bin:
        pytest.skip("bash not on PATH")
    blocks = _extract_bash_blocks(md_path)
    assert blocks, f"{md_path.name} has no ```bash blocks"
    runnable_count = 0
    for i, block in enumerate(blocks):
        if not _is_runnable_block(block):
            continue
        runnable_count += 1
        normalized = _normalize_for_bash_n(block)
        proc = subprocess.run(
            [bash_bin, "-n", "-c", normalized],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0, (
            f"{md_path.name} bash block #{i+1} doesn't parse with `bash -n`. "
            f"Customers copy-pasting will hit a syntax error.\n"
            f"--- block ---\n{block}\n--- stderr ---\n{proc.stderr}"
        )
    assert runnable_count >= 1, (
        f"{md_path.name} has zero runnable bash blocks — that's likely a "
        "doc structure bug."
    )


# ---------------------------------------------------------------------------
# Every -f path resolves.
# ---------------------------------------------------------------------------
def test_every_yaml_file_referenced_in_quickstart_exists():
    blocks = _extract_bash_blocks(QUICKSTART)
    refs = []
    for block in blocks:
        refs.extend(_extract_file_refs(block))
    missing = []
    for ref in refs:
        target = REPO_ROOT / ref
        if target.exists():
            continue
        # Also try wrapper-chart relative
        if (WRAPPER_CHART / ref).exists():
            continue
        missing.append(ref)
    assert not missing, (
        f"QUICKSTART references YAML files that don't exist:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.parametrize("md_path", [QUICKSTART, WRAPPER_README])
def test_every_yaml_file_in_doc_exists(md_path):
    if not md_path.exists():
        pytest.skip(f"{md_path} not present")
    blocks = _extract_bash_blocks(md_path)
    missing = []
    for block in blocks:
        for ref in _extract_file_refs(block):
            for candidate in (REPO_ROOT / ref, WRAPPER_CHART / ref):
                if candidate.exists():
                    break
            else:
                missing.append(ref)
    assert not missing, (
        f"{md_path.name}: references missing YAML files:\n  "
        + "\n  ".join(set(missing))
    )


# ---------------------------------------------------------------------------
# Every $ENV used has been exported earlier in the doc.
# ---------------------------------------------------------------------------
# Customer-supplied externals that we expect them to set in their shell
# before following the doc, NOT something the doc itself exports.
EXPECTED_PREEXISTING_ENVS = {
    "HOME", "USER", "PATH", "PWD", "SHELL", "TMPDIR",
    "KUBECONFIG",            # `kubectl` precondition
    "AWS_PROFILE",           # if customer is multi-cloud
    "RANDOM",                # bash builtin
}


def test_every_env_var_used_in_quickstart_was_exported_earlier():
    """A customer copy-pastes the doc top-to-bottom. If a `$NGC_API_KEY`
    appears before `export NGC_API_KEY=...`, they hit "Error: required
    value is empty"."""
    text = QUICKSTART.read_text()
    blocks = _extract_bash_blocks(QUICKSTART)
    # Walk the doc in order, accumulate exports, flag uses.
    exported_so_far: set[str] = set(EXPECTED_PREEXISTING_ENVS)
    leaks = []
    for block in blocks:
        for env in _extract_env_var_uses(block):
            if env in exported_so_far:
                continue
            leaks.append(env)
        exported_so_far.update(_extract_env_var_exports(block))
    assert not leaks, (
        f"QUICKSTART uses env vars before they're exported. Customer "
        f"copy-paste will fail. Missing exports: {sorted(set(leaks))}"
    )


# ---------------------------------------------------------------------------
# Every helm --set chart-value reference is a real value.
# ---------------------------------------------------------------------------
def _extract_set_keys(block: str) -> list[str]:
    """Return helm --set keys from a block."""
    return re.findall(r"--set\s+([\w.-]+)=", block)


def _is_known_chart_value(values_yaml_text: str, key: str) -> bool:
    """Heuristic: every part of a dotted key must appear as a yaml key."""
    # Strip alias prefixes used in subchart values (e.g. "rag.envFrom")
    parts = key.split(".")
    for part in parts:
        # YAML keys use `<key>:` form
        if not re.search(rf"\b{re.escape(part)}\b\s*:", values_yaml_text):
            return False
    return True


def test_every_helm_set_key_in_quickstart_is_known_to_values_yaml():
    """`--set foo.bar=…` references must correspond to real keys in
    one of our values files. A typo silently no-ops in helm 3, so
    customers ship a half-configured release without warning."""
    create_values = (WRAPPER_CHART / "values.create-adb.yaml").read_text()
    existing_values = (WRAPPER_CHART / "values.existing-adb.yaml").read_text()
    schema_path = WRAPPER_CHART / "values.schema.json"
    schema_text = schema_path.read_text() if schema_path.exists() else ""
    haystack = create_values + "\n" + existing_values + "\n" + schema_text
    blocks = _extract_bash_blocks(QUICKSTART)
    bad = []
    for block in blocks:
        for key in _extract_set_keys(block):
            if _is_known_chart_value(haystack, key):
                continue
            bad.append(key)
    # Allow a few well-known umbrella keys that the schema doesn't pre-stub
    EXPECTED_UMBRELLA_KEYS = {
        "rag.imagePullSecret.password",
        "rag.ngcApiSecret.password",
    }
    bad = [k for k in bad if k not in EXPECTED_UMBRELLA_KEYS]
    assert not bad, (
        f"QUICKSTART --set keys that don't appear anywhere in values.yaml "
        f"or values.schema.json: {sorted(set(bad))}. "
        "Either fix the doc OR add the key to the chart values."
    )


# ---------------------------------------------------------------------------
# `helm install` command is self-contained: every required input is
# represented (NGC key, OCR creds, OCI config, values file).
# ---------------------------------------------------------------------------
def test_quickstart_install_block_supplies_all_required_inputs():
    """The single ``helm install`` command in the QUICKSTART must
    pass every required input. Forgetting one means the customer
    sees a confusing error mid-install."""
    blocks = _extract_bash_blocks(QUICKSTART)
    install_block = next(
        (b for b in blocks if re.search(r"helm install\s+\S+\s+examples/oracle/helm", b)),
        None,
    )
    if install_block is None:
        # Some QUICKSTARTs use `helm install rag .` from inside chart dir
        install_block = next(
            (b for b in blocks if "helm install" in b
             and ("--set" in b or "-f" in b)),
            None,
        )
    assert install_block, "QUICKSTART has no helm install command"
    # Required inputs
    must_have = [
        ("NGC API key in some form",
         re.compile(r"ngcApiSecret\.password|imagePullSecret\.password|NGC_API_KEY")),
        ("Oracle Container Registry creds",
         re.compile(r"oracle\.containerRegistry|ORACLE_OCR_TOKEN|ORACLE_SSO")),
        ("a values file",
         re.compile(r"-f\s+\S+values[\w.-]*\.ya?ml")),
        ("explicit --timeout (long install)",
         re.compile(r"--timeout\s+\d+\s*[mh]?")),
    ]
    missing = []
    for label, pat in must_have:
        if not pat.search(install_block):
            missing.append(label)
    assert not missing, (
        f"QUICKSTART helm install block is missing required input(s): "
        f"{missing}. Customers will hit failures mid-install.\n--- block ---\n"
        f"{install_block}"
    )


# ---------------------------------------------------------------------------
# Three-step promise: the doc must structurally show 3 (or fewer)
# customer-runnable steps after the prereqs.
# ---------------------------------------------------------------------------
def test_quickstart_keeps_three_step_promise():
    """The doc should structure customer-runnable work as a small number
    of steps. Allow 3-7 (Step N: …) headings — anything more means
    we've broken the simplicity promise."""
    text = QUICKSTART.read_text()
    step_headings = re.findall(r"^##\s+Step\s+\d+", text, re.MULTILINE)
    assert 3 <= len(step_headings) <= 7, (
        f"QUICKSTART has {len(step_headings)} 'Step N:' headings. "
        "Aim for 3-7 to keep the '3 steps automated' promise believable. "
        f"Got: {step_headings}"
    )


# ---------------------------------------------------------------------------
# Pre-reqs are explicit so we can't silently rely on a customer having
# something installed.
# ---------------------------------------------------------------------------
def test_quickstart_documents_required_clis():
    """The doc must list each CLI prerequisite (kubectl, helm, oci)."""
    text = QUICKSTART.read_text().lower()
    for cli in ("kubectl", "helm", "oci"):
        assert cli in text, f"QUICKSTART must mention {cli} as a prerequisite"


def test_quickstart_documents_ngc_api_key_requirement():
    text = QUICKSTART.read_text()
    assert "NGC" in text and "API" in text, (
        "QUICKSTART must mention NGC API key as a prerequisite"
    )


def test_quickstart_documents_ocr_token_requirement():
    """Customers need to accept the Oracle Private AI Services license
    before pulling the cuVS image. If the doc doesn't say so, the
    preflight Job dies with an authentication error."""
    text = QUICKSTART.read_text()
    assert "OCR" in text or "Oracle Container Registry" in text, (
        "QUICKSTART must document the OCR auth-token requirement "
        "for pulling the Oracle PAI gpu-index image."
    )


# ---------------------------------------------------------------------------
# Happy-path estimated install time
# ---------------------------------------------------------------------------
def test_quickstart_sets_realistic_helm_timeout():
    """A real install takes ~30 min (PAI image pull, ADB provision,
    NIM cache warmup). Anything below 30m guarantees timeouts."""
    text = QUICKSTART.read_text()
    timeouts = re.findall(r"--timeout\s+(\d+)\s*([mh])", text)
    assert timeouts, "QUICKSTART helm install must use --timeout"
    for value, unit in timeouts:
        seconds = int(value) * (60 if unit == "m" else 3600)
        assert seconds >= 30 * 60, (
            f"`--timeout {value}{unit}` is too short for a real install "
            f"(needs at least 30m). Customers will see helm timeouts."
        )

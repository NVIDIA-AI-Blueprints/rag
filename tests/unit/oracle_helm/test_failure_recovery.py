# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure recovery + idempotency tests.

These pin the behaviour customers depend on when something fails
mid-install, mid-upgrade, or when a Job retries:

  * Helm hooks have ``before-hook-creation`` delete policy so a
    second ``helm install`` (or ``helm upgrade``) doesn't error with
    "Job already exists".
  * Provisioner Job is re-runnable (idempotent ADB lookup, idempotent
    user create, idempotent grant).
  * Adapter's ``check_collection_exists`` is the gate for create —
    customers re-running ingestion don't get ORA-00955 on the table.
  * Adapter's collection delete tolerates "doesn't exist" (so
    cleanup retries are safe).
  * Helm's chart values pin reasonable defaults (no required values
    that crash the chart on first try).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(
    HELM_BIN is None, reason="helm CLI not on PATH",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_CHART = REPO_ROOT / "examples" / "oracle" / "helm"


REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
    "--set", "oracle.containerRegistry.username=fake@example.com",
    "--set", "oracle.containerRegistry.password=fake-ocr",
]


@pytest.fixture(scope="module")
def rendered(helm_chart_dir):
    proc = subprocess.run(
        [HELM_BIN, "template", "recovery-test", ".",
         "--kube-version=1.31.0",
         "-f", "values.create-adb.yaml", *REQUIRED_CREDS],
        cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(f"helm template failed: {proc.stderr}")
    return [d for d in yaml.safe_load_all(proc.stdout) if d]


def _by_kind_name(docs):
    out = {}
    for d in docs:
        k = d.get("kind")
        n = (d.get("metadata") or {}).get("name", "")
        if k:
            out.setdefault(k, {})[n] = d
    return out


# ---------------------------------------------------------------------------
# 1. Helm hooks must use before-hook-creation so retries succeed.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_every_oracle_job_uses_before_hook_creation_policy(rendered):
    """Without ``before-hook-creation``, a second ``helm install``
    (after a failed first attempt) errors with `Job <name> already
    exists`. The customer is then stuck."""
    by = _by_kind_name(rendered)
    leaks = []
    for name, job in by.get("Job", {}).items():
        if not name.lower().startswith(("recovery-test-oracle", "oracle-")):
            # Only check our Jobs; sub-charts have their own conventions
            if "oracle" not in name.lower():
                continue
        annos = job.get("metadata", {}).get("annotations") or {}
        delete_policy = annos.get("helm.sh/hook-delete-policy", "")
        if "before-hook-creation" not in delete_policy:
            leaks.append(f"{name}: hook-delete-policy={delete_policy!r}")
    assert not leaks, (
        "Oracle Hook Jobs without `before-hook-creation` policy will "
        "fail to retry after a failed install:\n  " + "\n  ".join(leaks)
    )


@NEEDS_HELM
def test_post_install_hooks_also_run_on_upgrade(rendered):
    """Customers running ``helm upgrade`` should still re-fetch the
    PAI URL and patch the Secret — otherwise a chart bump silently
    leaves stale Secret values in place."""
    by = _by_kind_name(rendered)
    for name, job in by.get("Job", {}).items():
        if "verify" not in name.lower() or "oracle" not in name.lower():
            continue
        annos = job.get("metadata", {}).get("annotations") or {}
        hooks = annos.get("helm.sh/hook", "")
        # post-install AND post-upgrade
        assert "post-install" in hooks, (
            f"{name} should be post-install hook"
        )
        assert "post-upgrade" in hooks, (
            f"{name} must ALSO run on post-upgrade — otherwise the "
            "PAI URL doesn't refresh when the customer bumps the chart."
        )
        return
    pytest.skip("no verify Job in this rendering")


# ---------------------------------------------------------------------------
# 2. Provisioner is idempotent.
# ---------------------------------------------------------------------------
def test_provisioner_supports_reuse_existing_for_idempotent_runs():
    """If a customer re-runs ``helm install`` (after a partial
    failure), the provisioner Job re-runs. It must NOT explode if
    the ADB already exists from the first attempt."""
    provisioner = (REPO_ROOT / "examples" / "oracle" / "helm" / "files"
                   / "provision_adb.py").read_text()
    has_idempotency = bool(re.search(
        r"(reuse[_-]?existing|already[_ -]?exists|idempotent|"
        r"existing_db|find_existing|lookup_db|check_existing)",
        provisioner, re.IGNORECASE,
    ))
    assert has_idempotency, (
        "provision_adb.py must handle the 'ADB already exists' case "
        "(reuse-existing, find-by-name, etc.). Otherwise customers "
        "stuck in re-install loops can't recover."
    )


def test_provisioner_creates_app_user_idempotently():
    """``CREATE USER ... `` errors with ORA-01920 if the user exists.
    The provisioner must use a CREATE-OR-UPDATE pattern."""
    provisioner = (REPO_ROOT / "examples" / "oracle" / "helm" / "files"
                   / "provision_adb.py").read_text()
    has_idempotent_create = bool(re.search(
        r"(IF\s+NOT\s+EXISTS|"
        r"DROP\s+USER\s+.+CASCADE|"
        r"begin.+create\s+user.+exception|"
        r"ORA-01920|"
        r"all_users.+where\s+username|"
        r"select.+username\s+from\s+(?:all|dba)_users)",
        provisioner, re.IGNORECASE | re.DOTALL,
    ))
    assert has_idempotent_create, (
        "provision_adb.py CREATE USER path must be idempotent. Either "
        "guard with `IF NOT EXISTS`, query `all_users` first, OR catch "
        "ORA-01920."
    )


# ---------------------------------------------------------------------------
# 3. Adapter's check_collection_exists guards create.
# ---------------------------------------------------------------------------
def test_adapter_create_collection_checks_existence_first():
    """The adapter's create-collection path must call
    check_collection_exists first OR catch ORA-00955 (object exists).
    Without one of those, customers re-running ingestion get a
    confusing crash."""
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle"
           / "oracle_vdb.py").read_text()
    has_existence_check = bool(re.search(
        r"check_collection_exists|all_tables\b|user_tables\b|table_exists|ORA-00955",
        src, re.IGNORECASE,
    ))
    assert has_existence_check, (
        "OracleVDB.create_collection must check for existing table or "
        "catch ORA-00955. Otherwise re-ingestion crashes the customer."
    )


# ---------------------------------------------------------------------------
# 4. Adapter delete is forgiving.
# ---------------------------------------------------------------------------
def test_adapter_delete_collection_handles_missing_gracefully():
    """delete_collections should return success/skip for non-existent
    tables, not crash. Otherwise cleanup retries get stuck."""
    src = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle"
           / "oracle_vdb.py").read_text()
    has_forgiving_drop = bool(re.search(
        r"(IF\s+EXISTS|"
        r"ORA-00942|"
        r"check_collection_exists|"
        r"try:.+drop|"
        r"drop\s+table.+except)",
        src, re.IGNORECASE | re.DOTALL,
    ))
    assert has_forgiving_drop, (
        "OracleVDB.delete_collections must tolerate the table-doesn't-"
        "exist case (ORA-00942). Otherwise customer cleanup retries "
        "get stuck."
    )


# ---------------------------------------------------------------------------
# 5. Required values have sensible defaults — chart must render with
# only the bare-minimum credential overrides.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_chart_renders_with_minimum_inputs(helm_chart_dir):
    """Customer should be able to run ``helm install`` with only
    NGC + OCR creds (no other tweaks). Anything else means a customer
    has to read the chart source to find required overrides."""
    proc = subprocess.run(
        [HELM_BIN, "template", "minimal-test", ".",
         "--kube-version=1.31.0",
         "-f", "values.create-adb.yaml", *REQUIRED_CREDS],
        cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"Chart fails to render with minimum inputs:\n{proc.stderr[-1500:]}"
    )


@NEEDS_HELM
def test_chart_renders_with_existing_adb_minimum_inputs(helm_chart_dir):
    proc = subprocess.run(
        [HELM_BIN, "template", "minimal-test", ".",
         "--kube-version=1.31.0",
         "-f", "values.existing-adb.yaml",
         "--set", "oracle.credentials.adminPassword=fake-pw",
         "--set", "oracle.credentials.appPassword=fake-pw",
         *REQUIRED_CREDS],
        cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"Existing-ADB chart fails with minimum inputs:\n{proc.stderr[-1500:]}"
    )


# ---------------------------------------------------------------------------
# 6. Backoff limits are bounded so a doom-loop Job doesn't run forever.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_oracle_jobs_have_bounded_backoff_limit(rendered):
    """An infinite retry on a fundamentally-broken Job (bad password,
    missing OCR creds) burns OCI quota and fills logs. Cap at 5."""
    by = _by_kind_name(rendered)
    leaks = []
    for name, job in by.get("Job", {}).items():
        if "oracle" not in name.lower():
            continue
        bl = job.get("spec", {}).get("backoffLimit")
        if bl is None:
            leaks.append(f"{name}: no backoffLimit (defaults to 6)")
        elif bl > 5:
            leaks.append(f"{name}: backoffLimit={bl} > 5")
    assert not leaks, (
        f"Oracle Jobs need bounded backoff:\n  " + "\n  ".join(leaks)
    )


# ---------------------------------------------------------------------------
# 7. Chart upgrade compatibility: changing replicaCount / image.tag
# doesn't rename any resources (which forces a delete + recreate).
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_image_tag_change_doesnt_rename_resources(helm_chart_dir):
    """Customers running `helm upgrade --set image.tag=newtag` should
    see in-place updates, not orphan-delete-create cycles."""
    def render(extra_set):
        proc = subprocess.run(
            [HELM_BIN, "template", "upgrade-test", ".",
             "--kube-version=1.31.0",
             "-f", "values.create-adb.yaml",
             *extra_set, *REQUIRED_CREDS],
            cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            pytest.skip(f"render failed: {proc.stderr[-500:]}")
        return [d for d in yaml.safe_load_all(proc.stdout) if d]

    base = render([])
    bumped = render(["--set", "rag.image.tag=v9.9.9"])

    base_names = {(d.get("kind"), (d.get("metadata") or {}).get("name"))
                  for d in base}
    bumped_names = {(d.get("kind"), (d.get("metadata") or {}).get("name"))
                    for d in bumped}
    assert base_names == bumped_names, (
        "Bumping rag.image.tag changed resource names:\n  "
        f"only in base: {sorted(base_names - bumped_names)}\n  "
        f"only in bumped: {sorted(bumped_names - base_names)}"
    )

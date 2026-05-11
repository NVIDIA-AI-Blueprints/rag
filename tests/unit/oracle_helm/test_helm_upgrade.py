# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm upgrade-path tests.

A customer doesn't only run ``helm install`` once — they run
``helm upgrade`` to change config (rotate the OCR token, scale the
PAI Deployment, add a BYO entry, swap the LLM model). A "seamless"
deployment means upgrades work without:

  * resource conflicts (pre-install hooks re-running unnecessarily)
  * orphaned hook Jobs (deleted before completion)
  * Secret regeneration (which would break the running rag-server)
  * immutable-field changes (would force pod recreation)

We can't run a real ``helm upgrade`` without a cluster, but we CAN
diff-render the chart for two values combinations and assert the
shape difference is what we expect.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


HELM = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM is None, reason="helm not on PATH")
REQUIRED = [
    "--set", "ngcApiSecret.password=fake",
    "--set", "imagePullSecret.password=fake",
    "--set", "oracle.containerRegistry.username=u",
    "--set", "oracle.containerRegistry.password=p",
]


def _render(chart_dir: Path, *flags: str):
    p = subprocess.run(
        [HELM, "template", "rag-test", ".", *REQUIRED,
         "-f", "values.create-adb.yaml", *flags],
        cwd=str(chart_dir), capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return [d for d in yaml.safe_load_all(p.stdout) if d]


def _by_kind_name(docs):
    return {(d.get("kind"), d["metadata"]["name"]): d
            for d in docs if d.get("metadata")}


# ---------------------------------------------------------------------------
# Stable resource identity across config changes
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_changing_replicacount_does_not_recreate_resources(helm_chart_dir):
    """Scaling the PAI Deployment from 1 → 3 must produce the SAME
    resource set (just with replicas: 3). If the chart accidentally
    appends a release-name into the resource name, the upgrade would
    create a new Deployment and orphan the old one."""
    base = _by_kind_name(_render(helm_chart_dir))
    scaled = _by_kind_name(_render(
        helm_chart_dir, "--set", "oracle.gpuIndexOffload.replicaCount=3",
    ))
    assert set(base.keys()) == set(scaled.keys()), (
        "Scaling replicaCount changed the resource set:\n"
        f"  added: {set(scaled) - set(base)}\n"
        f"  removed: {set(base) - set(scaled)}"
    )
    pai = scaled.get(("Deployment", "oracle-pai-gpu-index"))
    assert pai and pai["spec"]["replicas"] == 3


@NEEDS_HELM
def test_changing_image_tag_does_not_change_resource_names(helm_chart_dir):
    """A model swap or PAI image bump must produce identical resource
    names — only the image: field changes."""
    base = _by_kind_name(_render(helm_chart_dir))
    upgraded = _by_kind_name(_render(
        helm_chart_dir, "--set", "oracle.gpuIndexOffload.image.tag=gpu-index-26.2.0.0.0",
    ))
    assert set(base.keys()) == set(upgraded.keys()), (
        f"Image tag bump changed the resource set:\n"
        f"  added: {set(upgraded) - set(base)}\n"
        f"  removed: {set(base) - set(upgraded)}"
    )
    pai = upgraded.get(("Deployment", "oracle-pai-gpu-index"))
    img = pai["spec"]["template"]["spec"]["containers"][0]["image"]
    assert "26.2.0.0.0" in img


@NEEDS_HELM
def test_immutable_fields_are_not_in_template(helm_chart_dir):
    """``Service.spec.clusterIP`` and ``Job.spec.template.spec.*`` are
    immutable after creation. If a chart sets them dynamically, helm
    upgrade fails with "field is immutable". Confirm we don't set
    clusterIP explicitly (Kubernetes assigns it)."""
    docs = _render(helm_chart_dir)
    for d in docs:
        if d.get("kind") != "Service":
            continue
        if not d["metadata"]["name"].startswith("oracle-"):
            continue
        ip = d["spec"].get("clusterIP")
        assert ip in (None, ""), (
            f"{d['metadata']['name']}: clusterIP={ip!r} hard-coded — "
            f"upgrade with `helm upgrade` would fail with 'field is immutable'"
        )


# ---------------------------------------------------------------------------
# Hook re-execution semantics
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_pre_install_hooks_also_run_pre_upgrade(helm_chart_dir):
    """The provisioner + preflight Jobs must run on every helm upgrade
    too — otherwise an OCR token rotation wouldn't trigger a fresh
    pull check."""
    docs = _render(helm_chart_dir)
    for d in docs:
        if d.get("kind") != "Job":
            continue
        if not d["metadata"]["name"].startswith("oracle-"):
            continue
        hook = d["metadata"].get("annotations", {}).get("helm.sh/hook", "")
        if "pre-install" in hook:
            assert "pre-upgrade" in hook, (
                f"{d['metadata']['name']}: pre-install hook is missing "
                f"pre-upgrade — upgrades will skip this Job"
            )


@NEEDS_HELM
def test_post_install_hooks_also_run_post_upgrade(helm_chart_dir):
    """Symmetrically, post-install Jobs (verify, byo-import) must
    re-run on upgrade to pick up new BYO entries / new LB IPs."""
    docs = _render(helm_chart_dir)
    for d in docs:
        if d.get("kind") != "Job":
            continue
        if not d["metadata"]["name"].startswith("oracle-"):
            continue
        hook = d["metadata"].get("annotations", {}).get("helm.sh/hook", "")
        if "post-install" in hook:
            assert "post-upgrade" in hook, (
                f"{d['metadata']['name']}: post-install hook is missing "
                f"post-upgrade — upgrade won't re-run it"
            )


@NEEDS_HELM
def test_hook_delete_policies_clean_up_old_runs(helm_chart_dir):
    """``before-hook-creation`` is required so the second helm upgrade
    doesn't fail with "Job already exists"."""
    docs = _render(helm_chart_dir)
    for d in docs:
        if d.get("kind") != "Job":
            continue
        if not d["metadata"]["name"].startswith("oracle-"):
            continue
        policy = d["metadata"].get("annotations", {}).get(
            "helm.sh/hook-delete-policy", "",
        )
        assert "before-hook-creation" in policy, (
            f"{d['metadata']['name']}: hook-delete-policy {policy!r} "
            f"missing 'before-hook-creation' — second `helm upgrade` "
            f"will fail with 'Job already exists'"
        )


# ---------------------------------------------------------------------------
# Helm upgrade --dry-run goes the same way as install --dry-run
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_helm_template_with_is_upgrade_renders_cleanly(helm_chart_dir):
    """``helm template --is-upgrade`` simulates upgrade-time rendering
    (some templates use ``.Release.IsUpgrade`` to gate logic). Confirm
    nothing breaks under that path."""
    p = subprocess.run(
        [HELM, "template", "rag-test", ".", *REQUIRED,
         "-f", "values.create-adb.yaml",
         "--is-upgrade"],
        cwd=str(helm_chart_dir), capture_output=True, text=True,
    )
    assert p.returncode == 0, f"helm template --is-upgrade failed:\n{p.stderr}"
    # Result is non-empty
    docs = list(yaml.safe_load_all(p.stdout))
    assert any(d for d in docs if d), "Empty render under --is-upgrade"

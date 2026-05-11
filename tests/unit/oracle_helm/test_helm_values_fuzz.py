# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuzz the chart with edge values for ports, replicas, sizes, etc.

The chart accepts a number of numeric / string overrides:

  * oracle.gpuIndexOffload.replicaCount
  * oracle.gpuIndexOffload.service.port
  * oracle.gpuIndexOffload.gpuMode (DETECT | FORCE | DISABLE)
  * oracle.gpuIndexOffload.image.tag
  * oracle.gpuIndexOffload.service.type (LoadBalancer | ClusterIP | NodePort)

We render a sweep of edge values for each. The point isn't to find SQL
bugs — it's to confirm Helm stamping doesn't break for plausible
operator inputs (e.g. a 5-digit port number, replicaCount=0 for
maintenance, exotic image tag strings).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM_BIN is None, reason="helm not on PATH")
REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
    "--set", "oracle.containerRegistry.username=u",
    "--set", "oracle.containerRegistry.password=p",
]


def _render(helm_chart_dir: Path, *flags: str) -> tuple[int, str, str]:
    cmd = [HELM_BIN, "template", "rag-test", ".", *REQUIRED_CREDS, *flags]
    p = subprocess.run(cmd, cwd=str(helm_chart_dir), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ---------------------------------------------------------------------------
# Numeric edge values for ports
# ---------------------------------------------------------------------------
PORT_CASES = [
    ("low",          80),
    ("low-priv-bad", 0),       # invalid by convention; should still render
    ("typical",      8080),
    ("alt",          18080),
    ("high",         32767),   # Kubernetes NodePort max
    ("very-high",    65535),
]


@NEEDS_HELM
@pytest.mark.parametrize(("label", "port"), PORT_CASES, ids=[c[0] for c in PORT_CASES])
def test_pai_port_renders_at_edge(helm_chart_dir, label, port):
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        "--set", f"oracle.gpuIndexOffload.service.port={port}",
    )
    if port == 0:
        # Port=0 isn't valid in K8s (0 is reserved). Helm itself doesn't
        # check this — kubelet rejects it at apply time. Either render
        # succeeds (we'll catch at validating-webhook) or fails fast in
        # validate-values.yaml. Both are acceptable; we just don't want
        # an opaque template panic.
        assert rc in (0, 1), f"{label}: unexpected exit {rc}\n{err}"
        return
    assert rc == 0, f"{label}: render failed:\n{err}"
    docs = list(yaml.safe_load_all(out))
    pai_svc = next(
        d for d in docs if d and d.get("kind") == "Service"
        and d["metadata"]["name"] == "oracle-pai-gpu-index"
    )
    assert pai_svc["spec"]["ports"][0]["port"] == port


# ---------------------------------------------------------------------------
# Replica counts
# ---------------------------------------------------------------------------
REPLICA_CASES = [0, 1, 2, 5, 10]


@NEEDS_HELM
@pytest.mark.parametrize("replicas", REPLICA_CASES)
def test_pai_replicas_render_correctly(helm_chart_dir, replicas):
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        "--set", f"oracle.gpuIndexOffload.replicaCount={replicas}",
    )
    assert rc == 0, f"replicas={replicas} render failed:\n{err}"
    docs = list(yaml.safe_load_all(out))
    dep = next(
        d for d in docs if d and d.get("kind") == "Deployment"
        and d["metadata"]["name"] == "oracle-pai-gpu-index"
    )
    assert dep["spec"]["replicas"] == replicas


# ---------------------------------------------------------------------------
# GPU mode enum
# ---------------------------------------------------------------------------
@NEEDS_HELM
@pytest.mark.parametrize("mode", ["DETECT", "FORCE", "DISABLE"])
def test_gpu_mode_propagates_to_env(helm_chart_dir, mode):
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        "--set", f"oracle.gpuIndexOffload.gpuMode={mode}",
    )
    assert rc == 0, err
    docs = list(yaml.safe_load_all(out))
    dep = next(
        d for d in docs if d and d.get("kind") == "Deployment"
        and d["metadata"]["name"] == "oracle-pai-gpu-index"
    )
    env = dep["spec"]["template"]["spec"]["containers"][0]["env"]
    pai_gpu = next(e for e in env if e["name"] == "PRIVATE_AI_GPU_MODE")
    assert pai_gpu["value"] == mode


# ---------------------------------------------------------------------------
# Service type
# ---------------------------------------------------------------------------
@NEEDS_HELM
@pytest.mark.parametrize("svc_type", ["LoadBalancer", "ClusterIP", "NodePort"])
def test_service_type_renders(helm_chart_dir, svc_type):
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        "--set", f"oracle.gpuIndexOffload.service.type={svc_type}",
    )
    assert rc == 0, err
    docs = list(yaml.safe_load_all(out))
    pai_svc = next(
        d for d in docs if d and d.get("kind") == "Service"
        and d["metadata"]["name"] == "oracle-pai-gpu-index"
    )
    assert pai_svc["spec"]["type"] == svc_type
    if svc_type != "LoadBalancer":
        ann = pai_svc["metadata"].get("annotations") or {}
        # OCI LB-specific annotations should NOT appear for non-LB types
        assert not any("oci-load-balancer" in k for k in ann.keys()), (
            f"OCI LB annotations leaked into a {svc_type} Service"
        )


# ---------------------------------------------------------------------------
# Image tag — accept anything tag-shaped. Helm pre-26 had a panic on
# tags starting with digits in some templates.
# ---------------------------------------------------------------------------
TAG_CASES = [
    "26.1.0.0.0",
    "latest",
    "26.1.0.0.0-rc1",
    "main",
    "sha-abc123",
    "26.1.0.0.0_dev",
]


@NEEDS_HELM
@pytest.mark.parametrize("tag", TAG_CASES)
def test_image_tag_propagates(helm_chart_dir, tag):
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        "--set", f"oracle.gpuIndexOffload.image.tag={tag}",
    )
    assert rc == 0, err
    assert f":{tag}" in out, f"tag {tag} not found in any image: ref"


# ---------------------------------------------------------------------------
# BYO entries with realistic counts (1, 5, 25)
# ---------------------------------------------------------------------------
@NEEDS_HELM
@pytest.mark.parametrize("n", [0, 1, 5, 25])
def test_byo_entries_at_scale(helm_chart_dir, n):
    """BYO Job stamps a single env var holding JSON-serialised
    importExistingTables. Confirm 25 entries don't blow past the 1MiB
    env-var limit and don't break YAML scalar parsing."""
    import json
    entries = [
        {
            "sourceTable": f"CUST_TABLE_{i}",
            "collectionName": f"cust_{i}",
            "columns": {"text": "BODY", "vector": "EMBED"},
        }
        for i in range(n)
    ]
    flags = []
    if n > 0:
        flags = ["--set-json", f"oracle.importExistingTables={json.dumps(entries)}"]
    rc, out, err = _render(
        helm_chart_dir,
        "-f", "values.existing-adb.yaml",
        "--set", "oracle.existing.user=RAG_APP",
        "--set", "oracle.existing.password=secret",
        "--set", "oracle.existing.connectString=existing_medium",
        *flags,
    )
    assert rc == 0, f"n={n} render failed:\n{err}"
    docs = list(yaml.safe_load_all(out))
    job = next(
        (d for d in docs if d and d.get("kind") == "Job"
         and d["metadata"]["name"] == "oracle-byo-import"),
        None,
    )
    assert job, "BYO Job missing"
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    byo_env = next(e for e in env if e["name"] == "BYO_TABLES_JSON")
    parsed = json.loads(byo_env["value"])
    assert len(parsed) == n

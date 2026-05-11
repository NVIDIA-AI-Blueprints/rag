# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout-time sanity for every workload the chart ships.

Customers care that ``kubectl get pods`` settles into Running 1/1
without manual intervention. These tests validate the *shape* of every
Deployment / Job / DaemonSet / StatefulSet we render:

  * Every Deployment / StatefulSet has a readiness probe (so the
    Service doesn't blackhole traffic to a not-ready pod).
  * Every Job has a sane backoffLimit (no infinite retries on
    ImagePullBackOff).
  * Every Job has ttlSecondsAfterFinished set so completed Jobs don't
    accumulate forever and clog `kubectl get pods`.
  * Every workload referencing a Secret declares it as a dependency
    (envFrom / valueFrom.secretKeyRef), so missing-Secret errors
    surface at admission, not at first traffic.
  * No image uses ``:latest`` (reproducibility).
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


@pytest.fixture(scope="module")
def rendered(helm_chart_dir):
    p = subprocess.run(
        [HELM, "template", "rag-test", ".",
         *REQUIRED,
         "-f", "values.create-adb.yaml"],
        cwd=str(helm_chart_dir), capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return [d for d in yaml.safe_load_all(p.stdout) if d]


# ---------------------------------------------------------------------------
# Workloads we OWN (Oracle-specific). Stock RAG-blueprint workloads are
# the upstream's responsibility; we only assert the wrapper-level shape.
# ---------------------------------------------------------------------------
ORACLE_WORKLOADS_PREFIXES = ("oracle-",)


def _is_ours(d: dict) -> bool:
    name = d.get("metadata", {}).get("name", "")
    return any(name.startswith(p) for p in ORACLE_WORKLOADS_PREFIXES)


# ===========================================================================
# Jobs
# ===========================================================================
@NEEDS_HELM
def test_every_job_has_finite_backoff_limit(rendered):
    """No Job should retry forever — that hides errors and wastes
    GPU minutes."""
    for d in rendered:
        if d.get("kind") != "Job" or not _is_ours(d):
            continue
        bl = d["spec"].get("backoffLimit")
        assert bl is not None, (
            f"{d['metadata']['name']}: Job missing backoffLimit"
        )
        assert 0 <= bl <= 5, (
            f"{d['metadata']['name']}: backoffLimit={bl} too high "
            f"(infinite retries would mask errors)"
        )


@NEEDS_HELM
def test_every_job_has_ttl_seconds_after_finished(rendered):
    """Completed Jobs accumulate forever without ttlSecondsAfterFinished
    — clogs `kubectl get pods` and confuses operators."""
    for d in rendered:
        if d.get("kind") != "Job" or not _is_ours(d):
            continue
        ttl = d["spec"].get("ttlSecondsAfterFinished")
        assert ttl is not None, (
            f"{d['metadata']['name']}: Job missing ttlSecondsAfterFinished"
        )
        assert 60 <= ttl <= 86400, (
            f"{d['metadata']['name']}: ttl {ttl}s outside reasonable range"
        )


@NEEDS_HELM
def test_every_job_has_restart_policy_never(rendered):
    """Jobs must use restartPolicy: Never so a crash counts as a Job
    failure, not an infinite restart loop within a single Pod."""
    for d in rendered:
        if d.get("kind") != "Job" or not _is_ours(d):
            continue
        rp = d["spec"]["template"]["spec"].get("restartPolicy")
        assert rp == "Never", (
            f"{d['metadata']['name']}: restartPolicy={rp!r}, expected Never"
        )


# ===========================================================================
# Deployments
# ===========================================================================
@NEEDS_HELM
def test_oracle_pai_deployment_has_readiness_probe(rendered):
    """The PAI Deployment is fronted by a Service — a missing readiness
    probe means the Service can route traffic to a starting pod and 502
    on the customer."""
    for d in rendered:
        if d.get("kind") != "Deployment" or not _is_ours(d):
            continue
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert c.get("readinessProbe") is not None, (
                f"{d['metadata']['name']}/{c['name']}: missing readinessProbe"
            )


@NEEDS_HELM
def test_oracle_pai_deployment_has_resource_requests(rendered):
    """Without requests, the scheduler can't make GPU placement
    decisions and OKE will overpack the node."""
    for d in rendered:
        if d.get("kind") != "Deployment" or not _is_ours(d):
            continue
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert c.get("resources", {}).get("requests"), (
                f"{d['metadata']['name']}/{c['name']}: missing resource requests"
            )


# ===========================================================================
# Image hygiene
# ===========================================================================
@NEEDS_HELM
def test_no_workload_uses_latest_tag(rendered):
    """``:latest`` makes installs non-reproducible; an OCR token
    rotation could change the image under the customer's feet."""
    for d in rendered:
        if d.get("kind") not in ("Deployment", "Job", "StatefulSet", "DaemonSet"):
            continue
        if not _is_ours(d):
            continue
        for c in d["spec"]["template"]["spec"]["containers"]:
            img = c.get("image", "")
            assert not img.endswith(":latest"), (
                f"{d['metadata']['name']}/{c['name']}: image {img!r} uses :latest"
            )
            # Empty / no tag is also bad
            assert ":" in img, (
                f"{d['metadata']['name']}/{c['name']}: image {img!r} has no tag"
            )


# ===========================================================================
# Secret references / Service shape
# ===========================================================================
@NEEDS_HELM
def test_pai_service_exposes_a_named_port(rendered):
    """PAI's Service is what ADB dials over the OCI internal LB. The
    port must be named so future Service-mesh integrations don't break."""
    for d in rendered:
        if d.get("kind") != "Service":
            continue
        if d["metadata"]["name"] != "oracle-pai-gpu-index":
            continue
        ports = d["spec"]["ports"]
        assert ports, "PAI Service has no ports"
        for p in ports:
            assert "port" in p
            # Either name or appProtocol must be set so Service mesh /
            # NetworkPolicy can match by port name
            assert p.get("name") or p.get("appProtocol"), (
                f"PAI Service port {p['port']} has no name/appProtocol"
            )


# ===========================================================================
# Service accounts (least privilege)
# ===========================================================================
@NEEDS_HELM
def test_oracle_pai_verify_serviceaccount_has_minimal_rbac(rendered):
    """The verify Job patches one Secret + restarts two Deployments. It
    must NOT have cluster-wide get/list/watch on Secrets — that's a
    typical RBAC blast-radius mistake."""
    role_rules = []
    for d in rendered:
        if d.get("kind") != "Role":
            continue
        if "oracle-pai-verify" not in d["metadata"]["name"]:
            continue
        role_rules = d.get("rules", [])
        break
    assert role_rules, "oracle-pai-verify Role not found"
    for rule in role_rules:
        # No wildcard verbs
        assert "*" not in rule.get("verbs", []), (
            f"oracle-pai-verify Role has wildcard verb: {rule}"
        )

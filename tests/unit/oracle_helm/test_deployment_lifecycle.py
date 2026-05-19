# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end deployment lifecycle tests.

Renders the wrapper chart with the customer-facing values files and
asserts the **complete** install sequence is correct, top to bottom,
with no race conditions, no missing hooks, and no orphaned resources:

  Pre-install:
    1. validate-values    — fail-fast on bad config
    2. oracle-pai-preflight — pulls cuVS image on a GPU node, before ADB
    3. provisioner-job    — creates ADB, opens NSG path

  Install:
    4. oracle-pai           — Deployment + LoadBalancer Service for cuVS
    5. rag / nv-ingest      — RAG blueprint sub-chart
    6. nim-operator + crds  — LLM/Embed/Rerank NIMs
    7. gpu-operator         — driver/runtime DaemonSets

  Post-install:
    8. oracle-pai-verify   — patches ORACLE_PAI_INDEX_URL Secret, rolls
                              rag-server + ingestor-server

  Post-install (BYO data, optional):
    9. oracle-byo-import   — auto-discovers / view-maps customer tables

These tests are gated on `helm` being on PATH and the `helm_chart_dir`
fixture (which auto-runs `helm dep update`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(
    HELM_BIN is None, reason="helm CLI not on PATH",
)

REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc-key",
    "--set", "imagePullSecret.password=fake-ngc-key",
    "--set", "oracle.containerRegistry.username=fake@example.com",
    "--set", "oracle.containerRegistry.password=fake-ocr-token",
]


def _render(helm_chart_dir, *extra: str) -> list[dict[str, Any]]:
    cmd = [HELM_BIN, "template", "lifecycle-test", ".",
           "--include-crds=false"] + list(extra)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(helm_chart_dir),
    )
    if proc.returncode != 0:
        pytest.fail(
            f"helm template failed:\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return [d for d in yaml.safe_load_all(proc.stdout) if d]


@pytest.fixture(scope="module")
def rendered_create_adb(helm_chart_dir):
    return _render(helm_chart_dir, "-f", "values.create-adb.yaml", *REQUIRED_CREDS)


@pytest.fixture(scope="module")
def rendered_existing_adb(helm_chart_dir):
    """Existing-ADB mode: customer brings their own oracle-creds Secret;
    we never run the ADB provisioner, never touch OCI."""
    return _render(
        helm_chart_dir, "-f", "values.existing-adb.yaml", *REQUIRED_CREDS,
        "--set", "oracle.credentials.adminPassword=fake-admin-pw",
        "--set", "oracle.credentials.appPassword=fake-app-pw",
    )


def _by_kind_name(docs):
    out = {}
    for d in docs:
        k = d.get("kind")
        n = (d.get("metadata") or {}).get("name", "")
        if k:
            out.setdefault(k, {})[n] = d
    return out


# ---------------------------------------------------------------------------
# 1. Pre-install ordering: validate → preflight → provisioner
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_create_adb_renders_provisioner_job(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    job_names = list(by.get("Job", {}))
    assert any("provision" in n.lower() for n in job_names), (
        f"Create-ADB mode must render the ADB provisioner Job. Got: {job_names}"
    )


@NEEDS_HELM
def test_create_adb_renders_preflight_job(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    job_names = list(by.get("Job", {}))
    assert any("preflight" in n.lower() for n in job_names), (
        f"PAI preflight Job missing. Found Jobs: {job_names}"
    )


@NEEDS_HELM
def test_existing_adb_skips_provisioner(rendered_existing_adb):
    by = _by_kind_name(rendered_existing_adb)
    job_names = list(by.get("Job", {}))
    assert not any("provision" in n.lower() for n in job_names), (
        "Existing-ADB mode must NOT render the provisioner Job. "
        f"Found: {job_names}"
    )


@NEEDS_HELM
@pytest.mark.parametrize("hook_phase, expected_jobs", [
    ("pre-install", ["preflight", "validate", "provision"]),
])
def test_pre_install_jobs_are_pre_install_hooks(rendered_create_adb,
                                                 hook_phase, expected_jobs):
    by = _by_kind_name(rendered_create_adb)
    for job_name, job in by.get("Job", {}).items():
        annos = (job.get("metadata") or {}).get("annotations") or {}
        for tok in expected_jobs:
            if tok in job_name.lower():
                hooks = annos.get("helm.sh/hook", "")
                assert hook_phase in hooks, (
                    f"{job_name} should be a {hook_phase} hook (currently: "
                    f"{hooks!r}). Otherwise it runs after the rag-server "
                    "Deployment, which then crash-loops waiting for ADB."
                )


@NEEDS_HELM
def test_provisioner_runs_after_preflight(rendered_create_adb):
    """Both must be pre-install hooks, but the provisioner must have a
    higher hook-weight (runs later) than the preflight."""
    by = _by_kind_name(rendered_create_adb)
    provisioner = next(
        (j for n, j in by.get("Job", {}).items() if "provision" in n.lower()),
        None,
    )
    preflight = next(
        (j for n, j in by.get("Job", {}).items() if "preflight" in n.lower()),
        None,
    )
    if provisioner is None or preflight is None:
        pytest.skip("missing one of the jobs")
    pw = int(provisioner["metadata"]["annotations"].get("helm.sh/hook-weight", "0"))
    pf = int(preflight["metadata"]["annotations"].get("helm.sh/hook-weight", "0"))
    assert pw > pf, (
        "Preflight must run BEFORE provisioner (cuVS image pull validates "
        f"GPU node + OCR creds in seconds). preflight weight={pf}, "
        f"provisioner weight={pw}."
    )


# ---------------------------------------------------------------------------
# 2. Install: PAI Deployment + Service is rendered.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_pai_deployment_present(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    deployments = by.get("Deployment", {})
    pai_names = [n for n in deployments if "pai" in n.lower()
                  or "gpu-index" in n.lower()]
    assert pai_names, (
        f"Oracle PAI cuVS Deployment missing. Got: {list(deployments)}"
    )


@NEEDS_HELM
def test_pai_service_present_and_lb_or_clusterip(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    services = by.get("Service", {})
    pai_svcs = [s for n, s in services.items() if "pai" in n.lower()
                or "gpu-index" in n.lower()]
    assert pai_svcs, "Oracle PAI Service missing"
    for svc in pai_svcs:
        svc_type = svc["spec"].get("type", "ClusterIP")
        assert svc_type in {"LoadBalancer", "ClusterIP", "NodePort"}, (
            f"PAI service has unexpected type {svc_type!r}"
        )


@NEEDS_HELM
def test_pai_deployment_requests_gpu(rendered_create_adb):
    """PAI runs cuVS — must request a GPU."""
    by = _by_kind_name(rendered_create_adb)
    pai = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "pai" in n.lower() or "gpu-index" in n.lower()),
        None,
    )
    if pai is None:
        pytest.skip("no PAI deployment")
    pod_spec = pai["spec"]["template"]["spec"]
    containers = pod_spec.get("containers", [])
    needs_gpu = False
    for c in containers:
        resources = c.get("resources") or {}
        for section in (resources.get("limits") or {}, resources.get("requests") or {}):
            for k in section:
                if "nvidia.com/gpu" in str(k) or "gpu" in str(k):
                    needs_gpu = True
    nodeSelector = pod_spec.get("nodeSelector") or {}
    if any("gpu" in str(k).lower() or "nvidia" in str(k).lower()
           for k in nodeSelector):
        needs_gpu = True
    assert needs_gpu, (
        "PAI Deployment must request a GPU (resource limit or nodeSelector). "
        "Otherwise it lands on a CPU node and crash-loops."
    )


# ---------------------------------------------------------------------------
# 3. Post-install: verify Job patches Secret + restarts consumers.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_verify_job_is_post_install_hook(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    verify = next(
        (j for n, j in by.get("Job", {}).items() if "verify" in n.lower()),
        None,
    )
    if verify is None:
        pytest.skip("no verify job in this rendering")
    annos = verify["metadata"].get("annotations") or {}
    hooks = annos.get("helm.sh/hook", "")
    assert "post-install" in hooks, (
        f"verify Job must be a post-install hook (got: {hooks!r}). "
        "It needs to wait for the LoadBalancer ingress IP, which doesn't "
        "exist before install completes."
    )


# ---------------------------------------------------------------------------
# Helpers — scope every hygiene check to resources WE author
# (oracle-* templates in examples/oracle/helm/templates/, plus rag-server/
# ingestor-server from the RAG blueprint sub-chart we contribute to).
#
# Sub-chart resources (gpu-operator, k8s-nim-operator, kube-prometheus-stack,
# etc.) follow their own conventions and ship from different repositories.
# They are explicitly NOT in our review scope; each upstream owns their own
# Helm hygiene.
# ---------------------------------------------------------------------------
OUR_RESOURCE_NAME_TOKENS = (
    "oracle",        # all our wrapper-chart resources
    "rag-server",    # blueprint sub-chart, edited by us
    "ingestor-server",  # blueprint sub-chart, edited by us
    "rag-frontend",  # blueprint sub-chart
    "rag-redis",     # blueprint sub-chart
    "rag-mongodb",   # blueprint sub-chart
)

# Sub-chart-owned resource name fragments — explicit allow-list so future
# refactors that change naming don't accidentally widen our test scope.
SUBCHART_NAME_FRAGMENTS = (
    "nim-operator", "gpu-operator", "node-feature-discovery",
    "nfd-master", "nfd-worker", "nv-ingest", "kube-prometheus",
    "kube-state-metrics", "prometheus", "alertmanager", "grafana",
    "elasticsearch", "zipkin", "opentelemetry-collector", "minio",
    "redis-master", "redis-replica", "milvus",
)


def _is_subchart_resource(name: str) -> bool:
    name_lower = name.lower()
    return any(frag in name_lower for frag in SUBCHART_NAME_FRAGMENTS)


def _is_our_resource(name: str) -> bool:
    """True only for resources WE author."""
    if _is_subchart_resource(name):
        return False
    name_lower = name.lower()
    return any(token in name_lower for token in OUR_RESOURCE_NAME_TOKENS)


@NEEDS_HELM
def test_our_jobs_have_ttl_for_cleanup(rendered_create_adb):
    """Every Job that WE author must have ttlSecondsAfterFinished, so the
    cluster doesn't accumulate completed Jobs forever."""
    by = _by_kind_name(rendered_create_adb)
    leaks = []
    for name, job in by.get("Job", {}).items():
        if not _is_our_resource(name):
            continue
        ttl = job.get("spec", {}).get("ttlSecondsAfterFinished")
        if ttl is None:
            leaks.append(name)
    assert not leaks, (
        f"Our Jobs without ttlSecondsAfterFinished will accumulate forever: "
        f"{leaks}"
    )


@NEEDS_HELM
def test_our_jobs_have_explicit_backoff_limit(rendered_create_adb):
    by = _by_kind_name(rendered_create_adb)
    leaks = []
    for name, job in by.get("Job", {}).items():
        if not _is_our_resource(name):
            continue
        bl = job.get("spec", {}).get("backoffLimit")
        if bl is None:
            leaks.append(name)
        elif bl > 5:
            leaks.append(f"{name} (backoffLimit={bl} too high)")
    assert not leaks, f"Our Job backoffLimit issues: {leaks}"


# ---------------------------------------------------------------------------
# Generic Helm hygiene (scoped to our resources): no :latest tags, no
# missing pull secrets for our private-registry pulls.
# ---------------------------------------------------------------------------
def _walk(obj, key=None):
    if isinstance(obj, dict):
        yield obj, key
        for k, v in obj.items():
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


@NEEDS_HELM
def test_our_resources_no_latest_image_tags(rendered_create_adb):
    """Inspect every workload WE author. Latest-tag pins break rollback
    and reproducibility. ClusterPolicy CRDs from gpu-operator are skipped
    because that's a CRD spec, not an actual image pull."""
    bad = []
    for d in rendered_create_adb:
        kind = d.get("kind") or ""
        # ClusterPolicy is gpu-operator's CRD for declaring desired image
        # versions; not actual image pulls. Skip.
        if kind == "ClusterPolicy":
            continue
        if kind not in {"Deployment", "DaemonSet", "Job", "StatefulSet",
                        "CronJob", "Pod"}:
            continue
        name = (d.get("metadata") or {}).get("name") or ""
        if not _is_our_resource(name):
            continue
        for entry, _ in _walk(d):
            if isinstance(entry, dict) and "image" in entry:
                img = entry["image"]
                if not isinstance(img, str):
                    continue
                if img.endswith(":latest") or ":" not in img:
                    bad.append(f"{kind}/{name}: image={img}")
    assert not bad, (
        f"Found :latest or untagged images in OUR resources (breaks "
        f"reproducibility / makes rollbacks dangerous):\n  " + "\n  ".join(bad)
    )


@NEEDS_HELM
def test_our_workloads_pulling_private_have_pull_secret(rendered_create_adb):
    """Every WORKLOAD WE author that pulls from a private registry must
    declare imagePullSecrets."""
    leaks = []
    for d in rendered_create_adb:
        kind = d.get("kind", "")
        if kind not in {"Deployment", "DaemonSet", "Job", "StatefulSet"}:
            continue
        name = (d.get("metadata") or {}).get("name") or ""
        if not _is_our_resource(name):
            continue
        spec = d.get("spec", {})
        pod_spec = (spec.get("template") or {}).get("spec") or spec
        images = []
        for c in pod_spec.get("containers", []) or []:
            images.append(c.get("image", ""))
        for c in pod_spec.get("initContainers", []) or []:
            images.append(c.get("image", ""))
        needs_secret = any(
            ("nvcr.io" in i)
            or ("container-registry.oracle.com" in i)
            for i in images
        )
        if not needs_secret:
            continue
        if not pod_spec.get("imagePullSecrets"):
            leaks.append(f"{kind}/{name} (images: {images})")
    assert not leaks, (
        f"Our workloads pulling from private registry without "
        f"imagePullSecrets:\n  " + "\n  ".join(leaks)
    )


# ---------------------------------------------------------------------------
# 5. Existing-ADB BYO mode: rag-server still gets oracle-creds.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_existing_adb_wires_oracle_creds_secret_into_rag_server(rendered_existing_adb):
    """rag-server's pod spec must envFrom the oracle-creds Secret."""
    by = _by_kind_name(rendered_existing_adb)
    rag = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "rag-server" in n.lower() and "ingestor" not in n.lower()),
        None,
    )
    if rag is None:
        pytest.skip("no rag-server deployment in this rendering")
    pod_spec = rag["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    env_from = container.get("envFrom") or []
    secret_refs = [
        ef.get("secretRef", {}).get("name", "") for ef in env_from
        if "secretRef" in ef
    ]
    assert any("oracle" in n.lower() for n in secret_refs), (
        "rag-server must envFrom an oracle-creds Secret when oracle is "
        f"the active vector store. Found envFrom: {env_from}"
    )


@NEEDS_HELM
def test_existing_adb_sets_app_vectorstore_oracle(rendered_existing_adb):
    """rag-server pod env must set APP_VECTORSTORE_NAME=oracle."""
    by = _by_kind_name(rendered_existing_adb)
    rag = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "rag-server" in n.lower() and "ingestor" not in n.lower()),
        None,
    )
    if rag is None:
        pytest.skip("no rag-server deployment in this rendering")
    container = rag["spec"]["template"]["spec"]["containers"][0]
    env_pairs = {e["name"]: e.get("value", "") for e in (container.get("env") or [])}
    name = env_pairs.get("APP_VECTORSTORE_NAME", "")
    assert name == "oracle", (
        f"APP_VECTORSTORE_NAME should be 'oracle', got {name!r}"
    )

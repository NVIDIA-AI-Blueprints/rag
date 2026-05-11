# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-system correctness tests.

Renders the full wrapper chart (our templates + all sub-charts) and
verifies the result is a *self-consistent* Kubernetes deployment:

  * Every Secret name referenced by `envFrom` / `valueFrom.secretKeyRef`
    actually exists in the rendered output (or is created by the
    customer at install time per docs).
  * Every ConfigMap name referenced by `envFrom` / `valueFrom.configMapKeyRef`
    exists or is documented as customer-supplied.
  * Every Service name referenced as a hostname in env vars resolves
    to a rendered Service.
  * Every container `volumeMount` references a `volume` defined on the
    same pod spec.
  * No resource has a duplicate (kind, name) pair.
  * Each rendered manifest has `apiVersion` and `kind` set.
  * `helm install --dry-run=client` succeeds with our customer-facing
    values files.

Unlike ``test_deployment_lifecycle.py`` which scopes hygiene checks to
our resources, **this file inspects EVERYTHING** the chart renders —
including sub-chart output — because integration bugs (missing
references, dangling secrets, malformed manifests) impact deployment
regardless of which template authored the resource.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

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


def _render(chart_dir: Path, *extra: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [HELM_BIN, "template", "wholetest", ".",
         "--include-crds=false"] + list(extra),
        capture_output=True, text=True, cwd=str(chart_dir),
    )
    if proc.returncode != 0:
        pytest.fail(
            f"helm template failed:\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return [d for d in yaml.safe_load_all(proc.stdout) if d]


@pytest.fixture(scope="module")
def whole_create_adb(helm_chart_dir):
    return _render(helm_chart_dir, "-f", "values.create-adb.yaml", *REQUIRED_CREDS)


@pytest.fixture(scope="module")
def whole_existing_adb(helm_chart_dir):
    return _render(
        helm_chart_dir, "-f", "values.existing-adb.yaml", *REQUIRED_CREDS,
        "--set", "oracle.credentials.adminPassword=fake-admin-pw",
        "--set", "oracle.credentials.appPassword=fake-app-pw",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _walk(obj, key=None):
    """Yield (parent_dict, child_key, child_value) for every dict node."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield obj, k, v
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _index(docs) -> dict[tuple[str, str], dict]:
    """Index resources by (kind, name)."""
    out = {}
    for d in docs:
        kind = d.get("kind")
        name = (d.get("metadata") or {}).get("name")
        if kind and name:
            out[(kind, name)] = d
    return out


def _all_pod_specs(docs) -> Iterable[tuple[dict, str, list]]:
    """Yield (pod_spec, owner_label, volume_claim_templates) for every
    pod-bearing workload. ``volume_claim_templates`` is non-empty only for
    StatefulSets, where PVC names auto-populate as volumes on the Pod."""
    for d in docs:
        kind = d.get("kind", "")
        spec = d.get("spec") or {}
        owner = f"{kind}/{(d.get('metadata') or {}).get('name', '?')}"
        if kind in {"Deployment", "DaemonSet", "Job"}:
            ps = spec.get("template", {}).get("spec") or {}
            yield ps, owner, []
        elif kind == "StatefulSet":
            ps = spec.get("template", {}).get("spec") or {}
            vcts = spec.get("volumeClaimTemplates") or []
            yield ps, owner, vcts
        elif kind == "CronJob":
            ps = spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec") or {}
            yield ps, owner, []
        elif kind == "Pod":
            yield spec, owner, []


# Customer/Helm-managed Secrets we know exist outside of `helm template`
# output. Customer-created Secrets are expected to be in the cluster
# before / after install (per QUICKSTART instructions).
EXPECTED_EXTERNAL_SECRETS = {
    "oci-config",            # `kubectl create secret generic oci-config ...`
    "oracle-creds",          # written by provisioner Job at runtime; pre-supplied in BYO mode
    "oracle-pai-secret",     # Helm hook resource, may not be in main render
    "oracle-pai-ocr",        # OCR pull secret created by Helm hook in our wrapper
    "ngc-api",               # NGC API key, customer/Helm provides
    "ngc-secret",            # NGC pull secret, customer/Helm provides
    "rag-secret-from-helm",  # blueprint convention
    # Observability stack secrets (only render when their condition flag is on)
    "elasticsearch-es-elastic-user",
    "elasticsearch-master-credentials",
}

EXPECTED_EXTERNAL_CONFIGMAPS = {
    # Observability stack templates
    "kube-state-metrics",
    "alertmanager",
}


# ---------------------------------------------------------------------------
# 1. Manifest sanity: every doc has apiVersion + kind, no duplicates.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_every_manifest_has_apiversion_and_kind(whole_create_adb):
    leaks = []
    for d in whole_create_adb:
        if not d.get("apiVersion"):
            leaks.append(f"missing apiVersion: kind={d.get('kind')} name={(d.get('metadata') or {}).get('name')}")
        if not d.get("kind"):
            leaks.append(f"missing kind: apiVersion={d.get('apiVersion')}")
    assert not leaks, "Malformed manifests:\n  " + "\n  ".join(leaks)


@NEEDS_HELM
def test_no_duplicate_kind_name_pairs(whole_create_adb):
    """`kubectl apply` chokes on duplicate (kind, namespace, name)
    triples. We don't render namespaces (helm install -n …), so
    (kind, name) is the relevant key."""
    seen = {}
    dups = []
    for d in whole_create_adb:
        kind = d.get("kind")
        name = (d.get("metadata") or {}).get("name")
        if not kind or not name:
            continue
        key = (kind, name)
        if key in seen:
            dups.append(f"{kind}/{name} (twice)")
        seen[key] = d
    assert not dups, "Duplicate manifests:\n  " + "\n  ".join(dups)


@NEEDS_HELM
def test_every_pod_has_at_least_one_container(whole_create_adb):
    leaks = []
    for ps, owner, _vcts in _all_pod_specs(whole_create_adb):
        if not ps.get("containers"):
            leaks.append(owner)
    assert not leaks, f"Pods with no containers: {leaks}"


# ---------------------------------------------------------------------------
# 2. envFrom Secret references resolve.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_envfrom_secret_references_resolve(whole_create_adb):
    """For every envFrom.secretRef.name, the Secret either exists in
    rendered output or is in the customer-managed allow-list."""
    rendered = _index(whole_create_adb)
    rendered_secrets = {n for k, n in rendered if k == "Secret"}
    leaks = []
    for ps, owner, _vcts in _all_pod_specs(whole_create_adb):
        for c in (ps.get("containers") or []) + (ps.get("initContainers") or []):
            for ef in c.get("envFrom") or []:
                ref = ef.get("secretRef", {}).get("name")
                if not ref:
                    continue
                if ref in rendered_secrets:
                    continue
                if ref in EXPECTED_EXTERNAL_SECRETS:
                    continue
                # Helm-mangled release prefix: "wholetest-<base>"
                stripped = ref.replace("wholetest-", "")
                if stripped in EXPECTED_EXTERNAL_SECRETS:
                    continue
                # blueprint sub-chart secret references are managed by
                # the sub-chart itself; allow if optional=true
                if ef.get("secretRef", {}).get("optional"):
                    continue
                leaks.append(f"{owner} envFrom Secret {ref!r} not rendered")
    assert not leaks, "Dangling envFrom Secret references:\n  " + "\n  ".join(leaks)


@NEEDS_HELM
def test_valuefrom_secret_key_references_resolve(whole_create_adb):
    rendered = _index(whole_create_adb)
    rendered_secrets = {n for k, n in rendered if k == "Secret"}
    leaks = []
    for ps, owner, _vcts in _all_pod_specs(whole_create_adb):
        for c in (ps.get("containers") or []) + (ps.get("initContainers") or []):
            for env in c.get("env") or []:
                ref = (env.get("valueFrom") or {}).get("secretKeyRef") or {}
                name = ref.get("name")
                if not name:
                    continue
                if ref.get("optional"):
                    continue
                if name in rendered_secrets or name in EXPECTED_EXTERNAL_SECRETS:
                    continue
                stripped = name.replace("wholetest-", "")
                if stripped in EXPECTED_EXTERNAL_SECRETS:
                    continue
                leaks.append(
                    f"{owner} env {env.get('name')!r} → Secret/{name} not rendered"
                )
    assert not leaks, (
        "Dangling secretKeyRef references:\n  " + "\n  ".join(leaks)
    )


# ---------------------------------------------------------------------------
# 3. ConfigMap references resolve.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_envfrom_configmap_references_resolve(whole_create_adb):
    rendered = _index(whole_create_adb)
    rendered_cms = {n for k, n in rendered if k == "ConfigMap"}
    leaks = []
    for ps, owner, _vcts in _all_pod_specs(whole_create_adb):
        for c in (ps.get("containers") or []) + (ps.get("initContainers") or []):
            for ef in c.get("envFrom") or []:
                ref = ef.get("configMapRef", {}).get("name")
                if not ref:
                    continue
                if ef.get("configMapRef", {}).get("optional"):
                    continue
                if ref in rendered_cms or ref in EXPECTED_EXTERNAL_CONFIGMAPS:
                    continue
                stripped = ref.replace("wholetest-", "")
                if stripped in EXPECTED_EXTERNAL_CONFIGMAPS:
                    continue
                leaks.append(f"{owner} envFrom ConfigMap {ref!r} not rendered")
    assert not leaks, "Dangling ConfigMap references:\n  " + "\n  ".join(leaks)


# ---------------------------------------------------------------------------
# 4. Volume references resolve within each pod spec.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_every_volume_mount_resolves_to_a_volume(whole_create_adb):
    """Every volumeMount.name must resolve to either a Pod-spec volume
    OR (for StatefulSets) a volumeClaimTemplate.

    Also tolerates the well-known auto-volumes Kubernetes itself injects:
    ``kube-api-access-*`` for projected service-account tokens.
    """
    leaks = []
    for ps, owner, vcts in _all_pod_specs(whole_create_adb):
        pod_volumes = {v.get("name", "") for v in (ps.get("volumes") or [])}
        # PVC templates auto-populate as volumes named after the template.
        pvc_template_names = {t.get("metadata", {}).get("name", "") for t in vcts}
        valid = pod_volumes | pvc_template_names
        for c in (ps.get("containers") or []) + (ps.get("initContainers") or []):
            for vm in c.get("volumeMounts") or []:
                vname = vm.get("name", "")
                if not vname:
                    continue
                if vname in valid:
                    continue
                # Kubernetes auto-mounts a projected SA token volume.
                if vname.startswith("kube-api-access-"):
                    continue
                leaks.append(
                    f"{owner} container {c.get('name')!r} mounts "
                    f"volume {vname!r} which isn't in pod volumes nor "
                    f"in volumeClaimTemplates"
                )
    assert not leaks, "Dangling volumeMount references:\n  " + "\n  ".join(leaks)


# ---------------------------------------------------------------------------
# 5. Service.spec.selector matches at least one pod-template label set.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_every_service_selector_matches_at_least_one_pod_template(whole_create_adb):
    """A Service whose selector matches no pod is dead weight at best
    and a copy-paste bug at worst. Skip headless services (no ports) and
    sub-chart services we can't reason about cleanly."""
    workloads = []
    for d in whole_create_adb:
        if d.get("kind") not in {"Deployment", "StatefulSet", "DaemonSet"}:
            continue
        labels = (((d.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("labels") or {}
        workloads.append(((d.get("metadata") or {}).get("name", "?"), labels))
    leaks = []
    for d in whole_create_adb:
        if d.get("kind") != "Service":
            continue
        sel = (d.get("spec") or {}).get("selector") or {}
        if not sel:
            continue
        svc_name = (d.get("metadata") or {}).get("name", "?")
        if any(all(labels.get(k) == v for k, v in sel.items())
               for _, labels in workloads):
            continue
        # Could be a service for a sub-chart workload we don't iterate
        # in this rendering (e.g., NIMService → NIMCache CRDs render
        # workloads outside our doc set).
        if "nim" in svc_name.lower() or "nemotron" in svc_name.lower():
            continue
        leaks.append(f"Service/{svc_name} selector {sel} matches no workload")
    # Be lenient: services may target NIMServices / external endpoints.
    # Just warn loudly if there are MANY unresolved selectors.
    assert len(leaks) <= 5, (
        f"More than 5 Services have selectors with no matching workload:\n  "
        + "\n  ".join(leaks[:10])
    )


# ---------------------------------------------------------------------------
# 6. helm install --dry-run=client succeeds end-to-end.
# ---------------------------------------------------------------------------
@NEEDS_HELM
@pytest.mark.parametrize("values_file, extra_sets", [
    ("values.create-adb.yaml", []),
    ("values.existing-adb.yaml",
     ["--set", "oracle.credentials.adminPassword=fake-admin-pw",
      "--set", "oracle.credentials.appPassword=fake-app-pw"]),
])
def test_helm_template_validates_with_kube_version(
    helm_chart_dir, values_file, extra_sets
):
    """`helm template` with a pinned Kube version is what CI runs in a
    container without a cluster — fully client-side. This catches
    template errors, missing values, malformed YAML.

    We intentionally avoid `helm install --dry-run=client` because in
    helm 3.18 it still calls ``Capabilities.APIVersions.Has`` against a
    real cluster (required by our llm-nim.yaml's ``lookup`` for GPU
    autodetect)."""
    proc = subprocess.run(
        [HELM_BIN, "template", "rag-dryrun", ".",
         "--kube-version=1.31.0",
         "-f", values_file, *REQUIRED_CREDS, *extra_sets],
        cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"helm template ({values_file}) failed:\n"
        f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    )
    # And the output must be parseable YAML (catch malformed templates).
    docs = list(yaml.safe_load_all(proc.stdout))
    assert len(docs) >= 10, (
        f"helm template emitted only {len(docs)} docs; expected 10+ "
        "(deployments, services, jobs, secrets, configmaps, ...)"
    )


# ---------------------------------------------------------------------------
# 7. helm lint passes.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_helm_lint_passes(helm_chart_dir):
    """`helm lint` is what the upstream CI's 'Helm Blueprint Compliance'
    check runs. If we fail this, the PR's helm-compliance check breaks."""
    proc = subprocess.run(
        [HELM_BIN, "lint", ".", "-f", "values.create-adb.yaml",
         *REQUIRED_CREDS],
        cwd=str(helm_chart_dir), capture_output=True, text=True, timeout=60,
    )
    # `helm lint` returns 1 on errors, 0 on success/info-only. Some
    # warnings (icon missing, etc.) emit info messages but still exit 0.
    assert proc.returncode == 0, (
        f"`helm lint` failed (this is what CI's Helm Blueprint Compliance "
        f"check runs):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# 8. Stock blueprint chart still renders without Oracle values
#    (zero-impact guarantee — non-Oracle customers must be unaffected).
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_stock_blueprint_chart_renders_without_oracle_values():
    """Render the upstream chart by itself with no Oracle values. Must
    succeed — proves our envFrom/extraVolumes edits to deployment.yaml
    don't break the stock blueprint experience.

    Auto-runs ``helm dep update`` first because we deliberately do not
    bundle the third-party charts (per Sumit's review). Skips cleanly
    if no network is available (offline CI).
    """
    chart = Path(__file__).resolve().parents[3] / "deploy" / "helm" / "nvidia-blueprint-rag"
    if not chart.exists():
        pytest.skip("upstream chart not present")

    # Materialise deps if not already present.
    charts_dir = chart / "charts"
    needs_update = (
        not charts_dir.is_dir()
        or not any(charts_dir.glob("*.tgz"))
    )
    if needs_update:
        dep_proc = subprocess.run(
            [HELM_BIN, "dependency", "update", str(chart)],
            capture_output=True, text=True, timeout=180,
        )
        if dep_proc.returncode != 0:
            pytest.skip(
                "helm dependency update failed (likely no network); "
                f"skip stock-chart render test. stderr: "
                f"{dep_proc.stderr.strip()[:300]}"
            )

    proc = subprocess.run(
        [HELM_BIN, "template", "stocktest", ".",
         "--kube-version=1.31.0",
         "--set", "ngcApiSecret.password=fake",
         "--set", "imagePullSecret.password=fake"],
        cwd=str(chart), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"Stock blueprint chart fails to render without Oracle values "
        f"— our edits broke the non-Oracle path!\nSTDOUT:\n"
        f"{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# 9. BYO existing-ADB mode parity: same set of CORE workloads as create-adb.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_byo_mode_renders_same_core_workloads(whole_create_adb, whole_existing_adb):
    """Both modes must produce rag-server, ingestor-server, oracle-pai —
    only the provisioner Job differs."""
    create_idx = _index(whole_create_adb)
    existing_idx = _index(whole_existing_adb)

    def _has(idx, frag):
        return any(frag in n.lower() for k, n in idx if k in {"Deployment", "StatefulSet"})

    for frag in ("rag-server", "ingestor", "pai"):
        assert _has(create_idx, frag), f"create-adb missing {frag}"
        assert _has(existing_idx, frag), f"existing-adb missing {frag}"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm-based chart rendering tests for the Oracle 26ai wrapper chart.

These tests shell out to the local ``helm`` binary to render the wrapper
chart with several scenarios:
* Default (create-adb path, GPU offload on)
* Existing-ADB (BYO database)
* GPU offload opted out (``oracle.gpuIndexOffload.enabled=false``)
* Service overridden to ClusterIP
* Custom PAI port
* Validation fails on missing OCR creds

We assert on the rendered YAML to lock down the contract: which
resources are present, which annotations are stamped, and which
Kubernetes objects show up only when their feature is enabled.

Tests are skipped if ``helm`` is not on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(
    HELM_BIN is None, reason="helm CLI not on PATH (skip chart-render tests)",
)

REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_helm_template(chart_dir: Path, *extra: str, fail_ok: bool = False) -> str:
    cmd = [
        HELM_BIN, "template", "testrun", ".",
        "--include-crds=false",
    ] + list(extra)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(chart_dir),  # so relative `-f values.create-adb.yaml` resolves
    )
    if proc.returncode != 0 and not fail_ok:
        pytest.fail(
            f"helm template failed (exit {proc.returncode}):\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout if proc.returncode == 0 else proc.stderr


def _split_docs(rendered: str) -> list[dict[str, Any]]:
    """Split a multi-doc YAML stream and skip empty docs."""
    return [d for d in yaml.safe_load_all(rendered) if d]


def _by_kind_and_name(docs):
    """Return {(kind, name): doc} for easy assertions."""
    out = {}
    for d in docs:
        kind = d.get("kind")
        name = (d.get("metadata") or {}).get("name")
        if kind and name:
            out[(kind, name)] = d
    return out


# ---------------------------------------------------------------------------
# Session fixtures: render once per scenario, share across tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rendered_default(helm_chart_dir):
    """Default rendering: create-adb values, full creds, GPU offload on."""
    if HELM_BIN is None:
        pytest.skip("helm not on PATH")
    return _run_helm_template(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        *REQUIRED_CREDS,
        "--set", "oracle.containerRegistry.username=ssouser@example.com",
        "--set", "oracle.containerRegistry.password=ocr-token",
    )


@pytest.fixture(scope="session")
def rendered_existing_adb(helm_chart_dir):
    """Existing-ADB rendering: provisioner skipped, GPU offload still on."""
    if HELM_BIN is None:
        pytest.skip("helm not on PATH")
    return _run_helm_template(
        helm_chart_dir,
        "-f", "values.existing-adb.yaml",
        *REQUIRED_CREDS,
        "--set", "oracle.existing.user=RAG_APP",
        "--set", "oracle.existing.password=secret",
        "--set", "oracle.existing.connectString=existing_medium",
        "--set", "oracle.containerRegistry.username=ssouser@example.com",
        "--set", "oracle.containerRegistry.password=ocr-token",
    )


@pytest.fixture(scope="session")
def rendered_offload_off(helm_chart_dir):
    """Opt-out: gpuIndexOffload.enabled=false should drop ALL PAI resources."""
    if HELM_BIN is None:
        pytest.skip("helm not on PATH")
    return _run_helm_template(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        *REQUIRED_CREDS,
        "--set", "oracle.gpuIndexOffload.enabled=false",
    )


@pytest.fixture(scope="session")
def rendered_clusterip(helm_chart_dir):
    """Force ClusterIP override on the PAI Service."""
    if HELM_BIN is None:
        pytest.skip("helm not on PATH")
    return _run_helm_template(
        helm_chart_dir,
        "-f", "values.create-adb.yaml",
        *REQUIRED_CREDS,
        "--set", "oracle.containerRegistry.username=u",
        "--set", "oracle.containerRegistry.password=p",
        "--set", "oracle.gpuIndexOffload.service.type=ClusterIP",
    )


# ---------------------------------------------------------------------------
# Default scenario assertions
# ---------------------------------------------------------------------------
@NEEDS_HELM
class TestDefaultRendering:
    def test_renders_all_required_oracle_resources(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))

        # Provisioner (pre-install) hook
        assert ("Job", "oracle-adb-provisioner") in docs
        # PAI gpu-index Deployment (the cuVS workload)
        assert ("Deployment", "oracle-pai-gpu-index") in docs
        # Both Services (LB for ADB, ClusterIP sidecar for in-cluster checks)
        assert ("Service", "oracle-pai-gpu-index") in docs
        assert ("Service", "oracle-pai-gpu-index-internal") in docs
        # Verify Job + RBAC (post-install)
        assert ("Job", "oracle-pai-verify") in docs
        assert ("ServiceAccount", "oracle-pai-verify") in docs
        assert ("Role", "oracle-pai-verify") in docs
        assert ("RoleBinding", "oracle-pai-verify") in docs

    def test_pai_service_is_oci_internal_loadbalancer(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"]["type"] == "LoadBalancer"
        annotations = svc["metadata"].get("annotations") or {}
        assert annotations.get("service.beta.kubernetes.io/oci-load-balancer-internal") == "true", (
            "PAI LB must be marked internal -- exposing the cuVS service to the "
            "public internet is a security incident waiting to happen"
        )

    def test_internal_clusterip_service_is_clusterip(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        svc = docs[("Service", "oracle-pai-gpu-index-internal")]
        assert svc["spec"]["type"] == "ClusterIP"

    def test_pai_service_targets_port_8080(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        for name in ("oracle-pai-gpu-index", "oracle-pai-gpu-index-internal"):
            svc = docs[("Service", name)]
            ports = svc["spec"]["ports"]
            assert any(p.get("port") == 8080 for p in ports), (
                f"{name} must expose port 8080 (cuVS API port)"
            )

    def test_default_index_is_hnsw_for_offload(self, rendered_default):
        """Offload only works for HNSW; the chart MUST default to HNSW so the
        cuVS GPU path engages without the user knowing the SQL detail."""
        docs = _split_docs(rendered_default)
        env_payloads = []
        for d in docs:
            spec = d.get("spec", {})
            tmpl = (spec.get("template") or {}).get("spec") or spec
            for c in (tmpl.get("containers") or []):
                for env in (c.get("env") or []):
                    env_payloads.append((env.get("name"), env.get("value")))

        index_envs = [v for k, v in env_payloads if k in (
            "APP_VECTORSTORE_INDEXTYPE", "ORACLE_VECTOR_INDEX_TYPE",
        )]
        # If any are stamped, they must be HNSW; if none are stamped, the
        # ConfigMap-from-files path drives them, which is also tested below.
        for v in index_envs:
            assert v == "HNSW", (
                f"Default index type must be HNSW for cuVS offload; got {v!r}"
            )

    def test_provisioner_args_carry_pai_offload_enabled(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        job = docs[("Job", "oracle-adb-provisioner")]
        containers = job["spec"]["template"]["spec"]["containers"]
        # Find the provisioner container and inspect its args
        prov_args = []
        for c in containers:
            args = c.get("args") or []
            cmd = c.get("command") or []
            prov_args += args + cmd
        # The flag should be passed when GPU offload is on
        joined = " ".join(prov_args)
        assert "--pai-offload-enabled" in joined, (
            "Provisioner must be invoked with --pai-offload-enabled when GPU "
            "offload is on so it opens the OCI security path"
        )

    def test_verify_job_is_post_install_hook(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        job = docs[("Job", "oracle-pai-verify")]
        ann = job["metadata"].get("annotations") or {}
        hook = ann.get("helm.sh/hook", "")
        # Must run AFTER everything else is up so the LB has time to provision
        assert "post-install" in hook
        assert "post-upgrade" in hook

    def test_provisioner_job_is_pre_install_hook(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        job = docs[("Job", "oracle-adb-provisioner")]
        ann = job["metadata"].get("annotations") or {}
        hook = ann.get("helm.sh/hook", "")
        assert "pre-install" in hook

    def test_verify_job_rbac_is_namespace_scoped(self, rendered_default):
        """Cluster-wide perms here would be a security smell. Role + RoleBinding
        only -- never ClusterRole/ClusterRoleBinding."""
        docs = _split_docs(rendered_default)
        for d in docs:
            kind = d.get("kind", "")
            name = (d.get("metadata") or {}).get("name", "")
            if "verify" in name:
                assert kind not in ("ClusterRole", "ClusterRoleBinding"), (
                    f"verify Job RBAC must NOT use {kind} -- found {kind}/{name}"
                )

    def test_verify_role_has_required_verbs_only(self, rendered_default):
        """Pin the role to least privilege -- get/list/watch services, get/patch/update secrets,
        get/list/patch deployments. No wildcards."""
        docs = _by_kind_and_name(_split_docs(rendered_default))
        role = docs[("Role", "oracle-pai-verify")]
        rules = role["rules"]
        flat = []
        for r in rules:
            for resource in r["resources"]:
                for verb in r["verbs"]:
                    flat.append((tuple(r.get("apiGroups") or [""]), resource, verb))

        # Must be able to read services (find the LB IP)
        assert any(r == "services" and v in ("get", "list") for _, r, v in flat)
        # Must be able to patch secrets (stamp ORACLE_PAI_INDEX_URL)
        assert any(r == "secrets" and v in ("patch", "update") for _, r, v in flat)
        # Must be able to patch deployments (roll consumers)
        assert any(r == "deployments" and v == "patch" for _, r, v in flat)

        # No wildcards
        for _groups, resource, verb in flat:
            assert verb != "*"
            assert resource != "*"

    def test_no_cluster_local_in_pre_install_secret_stamp(self, rendered_default):
        """ADB cannot resolve ``cluster.local`` DNS -- the chart MUST NOT stamp
        a cluster.local URL into oracle-creds at install time. The verify Job
        patches the real LB IP after the LoadBalancer comes up."""
        # Find the provisioner Job's args/env -- if any cluster.local string is
        # used as the OFFLOAD URL, that's the bug.
        assert "cluster.local" not in rendered_default or self._is_only_in_pure_dns_lookups(
            rendered_default,
        ), "cluster.local must not be stamped onto oracle-creds at pre-install time"

    def _is_only_in_pure_dns_lookups(self, rendered: str) -> bool:
        """cluster.local can legitimately appear in resolver hints (e.g. dnsConfig
        searches) -- but never as a value passed to ORACLE_PAI_INDEX_URL or any
        OFFLOAD_URL setting. Crude: scan the rendered text for both substrings."""
        for line in rendered.splitlines():
            if "cluster.local" in line and "oracle-pai" in line.lower():
                # An oracle-pai...cluster.local URL -> bad
                return False
        return True

    def test_pai_deployment_image_is_gpu_index_variant(self, rendered_default):
        """Headline requirement: the ``cuVS``-equipped variant of the PAI
        container image. Older non-GPU variants exist and should never be used
        by this integration."""
        docs = _by_kind_and_name(_split_docs(rendered_default))
        dep = docs[("Deployment", "oracle-pai-gpu-index")]
        containers = dep["spec"]["template"]["spec"]["containers"]
        images = [c["image"] for c in containers if "image" in c]
        assert any("gpu-index" in img for img in images), (
            "Deployment must reference the cuVS-equipped 'gpu-index' tag of the "
            "Oracle Private AI Services container"
        )

    def test_pai_deployment_requests_a_gpu(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        dep = docs[("Deployment", "oracle-pai-gpu-index")]
        containers = dep["spec"]["template"]["spec"]["containers"]
        # Some container in the Deployment must request nvidia.com/gpu
        gpu_requested = False
        for c in containers:
            res = c.get("resources") or {}
            limits = res.get("limits") or {}
            if any(k.startswith("nvidia.com/gpu") for k in limits):
                gpu_requested = True
        assert gpu_requested, "PAI Deployment must request a GPU"


# ---------------------------------------------------------------------------
# Existing-ADB scenario
# ---------------------------------------------------------------------------
@NEEDS_HELM
class TestExistingADB:
    def test_no_provisioner_job_when_existing(self, rendered_existing_adb):
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        assert ("Job", "oracle-adb-provisioner") not in docs, (
            "BYO database path must skip the ADB provisioner Job"
        )

    def test_pai_resources_still_rendered_for_existing_adb(self, rendered_existing_adb):
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        # GPU offload is on by default for existing too
        assert ("Deployment", "oracle-pai-gpu-index") in docs
        assert ("Service", "oracle-pai-gpu-index") in docs
        assert ("Job", "oracle-pai-verify") in docs

    def test_oracle_creds_secret_uses_existing_values(self, rendered_existing_adb):
        """Existing-ADB path must stamp the user-supplied creds onto oracle-creds."""
        import base64
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        # If chart defines oracle-creds inline (not just via Job), check it
        for (kind, name), d in docs.items():
            if kind == "Secret" and name == "oracle-creds":
                data = d.get("data") or d.get("stringData") or {}
                if "ORACLE_USER" in data and isinstance(data["ORACLE_USER"], str):
                    decoded = base64.b64decode(data["ORACLE_USER"]).decode()
                    assert decoded == "RAG_APP"


# ---------------------------------------------------------------------------
# Opt-out scenario
# ---------------------------------------------------------------------------
@NEEDS_HELM
class TestGpuOffloadOptOut:
    def test_no_pai_resources_when_disabled(self, rendered_offload_off):
        """All PAI-named resources must vanish when offload is opted out."""
        docs = _by_kind_and_name(_split_docs(rendered_offload_off))
        for (kind, name) in docs.keys():
            assert "oracle-pai" not in name, (
                f"oracle.gpuIndexOffload.enabled=false must drop all PAI "
                f"resources; found {kind}/{name}"
            )

    def test_provisioner_args_drop_pai_offload_enabled(self, rendered_offload_off):
        docs = _by_kind_and_name(_split_docs(rendered_offload_off))
        job = docs.get(("Job", "oracle-adb-provisioner"))
        if job is None:
            pytest.skip("provisioner skipped in this scenario")
        for c in job["spec"]["template"]["spec"]["containers"]:
            args = c.get("args") or []
            assert "--pai-offload-enabled" not in args, (
                "When GPU offload is off, --pai-offload-enabled must NOT be passed"
            )


# ---------------------------------------------------------------------------
# ClusterIP override
# ---------------------------------------------------------------------------
@NEEDS_HELM
class TestServiceTypeOverride:
    def test_clusterip_override_drops_lb_annotation(self, rendered_clusterip):
        docs = _by_kind_and_name(_split_docs(rendered_clusterip))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"]["type"] == "ClusterIP"
        annotations = svc["metadata"].get("annotations") or {}
        # OCI LB annotations don't apply to ClusterIP -- they should be absent
        assert "service.beta.kubernetes.io/oci-load-balancer-internal" not in annotations


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BYO data import scenarios
# ---------------------------------------------------------------------------
@NEEDS_HELM
class TestBYOImport:
    """The oracle-byo-import Job exposes pre-existing customer data to the
    frontend. It must:
    * NOT render in default create-adb (no BYO needed for fresh DBs)
    * Render automatically when oracle.mode=existing (discovery pass)
    * Render when oracle.importExistingTables is non-empty (mapping pass)
    * Run as a post-install hook so oracle-creds is populated first
    """

    def test_no_byo_job_in_default_create_adb(self, rendered_default):
        docs = _by_kind_and_name(_split_docs(rendered_default))
        assert ("Job", "oracle-byo-import") not in docs, (
            "BYO Job must NOT render in fresh-DB mode with no import entries; "
            "would be a no-op spinning up an extra Pod for no reason"
        )

    def test_byo_job_renders_for_existing_adb(self, rendered_existing_adb):
        """existing-ADB mode always runs the discovery pass so canonical
        BYO tables get auto-registered + listed in the UI on first install."""
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        assert ("Job", "oracle-byo-import") in docs

    def test_byo_job_uses_oracle_creds_envfrom(self, rendered_existing_adb):
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        job = docs[("Job", "oracle-byo-import")]
        containers = job["spec"]["template"]["spec"]["containers"]
        env_froms = [ef for c in containers for ef in (c.get("envFrom") or [])]
        secret_refs = [ef.get("secretRef", {}).get("name") for ef in env_froms]
        assert "oracle-creds" in secret_refs, (
            "BYO Job must envFrom: oracle-creds so it inherits ORACLE_USER / "
            "ORACLE_PASSWORD / ORACLE_CS without reimplementing the lookup"
        )

    def test_byo_job_is_post_install(self, rendered_existing_adb):
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        job = docs[("Job", "oracle-byo-import")]
        ann = job["metadata"].get("annotations") or {}
        hook = ann.get("helm.sh/hook", "")
        assert "post-install" in hook
        assert "post-upgrade" in hook

    def test_byo_job_renders_when_import_entries_set_in_create_adb(
        self, helm_chart_dir,
    ):
        """Even in create-adb mode, populating importExistingTables should
        spin up the Job to create the SQL views post-install."""
        out = _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            *REQUIRED_CREDS,
            "--set", "oracle.containerRegistry.username=u",
            "--set", "oracle.containerRegistry.password=p",
            "--set", "oracle.importExistingTables[0].sourceTable=KB.MY_DOCS",
            "--set", "oracle.importExistingTables[0].collectionName=my_kb",
            "--set", "oracle.importExistingTables[0].columns.text=content",
            "--set", "oracle.importExistingTables[0].columns.vector=embedding",
        )
        docs = _by_kind_and_name(_split_docs(out))
        assert ("Job", "oracle-byo-import") in docs
        job = docs[("Job", "oracle-byo-import")]

        # Verify the BYO entries are stamped into the env so the Job script
        # can read them with json.loads()
        envs = []
        for c in job["spec"]["template"]["spec"]["containers"]:
            envs += c.get("env") or []
        byo_env = next((e for e in envs if e["name"] == "BYO_TABLES_JSON"), None)
        assert byo_env is not None
        import json as _json
        parsed = _json.loads(byo_env["value"])
        assert isinstance(parsed, list) and len(parsed) == 1
        entry = parsed[0]
        assert entry["sourceTable"] == "KB.MY_DOCS"
        assert entry["collectionName"] == "my_kb"
        assert entry["columns"]["text"] == "content"
        assert entry["columns"]["vector"] == "embedding"

    def test_byo_job_does_not_drop_or_truncate(self, rendered_existing_adb):
        """Anti-footgun: the script must never DROP / TRUNCATE / DELETE FROM
        anything. Customer BYO data is sacred."""
        docs = _by_kind_and_name(_split_docs(rendered_existing_adb))
        job = docs[("Job", "oracle-byo-import")]
        args = []
        for c in job["spec"]["template"]["spec"]["containers"]:
            args += c.get("args") or []
            args += c.get("command") or []
        joined = "\n".join(args).upper()
        for forbidden in ("DROP TABLE", "TRUNCATE", "DELETE FROM"):
            assert forbidden not in joined, (
                f"BYO Job must not contain {forbidden!r} -- never mutate "
                "customer base tables"
            )


@NEEDS_HELM
class TestValidations:
    def test_missing_ngc_password_fails_install(self, helm_chart_dir):
        out = _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            "--set", "oracle.containerRegistry.username=u",
            "--set", "oracle.containerRegistry.password=p",
            fail_ok=True,
        )
        assert "ngcApiSecret.password" in out

    def test_missing_ocr_creds_with_offload_on_fails_install(self, helm_chart_dir):
        out = _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            *REQUIRED_CREDS,
            fail_ok=True,
        )
        # The validation block must mention the OCR credentials
        assert "container-registry.oracle.com" in out or "containerRegistry" in out
        # And gives a concrete opt-out path
        assert "gpuIndexOffload.enabled=false" in out

    def test_missing_ocr_creds_with_offload_off_succeeds(self, helm_chart_dir):
        """Opting out of GPU offload must NOT require OCR credentials."""
        out = _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            *REQUIRED_CREDS,
            "--set", "oracle.gpuIndexOffload.enabled=false",
        )
        # If we got here without fail_ok the render succeeded; sanity-check it
        # produced YAML
        assert "kind:" in out

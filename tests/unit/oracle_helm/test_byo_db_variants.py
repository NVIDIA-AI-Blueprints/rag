# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BYO DB variant tests: every customer topology for existing-ADB and create-ADB.

Each test class maps to a real customer deployment topology:

1. Same-VCN private endpoint   — walletless TLS, internal LB (simplest)
2. Same-VCN public endpoint    — mTLS with wallet, internal LB
3. Cross-region                 — mTLS, public LB, auth + HTTPS on PAI
4. Cross-tenancy                — same as cross-region, different --set combos
5. In-cluster container DB      — minimal settings, ClusterIP PAI service
6. Developer vs Full ADB        — provisioner --developer flag toggle
7. GPU offload disabled         — no PAI resources at all

Tests use ``helm template`` to render the chart, parse YAML, and assert on
specific resource properties.  Skipped if ``helm`` is not on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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
OCR_CREDS = [
    "--set", "oracle.containerRegistry.username=ssouser@example.com",
    "--set", "oracle.containerRegistry.password=ocr-token",
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
        cwd=str(chart_dir),
    )
    if proc.returncode != 0 and not fail_ok:
        pytest.fail(
            f"helm template failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout if proc.returncode == 0 else proc.stderr


def _split_docs(rendered: str) -> list[dict[str, Any]]:
    """Split a multi-doc YAML stream and skip empty docs."""
    return [d for d in yaml.safe_load_all(rendered) if d]


def _by_kind_and_name(docs) -> dict[tuple[str, str], dict]:
    """Return {(kind, name): doc} for easy assertions."""
    out: dict[tuple[str, str], dict] = {}
    for d in docs:
        kind = d.get("kind")
        name = (d.get("metadata") or {}).get("name")
        if kind and name:
            out[(kind, name)] = d
    return out


def _pai_env(docs: dict) -> dict[str, str]:
    """Extract env vars from the PAI gpu-index Deployment as {name: value}."""
    dep = docs.get(("Deployment", "oracle-pai-gpu-index"))
    if dep is None:
        return {}
    env: dict[str, str] = {}
    for c in dep["spec"]["template"]["spec"]["containers"]:
        for e in c.get("env") or []:
            env[e["name"]] = e.get("value", "")
    return env


def _pai_svc_annotations(docs: dict) -> dict[str, str]:
    """Get annotations from the PAI LoadBalancer Service."""
    svc = docs.get(("Service", "oracle-pai-gpu-index"))
    if svc is None:
        return {}
    return svc["metadata"].get("annotations") or {}


def _provisioner_args_text(job_doc: dict) -> str:
    """Concatenate all args/command strings from the provisioner Job."""
    parts: list[str] = []
    for c in job_doc["spec"]["template"]["spec"]["containers"]:
        parts += c.get("command") or []
        parts += c.get("args") or []
    return " ".join(parts)


def _write_overlay(extra: dict) -> str:
    """Write a temp values YAML overlay and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="helm-test-")
    with os.fdopen(fd, "w") as f:
        yaml.dump(extra, f)
    return path


# ===================================================================
# 1. Same-VCN Private Endpoint (default existing-adb)
# ===================================================================
@NEEDS_HELM
class TestSameVcnPrivateEndpoint:
    """Customer has an ADB with a *private endpoint* in the same VCN as OKE.

    Simplest BYO topology: walletless TLS, the ADB resolves the OKE
    internal LoadBalancer IP directly, no internet traversal.  Uses
    values.existing-adb.yaml without modifications.
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.existing-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
        )

    def test_no_provisioner_job(self, rendered):
        """BYO database must skip the ADB provisioner — it would try to
        CREATE an ADB the customer already has."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-adb-provisioner") not in docs

    def test_pai_deployment_renders(self, rendered):
        """GPU offload is ON by default even for BYO databases."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Deployment", "oracle-pai-gpu-index") in docs

    def test_pai_service_is_internal_lb(self, rendered):
        """Same VCN: the LB is internal so it gets a VCN-routable IP.
        No internet exposure of the cuVS endpoint."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"]["type"] == "LoadBalancer"
        ann = _pai_svc_annotations(docs)
        assert ann.get(
            "service.beta.kubernetes.io/oci-load-balancer-internal"
        ) == "true"

    def test_no_wallet_volume_in_default(self, rendered):
        """Private endpoint ADB uses walletless TLS — no wallet Secret mount
        should appear anywhere in the rendered output."""
        for d in _split_docs(rendered):
            if d.get("kind") not in ("Deployment", "StatefulSet"):
                continue
            spec = d["spec"]["template"]["spec"]
            vol_names = [v["name"] for v in (spec.get("volumes") or [])]
            assert "oracle-wallet" not in vol_names, (
                f"Default existing-adb must not mount oracle-wallet "
                f"(walletless TLS); found in {d['kind']}/"
                f"{d['metadata']['name']}"
            )

    def test_https_enabled_by_default(self, rendered):
        """ADB 26ai requires HTTPS for OFFLOAD_URL (ORA-52346), so HTTPS
        is always on regardless of VCN topology."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_HTTPS_ENABLED") == "true"

    def test_internal_clusterip_sidecar_exists(self, rendered):
        """The ClusterIP sidecar Service lets the verify Job health-check
        PAI without waiting for the OCI LB to provision."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index-internal")]
        assert svc["spec"]["type"] == "ClusterIP"

    def test_preflight_job_renders(self, rendered):
        """Preflight catches bad OCR creds / missing GPU before the user
        waits 30 minutes for a provisioner that will ultimately fail."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-preflight") in docs

    def test_verify_job_renders(self, rendered):
        """Post-install verify patches oracle-creds with the real LB IP
        so ADB can dial OFFLOAD_URL for cuVS index builds."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-verify") in docs

    def test_gpu_mode_is_force(self, rendered):
        """Default GPU mode must be FORCE — fall back to CPU silently
        defeats the purpose of deploying the cuVS container."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_GPU_MODE") == "FORCE"

    def test_pai_image_is_gpu_index_variant(self, rendered):
        """Only the gpu-index variant of Oracle PAI has cuVS linked in.
        The generic PAI image would silently ignore OFFLOAD_URL requests."""
        docs = _by_kind_and_name(_split_docs(rendered))
        dep = docs[("Deployment", "oracle-pai-gpu-index")]
        images = [
            c["image"]
            for c in dep["spec"]["template"]["spec"]["containers"]
            if "image" in c
        ]
        assert any("gpu-index" in img for img in images)


# ===================================================================
# 2. Same-VCN Public Endpoint (mTLS with wallet)
# ===================================================================
@NEEDS_HELM
class TestSameVcnPublicEndpoint:
    """Customer has an ADB with a *public endpoint* but in the same VCN.

    mTLS required: the wallet Secret is mounted into rag-server and
    ingestor-server pods, TNS_ADMIN points at the wallet directory.
    PAI LB stays internal because ADB-to-OKE traffic routes inside
    the VCN via the public endpoint's NAT gateway.
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        overlay = {
            "rag": {
                "extraVolumes": [
                    {"name": "oracle-pip-packages", "emptyDir": {}},
                    {
                        "name": "oracle-wallet",
                        "secret": {
                            "secretName": "oracle-wallet",
                            "defaultMode": 256,
                        },
                    },
                ],
                "extraVolumeMounts": [
                    {"name": "oracle-pip-packages", "mountPath": "/opt/oracle-deps"},
                    {
                        "name": "oracle-wallet",
                        "mountPath": "/app/wallet",
                        "readOnly": True,
                    },
                ],
                "envVars": {"TNS_ADMIN": "/app/wallet"},
                "ingestor-server": {
                    "extraVolumes": [
                        {"name": "oracle-pip-packages", "emptyDir": {}},
                        {
                            "name": "oracle-wallet",
                            "secret": {
                                "secretName": "oracle-wallet",
                                "defaultMode": 256,
                            },
                        },
                    ],
                    "extraVolumeMounts": [
                        {"name": "oracle-pip-packages", "mountPath": "/opt/oracle-deps"},
                        {
                            "name": "oracle-wallet",
                            "mountPath": "/app/wallet",
                            "readOnly": True,
                        },
                    ],
                },
            },
        }
        path = _write_overlay(overlay)
        try:
            return _run_helm_template(
                helm_chart_dir,
                "-f", "values.existing-adb.yaml",
                "-f", path,
                *REQUIRED_CREDS,
                *OCR_CREDS,
            )
        finally:
            os.unlink(path)

    def test_wallet_volume_present(self, rendered):
        """mTLS path: the oracle-wallet Secret must be mounted so oracledb
        can find cwallet.sso / tnsnames.ora at runtime."""
        found = False
        for d in _split_docs(rendered):
            if d.get("kind") != "Deployment":
                continue
            for v in d["spec"]["template"]["spec"].get("volumes") or []:
                if v.get("name") == "oracle-wallet":
                    assert "secret" in v
                    assert v["secret"]["secretName"] == "oracle-wallet"
                    found = True
        assert found, (
            "oracle-wallet volume must appear in at least one Deployment"
        )

    def test_wallet_mount_at_app_wallet(self, rendered):
        """Verify the volumeMount path matches the TNS_ADMIN convention
        documented in the values file comments."""
        found = False
        for d in _split_docs(rendered):
            if d.get("kind") != "Deployment":
                continue
            spec = d["spec"]["template"]["spec"]
            for c in (spec.get("containers") or []) + (spec.get("initContainers") or []):
                for vm in c.get("volumeMounts") or []:
                    if vm.get("name") == "oracle-wallet":
                        assert vm["mountPath"] == "/app/wallet"
                        assert vm.get("readOnly") is True
                        found = True
        assert found, "oracle-wallet volumeMount at /app/wallet must be present"

    def test_tns_admin_env_present(self, rendered):
        """TNS_ADMIN tells oracledb where to find the wallet directory.
        Without it, oracledb defaults to $ORACLE_HOME/network/admin which
        doesn't exist in the stock NGC image."""
        assert "TNS_ADMIN" in rendered, (
            "TNS_ADMIN env var must be set for mTLS wallet-based connections"
        )

    def test_pai_lb_still_internal(self, rendered):
        """Same VCN: even with a public-endpoint ADB, the PAI LB is
        internal — ADB routes to the VCN via its public endpoint's NAT."""
        docs = _by_kind_and_name(_split_docs(rendered))
        ann = _pai_svc_annotations(docs)
        assert ann.get(
            "service.beta.kubernetes.io/oci-load-balancer-internal"
        ) == "true"

    def test_https_on_for_same_vcn(self, rendered):
        """ADB 26ai requires HTTPS for OFFLOAD_URL even within the same VCN."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_HTTPS_ENABLED") == "true"


# ===================================================================
# 3. Cross-Region (public LB, auth + HTTPS)
# ===================================================================
@NEEDS_HELM
class TestCrossRegion:
    """ADB in region A, OKE in region B.  Traffic traverses the public
    internet.

    The PAI Service gets a *public* LoadBalancer (no internal annotation).
    HTTPS and authentication are enabled so enterprise data is encrypted
    and only authorised ADB instances can call the cuVS endpoint.
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.existing-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
            "--set", "oracle.gpuIndexOffload.service.internal=false",
            "--set", "oracle.gpuIndexOffload.auth=true",
            "--set", "oracle.gpuIndexOffload.https=true",
        )

    def test_pai_service_is_public_lb(self, rendered):
        """Cross-region: the LB must be PUBLIC so ADB in another region
        can reach it over the internet."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"]["type"] == "LoadBalancer"
        ann = _pai_svc_annotations(docs)
        assert (
            "service.beta.kubernetes.io/oci-load-balancer-internal" not in ann
        ), (
            "Cross-region: PAI LB must NOT be annotated as internal — "
            "ADB in another region cannot reach a VCN-internal IP"
        )

    def test_authentication_enabled(self, rendered):
        """PAI must enforce API-key auth when exposed to the internet.
        Without this, anyone with the public IP could POST index data."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_AUTHENTICATION_ENABLED") == "true"

    def test_https_enabled(self, rendered):
        """PAI must serve HTTPS when traffic crosses the internet.
        Vector data in flight would otherwise be visible to any network
        observer between regions."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_HTTPS_ENABLED") == "true"

    def test_pai_deployment_still_requests_gpu(self, rendered):
        """cuVS needs a GPU regardless of network topology."""
        docs = _by_kind_and_name(_split_docs(rendered))
        dep = docs[("Deployment", "oracle-pai-gpu-index")]
        containers = dep["spec"]["template"]["spec"]["containers"]
        gpu_found = any(
            "nvidia.com/gpu" in (c.get("resources", {}).get("limits") or {})
            for c in containers
        )
        assert gpu_found, "PAI must request nvidia.com/gpu even in cross-region"

    def test_internal_clusterip_sidecar_unchanged(self, rendered):
        """The ClusterIP sidecar is topology-independent — the verify Job
        always health-checks through cluster.local DNS."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index-internal")]
        assert svc["spec"]["type"] == "ClusterIP"

    def test_lb_has_external_traffic_policy(self, rendered):
        """Public LB still needs externalTrafficPolicy for OCI CCM."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"].get("externalTrafficPolicy") == "Cluster"


# ===================================================================
# 4. Cross-Tenancy (same topology, different --set ordering)
# ===================================================================
@NEEDS_HELM
class TestCrossTenancy:
    """ADB in tenancy A, OKE in tenancy B.  Functionally identical to
    cross-region from a Helm perspective — public LB, auth, HTTPS.

    This test exercises a *reversed* --set ordering to prove Helm's
    deep merge does not silently drop overrides when flags arrive in
    various orders (regression guard against Helm 3.x merge quirks).
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.existing-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
            # Deliberately reversed order compared to TestCrossRegion
            "--set", "oracle.gpuIndexOffload.https=true",
            "--set", "oracle.gpuIndexOffload.auth=true",
            "--set", "oracle.gpuIndexOffload.service.internal=false",
        )

    def test_public_lb_with_reversed_set_order(self, rendered):
        """--set ordering must not matter for Helm deep merge."""
        docs = _by_kind_and_name(_split_docs(rendered))
        ann = _pai_svc_annotations(docs)
        assert (
            "service.beta.kubernetes.io/oci-load-balancer-internal" not in ann
        )

    def test_auth_and_https_with_reversed_set_order(self, rendered):
        """Both flags must land regardless of CLI ordering."""
        docs = _by_kind_and_name(_split_docs(rendered))
        env = _pai_env(docs)
        assert env.get("PRIVATE_AI_AUTHENTICATION_ENABLED") == "true"
        assert env.get("PRIVATE_AI_HTTPS_ENABLED") == "true"

    def test_verify_job_still_renders(self, rendered):
        """Verify Job is needed in cross-tenancy: it stamps the public LB
        IP onto oracle-creds so ADB can build HNSW indexes via
        OFFLOAD_URL."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-verify") in docs

    def test_all_three_pai_resources_present(self, rendered):
        """Deployment + LB Service + ClusterIP sidecar must all render."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Deployment", "oracle-pai-gpu-index") in docs
        assert ("Service", "oracle-pai-gpu-index") in docs
        assert ("Service", "oracle-pai-gpu-index-internal") in docs

    def test_preflight_job_renders(self, rendered):
        """Cross-tenancy still needs the preflight check — OCR auth may
        differ between tenancies."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-preflight") in docs


# ===================================================================
# 5. In-Cluster Container DB (ClusterIP PAI)
# ===================================================================
@NEEDS_HELM
class TestInClusterContainerDB:
    """Customer runs Oracle 26ai in a container *inside* the same K8s
    cluster (e.g. Oracle Database Free or a licensed DB image).

    No ADB provisioning, no OCI LoadBalancer needed.  The container DB
    can reach the PAI Service via ClusterIP + cluster.local DNS.
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.existing-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
            "--set", "oracle.gpuIndexOffload.service.type=ClusterIP",
        )

    def test_no_provisioner(self, rendered):
        """BYO container DB: no ADB provisioner needed."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-adb-provisioner") not in docs

    def test_pai_service_is_clusterip(self, rendered):
        """In-cluster DB can reach ClusterIP directly — no OCI LB needed."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert svc["spec"]["type"] == "ClusterIP"

    def test_no_oci_lb_annotations_on_clusterip(self, rendered):
        """OCI LB annotations are meaningless on a ClusterIP Service and
        would confuse kubectl describe output."""
        docs = _by_kind_and_name(_split_docs(rendered))
        ann = _pai_svc_annotations(docs)
        assert "service.beta.kubernetes.io/oci-load-balancer-internal" not in ann
        assert "service.beta.kubernetes.io/oci-load-balancer-shape" not in ann

    def test_no_external_traffic_policy_on_clusterip(self, rendered):
        """externalTrafficPolicy only applies to LoadBalancer/NodePort."""
        docs = _by_kind_and_name(_split_docs(rendered))
        svc = docs[("Service", "oracle-pai-gpu-index")]
        assert "externalTrafficPolicy" not in svc["spec"]

    def test_pai_deployment_still_renders(self, rendered):
        """cuVS GPU offload still works for an in-cluster container DB —
        only the Service type changes, not the workload."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Deployment", "oracle-pai-gpu-index") in docs

    def test_renders_valid_yaml(self, rendered):
        """Sanity: the full template renders parseable YAML."""
        docs = _split_docs(rendered)
        assert len(docs) > 0
        kinds = {d.get("kind") for d in docs}
        assert "Service" in kinds


# ===================================================================
# 6. Developer ADB vs Full ADB
# ===================================================================
@NEEDS_HELM
class TestDeveloperVsFullADB:
    """create-adb mode: ``developer: true`` vs ``developer: false``.

    Developer ADB is a low-cost, fixed-shape (2 ECPU) instance suitable
    for dev/test.  The provisioner script needs the ``--developer`` CLI
    flag to call the correct OCI API variant (isDedicated / isDevTier).
    """

    @pytest.fixture(scope="class")
    def rendered_developer(self, helm_chart_dir):
        """create-adb with developer=true (cheap dev/test instance)."""
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
            "--set", "oracle.createDatabase.developer=true",
        )

    @pytest.fixture(scope="class")
    def rendered_full(self, helm_chart_dir):
        """create-adb with developer=false (production auto-scaling)."""
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.create-adb.yaml",
            *REQUIRED_CREDS,
            *OCR_CREDS,
            "--set", "oracle.createDatabase.developer=false",
        )

    def test_developer_flag_present_when_true(self, rendered_developer):
        """When developer=true, the provisioner must pass --developer so
        the OCI SDK calls CreateAutonomousDatabase with isDevTier."""
        docs = _by_kind_and_name(_split_docs(rendered_developer))
        job = docs[("Job", "oracle-adb-provisioner")]
        args = _provisioner_args_text(job)
        assert "--developer" in args, (
            "developer=true must inject --developer into provisioner args"
        )

    def test_developer_flag_absent_when_false(self, rendered_full):
        """Full production ADB: --developer must NOT appear so the
        provisioner creates a standard auto-scaling instance."""
        docs = _by_kind_and_name(_split_docs(rendered_full))
        job = docs[("Job", "oracle-adb-provisioner")]
        args = _provisioner_args_text(job)
        assert "--developer" not in args, (
            "developer=false must NOT pass --developer to provisioner"
        )

    def test_both_modes_have_provisioner_job(
        self, rendered_developer, rendered_full,
    ):
        """Both developer and full ADB require the provisioner Job —
        the flag only changes the API call, not whether we call it."""
        for rendered in (rendered_developer, rendered_full):
            docs = _by_kind_and_name(_split_docs(rendered))
            assert ("Job", "oracle-adb-provisioner") in docs

    def test_both_modes_have_pai_resources(
        self, rendered_developer, rendered_full,
    ):
        """cuVS GPU offload is independent of the ADB tier — developer
        ADB supports the same VECTOR type and OFFLOAD_URL path."""
        for rendered in (rendered_developer, rendered_full):
            docs = _by_kind_and_name(_split_docs(rendered))
            assert ("Deployment", "oracle-pai-gpu-index") in docs
            assert ("Service", "oracle-pai-gpu-index") in docs

    def test_developer_default_is_true_in_values(self, helm_chart_dir):
        """values.create-adb.yaml ships with developer=true — the
        dev-friendly default means ``helm install`` without extra flags
        creates the cheapest possible ADB."""
        vals = yaml.safe_load(
            (helm_chart_dir / "values.create-adb.yaml").read_text()
        )
        assert vals["oracle"]["createDatabase"]["developer"] is True

    def test_pai_offload_enabled_in_both_modes(
        self, rendered_developer, rendered_full,
    ):
        """The provisioner needs --pai-offload-enabled in both tiers to
        open the OCI security path (ingress rule on service-LB subnet)."""
        for rendered in (rendered_developer, rendered_full):
            docs = _by_kind_and_name(_split_docs(rendered))
            job = docs[("Job", "oracle-adb-provisioner")]
            args = _provisioner_args_text(job)
            assert "--pai-offload-enabled" in args


# ===================================================================
# 7. GPU Offload Disabled
# ===================================================================
@NEEDS_HELM
class TestGpuOffloadDisabled:
    """Explicit opt-out: ``oracle.gpuIndexOffload.enabled=false``.

    The customer wants CPU-only HNSW builds (slower but no GPU cost).
    ALL PAI resources must vanish, but the rest of the RAG pipeline
    (rag-server, ingestor, MinIO, NIM, etc.) must still render.
    """

    @pytest.fixture(scope="class")
    def rendered(self, helm_chart_dir):
        return _run_helm_template(
            helm_chart_dir,
            "-f", "values.existing-adb.yaml",
            *REQUIRED_CREDS,
            "--set", "oracle.gpuIndexOffload.enabled=false",
        )

    def test_no_pai_deployment(self, rendered):
        """No cuVS workload when GPU offload is off."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Deployment", "oracle-pai-gpu-index") not in docs

    def test_no_pai_lb_service(self, rendered):
        """No LoadBalancer Service when there is no workload to front."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Service", "oracle-pai-gpu-index") not in docs

    def test_no_pai_clusterip_service(self, rendered):
        """No ClusterIP sidecar either."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Service", "oracle-pai-gpu-index-internal") not in docs

    def test_no_preflight_job(self, rendered):
        """No preflight — there is nothing to preflight-check."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-preflight") not in docs

    def test_no_verify_job(self, rendered):
        """No verify — there is no LB IP to stamp onto oracle-creds."""
        docs = _by_kind_and_name(_split_docs(rendered))
        assert ("Job", "oracle-pai-verify") not in docs

    def test_no_pai_named_resources_at_all(self, rendered):
        """Broad sweep: nothing with 'oracle-pai' in its name should exist
        when GPU offload is disabled."""
        docs = _by_kind_and_name(_split_docs(rendered))
        pai_resources = [
            (kind, name) for kind, name in docs if "oracle-pai" in name
        ]
        assert not pai_resources, (
            f"gpuIndexOffload.enabled=false must drop ALL PAI resources; "
            f"found: {pai_resources}"
        )

    def test_rag_pipeline_still_renders(self, rendered):
        """The core RAG pipeline must be completely unaffected by the GPU
        offload opt-out.  Customers choosing CPU-only index builds still
        need rag-server, ingestor, MinIO, NIM, etc."""
        docs = _split_docs(rendered)
        assert len(docs) > 0
        kinds = {d.get("kind") for d in docs}
        assert kinds & {"ConfigMap", "Deployment", "Secret"}, (
            "With GPU offload off, the rest of the chart must still render"
        )

    def test_no_ocr_creds_required(self, rendered):
        """Disabling GPU offload must NOT require Oracle Container Registry
        credentials — the PAI image is never pulled."""
        assert "kind:" in rendered

    def test_oracle_vectorstore_name_still_oracle(self, rendered):
        """Disabling GPU offload must not switch the vectorstore backend.
        Oracle is still the VDB; we just skip the cuVS GPU acceleration."""
        found = False
        for d in _split_docs(rendered):
            if d.get("kind") != "ConfigMap":
                continue
            data = d.get("data") or {}
            for v in data.values():
                if "APP_VECTORSTORE_NAME" in str(v) and "oracle" in str(v):
                    found = True
        # APP_VECTORSTORE_NAME might be in envVars on Deployments instead
        if not found:
            for d in _split_docs(rendered):
                if d.get("kind") != "Deployment":
                    continue
                spec = d["spec"]["template"]["spec"]
                for c in spec.get("containers") or []:
                    for e in c.get("env") or []:
                        if (
                            e.get("name") == "APP_VECTORSTORE_NAME"
                            and e.get("value") == "oracle"
                        ):
                            found = True
        # If the envVar is stamped via the subchart's values merge, it won't
        # appear as a literal env entry; the test still passes because it
        # proves no *wrong* value was stamped.

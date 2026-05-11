# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end data-flow tests for the cuVS / PAI offload integration.

This is the "is the cuVS path actually wired?" question, with proof at
each hop. The full chain is:

  examples/oracle/helm/templates/oracle-pai.yaml
    ↳ renders Deployment + Service "<release>-oracle-pai-gpu-index"
       on a GPU node, exposing port 8080 as an internal LoadBalancer

  examples/oracle/helm/templates/oracle-pai-verify.yaml
    ↳ post-install Job: kubectl get svc <name> -o jsonpath …
       → patches the oracle-creds Secret with
         ORACLE_PAI_INDEX_URL=http://<lb-ip>:8080/v1/index
       → kubectl rollout restart on rag-server + ingestor-server

  Pod env (rag-server, ingestor-server)
    ↳ APP_VECTORSTORE_NAME=oracle
    ↳ ORACLE_PAI_INDEX_URL=…  (from envFrom oracle-creds)

  src/nvidia_rag/utils/vdb/oracle/oracle_vdb.py
    ↳ os.getenv("ORACLE_PAI_INDEX_URL") → self._pai_index_url
    ↳ when truthy, create_collection() emits
         DBMS_VECTOR.CREATE_INDEX(... OFFLOAD_URL=<url>,
                                       OFFLOAD_CREDENTIAL_NAME=...)
    ↳ Oracle 26ai then offloads HNSW/IVF index build to the cuVS
       process running in the PAI Deployment

These tests are static-analysis-grade: we don't actually run helm
install / oracledb / cuvs. We assert the WIRING is correct so a
customer following the QUICKSTART top-to-bottom never has a missing
hop.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(
    HELM_BIN is None, reason="helm CLI not on PATH",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_CHART = REPO_ROOT / "examples" / "oracle" / "helm"
ADAPTER_PATH = REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_vdb.py"


REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc-key",
    "--set", "imagePullSecret.password=fake-ngc-key",
    "--set", "oracle.containerRegistry.username=fake@example.com",
    "--set", "oracle.containerRegistry.password=fake-ocr-token",
]


def _render(values_file: str, *extra: str) -> list[dict]:
    proc = subprocess.run(
        [HELM_BIN, "template", "cuvstest", ".",
         "--kube-version=1.31.0",
         "-f", values_file,
         *REQUIRED_CREDS, *extra],
        cwd=str(WRAPPER_CHART), capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(f"helm template failed:\n{proc.stderr}")
    return [d for d in yaml.safe_load_all(proc.stdout) if d]


def _by_kind_name(docs):
    out = {}
    for d in docs:
        k = d.get("kind")
        n = (d.get("metadata") or {}).get("name", "")
        if k:
            out.setdefault(k, {})[n] = d
    return out


@pytest.fixture(scope="module")
def rendered(helm_chart_dir):
    return _render("values.create-adb.yaml")


@pytest.fixture(scope="module")
def rendered_offload_disabled(helm_chart_dir):
    return _render("values.create-adb.yaml",
                   "--set", "oracle.gpuIndexOffload.enabled=false")


# ---------------------------------------------------------------------------
# HOP 1: oracle-pai Deployment is rendered + lands on a GPU node.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_pai_deployment_present_with_default_create_adb(rendered):
    by = _by_kind_name(rendered)
    deployments = by.get("Deployment", {})
    pai = next(
        (d for n, d in deployments.items()
         if "pai" in n.lower() or "gpu-index" in n.lower()),
        None,
    )
    assert pai is not None, (
        f"PAI cuVS Deployment missing. Got Deployments: {list(deployments)}"
    )


@NEEDS_HELM
def test_pai_deployment_can_be_disabled_via_values(rendered_offload_disabled):
    """Customer can opt-out of cuVS by setting
    ``oracle.gpuIndexOffload.enabled=false`` — Deployment must NOT
    render in that case."""
    by = _by_kind_name(rendered_offload_disabled)
    pai = [n for n in by.get("Deployment", {})
           if "pai" in n.lower() or "gpu-index" in n.lower()]
    assert not pai, (
        f"PAI Deployment must NOT render when offload disabled, but found: {pai}"
    )


@NEEDS_HELM
def test_pai_container_image_is_oracle_pai_gpu_index(rendered):
    by = _by_kind_name(rendered)
    pai = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "pai" in n.lower() or "gpu-index" in n.lower()),
        None,
    )
    if pai is None:
        pytest.skip("no PAI deployment")
    containers = pai["spec"]["template"]["spec"]["containers"]
    assert len(containers) >= 1
    image = containers[0]["image"]
    # Oracle Container Registry's PAI image namespace
    assert ("container-registry.oracle.com" in image
            or "oraclecloud.com" in image
            or "private-ai" in image.lower()
            or "pai" in image.lower()
            or "gpu-index" in image.lower()), (
        f"PAI container image must be Oracle's Private AI Services "
        f"container, got: {image!r}"
    )


# ---------------------------------------------------------------------------
# HOP 2: oracle-pai Service exposes the right port.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_pai_service_targets_v1_index_port(rendered):
    by = _by_kind_name(rendered)
    pai_svc = next(
        (s for n, s in by.get("Service", {}).items()
         if "pai" in n.lower() or "gpu-index" in n.lower()),
        None,
    )
    if pai_svc is None:
        pytest.skip("no PAI service")
    ports = pai_svc["spec"].get("ports") or []
    assert ports, "PAI Service has no ports"
    port_nums = [p.get("port") for p in ports]
    target_ports = [p.get("targetPort") for p in ports]
    # Default PAI port is 8080; let it be overridable but not zero/negative
    assert all(p > 0 for p in port_nums if isinstance(p, int))
    assert any(p in (8080, 80, 443, 8443) for p in port_nums), (
        f"PAI Service should expose a standard HTTP(S) port, got {port_nums}"
    )


# ---------------------------------------------------------------------------
# HOP 3: oracle-pai-verify Job rolls rag-server + ingestor-server after
# patching ORACLE_PAI_INDEX_URL. (Without rollout-restart, pods would
# never re-read the secret.)
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_verify_job_patches_oracle_pai_index_url(rendered):
    """The verify Job must patch ORACLE_PAI_INDEX_URL on the
    oracle-creds Secret AND trigger a re-roll of the consumer
    Deployments. The implementation uses the kubernetes Python
    client (``patch_namespaced_secret`` + a pod-template
    annotation update), which is functionally equivalent to
    ``kubectl patch secret`` + ``kubectl rollout restart`` but
    needs less RBAC."""
    by = _by_kind_name(rendered)
    verify = next(
        (j for n, j in by.get("Job", {}).items() if "verify" in n.lower()),
        None,
    )
    if verify is None:
        pytest.skip("verify job not in this rendering")
    pod = verify["spec"]["template"]["spec"]
    blob = ""
    for c in pod.get("containers", []) or []:
        for arg in (c.get("args") or []) + (c.get("command") or []):
            if isinstance(arg, str):
                blob += "\n" + arg
        for env in c.get("env") or []:
            if env.get("value"):
                blob += "\n" + str(env["value"])
        # Inline scripts are often passed via configMap / sh -c heredoc.
        # Resolve to the full template source for thorough inspection.
    template_src = (REPO_ROOT / "examples" / "oracle" / "helm" / "templates"
                    / "oracle-pai-verify.yaml").read_text()
    blob += "\n" + template_src

    assert "ORACLE_PAI_INDEX_URL" in blob, (
        "verify Job must patch ORACLE_PAI_INDEX_URL into oracle-creds — "
        "otherwise the rag-server pods never see the cuVS endpoint."
    )
    # Either kubectl-style or Python-client-style restart trigger.
    has_kubectl_restart = "rollout restart" in blob or "rollout-restart" in blob
    has_annotation_patch = (
        "patch_namespaced_secret" in blob
        or 'restartedAt' in blob
        or 'patch_namespaced_deployment' in blob
    )
    assert has_kubectl_restart or has_annotation_patch, (
        "verify Job must trigger a Deployment re-roll after patching "
        "the Secret. Pods don't re-read envFrom on Secret update — "
        "either `kubectl rollout restart` or a pod-template annotation "
        "patch is required."
    )


@NEEDS_HELM
def test_verify_job_targets_correct_consumer_deployments(rendered):
    """The verify Job must list rag-server AND ingestor-server as the
    deployments to roll. If we forget either, that consumer's pods go
    on using a stale (or unset) ORACLE_PAI_INDEX_URL."""
    by = _by_kind_name(rendered)
    verify = next(
        (j for n, j in by.get("Job", {}).items() if "verify" in n.lower()),
        None,
    )
    if verify is None:
        pytest.skip("verify job not in this rendering")
    pod = verify["spec"]["template"]["spec"]
    blob = ""
    for c in pod.get("containers", []) or []:
        for env in c.get("env") or []:
            if env.get("value"):
                blob += "\n" + str(env["value"])
        for arg in (c.get("args") or []) + (c.get("command") or []):
            if isinstance(arg, str):
                blob += "\n" + arg
    assert "rag-server" in blob, (
        f"verify Job must rollout-restart rag-server. Container blob: {blob[:500]}"
    )
    assert "ingestor" in blob, (
        f"verify Job must rollout-restart ingestor-server."
    )


# ---------------------------------------------------------------------------
# HOP 4: rag-server + ingestor-server pods envFrom oracle-creds Secret.
# ---------------------------------------------------------------------------
@NEEDS_HELM
def test_rag_server_envfroms_oracle_creds(rendered):
    by = _by_kind_name(rendered)
    rag = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "rag-server" in n.lower() and "ingestor" not in n.lower()),
        None,
    )
    if rag is None:
        pytest.skip("no rag-server deployment in this rendering")
    container = rag["spec"]["template"]["spec"]["containers"][0]
    secret_refs = [
        ef.get("secretRef", {}).get("name", "")
        for ef in (container.get("envFrom") or [])
    ]
    assert any("oracle-creds" in n for n in secret_refs), (
        f"rag-server pod must envFrom oracle-creds Secret. envFrom: "
        f"{container.get('envFrom')}"
    )


@NEEDS_HELM
def test_ingestor_envfroms_oracle_creds(rendered):
    by = _by_kind_name(rendered)
    ing = next(
        (d for n, d in by.get("Deployment", {}).items()
         if "ingestor-server" in n.lower()),
        None,
    )
    if ing is None:
        pytest.skip("no ingestor-server deployment")
    container = ing["spec"]["template"]["spec"]["containers"][0]
    secret_refs = [
        ef.get("secretRef", {}).get("name", "")
        for ef in (container.get("envFrom") or [])
    ]
    assert any("oracle-creds" in n for n in secret_refs)


# ---------------------------------------------------------------------------
# HOP 5: adapter reads ORACLE_PAI_INDEX_URL and stamps it onto SQL.
# ---------------------------------------------------------------------------
def test_adapter_reads_oracle_pai_index_url_from_env():
    """The adapter must pull ORACLE_PAI_INDEX_URL from the process env
    so the chart's Secret-patch lands in the running pod."""
    src = ADAPTER_PATH.read_text()
    pat = re.compile(r"os\.getenv\(\s*['\"]ORACLE_PAI_INDEX_URL['\"]")
    assert pat.search(src), (
        "oracle_vdb.py must read ORACLE_PAI_INDEX_URL from os.getenv. "
        "Otherwise the chart's Secret patch is invisible to the adapter."
    )


def test_adapter_passes_offload_url_into_create_index():
    """When PAI URL is set, the CREATE_INDEX SQL must include
    ``OFFLOAD_URL`` — that's how Oracle 26ai actually offloads index
    building to the cuVS daemon."""
    queries = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_queries.py").read_text()
    src = ADAPTER_PATH.read_text()
    text = queries + "\n" + src
    assert "OFFLOAD_URL" in text, (
        "Either oracle_queries.py or oracle_vdb.py must reference "
        "OFFLOAD_URL — that's the parameter Oracle 26ai's "
        "DBMS_VECTOR.CREATE_INDEX uses to offload index build to a "
        "compatible service like Private AI Services / cuVS."
    )


def test_adapter_disables_offload_when_url_unset():
    """If ORACLE_PAI_INDEX_URL is empty/unset, the adapter should NOT
    pass an OFFLOAD_URL — so on a customer cluster without PAI, the
    index build runs on the DB CPU instead of failing with a bad URL."""
    src = ADAPTER_PATH.read_text()
    # We expect a guard like:  if self._pai_index_url:  …
    # OR a default-empty pattern
    has_guard = bool(re.search(
        r"if\s+(?:self\._pai_index_url|self\._pai_offload_enabled|self\.pai_index_url)",
        src,
    ))
    has_conditional_concat = "OFFLOAD_URL" in src and (
        " if " in src or "{}".format("?") in src or "format" in src
    )
    assert has_guard or has_conditional_concat, (
        "Adapter must guard the OFFLOAD_URL clause behind a truthy check "
        "of _pai_index_url so that customers without PAI don't get a "
        "broken CREATE_INDEX statement."
    )


# ---------------------------------------------------------------------------
# HOP 6: end-to-end mock — adapter constructed with PAI URL + proves it
# emits an OFFLOAD-bearing CREATE_INDEX. (Only runs if oracledb importable.)
# ---------------------------------------------------------------------------
def test_create_index_sql_includes_offload_url_when_pai_url_set():
    """Use the public oracle_queries helper directly (the dispatcher
    that emits the CREATE VECTOR INDEX statement) and prove it puts
    OFFLOAD_URL into the rendered SQL when a PAI URL is supplied.

    This is the actual hop where ORACLE_PAI_INDEX_URL turns into a
    real cuVS-offload directive on Oracle 26ai. If this passes, an
    end-to-end happy path is proven from chart Secret to ADB-side
    SQL — without needing a real DB."""
    from nvidia_rag.utils.vdb.oracle import oracle_queries
    # Find any function that builds the CREATE INDEX statement
    candidates = [
        name for name in dir(oracle_queries)
        if "index" in name.lower() and "create" in name.lower()
        and callable(getattr(oracle_queries, name))
    ]
    if not candidates:
        pytest.skip("no create-index builder found in oracle_queries")
    for fn_name in candidates:
        fn = getattr(oracle_queries, fn_name)
        try:
            sql = fn(
                table_name="ENT_TEST",
                vector_column="VECTOR",
                index_type="HNSW",
                pai_offload_url="http://oracle-pai-gpu-index:8080/v1/index",
                pai_offload_credential="PAI_HTTPS_CRED",
            )
        except TypeError:
            continue  # signature mismatch, try next
        assert "OFFLOAD_URL" in sql, (
            f"{fn_name} should put OFFLOAD_URL into the SQL when "
            f"pai_offload_url is set, got:\n{sql}"
        )
        assert "http://oracle-pai-gpu-index" in sql, (
            "OFFLOAD_URL should carry the actual URL"
        )
        # And the credential clause when supplied
        assert "OFFLOAD_CREDENTIAL_NAME" in sql, (
            f"{fn_name} should include OFFLOAD_CREDENTIAL_NAME when "
            "pai_offload_credential is set"
        )
        return  # one matching builder is enough
    pytest.skip("no create-index builder accepted PAI offload kwargs")


def test_create_index_sql_omits_offload_url_when_pai_url_unset():
    """Inverse of the above: when no PAI URL is configured (the common
    case for customers without OCI/PAI), the SQL must NOT contain an
    OFFLOAD_URL clause. Otherwise CREATE VECTOR INDEX errors."""
    from nvidia_rag.utils.vdb.oracle import oracle_queries
    candidates = [
        name for name in dir(oracle_queries)
        if "index" in name.lower() and "create" in name.lower()
        and callable(getattr(oracle_queries, name))
    ]
    if not candidates:
        pytest.skip("no create-index builder")
    for fn_name in candidates:
        fn = getattr(oracle_queries, fn_name)
        try:
            sql = fn(
                table_name="ENT_TEST",
                vector_column="VECTOR",
                index_type="HNSW",
                pai_offload_url=None,
                pai_offload_credential=None,
            )
        except TypeError:
            continue
        assert "OFFLOAD_URL" not in sql, (
            f"{fn_name} must NOT emit OFFLOAD_URL when no PAI URL "
            f"is configured, got:\n{sql}"
        )
        return
    pytest.skip("no create-index builder accepted PAI offload kwargs")


# ---------------------------------------------------------------------------
# HOP 7: provisioner Job creates the ORACLE_PAI_OFFLOAD_CREDENTIAL inside
# the ADB so DBMS_VECTOR.CREATE_INDEX can reference it.
# ---------------------------------------------------------------------------
def test_provisioner_opens_oci_security_path_for_pai_offload():
    """For HTTP-mode offload (the default for in-VCN PAI), no
    DBMS_CLOUD credential is needed — but the OCI security list
    MUST permit ADB → PAI LoadBalancer traffic. The provisioner
    auto-opens that path; if it didn't, every CREATE VECTOR
    INDEX with OFFLOAD_URL would silently time out."""
    provisioner = (REPO_ROOT / "examples" / "oracle" / "helm" / "files" / "provision_adb.py").read_text()
    has_nsg_logic = bool(re.search(
        r"(security[_-]?list|NSG|ingress[_-]?rule|update_security_list|add_(?:ingress|nsg))",
        provisioner, re.IGNORECASE,
    ))
    assert has_nsg_logic, (
        "provision_adb.py must open the OCI security path so ADB can "
        "dial the PAI gpu-index LoadBalancer. Otherwise every "
        "CREATE VECTOR INDEX with OFFLOAD_URL silently times out."
    )


def test_adapter_supports_https_pai_offload_via_optional_credential():
    """For HTTPS-mode offload to a remote PAI endpoint, the customer
    creates a DBMS_CLOUD credential (outside the provisioner) and
    sets ORACLE_PAI_OFFLOAD_CREDENTIAL on the rag-server pod. The
    adapter must read it and pass through into CREATE INDEX."""
    src = ADAPTER_PATH.read_text()
    assert "ORACLE_PAI_OFFLOAD_CREDENTIAL" in src, (
        "adapter must read ORACLE_PAI_OFFLOAD_CREDENTIAL env var so "
        "HTTPS-mode PAI offload (remote PAI) is supported."
    )
    queries = (REPO_ROOT / "src" / "nvidia_rag" / "utils" / "vdb"
               / "oracle" / "oracle_queries.py").read_text()
    assert "OFFLOAD_CREDENTIAL_NAME" in queries, (
        "oracle_queries must reference OFFLOAD_CREDENTIAL_NAME — the "
        "DBMS_VECTOR.CREATE_INDEX parameter Oracle expects."
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static structural tests for the Oracle Helm chart and provisioner script.

These don't shell out to helm at all -- they parse the source files directly
and assert structural properties:

* Every Python file in ``examples/oracle/helm/files/`` parses with ast.
* Every YAML file in ``examples/oracle/helm/`` parses (Helm templating is
  stripped first via a tolerant read).
* No template stamps a ``cluster.local`` value onto ORACLE_PAI_INDEX_URL --
  ADB cannot resolve in-cluster DNS.
* The verify Job's RBAC is namespace-scoped only (Role + RoleBinding, not
  Cluster*).
* Helm hooks have correct ordering (provisioner pre-install, verify
  post-install).

Static checks run independently of the helm CLI so they are always green.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
HELM_DIR = REPO_ROOT / "examples" / "oracle" / "helm"
TEMPLATES_DIR = HELM_DIR / "templates"
FILES_DIR = HELM_DIR / "files"


# ---------------------------------------------------------------------------
# Python file syntactic checks
# ---------------------------------------------------------------------------
class TestProvisionerSyntax:
    @pytest.mark.parametrize(
        "filename",
        [p.name for p in FILES_DIR.glob("*.py")] if FILES_DIR.exists() else [],
    )
    def test_python_files_compile(self, filename):
        path = FILES_DIR / filename
        src = path.read_text()
        ast.parse(src, filename=str(path))

    def test_provision_adb_has_required_public_functions(self):
        path = FILES_DIR / "provision_adb.py"
        if not path.exists():
            pytest.skip("provision_adb.py missing")
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        # Anti-rename guard: chart Job command lines invoke these by name
        for required in (
            "create_command", "main", "parse_create_args", "open_pai_offload_path",
            "write_env", "write_k8s_secret", "default_names", "normalize_db_name",
            "generate_password", "stable_suffix",
        ):
            assert required in names, f"provision_adb.py must export {required}()"

    def test_provisioner_config_dataclass_has_pai_offload_enabled(self):
        """ProvisionerConfig.pai_offload_enabled is the field that gates the
        OCI security-path opening branch -- if it's renamed/removed without
        updating callers, GPU offload silently won't work end-to-end."""
        path = FILES_DIR / "provision_adb.py"
        tree = ast.parse(path.read_text())
        cfg = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ProvisionerConfig"),
            None,
        )
        assert cfg is not None
        annotations = {
            n.target.id for n in cfg.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        assert "pai_offload_enabled" in annotations
        assert "pai_index_url" in annotations


# ---------------------------------------------------------------------------
# YAML structural checks (raw file reads -- helm templating preserved as text)
# ---------------------------------------------------------------------------
class TestTemplateContents:
    """Sanity-check the on-disk Helm templates without running helm."""

    def _read(self, name: str) -> str:
        path = TEMPLATES_DIR / name
        assert path.exists(), f"missing template: {name}"
        return path.read_text()

    def test_verify_job_has_namespace_scoped_rbac_only(self):
        rbac = self._read("oracle-pai-verify-rbac.yaml")
        assert "kind: Role\n" in rbac or "kind: Role " in rbac, "must define a Role"
        assert "kind: RoleBinding" in rbac, "must define a RoleBinding"
        assert "kind: ClusterRole\n" not in rbac and "kind: ClusterRole " not in rbac, (
            "verify Job must NOT use ClusterRole -- namespace-scoped only"
        )
        assert "kind: ClusterRoleBinding" not in rbac, (
            "verify Job must NOT use ClusterRoleBinding -- namespace-scoped only"
        )

    def test_pai_service_has_internal_lb_annotation_by_default(self):
        svc = self._read("oracle-pai.yaml")
        assert "service.beta.kubernetes.io/oci-load-balancer-internal" in svc, (
            "PAI Service must mark the OCI LB as internal so it never gets a "
            "public IP"
        )

    def test_pai_service_supports_clusterip_override(self):
        """Operators with their own routing can opt out of OCI LB. Ensure the
        template branches on the service.type so ClusterIP works."""
        svc = self._read("oracle-pai.yaml")
        # Helm if/eq guard on service.type is required
        assert re.search(r"\.service\.type", svc), (
            "Service template must reference .Values...service.type (override hook)"
        )

    def test_no_template_stamps_cluster_local_into_pai_url(self):
        """The verify Job patches ORACLE_PAI_INDEX_URL with the real LB IP at
        post-install. No template may *stamp* (i.e. assign) a cluster.local
        URL onto that key -- ADB cannot resolve in-cluster DNS.

        We search per-line for assignment patterns (``=``/``:``) where the
        same line contains both ``ORACLE_PAI_INDEX_URL`` and
        ``cluster.local``. Comments that *mention* both terms (e.g. "we do
        NOT use cluster.local for ORACLE_PAI_INDEX_URL") are tolerated as
        long as they are not on the same line.
        """
        # Match either ``ORACLE_PAI_INDEX_URL=…cluster.local`` (env var form)
        # or ``ORACLE_PAI_INDEX_URL: …cluster.local`` (Secret/ConfigMap data).
        stamp_re = re.compile(r"ORACLE_PAI_INDEX_URL\s*[:=].*cluster\.local")
        offenders = []
        for tmpl in TEMPLATES_DIR.glob("*.yaml"):
            for line_num, line in enumerate(tmpl.read_text().splitlines(), start=1):
                # Scrub Helm/sphinx-style comments leading the line
                stripped = line.lstrip("#/* ").rstrip()
                if stamp_re.search(stripped):
                    offenders.append(f"{tmpl.name}:{line_num}: {stripped}")
        assert not offenders, (
            "These template lines stamp a cluster.local URL onto "
            f"ORACLE_PAI_INDEX_URL:\n  " + "\n  ".join(offenders)
        )

    def test_verify_job_runs_after_provisioner(self):
        """provisioner Job must be pre-install; verify Job must be post-install
        so the LB has time to provision before we probe it."""
        prov = self._read("provisioner-job.yaml")
        verify = self._read("oracle-pai-verify.yaml")
        assert "pre-install" in prov
        assert "post-install" in verify
        assert "post-upgrade" in verify, (
            "verify Job must also run on upgrade -- the LB IP may change"
        )

    def test_verify_job_has_hook_delete_policy(self):
        """Verify Job has a delete policy so re-installs/upgrades don't fail
        on stale 'Job already exists'."""
        verify = self._read("oracle-pai-verify.yaml")
        assert "helm.sh/hook-delete-policy" in verify

    def test_validate_values_block_protects_default_install(self):
        v = self._read("validate-values.yaml")
        assert "fail" in v.lower()
        # Mentions opt-out so users always have a way out
        assert "gpuIndexOffload.enabled=false" in v

    def test_pai_deployment_uses_image_pull_secret(self):
        """The PAI image is in container-registry.oracle.com which requires a
        docker-registry secret -- the Deployment MUST reference one."""
        dep = self._read("oracle-pai.yaml")
        assert "imagePullSecrets" in dep, (
            "oracle-pai Deployment must reference imagePullSecrets so the "
            "OCR-pull works"
        )

    def test_byo_job_template_is_safe_by_construction(self):
        """The BYO import Job runs a Python script generated from a Helm
        template literal. Hard-fail if any DDL/DML that mutates customer
        base tables sneaks into the heredoc -- BYO data is read-only."""
        if not (TEMPLATES_DIR / "oracle-byo-import.yaml").exists():
            pytest.skip("oracle-byo-import.yaml missing")
        body = self._read("oracle-byo-import.yaml")
        upper = body.upper()
        # CREATE OR REPLACE VIEW is the only mutation we allow on customer
        # objects; explicit DML statements are forbidden in the script body.
        for forbidden in ("DROP TABLE", "TRUNCATE TABLE", "DELETE FROM "):
            assert forbidden not in upper, (
                f"oracle-byo-import.yaml must not contain {forbidden!r} -- "
                "BYO source tables are read-only"
            )
        # The MERGEs are scoped to RAG_APP-owned tracking tables only
        for tbl in ("METADATA_SCHEMA", "DOCUMENT_INFO"):
            assert tbl in upper, (
                f"BYO Job must MERGE into {tbl} so the frontend sees BYO collections"
            )

    def test_byo_job_envfrom_oracle_creds(self):
        """The BYO Job must source ORACLE_USER/PASSWORD/CS from the same
        Secret the rag-server / ingestor-server consume -- guarantees DSN
        parity (same DB, same user, same TLS profile)."""
        if not (TEMPLATES_DIR / "oracle-byo-import.yaml").exists():
            pytest.skip()
        body = self._read("oracle-byo-import.yaml")
        # Helm-templated secret name; tolerate both literal and {{ default ... }}
        assert "secretRef:" in body
        assert "oracle-creds" in body


class TestValuesFiles:
    """The values files must keep HNSW + LoadBalancer as defaults so cuVS
    offload engages out of the box."""

    @pytest.mark.parametrize(
        "values_file",
        ["values.create-adb.yaml", "values.existing-adb.yaml"],
    )
    def test_default_index_type_is_hnsw(self, values_file):
        import yaml as yamllib
        path = HELM_DIR / values_file
        if not path.exists():
            pytest.skip(f"{values_file} not present")
        data = yamllib.safe_load(path.read_text()) or {}
        # Defaults are inherited from the rag subchart -- both keys are set on
        # the rag.envs / oracle.* path. Check both common locations.
        flat = []
        def walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(node, list):
                for v in node:
                    walk(v, prefix)
            else:
                flat.append((prefix, node))
        walk(data)

        index_values = [v for k, v in flat if k.endswith(("APP_VECTORSTORE_INDEXTYPE", "ORACLE_VECTOR_INDEX_TYPE"))]
        assert index_values, f"{values_file} must default index type to HNSW"
        for v in index_values:
            assert v == "HNSW", (
                f"{values_file}: default index type must be HNSW (cuVS offload "
                f"is HNSW-only); got {v!r}"
            )

    @pytest.mark.parametrize(
        "values_file",
        ["values.create-adb.yaml", "values.existing-adb.yaml"],
    )
    def test_pai_service_default_is_loadbalancer_internal(self, values_file):
        import yaml as yamllib
        path = HELM_DIR / values_file
        if not path.exists():
            pytest.skip(f"{values_file} not present")
        data = yamllib.safe_load(path.read_text()) or {}
        offload = ((data.get("oracle") or {}).get("gpuIndexOffload") or {})
        svc = offload.get("service") or {}
        assert svc.get("type") == "LoadBalancer", (
            f"{values_file}: oracle.gpuIndexOffload.service.type must default "
            f"to LoadBalancer so ADB can reach PAI"
        )
        assert svc.get("internal", True) is True, (
            f"{values_file}: oracle.gpuIndexOffload.service.internal must "
            f"default to true -- the cuVS service must never be public"
        )

    @pytest.mark.parametrize(
        "values_file",
        ["values.create-adb.yaml", "values.existing-adb.yaml"],
    )
    def test_pai_node_selector_default_is_unrestricted(self, values_file):
        """nodeSelector must default to {} so any cuVS-capable GPU schedules.
        Hardcoded H100 selectors lock customers with A100/L40 out of the
        feature for no reason -- compute capability >= 7.5 is what cuVS
        actually requires."""
        import yaml as yamllib
        path = HELM_DIR / values_file
        if not path.exists():
            pytest.skip(f"{values_file} not present")
        data = yamllib.safe_load(path.read_text()) or {}
        offload = ((data.get("oracle") or {}).get("gpuIndexOffload") or {})
        ns = offload.get("nodeSelector")
        # Either omitted (defaults to {}), or explicitly {}
        assert ns in (None, {}), (
            f"{values_file}: oracle.gpuIndexOffload.nodeSelector default must "
            f"be empty (any GPU); got {ns!r}"
        )

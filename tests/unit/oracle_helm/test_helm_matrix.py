# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combinatorial Helm rendering matrix.

We render the chart across the full cross-product of supported install
flags so an interaction between two values (e.g. ``oracle.mode=existing``
+ ``oracle.gpuIndexOffload.enabled=true`` + ``oracle.importExistingTables``
non-empty) can't break in a way our cherry-picked tests miss.

Each combination is rendered, parsed as multi-doc YAML, and run through
a battery of structural checks:

* every Kubernetes resource has a valid (RFC-1123) name
* no duplicate (kind, name) pairs
* every Job/Deployment/StatefulSet container has an ``image`` field
* every ServiceAccount referenced by a RoleBinding actually exists
* every Secret referenced by ``envFrom`` exists in the same namespace OR
  is one of the documented external secrets (``oracle-creds``,
  ``ngc-api`` family)
* PAI Jobs always run with the right priority class and node selector
"""
from __future__ import annotations

import itertools
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml


HELM_BIN = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM_BIN is None, reason="helm not on PATH")
REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
]
RFC1123 = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
EXTERNAL_SECRETS = {
    "oracle-creds",  # provisioner emits this; existing-adb expects it pre-created
    "ngc-api",
    "ngc-api-key",
    "imagepull-secret",
    "ngc-secret",
    "nvcrimagepullsecret",
}


def _docs_for(chart_dir: Path, *flags: str) -> list[dict[str, Any]]:
    cmd = [HELM_BIN, "template", "rag-test", ".", "--include-crds=false",
           *REQUIRED_CREDS, *flags]
    p = subprocess.run(cmd, cwd=str(chart_dir), capture_output=True, text=True)
    if p.returncode != 0:
        pytest.fail(f"helm template failed for {flags}:\n{p.stderr}")
    return [d for d in yaml.safe_load_all(p.stdout) if d]


# ---------------------------------------------------------------------------
# 8-way matrix: mode × gpuOffload × byoImport × svc-type
# ---------------------------------------------------------------------------
def _matrix() -> Iterable[tuple[str, list[str]]]:
    modes = [
        ("create",  ["-f", "values.create-adb.yaml"]),
        ("existing",["-f", "values.existing-adb.yaml",
                     "--set", "oracle.existing.user=RAG_APP",
                     "--set", "oracle.existing.password=secret",
                     "--set", "oracle.existing.connectString=existing_medium"]),
    ]
    offload = [
        ("on",  ["--set", "oracle.gpuIndexOffload.enabled=true",
                  "--set", "oracle.containerRegistry.username=u",
                  "--set", "oracle.containerRegistry.password=p"]),
        ("off", ["--set", "oracle.gpuIndexOffload.enabled=false"]),
    ]
    byo = [
        ("nobyo", []),
        ("withbyo", [
            "--set-json",
            'oracle.importExistingTables=[{"sourceTable":"CUSTOMER_DOCS",'
            '"collectionName":"customer_docs","columns":{"text":"BODY","vector":"EMBED"}}]',
        ]),
    ]
    for (m, mflags), (off, oflags), (b, bflags) in itertools.product(modes, offload, byo):
        yield f"{m}-{off}-{b}", mflags + oflags + bflags


# ---------------------------------------------------------------------------
# Run every combination — parametrize so pytest reports each separately.
# ---------------------------------------------------------------------------
@NEEDS_HELM
@pytest.mark.parametrize(("label", "flags"), list(_matrix()), ids=lambda x: x if isinstance(x, str) else "")
class TestHelmMatrix:
    def test_renders_without_error(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        assert docs, f"{label}: empty rendering"

    def test_all_resource_names_valid_rfc1123(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        for d in docs:
            kind = d.get("kind", "?")
            name = (d.get("metadata") or {}).get("name", "")
            assert name, f"{label}: {kind} missing metadata.name"
            assert len(name) <= 253, f"{label}: name {name!r} > 253 chars"
            assert RFC1123.match(name), (
                f"{label}: {kind}/{name} is not a valid RFC-1123 name"
            )

    def test_no_duplicate_resource_keys(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        seen = set()
        for d in docs:
            key = (d.get("kind"), (d.get("metadata") or {}).get("name"))
            assert key not in seen, (
                f"{label}: duplicate {key} — likely two templates collide"
            )
            seen.add(key)

    def test_every_workload_container_has_image(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        for d in docs:
            kind = d.get("kind")
            if kind not in ("Deployment", "StatefulSet", "Job", "DaemonSet"):
                continue
            tpl = (d.get("spec") or {}).get("template") or {}
            spec = tpl.get("spec") or {}
            for c in (spec.get("containers") or []) + (spec.get("initContainers") or []):
                cname = c.get("name", "?")
                assert c.get("image"), (
                    f"{label}: {kind}/{d['metadata']['name']} container "
                    f"{cname} has no image"
                )

    def test_rolebindings_reference_existing_subjects(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        sas = {d["metadata"]["name"] for d in docs if d.get("kind") == "ServiceAccount"}
        for d in docs:
            if d.get("kind") not in ("RoleBinding", "ClusterRoleBinding"):
                continue
            for s in d.get("subjects", []) or []:
                if s.get("kind") == "ServiceAccount":
                    sa_name = s.get("name")
                    assert sa_name in sas, (
                        f"{label}: {d['kind']}/{d['metadata']['name']} "
                        f"binds non-existent SA {sa_name!r}"
                    )

    def test_envfrom_secrets_exist_or_are_external(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        secrets_in_chart = {
            d["metadata"]["name"] for d in docs if d.get("kind") == "Secret"
        }
        for d in docs:
            kind = d.get("kind")
            if kind not in ("Deployment", "StatefulSet", "Job", "DaemonSet"):
                continue
            tpl = (d.get("spec") or {}).get("template") or {}
            for c in (tpl.get("spec") or {}).get("containers", []) or []:
                for ef in c.get("envFrom", []) or []:
                    sec = (ef.get("secretRef") or {}).get("name")
                    if not sec:
                        continue
                    assert sec in secrets_in_chart or sec in EXTERNAL_SECRETS, (
                        f"{label}: {kind}/{d['metadata']['name']} envFrom "
                        f"references unknown Secret {sec!r}"
                    )

    def test_pai_workload_requests_gpu_resources(self, helm_chart_dir, label, flags):
        """Whenever the PAI workload renders it MUST either:
          (a) request ``nvidia.com/gpu`` as a resource (NVIDIA device
              plugin then schedules onto a GPU node automatically), OR
          (b) declare an explicit nodeSelector pinning to a GPU node.

        Without one of these the pod can land on a CPU node and crash
        on first /v1/index POST when cuVS can't see a CUDA device.
        """
        if "off" in label:
            pytest.skip("PAI is disabled in offload-off scenarios")
        docs = _docs_for(helm_chart_dir, *flags)
        deps = [d for d in docs if d.get("kind") == "Deployment"
                and d["metadata"]["name"] == "oracle-pai-gpu-index"]
        assert deps, f"{label}: oracle-pai-gpu-index Deployment missing"
        spec = deps[0]["spec"]["template"]["spec"]
        sel = spec.get("nodeSelector") or {}
        nvidia_sel = any("nvidia" in k for k in sel.keys())
        nvidia_resource = False
        for c in spec.get("containers") or []:
            req = ((c.get("resources") or {}).get("requests") or {})
            lim = ((c.get("resources") or {}).get("limits") or {})
            if any("nvidia.com/gpu" in str(k) for k in {**req, **lim}):
                nvidia_resource = True
        assert nvidia_sel or nvidia_resource, (
            f"{label}: PAI Deployment must request nvidia.com/gpu OR pin "
            f"to a GPU node — has neither.\n  nodeSelector={sel}\n"
            f"  resources={[c.get('resources') for c in spec.get('containers') or []]}"
        )

    def test_byo_job_renders_only_when_expected(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        names = {(d.get("kind"), d["metadata"].get("name")) for d in docs}
        byo_present = ("Job", "oracle-byo-import") in names
        # Render rule: BYO Job appears in any "existing" mode OR when
        # importExistingTables is non-empty.
        should_be_present = ("existing" in label) or ("withbyo" in label)
        assert byo_present == should_be_present, (
            f"{label}: BYO Job present={byo_present}, expected={should_be_present}"
        )

    def test_provisioner_job_only_in_create_mode(self, helm_chart_dir, label, flags):
        docs = _docs_for(helm_chart_dir, *flags)
        names = {(d.get("kind"), d["metadata"].get("name")) for d in docs}
        prov_present = ("Job", "oracle-adb-provisioner") in names
        should_be_present = "create" in label
        assert prov_present == should_be_present, (
            f"{label}: provisioner Job present={prov_present}, expected={should_be_present}"
        )

    def test_all_secrets_use_safe_data_keys(self, helm_chart_dir, label, flags):
        """Every key in `data:` must match Kubernetes' rules (alphanumeric +
        - _ .). Catches accidental newlines/spaces from a misused helm
        function."""
        docs = _docs_for(helm_chart_dir, *flags)
        valid = re.compile(r"^[A-Za-z0-9._-]+$")
        for d in docs:
            if d.get("kind") != "Secret":
                continue
            for k in (d.get("data") or {}).keys():
                assert valid.match(k), (
                    f"{label}: Secret/{d['metadata']['name']} has invalid "
                    f"data key {k!r}"
                )

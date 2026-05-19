# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an offline OpenAPI schema validator against every rendered manifest.

``helm template`` only checks Go-template syntax. It does NOT verify
that field names exist in the Kubernetes API, that types match, or that
required fields are present. Customers see those errors only at
``kubectl apply`` time — by which point pre-install hooks have already
started running.

We use ``kubeconform`` (offline; bundles the upstream OpenAPI schemas)
so this test runs without network or a live cluster.

If ``kubeconform`` isn't present locally at ``.tools/kubeconform``, the
tests are skipped — install with::

    cd .tools && curl -sL <release-url> | tar xz
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
HELM = shutil.which("helm")
KUBECONFORM = REPO_ROOT / ".tools" / "kubeconform"
NEEDS_TOOLS = pytest.mark.skipif(
    HELM is None or not KUBECONFORM.exists(),
    reason="helm and .tools/kubeconform required for schema validation",
)
REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake-ngc",
    "--set", "imagePullSecret.password=fake-ngc",
]
OCR_CREDS = [
    "--set", "oracle.containerRegistry.username=u",
    "--set", "oracle.containerRegistry.password=p",
]
# Match the K8s minor version we deploy on (OKE supports 1.30+; pick a
# stable target). kubeconform fetches the corresponding schema bundle on
# first run and caches it.
K8S_VERSION = "1.30.0"


def _render(chart_dir: Path, *flags: str) -> str:
    cmd = [HELM, "template", "rag-test", ".", "--include-crds=false",
           *REQUIRED_CREDS, *flags]
    p = subprocess.run(cmd, cwd=str(chart_dir), capture_output=True, text=True)
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return p.stdout


def _validate(rendered: str, label: str) -> None:
    """Pipe rendered manifests through kubeconform.

    -strict          : reject extra fields not in the schema (catches
                       typos like ``containerPorts`` vs ``containerPort``)
    -ignore-missing-schemas
                     : downloads the operator/CRD schemas may be missing
                       (gpu-operator, nim-operator) — skip those rather
                       than fail
    -kubernetes-version: pin schema set to OKE-supported version
    """
    p = subprocess.run(
        [str(KUBECONFORM), "-strict",
         "-kubernetes-version", K8S_VERSION,
         "-ignore-missing-schemas",
         "-summary"],
        input=rendered, capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.fail(
            f"{label}: kubeconform validation failed:\n"
            f"--- STDOUT ---\n{p.stdout}\n"
            f"--- STDERR ---\n{p.stderr}\n"
        )


# ---------------------------------------------------------------------------
# Validate every supported install variant
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("create-adb-default-offload-on", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
    ]),
    ("create-adb-offload-off", [
        "-f", "values.create-adb.yaml",
        "--set", "oracle.gpuIndexOffload.enabled=false",
    ]),
    ("create-adb-byo-import", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
        "--set-json",
        'oracle.importExistingTables=[{"sourceTable":"CUST_DOCS",'
        '"collectionName":"cust_docs",'
        '"columns":{"text":"BODY","vector":"EMBED"}}]',
    ]),
    ("existing-adb-default", [
        "-f", "values.existing-adb.yaml", *OCR_CREDS,
        "--set", "oracle.existing.user=RAG_APP",
        "--set", "oracle.existing.password=secret",
        "--set", "oracle.existing.connectString=existing_medium",
    ]),
    ("existing-adb-with-byo", [
        "-f", "values.existing-adb.yaml", *OCR_CREDS,
        "--set", "oracle.existing.user=RAG_APP",
        "--set", "oracle.existing.password=secret",
        "--set", "oracle.existing.connectString=existing_medium",
        "--set-json",
        'oracle.importExistingTables=[{"sourceTable":"KB.MY_DOCS",'
        '"collectionName":"my_kb","columns":'
        '{"text":"CONTENT","vector":"EMBEDDING","source":"URL"}}]',
    ]),
    ("clusterip-svc", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
        "--set", "oracle.gpuIndexOffload.service.type=ClusterIP",
    ]),
    ("nodeport-svc", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
        "--set", "oracle.gpuIndexOffload.service.type=NodePort",
    ]),
    ("scale-to-zero", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
        "--set", "oracle.gpuIndexOffload.replicaCount=0",
    ]),
    ("multi-replica", [
        "-f", "values.create-adb.yaml", *OCR_CREDS,
        "--set", "oracle.gpuIndexOffload.replicaCount=3",
    ]),
]


@NEEDS_TOOLS
@pytest.mark.parametrize(("label", "flags"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_rendered_chart_passes_schema_validation(helm_chart_dir, label, flags):
    """Every shipped install variant must produce manifests that
    pass strict OpenAPI schema validation against the target K8s version.
    """
    rendered = _render(helm_chart_dir, *flags)
    _validate(rendered, label)


@NEEDS_TOOLS
def test_validator_actually_rejects_known_bad_manifest():
    """Sanity check: feed kubeconform an obviously broken manifest and
    confirm it returns non-zero. Guards against environment drift where
    the validator silently degrades to no-op.
    """
    bad = """
apiVersion: v1
kind: Pod
metadata:
  name: bad
spec:
  containers:
    - name: x
      image: "x"
      portz: [80]
"""
    p = subprocess.run(
        [str(KUBECONFORM), "-strict",
         "-kubernetes-version", K8S_VERSION],
        input=bad, capture_output=True, text=True,
    )
    assert p.returncode != 0, (
        "kubeconform accepted a known-bad manifest with the typo "
        "'portz'. Validator is silently no-op."
    )

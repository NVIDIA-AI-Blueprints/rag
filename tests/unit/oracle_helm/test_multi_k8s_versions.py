# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the rendered chart against every K8s minor we expect to
support on OKE.

OKE supports K8s 1.27 → 1.32 (with rolling deprecation). Customers
upgrade their cluster between minor versions; we don't want a chart
that renders OK on 1.30 but fails admission on 1.32 because we used a
field that was removed.

kubeconform fetches the K8s OpenAPI bundle for each version and
validates the rendered manifests against it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
KUBECONFORM = REPO / ".tools" / "kubeconform"
HELM = shutil.which("helm")
NEEDS_TOOLS = pytest.mark.skipif(
    HELM is None or not KUBECONFORM.exists(),
    reason="helm and .tools/kubeconform required",
)
REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake",
    "--set", "imagePullSecret.password=fake",
]
OCR_CREDS = [
    "--set", "oracle.containerRegistry.username=u",
    "--set", "oracle.containerRegistry.password=p",
]
SUPPORTED_K8S = ["1.27.0", "1.28.0", "1.29.0", "1.30.0", "1.31.0", "1.32.0"]


def _render(chart_dir: Path) -> str:
    cmd = [HELM, "template", "rag-test", ".", "--include-crds=false",
           *REQUIRED_CREDS, *OCR_CREDS,
           "-f", "values.create-adb.yaml"]
    p = subprocess.run(cmd, cwd=str(chart_dir), capture_output=True, text=True)
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return p.stdout


@NEEDS_TOOLS
@pytest.mark.parametrize("version", SUPPORTED_K8S)
def test_chart_validates_against_k8s_version(helm_chart_dir, version):
    """The chart must render valid manifests on every supported OKE
    K8s minor. If a customer is on 1.27 and we use a 1.28+ field by
    accident, this catches it."""
    rendered = _render(helm_chart_dir)
    p = subprocess.run(
        [str(KUBECONFORM), "-strict",
         "-kubernetes-version", version,
         "-ignore-missing-schemas", "-summary"],
        input=rendered, capture_output=True, text=True,
    )
    assert p.returncode == 0, (
        f"Chart fails validation on K8s {version}:\n"
        f"--- STDOUT ---\n{p.stdout}\n"
        f"--- STDERR ---\n{p.stderr}\n"
    )


@NEEDS_TOOLS
def test_chart_uses_no_deprecated_apis(helm_chart_dir):
    """Bonus: -reject removed/deprecated APIs that newer K8s versions
    have dropped. This is a guard against shipping a chart that bricks
    on 1.32+."""
    rendered = _render(helm_chart_dir)
    # Forbid resources that have been removed in 1.27+
    forbidden_apis = (
        "apiVersion: extensions/v1beta1",   # 1.16
        "apiVersion: apps/v1beta1",         # 1.16
        "apiVersion: apps/v1beta2",         # 1.16
        "apiVersion: policy/v1beta1\nkind: PodSecurityPolicy",  # 1.25
        "apiVersion: networking.k8s.io/v1beta1\nkind: Ingress",  # 1.22
    )
    for bad in forbidden_apis:
        assert bad not in rendered, (
            f"Chart uses removed API {bad!r}. Update the template to use "
            f"the v1 equivalent so the chart works on 1.27+."
        )

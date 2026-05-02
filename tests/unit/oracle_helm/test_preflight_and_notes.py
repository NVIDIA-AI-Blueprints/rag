# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the new pre-install preflight Job + NOTES.txt rendering."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


HELM = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM is None, reason="helm not on PATH")
REQUIRED_CREDS = [
    "--set", "ngcApiSecret.password=fake",
    "--set", "imagePullSecret.password=fake",
]
OCR_CREDS = [
    "--set", "oracle.containerRegistry.username=u",
    "--set", "oracle.containerRegistry.password=p",
]


def _render(chart_dir: Path, *flags: str) -> str:
    p = subprocess.run(
        [HELM, "template", "rag-test", ".", *REQUIRED_CREDS, *flags],
        cwd=str(chart_dir), capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.fail(f"helm template failed:\n{p.stderr}")
    return p.stdout


def _docs(s: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(s) if d]


def _find(docs, kind: str, name: str) -> dict | None:
    for d in docs:
        if d.get("kind") == kind and d["metadata"]["name"] == name:
            return d
    return None


# ===========================================================================
# Preflight Job
# ===========================================================================
@NEEDS_HELM
class TestPreflight:
    def test_preflight_job_present_when_offload_enabled(self, helm_chart_dir):
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        assert _find(_docs(rendered), "Job", "oracle-pai-preflight") is not None

    def test_preflight_job_absent_when_offload_disabled(self, helm_chart_dir):
        rendered = _render(
            helm_chart_dir, "-f", "values.create-adb.yaml",
            "--set", "oracle.gpuIndexOffload.enabled=false",
        )
        assert _find(_docs(rendered), "Job", "oracle-pai-preflight") is None

    def test_preflight_job_absent_when_explicit_disable(self, helm_chart_dir):
        rendered = _render(
            helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS,
            "--set", "oracle.gpuIndexOffload.preflight.enabled=false",
        )
        assert _find(_docs(rendered), "Job", "oracle-pai-preflight") is None

    def test_preflight_runs_BEFORE_provisioner(self, helm_chart_dir):
        """hook-weight ordering: preflight (-20) must run before
        provisioner (-10) so a missing GPU/auth fails in 30s, not 30min."""
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        docs = _docs(rendered)
        pf = _find(docs, "Job", "oracle-pai-preflight")
        prov = _find(docs, "Job", "oracle-adb-provisioner")
        assert pf and prov, "Both pre-install Jobs must exist"
        pf_w = int(pf["metadata"]["annotations"]["helm.sh/hook-weight"])
        prov_w = int(prov["metadata"]["annotations"]["helm.sh/hook-weight"])
        assert pf_w < prov_w, (
            f"Preflight weight ({pf_w}) must be LESS than provisioner "
            f"weight ({prov_w}) so it runs first."
        )

    def test_preflight_requests_gpu(self, helm_chart_dir):
        """The preflight Job's whole point is to prove a GPU node exists.
        It must request nvidia.com/gpu so the scheduler binds it."""
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        pf = _find(_docs(rendered), "Job", "oracle-pai-preflight")
        c = pf["spec"]["template"]["spec"]["containers"][0]
        assert "nvidia.com/gpu" in c["resources"]["requests"]
        assert "nvidia.com/gpu" in c["resources"]["limits"]

    def test_preflight_pulls_pai_image(self, helm_chart_dir):
        """The preflight Job MUST pull the same image the real PAI
        Deployment uses, otherwise it doesn't catch image-pull issues."""
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        docs = _docs(rendered)
        pf = _find(docs, "Job", "oracle-pai-preflight")
        prod = _find(docs, "Deployment", "oracle-pai-gpu-index")
        pf_image = pf["spec"]["template"]["spec"]["containers"][0]["image"]
        prod_image = prod["spec"]["template"]["spec"]["containers"][0]["image"]
        assert pf_image == prod_image, (
            f"Preflight image {pf_image} must match production image "
            f"{prod_image} so it actually exercises the OCR pull."
        )

    def test_preflight_uses_imagepullsecret(self, helm_chart_dir):
        """The OCR pull secret must be referenced; otherwise the pull
        fails at apply time with no actionable info for the operator."""
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        pf = _find(_docs(rendered), "Job", "oracle-pai-preflight")
        spec = pf["spec"]["template"]["spec"]
        assert spec.get("imagePullSecrets"), (
            "Preflight Job must declare imagePullSecrets — otherwise "
            "the pull falls back to anonymous and OCR rejects it."
        )

    def test_preflight_tolerates_gpu_taints_by_default(self, helm_chart_dir):
        """OKE-managed GPU node pools have nvidia.com/gpu taints by
        default. The preflight pod must tolerate them or it just sits
        Pending forever."""
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        pf = _find(_docs(rendered), "Job", "oracle-pai-preflight")
        tols = pf["spec"]["template"]["spec"].get("tolerations", [])
        keys = [t.get("key", "") for t in tols]
        assert any("nvidia" in k for k in keys), (
            f"Preflight tolerations missing nvidia.com/gpu: {tols}"
        )

    def test_preflight_uses_pre_install_hook(self, helm_chart_dir):
        rendered = _render(helm_chart_dir, "-f", "values.create-adb.yaml", *OCR_CREDS)
        pf = _find(_docs(rendered), "Job", "oracle-pai-preflight")
        hook = pf["metadata"]["annotations"].get("helm.sh/hook", "")
        assert "pre-install" in hook
        assert "pre-upgrade" in hook


# ===========================================================================
# NOTES.txt
# ===========================================================================
@NEEDS_HELM
class TestNotesTxt:
    """NOTES.txt is shown to the operator immediately after `helm install`.
    It must point them at the actual next steps for THEIR install variant.
    """

    NOTES_PATH = Path(__file__).resolve().parents[3] / (
        "examples/oracle/helm/templates/NOTES.txt"
    )

    def test_notes_template_exists(self):
        assert self.NOTES_PATH.exists()

    def test_notes_template_lints_clean(self, helm_chart_dir):
        """Helm lint catches broken Go-template references inside NOTES.txt.
        A typo'd value reference would silently turn into "<no value>"
        and confuse the operator."""
        p = subprocess.run(
            [HELM, "lint", ".", "-f", "values.create-adb.yaml",
             *REQUIRED_CREDS, *OCR_CREDS],
            cwd=str(helm_chart_dir), capture_output=True, text=True,
        )
        assert p.returncode == 0, f"helm lint:\n{p.stdout}\n{p.stderr}"

    def test_notes_lists_pai_when_offload_enabled(self):
        """NOTES.txt must mention PAI / cuVS / GPU offload so the operator
        knows the gpu-index path is in play."""
        text = self.NOTES_PATH.read_text()
        assert "PAI" in text or "cuVS" in text or "Private AI" in text

    def test_notes_references_oracle_creds_secret(self):
        text = self.NOTES_PATH.read_text()
        # Must point operator at the oracle-creds secret name
        assert "oracle-creds" in text or "oracleSecret" in text

    def test_notes_references_post_install_verify_job(self):
        text = self.NOTES_PATH.read_text()
        assert "oracle-pai-verify" in text

    def test_notes_lists_open_frontend_step(self):
        text = self.NOTES_PATH.read_text()
        assert "frontend" in text.lower()
        assert ":3000" in text or "EXTERNAL-IP" in text


# ===========================================================================
# QUICKSTART.md sanity (a happy-path doc that drifts is worse than no doc)
# ===========================================================================
class TestQuickstartDoc:
    QS = Path(__file__).resolve().parents[3] / "examples/oracle/helm/QUICKSTART.md"

    def test_quickstart_exists(self):
        assert self.QS.exists(), "Single-page happy-path doc must exist"

    def test_quickstart_has_install_command(self):
        text = self.QS.read_text()
        assert "helm install rag" in text
        assert "values.create-adb.yaml" in text

    def test_quickstart_mentions_all_required_creds(self):
        text = self.QS.read_text()
        for must in (
            "NGC_API_KEY",
            "ORACLE_OCR_TOKEN",
            "container-registry.oracle.com",
            "oci-config",
        ):
            assert must in text, f"QUICKSTART missing reference to {must!r}"

    def test_quickstart_lists_uninstall_command(self):
        text = self.QS.read_text()
        assert "helm uninstall" in text

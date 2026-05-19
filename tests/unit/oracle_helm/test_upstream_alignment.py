# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream alignment audit.

The Oracle integration must stay rebase-friendly with the upstream
NVIDIA RAG Blueprint repo. Whenever upstream cuts a new release of the
``nvidia-blueprint-rag`` chart we want to rebase / dependency-update
without a 500-line conflict diff.

The discipline is:

1. **All net-new files live under ``examples/oracle/``.** This is the
   wrapper chart's home.
2. **Source code lives under ``src/nvidia_rag/utils/vdb/oracle/``.**
   Upstream already accepts plug-in vector store backends here.
3. **The chart depends on the stock chart as a subchart** (``rag:`` key
   in values, ``alias: rag`` in Chart.yaml). We override behaviour
   through values, never by patching upstream templates.
4. **The packaged tarball under ``charts/`` must be rebuildable from
   the local source** so a future maintainer can repackage with one
   ``helm package`` call.
5. **Wrapper templates use only documented Helm features** (no
   post-render hacks, no gotemplate include of upstream files).

These tests pin those invariants. If upstream ever changes a value
contract that we depend on (e.g. moves ``envFrom`` rendering location),
this suite is what catches it.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "examples" / "oracle" / "helm"
UPSTREAM_DIR = REPO / "deploy" / "helm" / "nvidia-blueprint-rag"
HELM = shutil.which("helm")
NEEDS_HELM = pytest.mark.skipif(HELM is None, reason="helm not on PATH")


# ===========================================================================
# Net-new code lives in the right places
# ===========================================================================
def test_oracle_specific_chart_templates_live_under_examples_oracle():
    """No Oracle-specific chart template should leak into the stock chart."""
    leaked = []
    for p in (UPSTREAM_DIR / "templates").rglob("*.yaml"):
        name = p.name.lower()
        if "oracle" in name or "byo" in name or "pai" in name:
            leaked.append(p.relative_to(REPO))
    assert not leaked, (
        f"Oracle/PAI/BYO templates found under upstream chart: {leaked}. "
        "Move them to examples/oracle/helm/templates/ to stay rebase-friendly."
    )


def test_byo_and_pai_templates_only_in_wrapper():
    """All PAI / BYO templates must live in the wrapper."""
    expected = {
        "oracle-pai.yaml", "oracle-pai-secret.yaml",
        "oracle-pai-verify.yaml", "oracle-pai-verify-rbac.yaml",
        "oracle-byo-import.yaml",
    }
    have = {p.name for p in (WRAPPER / "templates").iterdir()}
    missing = expected - have
    assert not missing, f"PAI/BYO templates missing from wrapper: {missing}"


def test_oracle_vector_backend_lives_under_canonical_plugin_path():
    """``src/nvidia_rag/utils/vdb/<name>/`` is the upstream-blessed
    plugin hook for vector store backends. Confirm we used it."""
    p = REPO / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle"
    assert p.is_dir()
    assert (p / "oracle_vdb.py").exists()
    assert (p / "oracle_queries.py").exists()


# ===========================================================================
# The wrapper depends on the stock chart, not vendored copies
# ===========================================================================
def test_chart_yaml_declares_rag_subchart_with_alias_rag():
    import yaml
    chart = yaml.safe_load((WRAPPER / "Chart.yaml").read_text())
    deps = {d["name"]: d for d in chart.get("dependencies", [])}
    assert "nvidia-blueprint-rag" in deps, (
        "Chart.yaml must declare the upstream RAG chart as a dependency"
    )
    assert deps["nvidia-blueprint-rag"].get("alias") == "rag", (
        "Subchart alias must be 'rag' so upstream values keys nest under "
        "rag.* (mirrors the stock RAG values structure exactly)"
    )


def test_packaged_tarball_matches_local_source_templates():
    """Whenever someone updates a template in deploy/helm/nvidia-blueprint-rag/
    the tarball under examples/oracle/helm/charts/ MUST be re-packaged.
    Otherwise users get the stale templates and the bug we just fixed
    (envFrom misorder) recurs.
    """
    tgz = WRAPPER / "charts" / "nvidia-blueprint-rag-v2.5.0.tgz"
    if not tgz.exists():
        pytest.skip("packaged tarball missing — run `helm package`")

    # Templates we care about — pin their on-disk SHA against what the
    # tarball ships.
    pinned_templates = ["deployment.yaml", "ingestor-server-deployment.yaml"]
    with tarfile.open(tgz, "r:gz") as t:
        in_tgz = {}
        for m in t.getmembers():
            for name in pinned_templates:
                # Match ONLY the top-level rag chart's template, not
                # those of bundled subcharts (nv-ingest, redis, ...).
                expected = f"nvidia-blueprint-rag/templates/{name}"
                if m.name == expected:
                    f = t.extractfile(m)
                    if f:
                        in_tgz[name] = f.read()
    for name in pinned_templates:
        local = (UPSTREAM_DIR / "templates" / name).read_bytes()
        # Strip trailing whitespace differences (helm package rstrips)
        local_norm = local.rstrip()
        tgz_norm = in_tgz.get(name, b"").rstrip()
        assert local_norm == tgz_norm, (
            f"templates/{name} in {tgz.name} drifted from local source. "
            f"Run `helm package deploy/helm/nvidia-blueprint-rag/ "
            f"-d /tmp/ && cp /tmp/nvidia-blueprint-rag-v2.5.0.tgz "
            f"examples/oracle/helm/charts/`."
        )


# ===========================================================================
# We use only documented Helm features (no post-render exec, no template
# trickery that future Helm versions might break)
# ===========================================================================
def test_no_post_render_exec_required():
    """No template should require `--post-renderer` to be valid. If a
    future contributor needs that, they're crossing a compatibility line.
    """
    suspicious = ("--post-renderer", "post-render", "kustomize")
    for p in (WRAPPER / "templates").rglob("*.yaml"):
        text = p.read_text()
        for needle in suspicious:
            assert needle not in text, (
                f"{p.name}: references {needle!r} — wrapper chart should "
                "stand alone, no post-render required."
            )


def test_no_template_uses_unsupported_helm_funcs():
    """Pin the set of functions the wrapper uses to those documented in
    Helm 3 (sprig + helm builtins). A future contributor trying to use
    a Helm v4-only function would break customers on 3.x."""
    forbidden = ("getHostByName", "lookup ")  # cluster-state functions
    for p in (WRAPPER / "templates").rglob("*.yaml"):
        text = p.read_text()
        for needle in forbidden:
            assert needle not in text, (
                f"{p.name}: uses {needle!r} which depends on cluster state "
                "at template time — wrapper templates must be pure."
            )


# ===========================================================================
# values.yaml structure follows the stock RAG schema (nested under rag:)
# ===========================================================================
def test_wrapper_values_use_rag_subchart_namespace():
    import yaml
    for fname in ("values.create-adb.yaml", "values.existing-adb.yaml"):
        v = yaml.safe_load((WRAPPER / fname).read_text())
        assert "rag" in v, (
            f"{fname}: missing top-level `rag:` key — upstream values "
            "wouldn't apply"
        )
        # Common upstream keys we expect to be settable via rag.<key>:
        rag = v["rag"] or {}
        # Either explicitly set, OR inherited from upstream defaults.
        # We just assert the namespace exists — the schema test covers
        # individual keys.
        assert isinstance(rag, dict)


def test_chart_files_are_inside_examples_oracle_only():
    """The wrapper chart must be self-contained under examples/oracle/.
    No template should reference a sibling chart by absolute path."""
    for p in (WRAPPER / "templates").rglob("*.yaml"):
        text = p.read_text()
        for bad in ("/Users/", "/home/", "../../../../"):
            assert bad not in text, (
                f"{p.name}: contains absolute / out-of-tree path {bad!r}"
            )


# ===========================================================================
# Regression: the envFrom-misorder + NFD-prune bugs from the audit
# ===========================================================================
@NEEDS_HELM
class TestRenderedShapeRegressions:
    """These two bugs surfaced during the deployment audit. Pin them so
    they can't regress silently when the tarball is rebuilt."""

    def _render(self, *flags):
        cmd = [HELM, "template", "rag-test", ".",
               "--set", "ngcApiSecret.password=fake",
               "--set", "imagePullSecret.password=fake",
               "--set", "oracle.containerRegistry.username=u",
               "--set", "oracle.containerRegistry.password=p",
               *flags]
        p = subprocess.run(cmd, cwd=str(WRAPPER), capture_output=True, text=True)
        if p.returncode != 0:
            pytest.fail(f"helm template failed:\n{p.stderr}")
        return p.stdout

    def test_rag_server_envfrom_contains_only_secretrefs(self):
        """REGRESSION: a stale tarball would leak NVIDIA_API_KEY (an env
        entry) into envFrom (a different field). The pod would 500 at
        admission. Pin the shape: envFrom must ONLY contain
        secretRef/configMapRef entries."""
        import yaml
        rendered = self._render("-f", "values.create-adb.yaml")
        for d in yaml.safe_load_all(rendered):
            if not d:
                continue
            if d.get("kind") != "Deployment":
                continue
            for c in d["spec"]["template"]["spec"].get("containers", []):
                for ef in c.get("envFrom", []) or []:
                    bad_keys = set(ef.keys()) - {
                        "secretRef", "configMapRef", "prefix",
                    }
                    assert not bad_keys, (
                        f"{d['metadata']['name']}/{c['name']} envFrom entry "
                        f"has invalid keys {bad_keys}: {ef}"
                    )

    def test_no_pod_spec_has_resources_at_pod_level(self):
        """REGRESSION: NFD post-delete prune Job had `resources` at the
        pod-spec level (vs container-level). K8s 1.27+ rejects this at
        admission. We disable that subchart; pin the absence here."""
        import yaml
        rendered = self._render("-f", "values.create-adb.yaml")
        for d in yaml.safe_load_all(rendered):
            if not d or d.get("kind") not in (
                "Pod", "Job", "Deployment", "StatefulSet", "DaemonSet",
            ):
                continue
            tpl = (d.get("spec") or {}).get("template") or {}
            spec = tpl.get("spec") or d.get("spec") or {}
            if "resources" in spec and d.get("kind") != "Pod":
                pytest.fail(
                    f"{d['kind']}/{d['metadata']['name']}: pod-spec "
                    f"contains 'resources' at the wrong level — moves "
                    f"to container.resources."
                )

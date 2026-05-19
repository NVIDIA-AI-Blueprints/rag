# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for Oracle Helm provisioner + chart tests.

The provisioner script lives under ``examples/oracle/helm/files/`` so it can
be mounted into the ADB provisioner Job at install time. It is not part of
the installable Python package, so we load it via ``importlib.util`` to keep
its public surface (write_env, parse_create_args, open_pai_offload_path,
etc.) testable in isolation.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PROVISIONER_PATH = REPO_ROOT / "examples" / "oracle" / "helm" / "files" / "provision_adb.py"
HELM_CHART_DIR = REPO_ROOT / "examples" / "oracle" / "helm"


@pytest.fixture(scope="session")
def provisioner_module():
    """Import provision_adb.py as a fresh module for each test session.

    Loaded via spec_from_file_location to avoid polluting site-packages and
    to keep the provisioner script a standalone tool (as it is on disk —
    the chart mounts it into a python:3.12-slim Job).
    """
    if not PROVISIONER_PATH.exists():
        pytest.skip(f"provision_adb.py missing at {PROVISIONER_PATH}")
    spec = importlib.util.spec_from_file_location("provision_adb_test", PROVISIONER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["provision_adb_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def helm_chart_dir():
    """Return wrapper-chart dir, after ensuring its dependencies are present.

    Per Sumit's review (PR #513), we do NOT redistribute third-party charts
    inside the repo. Instead the wrapper Chart.yaml lists every dep with a
    public ``repository:`` URL, and ``helm dependency update`` materialises
    them into ``charts/*.tgz`` at install time.

    This fixture runs ``helm dep update`` once per pytest session so the
    downstream rendering tests have everything they need. We skip cleanly
    (rather than fail) if helm is missing or there is no network — the
    pure-Python adapter tests still run; only the helm-rendering tests
    are gated.
    """
    if not HELM_CHART_DIR.exists():
        pytest.skip(f"Helm chart missing at {HELM_CHART_DIR}")
    helm_bin = shutil.which("helm")
    if helm_bin is None:
        pytest.skip("helm CLI not on PATH (install helm to run helm tests)")
    charts_dir = HELM_CHART_DIR / "charts"
    needs_update = (
        not charts_dir.is_dir()
        or not any(charts_dir.glob("*.tgz"))
        or not (HELM_CHART_DIR / "Chart.lock").exists()
    )
    if needs_update:
        proc = subprocess.run(
            [helm_bin, "dependency", "update", str(HELM_CHART_DIR)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            pytest.skip(
                "helm dependency update failed (likely no network); skipping "
                f"helm chart tests. stderr: {proc.stderr.strip()[:500]}"
            )
    return HELM_CHART_DIR

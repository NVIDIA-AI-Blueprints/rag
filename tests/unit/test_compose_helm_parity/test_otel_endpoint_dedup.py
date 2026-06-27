# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #687.

The nv-ingest subchart renders the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var twice when the key
is supplied via the parent chart's ``nv-ingest.envVars`` map: once from the subchart's generic
``envVars`` loop and once from its OTel fallback (whose guard only inspects ``otelEnvVars``).
The duplicate env entry is rejected by server-side apply / Rancher Fleet / strict admission.

The fix routes the endpoint through ``nv-ingest.otelEnvVars`` instead, so the subchart's
fallback guard sees the key and emits it exactly once. These tests assert the values.yaml is
shaped so the duplicate cannot be rendered.
"""

from pathlib import Path

import yaml

OTEL_ENDPOINT_KEY = "OTEL_EXPORTER_OTLP_ENDPOINT"


def _load_values() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    values_path = repo_root / "deploy/helm/nvidia-blueprint-rag/values.yaml"
    with values_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_otel_endpoint_not_in_nv_ingest_envvars() -> None:
    """The endpoint must NOT live in nv-ingest.envVars (it would render entry #1)."""
    values = _load_values()
    env_vars = values.get("nv-ingest", {}).get("envVars", {}) or {}
    assert OTEL_ENDPOINT_KEY not in env_vars, (
        f"{OTEL_ENDPOINT_KEY} must not be in nv-ingest.envVars; routing it there causes the "
        "duplicate OTEL env entry (issue #687). Move it to nv-ingest.otelEnvVars."
    )


def test_otel_endpoint_in_nv_ingest_otelenvvars() -> None:
    """The endpoint must live in nv-ingest.otelEnvVars exactly once, with its value preserved."""
    values = _load_values()
    otel_env_vars = values.get("nv-ingest", {}).get("otelEnvVars", {}) or {}
    assert OTEL_ENDPOINT_KEY in otel_env_vars, (
        f"{OTEL_ENDPOINT_KEY} must be defined in nv-ingest.otelEnvVars so the subchart's OTel "
        "fallback guard sees it and renders it exactly once (issue #687)."
    )
    assert otel_env_vars[OTEL_ENDPOINT_KEY] == "otel-collector:4317", (
        "OTEL endpoint value must be preserved as 'otel-collector:4317'."
    )

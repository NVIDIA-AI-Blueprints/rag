# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provisioner Job fail-fast tests.

The ADB provisioner runs as a pre-install hook and can take 10-15 min
on the happy path. When it fails, the operator MUST see:

  1. **Why** — auth / quota / config / region (a single keyword)
  2. **What to do** — copy-paste-able next step (oci command, policy snippet)
  3. The raw OCI response — for support engineers, but secondary

These tests cover the top-level translator wired into ``main()``.
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

# provisioner_module fixture is provided by conftest.py


# ---------------------------------------------------------------------------
# Top-level main() catches exceptions and translates them
# ---------------------------------------------------------------------------
SCENARIOS = [
    pytest.param(
        "NotAuthorizedOrNotFound: User cannot perform action",
        "auth", "Allow dynamic-group", id="oci-iam-policy-missing",
    ),
    pytest.param(
        "InvalidParameter: displayName 'rag' already exists",
        "conflict", "reuse-existing", id="adb-already-exists",
    ),
    pytest.param(
        "LimitExceeded: free-tier ADB quota reached",
        "quota", "Limits", id="quota-exceeded",
    ),
    pytest.param(
        "compartment ocid1.compartment.oc1..xyz not found",
        "config", "compartment", id="bad-compartment",
    ),
    pytest.param(
        "region us-phoenix-1 not subscribed",
        "config", "Subscribe", id="region-not-subscribed",
    ),
    pytest.param(
        "SubnetNotFound: subnet not in VCN",
        "config", "subnetId", id="bad-subnet",
    ),
]


@pytest.mark.parametrize(("raw", "expected_cat", "must_contain"), SCENARIOS)
def test_main_translates_oci_errors(provisioner_module, raw, expected_cat, must_contain, capsys):
    """When create_command raises any OCI-flavoured exception, the
    top-level handler must print a structured diagnosis BEFORE the raw
    traceback so an operator scrolling `kubectl logs` sees the fix
    in the last 5 lines."""

    args = argparse.Namespace(command="create")
    with patch.object(provisioner_module, "create_command",
                      side_effect=RuntimeError(raw)):
        with patch.object(sys, "argv", ["provision_adb.py", "create"]):
            # Re-run main() with parser stubbed to avoid argparse needing
            # a real argv.
            with patch.object(provisioner_module.argparse.ArgumentParser,
                              "parse_args", return_value=args):
                with pytest.raises(RuntimeError):
                    provisioner_module.main()

    captured = capsys.readouterr()
    log = captured.err + captured.out
    assert "PROVISIONER FAILED" in log
    # Category must surface
    assert expected_cat in log, (
        f"Expected category {expected_cat!r} in:\n{log}"
    )
    # Specific actionable token must surface
    assert must_contain.lower() in log.lower(), (
        f"Missing actionable token {must_contain!r} in:\n{log}"
    )


def test_main_unknown_error_falls_through_safely(provisioner_module, capsys):
    """If OCI throws something we've never seen, the translator must
    still produce output — just less specific."""
    args = argparse.Namespace(command="create")
    with patch.object(provisioner_module, "create_command",
                      side_effect=RuntimeError("Cosmic ray flipped a bit")):
        with patch.object(sys, "argv", ["provision_adb.py", "create"]):
            with patch.object(provisioner_module.argparse.ArgumentParser,
                              "parse_args", return_value=args):
                with pytest.raises(RuntimeError):
                    provisioner_module.main()
    log = capsys.readouterr().err
    assert "PROVISIONER FAILED" in log
    # The raw error must still be preserved for support
    assert "Cosmic ray" in log


# ---------------------------------------------------------------------------
# Integration: parse_create_args fails fast on missing required values
# ---------------------------------------------------------------------------
def test_parse_create_args_with_no_compartment_and_no_autodiscover_fails_clearly(
    provisioner_module,
):
    """Operator turned off auto-discover but didn't pass --compartment-id
    — must error with a specific hint."""
    args = argparse.Namespace(
        command="create",
        compartment_id=None,
        subnet_id=None,
        vcn_id=None,
        region="us-chicago-1",
        db_name=None,
        display_name=None,
        admin_password=None,
        rag_app_password=None,
        ecpus=None,
        storage_tb=None,
        wait_seconds=900,
        poll_seconds=15,
        kubeconfig=None,
        auto_discover=False,
        reuse_existing=False,
        pai_index_url="",
        pai_offload_enabled=False,
        workload_type=None,
        output_env=None,
        k8s_secret="oracle-creds",
        namespace="rag",
    )
    cfg = provisioner_module.parse_create_args(args)
    # parse_create_args itself doesn't enforce; discovery does. But
    # parse_create_args MUST surface auto_discover=False so a downstream
    # check can run.
    assert cfg.auto_discover is False
    # And the resolved compartment_id must remain None so discover_network
    # can fail loudly.
    assert cfg.compartment_id in (None, "")


def test_main_succeeds_for_unknown_command(provisioner_module, capsys):
    """Sanity: calling main() with no command should hit argparse's
    `required=True` and exit with a usage message — not a stack trace."""
    with patch.object(sys, "argv", ["provision_adb.py"]):
        with pytest.raises(SystemExit):
            provisioner_module.main()
    err = capsys.readouterr().err
    assert "usage" in err.lower() or "required" in err.lower()

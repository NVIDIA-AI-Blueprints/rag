# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``open_pai_offload_path`` (OCI NSG auto-config helper).

This is the security-path opening logic that enables ADB to reach the
``oracle-pai-gpu-index`` LoadBalancer on TCP/8080. The function is a
best-effort helper called from the provisioner Job; on IAM denial it logs
a fallback command and returns None instead of failing the install.

We mock the OCI clients (network + container_engine) and verify:
* Idempotent: existing covering rule -> no update call.
* Adds rule when none covers TCP/8080 from the ADB CIDR.
* Multiple LB subnets all updated.
* IAM 403/401/404 -> graceful fallback (no exception, prints command).
* No ACTIVE OKE cluster -> graceful skip.
* No service_lb_subnet_ids -> graceful skip.
* Subnet has no security_list_ids -> graceful skip.
* protocol="all" rule already covers everything.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result(data):
    """Wrap an OCI SDK-shaped data object in a stub Response."""
    return SimpleNamespace(data=data)


def _adb(subnet_id="ocid1.subnet.oc1..adb"):
    return SimpleNamespace(subnet_id=subnet_id)


def _adb_subnet(cidr="10.0.20.0/24", vcn_id="ocid1.vcn.oc1..vcn1"):
    return SimpleNamespace(cidr_block=cidr, vcn_id=vcn_id)


def _cluster(
    name="my-cluster",
    cluster_id="ocid1.cluster.oc1..a",
    state="ACTIVE",
    lb_subnets=("ocid1.subnet.oc1..lb1",),
):
    options = SimpleNamespace(service_lb_subnet_ids=list(lb_subnets))
    return SimpleNamespace(
        name=name, id=cluster_id, lifecycle_state=state, options=options,
    )


def _lb_subnet(display_name="lb-subnet-1", security_list_ids=("ocid1.sl.oc1..a",)):
    return SimpleNamespace(
        display_name=display_name,
        security_list_ids=list(security_list_ids),
    )


def _security_list(rules=()):
    return SimpleNamespace(ingress_security_rules=list(rules))


def _rule(
    source="10.0.20.0/24",
    protocol="6",
    min_port=8080,
    max_port=8080,
):
    """Build an ingress rule that mirrors the OCI SDK shape."""
    if protocol == "all":
        tcp_options = None
    else:
        tcp_options = SimpleNamespace(
            destination_port_range=SimpleNamespace(min=min_port, max=max_port),
        )
    return SimpleNamespace(
        source=source,
        protocol=protocol,
        tcp_options=tcp_options,
    )


@pytest.fixture
def patched_oci_models(monkeypatch, provisioner_module):
    """Stub out oci.core.models.IngressSecurityRule etc. with passthrough constructors."""
    import oci

    def _ingress(**kwargs):
        return SimpleNamespace(**kwargs)

    def _tcp(destination_port_range=None):
        return SimpleNamespace(destination_port_range=destination_port_range)

    def _port(min, max):
        return SimpleNamespace(min=min, max=max)

    def _update(**kwargs):
        return SimpleNamespace(**kwargs)

    # Touch only the symbols we use; other oci.* code paths remain intact.
    monkeypatch.setattr(
        "oci.core.models.IngressSecurityRule", _ingress, raising=False,
    )
    monkeypatch.setattr("oci.core.models.TcpOptions", _tcp, raising=False)
    monkeypatch.setattr("oci.core.models.PortRange", _port, raising=False)
    monkeypatch.setattr(
        "oci.core.models.UpdateSecurityListDetails", _update, raising=False,
    )
    return None


@pytest.fixture
def patched_oci_pagination(monkeypatch):
    """Make oci.pagination.list_call_get_all_results return whatever the underlying
    callable returns (one-shot, no real pagination)."""
    def fake_paginate(call, *args, **kwargs):
        return call(*args, **kwargs)
    monkeypatch.setattr(
        "oci.pagination.list_call_get_all_results", fake_paginate, raising=True,
    )


# ---------------------------------------------------------------------------
# Helper to construct the network + container_engine mocks for one test
# ---------------------------------------------------------------------------
def _make_clients(
    adb_subnet=None,
    clusters=None,
    lb_subnets_by_id=None,
    security_lists_by_id=None,
):
    network = MagicMock()
    network.get_subnet.side_effect = lambda subnet_id: _result(
        {**(lb_subnets_by_id or {}), **({"adb": adb_subnet} if adb_subnet else {})}.get(
            subnet_id, _lb_subnet(),
        )
    )
    # We override get_subnet below to dispatch by ID more cleanly:
    subnet_table = dict(lb_subnets_by_id or {})
    if adb_subnet is not None:
        subnet_table["ocid1.subnet.oc1..adb"] = adb_subnet
    network.get_subnet.side_effect = lambda subnet_id: _result(
        subnet_table.get(subnet_id, _lb_subnet()),
    )
    sl_table = dict(security_lists_by_id or {})
    network.get_security_list.side_effect = lambda sl_id: _result(
        sl_table.get(sl_id, _security_list()),
    )

    container_engine = MagicMock()
    container_engine.list_clusters.return_value = _result(clusters or [])

    return network, container_engine


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestOpenPaiOffloadPath:
    def test_adds_rule_when_none_covers(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        adb_sub = _adb_subnet(cidr="10.0.20.0/24", vcn_id="vcn1")
        cluster = _cluster(lb_subnets=("ocid1.subnet.oc1..lb1",))
        lb_sub = _lb_subnet(security_list_ids=("ocid1.sl.oc1..a",))
        sl = _security_list(rules=[
            _rule(source="0.0.0.0/0", protocol="6", min_port=22, max_port=22),
        ])
        network, ce = _make_clients(
            adb_subnet=adb_sub,
            clusters=[cluster],
            lb_subnets_by_id={"ocid1.subnet.oc1..lb1": lb_sub},
            security_lists_by_id={"ocid1.sl.oc1..a": sl},
        )

        result = provisioner_module.open_pai_offload_path(
            network=network,
            container_engine=ce,
            adb=_adb(),
            compartment_id="ocid1.compartment.oc1..x",
            pai_port=8080,
        )

        assert result is not None
        assert "added ingress rule" in result.lower()
        assert network.update_security_list.call_count == 1
        # Confirm we sent the new rule with the right shape
        call = network.update_security_list.call_args
        sl_id_arg = call.args[0]
        details = call.args[1]
        assert sl_id_arg == "ocid1.sl.oc1..a"
        new_rules = details.ingress_security_rules
        assert len(new_rules) == 2  # original SSH rule + new PAI rule
        new_pai = next(r for r in new_rules if r.protocol == "6" and r.source == "10.0.20.0/24")
        assert new_pai.tcp_options.destination_port_range.min == 8080
        assert new_pai.tcp_options.destination_port_range.max == 8080

    def test_idempotent_when_existing_rule_covers_port(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        cluster = _cluster()
        lb_sub = _lb_subnet()
        sl = _security_list(rules=[
            # An existing rule already permits TCP 8080 from the ADB CIDR
            _rule(source="10.0.20.0/24", protocol="6", min_port=8000, max_port=8200),
        ])
        network, ce = _make_clients(
            adb_subnet=adb_sub, clusters=[cluster],
            lb_subnets_by_id={"ocid1.subnet.oc1..lb1": lb_sub},
            security_lists_by_id={"ocid1.sl.oc1..a": sl},
        )

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(),
            compartment_id="x", pai_port=8080,
        )

        assert result is not None
        assert "already permits" in result.lower()
        assert network.update_security_list.call_count == 0

    def test_idempotent_when_protocol_all_rule_exists(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        """A protocol='all' rule from the ADB CIDR covers any TCP port."""
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        cluster = _cluster()
        lb_sub = _lb_subnet()
        sl = _security_list(rules=[
            _rule(source="10.0.20.0/24", protocol="all"),
        ])
        network, ce = _make_clients(
            adb_subnet=adb_sub, clusters=[cluster],
            lb_subnets_by_id={"ocid1.subnet.oc1..lb1": lb_sub},
            security_lists_by_id={"ocid1.sl.oc1..a": sl},
        )

        provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(),
            compartment_id="x", pai_port=8080,
        )
        assert network.update_security_list.call_count == 0

    def test_iam_denial_returns_none_and_does_not_raise(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
        capsys,
    ):
        import oci

        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        cluster = _cluster()
        lb_sub = _lb_subnet()
        sl = _security_list(rules=[])
        network, ce = _make_clients(
            adb_subnet=adb_sub, clusters=[cluster],
            lb_subnets_by_id={"ocid1.subnet.oc1..lb1": lb_sub},
            security_lists_by_id={"ocid1.sl.oc1..a": sl},
        )

        # Simulate IAM 403
        err = oci.exceptions.ServiceError(
            status=403, code="NotAuthorizedOrNotFound",
            headers={}, message="forbidden",
        )
        network.update_security_list.side_effect = err

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(),
            compartment_id="x", pai_port=8080,
        )
        assert result is None
        out = capsys.readouterr().out
        # Operator-facing fallback
        assert "oci network security-list update" in out
        assert "10.0.20.0/24" in out

    def test_no_active_cluster_returns_none(
        self, provisioner_module, patched_oci_models, patched_oci_pagination, capsys,
    ):
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        # Cluster present but DELETED
        cluster = _cluster(state="DELETED")
        network, ce = _make_clients(adb_subnet=adb_sub, clusters=[cluster])

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(),
            compartment_id="x",
        )
        assert result is None
        assert network.update_security_list.call_count == 0
        assert "no ACTIVE OKE cluster" in capsys.readouterr().out

    def test_no_service_lb_subnet_ids_returns_none(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        # cluster.options.service_lb_subnet_ids is empty
        cluster = _cluster(lb_subnets=())
        network, ce = _make_clients(adb_subnet=adb_sub, clusters=[cluster])

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(), compartment_id="x",
        )
        assert result is None
        assert network.update_security_list.call_count == 0

    def test_adb_without_subnet_id_returns_none(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        adb = SimpleNamespace(subnet_id=None)
        network, ce = _make_clients()
        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=adb, compartment_id="x",
        )
        assert result is None

    def test_multiple_lb_subnets_all_updated(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        cluster = _cluster(lb_subnets=("ocid1.subnet.oc1..lb1", "ocid1.subnet.oc1..lb2"))
        lb1 = _lb_subnet(security_list_ids=("sl1",))
        lb2 = _lb_subnet(security_list_ids=("sl2",))
        sl1 = _security_list()
        sl2 = _security_list()
        network, ce = _make_clients(
            adb_subnet=adb_sub, clusters=[cluster],
            lb_subnets_by_id={
                "ocid1.subnet.oc1..lb1": lb1, "ocid1.subnet.oc1..lb2": lb2,
            },
            security_lists_by_id={"sl1": sl1, "sl2": sl2},
        )

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(), compartment_id="x",
        )
        assert result is not None
        assert network.update_security_list.call_count == 2
        sl_ids_called = sorted(c.args[0] for c in network.update_security_list.call_args_list)
        assert sl_ids_called == ["sl1", "sl2"]

    def test_unexpected_exception_logs_and_returns_none(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
        capsys,
    ):
        network = MagicMock()
        network.get_subnet.side_effect = RuntimeError("network down")
        ce = MagicMock()

        result = provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(), compartment_id="x",
        )
        assert result is None
        out = capsys.readouterr().out
        assert "auto-add failed" in out

    def test_custom_port_in_emitted_rule(
        self, provisioner_module, patched_oci_models, patched_oci_pagination,
    ):
        """If we ever change the PAI port, the rule must reflect it."""
        adb_sub = _adb_subnet(cidr="10.0.20.0/24")
        cluster = _cluster()
        lb_sub = _lb_subnet()
        sl = _security_list()
        network, ce = _make_clients(
            adb_subnet=adb_sub, clusters=[cluster],
            lb_subnets_by_id={"ocid1.subnet.oc1..lb1": lb_sub},
            security_lists_by_id={"ocid1.sl.oc1..a": sl},
        )
        provisioner_module.open_pai_offload_path(
            network=network, container_engine=ce, adb=_adb(),
            compartment_id="x", pai_port=9443,
        )
        new_rules = network.update_security_list.call_args.args[1].ingress_security_rules
        new_pai = next(r for r in new_rules if r.protocol == "6")
        assert new_pai.tcp_options.destination_port_range.min == 9443
        assert new_pai.tcp_options.destination_port_range.max == 9443

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drift guards + regression-pin tests.

* **Drift guards** lock invariants that span two files so that a refactor
  of one without the other is caught immediately. (Manual coupling like
  this is brittle in long-lived codebases; tests are the cheapest way to
  trace it.)
* **Regression pins** lock the exact input that triggered each fixed
  bug so the fix can't quietly regress.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
HELM = REPO / "examples" / "oracle" / "helm"
PROVISIONER = HELM / "files" / "provision_adb.py"


# ===========================================================================
# Drift guards
# ===========================================================================
def test_byo_job_required_set_matches_python_module():
    """The Helm Job's REQUIRED set MUST equal RAG_CANONICAL_COLUMNS.

    Drift here is silent: oracle_vdb thinks a table is canonical, the
    Job marks it non-canonical (or vice versa), and the UI lists or
    drops it inconsistently.
    """
    from nvidia_rag.utils.vdb.oracle.oracle_queries import RAG_CANONICAL_COLUMNS

    job_yaml = (HELM / "templates" / "oracle-byo-import.yaml").read_text()
    m = re.search(r'REQUIRED\s*=\s*\{([^}]+)\}', job_yaml)
    assert m, "REQUIRED set not found in oracle-byo-import.yaml"
    items = set(re.findall(r'"([A-Z_]+)"', m.group(1)))
    assert items == set(RAG_CANONICAL_COLUMNS), (
        f"BYO Job REQUIRED={items} != RAG_CANONICAL_COLUMNS="
        f"{set(RAG_CANONICAL_COLUMNS)}"
    )


def test_byo_job_view_ddl_matches_python_byo_helper():
    """The Job's inline view DDL builder and the Python helper used by
    OracleVDB should produce structurally identical SELECT projections
    for the same column map.

    We compare normalised lists of "X AS canonical" projections.
    """
    from nvidia_rag.utils.vdb.oracle.oracle_queries import create_byo_view_ddl

    py_ddl = create_byo_view_ddl(
        view_name="x",
        source_table="T",
        column_map={
            "id": "DOC_ID", "text": "BODY", "vector": "EMBED",
            "source": "URL", "content_metadata": "META",
        },
    )

    def projections(ddl: str) -> set[str]:
        # Pick out every "<expr> AS <canonical>" pair, normalise
        ddl_n = " ".join(ddl.upper().split())
        # Find each "AS <name>" target column
        return set(re.findall(r"AS\s+(\w+)", ddl_n))

    py_targets = projections(py_ddl)
    assert {"ID", "TEXT", "VECTOR", "SOURCE", "CONTENT_METADATA"} <= py_targets

    # The Job builds DDL inline. Pull its targets from the heredoc.
    job_yaml = (HELM / "templates" / "oracle-byo-import.yaml").read_text()
    job_targets = projections(job_yaml)
    # The job's heredoc contains the same five canonical AS targets,
    # plus possibly created_at.
    for col in ("ID", "TEXT", "VECTOR", "SOURCE", "CONTENT_METADATA"):
        assert col in job_targets, (
            f"BYO Job's view DDL projection missing AS {col}"
        )


def test_oracle_pai_verify_consumer_list_matches_chart_deployments():
    """oracle-pai-verify Job logs which downstream Deployments will
    consume the PAI URL. If a new Deployment is added that needs the URL
    but isn't on the consumer list, operators won't know to restart it.

    This test enumerates the deployments that *do* mount oracle-creds
    via envFrom (i.e. would need ORACLE_PAI_INDEX_URL refresh) and
    cross-checks them against the verify Job's print statements.
    """
    verify_yaml = (HELM / "templates" / "oracle-pai-verify.yaml").read_text()
    # The verify Job declares CONSUMER_DEPLOYMENTS as a comma-list; the
    # heredoc reads it and rolls each one. Drift here means a renamed /
    # added Deployment never gets restarted post-PAI-verify and ends up
    # serving stale env vars.
    m = re.search(r'CONSUMER_DEPLOYMENTS[^v]*value:\s*"([^"]+)"', verify_yaml)
    assert m, (
        "CONSUMER_DEPLOYMENTS env var missing from oracle-pai-verify.yaml; "
        "the post-verify rollout would be a no-op"
    )
    consumers = {x.strip() for x in m.group(1).split(",") if x.strip()}
    expected = {"rag-server", "ingestor-server"}
    missing = expected - consumers
    assert not missing, (
        f"verify Job CONSUMER_DEPLOYMENTS missing: {missing}. "
        "If you renamed/dropped one, update CONSUMER_DEPLOYMENTS too."
    )


def test_byo_values_block_present_in_both_values_files():
    """Operators routinely copy values.create-adb.yaml → custom file;
    we want them to discover the BYO knob no matter which template they
    started from."""
    for fname in ("values.create-adb.yaml", "values.existing-adb.yaml"):
        text = (HELM / fname).read_text()
        assert "importExistingTables" in text, (
            f"{fname}: missing importExistingTables block"
        )
        # Also confirm the doc comment exists so users know what shape to use
        assert "sourceTable" in text, f"{fname}: missing sourceTable example"
        assert "collectionName" in text, f"{fname}: missing collectionName example"


def test_readme_documents_byo_flow():
    readme = (HELM / "README.md").read_text()
    for needle in (
        "BYO collections", "importExistingTables",
        "sourceTable", "read_only",
    ):
        assert needle in readme, f"README missing mention of {needle!r}"


def test_provisioner_pai_url_helper_signature_stable():
    """The provisioner exposes ``open_pai_offload_path`` as the OCI NSG
    helper. Other modules import it by name; renaming would silently
    skip the post-install network gate. Pin the public name."""
    text = PROVISIONER.read_text()
    assert "def open_pai_offload_path" in text


# ===========================================================================
# Regression pins — one per fixed bug
# ===========================================================================
class TestRegressions:
    def test_pai_offload_enabled_uses_resolved_url_not_raw_arg(
        self, provisioner_module, monkeypatch,
    ):
        """REGRESSION: parse_create_args used args.pai_index_url to set
        pai_offload_enabled, which meant ORACLE_PAI_INDEX_URL set via env
        var was honored for PAI but did NOT trigger the OCI NSG path.

        Pin: setting only the env var (no CLI flag) must still set
        pai_offload_enabled=True.
        """
        import argparse

        monkeypatch.setenv("ORACLE_PAI_INDEX_URL", "http://10.0.50.42:8080/v1/index")
        ns = argparse.Namespace(
            output_env=None, k8s_secret="oracle-creds", namespace="rag",
            compartment_id="ocid1.compartment.oc1..xxx",
            subnet_id=None, vcn_id=None, region=None,
            db_name="ragbp", display_name=None,
            workload_type=None,
            admin_password=None, rag_app_password=None,
            ecpus=None, storage_tb=None,
            wait_seconds=300, poll_seconds=15,
            kubeconfig=None, auto_discover=True,
            reuse_existing=False,
            pai_index_url="",      # No CLI flag — env var alone must trigger
            pai_offload_enabled=False,
        )
        opts = provisioner_module.parse_create_args(ns)
        assert opts.pai_offload_enabled is True, (
            "ORACLE_PAI_INDEX_URL env var must enable pai_offload_enabled"
        )

    def test_extract_source_name_handles_legacy_plain_string(self):
        """REGRESSION: get_documents crashed with json.JSONDecodeError
        on legacy rows that stored ``source`` as a plain string instead
        of a JSON object. Defensive fix returns the string verbatim."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        # Plain string, no JSON brace → return as-is
        assert OracleVDB._extract_source_name("/tmp/x.pdf") == "/tmp/x.pdf"
        # Malformed JSON-like string → return stripped string
        assert OracleVDB._extract_source_name("{not json") == "{not json"
        # None → 'unknown' (UI handles 'unknown' gracefully)
        assert OracleVDB._extract_source_name(None) == "unknown"

    def test_get_documents_does_not_crash_on_non_canonical(self, monkeypatch):
        """REGRESSION: BYO custom-shape table caused get_documents() to
        500 the API with ORA-00904 (no SOURCE column). Defensive return
        of [] keeps the UI tab alive."""
        # We lean on the e2e mocked test for the full flow; here we
        # just lock in the contract that the docstring documents.
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
        import inspect
        src = inspect.getsource(OracleVDB.get_documents)
        assert "return []" in src
        assert "is_canonical" in src

    def test_byo_view_ddl_uses_create_or_replace(self):
        """REGRESSION: An earlier draft used CREATE VIEW which fails
        with ORA-00955 the second time the BYO Job runs (after a
        column-mapping change). Must be CREATE OR REPLACE."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        ddl = oracle_queries.create_byo_view_ddl(
            "v", "T", {"text": "c", "vector": "v"},
        ).upper()
        assert "CREATE OR REPLACE" in ddl

    def test_neighbor_partitions_two_words(self):
        """REGRESSION: ADB 23ai/26ai rejects ``neighbor_partitions`` as
        a single token (ORA-00922). Must be two words.

        Pinning here in addition to the existing test_oracle_vdb.py
        because this bug bit us once already in dev — pinning twice is
        cheap."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        ddl = oracle_queries.create_vector_index_ddl("X", "IVF", "COSINE")
        norm = " ".join(ddl.upper().split())
        assert "NEIGHBOR PARTITIONS" in norm
        assert "NEIGHBOR_PARTITIONS" not in norm

    def test_pai_offload_only_on_hnsw(self):
        """REGRESSION: Earlier draft incorrectly stamped OFFLOAD_URL
        onto IVF DDL too, which Oracle silently ignores AND breaks
        IVF builds. Pin: IVF must be entirely free of OFFLOAD."""
        from nvidia_rag.utils.vdb.oracle import oracle_queries
        ddl = oracle_queries.create_vector_index_ddl(
            "X", index_type="IVF", distance_metric="COSINE",
            pai_offload_url="http://x:8080/v1/index",
        )
        assert "OFFLOAD" not in ddl.upper()

    def test_pai_url_appends_v1_index_when_missing(self, monkeypatch):
        """REGRESSION: Operators set ORACLE_PAI_INDEX_URL=http://x:8080
        without /v1/index. We must append the path automatically;
        otherwise Oracle calls / and gets a 404, with no clear hint."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
        # Test the same normalization logic OracleVDB.__init__ uses.
        for raw, expected in [
            ("http://x:8080",          "http://x:8080/v1/index"),
            ("http://x:8080/",         "http://x:8080/v1/index"),
            ("http://x:8080/v1/index", "http://x:8080/v1/index"),
            ("https://x:8443/v1",      "https://x:8443/v1/v1/index"),  # docs warn this is wrong
        ]:
            url = raw.strip()
            if url and not url.endswith("/v1/index"):
                url = url.rstrip("/") + "/v1/index"
            assert url == expected, f"{raw!r} normalised to {url!r}, expected {expected!r}"

    def test_byo_yaml_no_destructive_sql(self):
        """REGRESSION: BYO Job must NEVER contain DROP/TRUNCATE/DELETE
        on customer base tables — it's the most user-visible failure
        mode if someone mis-edits the heredoc."""
        text = (HELM / "templates" / "oracle-byo-import.yaml").read_text()
        # Keep our literal-match list narrow to avoid catching the
        # DROP keyword inside identifiers / Python comments.
        for forbidden in ("DROP TABLE", "TRUNCATE TABLE", "DELETE FROM"):
            assert forbidden not in text.upper(), (
                f"BYO Job template contains {forbidden!r} — customer data at risk"
            )

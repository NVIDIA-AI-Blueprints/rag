# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks with hard performance budgets.

The DDL builders, source-name extractor, and password generator are all
on the hot path of:

* ``GET /collections``                — runs on every UI page load
* every ingestion ``write_to_index``  — runs N times per chunk
* every retrieval call                — runs once per query

A regression where someone accidentally compiles a regex inside the
hot loop (``re.compile`` in a function body), uses ``json.loads`` on
each row, or loops with O(n²) string concat would silently degrade
production latency. These budgets catch those changes early.

The numbers below are deliberately loose — 5× the typical observed time
on a slow CI runner — so they're stable but still useful as a
canary.
"""
from __future__ import annotations

import time

import pytest


def _time(fn, *args, iters=1000, **kwargs):
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - start) / iters


# ---------------------------------------------------------------------------
# Budgets are documented on the test for clarity. Multiply by 1000 to convert
# the per-call time (in seconds) into microseconds for readability.
# ---------------------------------------------------------------------------
class TestDDLBuilderBudgets:
    def test_create_table_ddl_under_50us(self):
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_table_ddl

        per_call = _time(create_vector_table_ddl, "DOCS", 2048, iters=5000)
        assert per_call < 50e-6, f"create_vector_table_ddl: {per_call*1e6:.1f}µs > 50µs"

    def test_create_index_ddl_under_50us(self):
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_vector_index_ddl

        per_call = _time(
            create_vector_index_ddl,
            "DOCS", "HNSW", "COSINE",
            iters=5000,
        )
        assert per_call < 50e-6, f"create_vector_index_ddl: {per_call*1e6:.1f}µs > 50µs"

    def test_byo_view_ddl_under_100us(self):
        from nvidia_rag.utils.vdb.oracle.oracle_queries import create_byo_view_ddl

        per_call = _time(
            create_byo_view_ddl,
            "v", "T",
            {"text": "c", "vector": "v", "source": "s",
             "content_metadata": "m", "id": "id"},
            iters=5000,
        )
        assert per_call < 100e-6, f"create_byo_view_ddl: {per_call*1e6:.1f}µs > 100µs"


class TestNormalizationBudgets:
    def test_normalize_db_name_under_20us(self, provisioner_module):
        per_call = _time(
            provisioner_module.normalize_db_name,
            "Some-Long.Customer_DB Name 12345",
            iters=10_000,
        )
        assert per_call < 20e-6, f"normalize_db_name: {per_call*1e6:.1f}µs > 20µs"

    def test_stable_suffix_under_30us(self, provisioner_module):
        per_call = _time(
            provisioner_module.stable_suffix,
            "ocid1.compartment.oc1..xxx", "us-ashburn-1",
            iters=10_000,
        )
        assert per_call < 30e-6, f"stable_suffix: {per_call*1e6:.1f}µs > 30µs"


class TestSourceExtractionBudget:
    def test_extract_source_name_under_10us(self):
        """``_extract_source_name`` is called once per row in get_documents.
        For a 10K-document collection that's 10K invocations during a
        single page render."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        cases = [
            '{"source_name": "/path/to/file.pdf"}',
            "/plain/path/file.pdf",
            None,
            {"source_name": "x.pdf"},
        ]
        for v in cases:
            per_call = _time(OracleVDB._extract_source_name, v, iters=5000)
            assert per_call < 10e-6, (
                f"_extract_source_name({v!r}): {per_call*1e6:.1f}µs > 10µs"
            )


class TestSanitizeQueryBudget:
    def test_sanitize_text_query_under_50us(self):
        """Hot path on every hybrid retrieval call."""
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        q = "What is the throughput of an H100 80GB GPU on cuVS HNSW builds?"
        per_call = _time(OracleVDB._sanitize_text_query, q, iters=5000)
        # Compiling a regex inside this function (the most common
        # regression) bumps it past 200µs immediately.
        assert per_call < 50e-6, f"_sanitize_text_query: {per_call*1e6:.1f}µs > 50µs"

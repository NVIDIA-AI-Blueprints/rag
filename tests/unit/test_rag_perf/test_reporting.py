# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``rag_perf.reporting``: percentile, stage breakdown, citation quality, summary aggregation."""

from __future__ import annotations

import pytest
from rag_perf.reporting import MetricsAggregator, ProfileRecord, RagMetricsSummary

from tests.unit.test_rag_perf.utils import fake_aiperf_json, make_profile_record

# Bind the MetricsAggregator static methods to module-level names so the tests
# below can call them as plain functions.
compute_from_profiler = MetricsAggregator.from_profiler
enrich_with_aiperf = MetricsAggregator.enrich_with_aiperf
_compute_stage_breakdown = MetricsAggregator._compute_stage_breakdown
_compute_citation_quality = MetricsAggregator._compute_citation_quality
_percentile = MetricsAggregator._percentile
_ms = MetricsAggregator._ms


class TestMetricsAggregator:
    """Tests for ``rag_perf.reporting.MetricsAggregator`` and its helpers."""

    # ── _percentile ───────────────────────────────────────────────────────────

    def test_percentile_p50_of_sorted(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == pytest.approx(3.0)

    def test_percentile_p0_is_min(self):
        vals = [10.0, 20.0, 30.0]
        assert _percentile(vals, 0) == pytest.approx(10.0)

    def test_percentile_p100_is_max(self):
        vals = [10.0, 20.0, 30.0]
        assert _percentile(vals, 100) == pytest.approx(30.0)

    def test_percentile_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)
        assert _percentile([42.0], 99) == pytest.approx(42.0)

    def test_percentile_empty_returns_none(self):
        assert _percentile([], 50) is None

    def test_percentile_p90_larger_list(self):
        result = _percentile([float(v) for v in range(1, 101)], 90)
        assert result == pytest.approx(90.0, abs=1.0)

    # ── _ms ───────────────────────────────────────────────────────────────────

    def test_ms_passthrough_for_ms_range(self):
        assert _ms(1234.5) == pytest.approx(1234.5)

    def test_ms_converts_nanoseconds(self):
        assert _ms(3_240_000_000) == pytest.approx(3240.0)

    def test_ms_none_returns_none(self):
        assert _ms(None) is None

    # ── Stage breakdown ───────────────────────────────────────────────────────

    def test_compute_stage_breakdown_means(self, profile_records):
        bd = _compute_stage_breakdown(profile_records)
        assert bd.retrieval_ms == pytest.approx(149.0, abs=10.0)
        assert bd.reranking_ms == pytest.approx(2285.0, abs=200.0)
        assert bd.llm_ttft_ms == pytest.approx(967.5, abs=100.0)

    def test_compute_stage_breakdown_fractions_sum_to_one(self, profile_records):
        bd = _compute_stage_breakdown(profile_records)
        total_frac = (
            (bd.retrieval_frac or 0) + (bd.reranking_frac or 0) + (bd.llm_frac or 0)
        )
        assert total_frac == pytest.approx(1.0, abs=0.15)

    def test_compute_stage_breakdown_identifies_reranking_bottleneck(
        self, profile_records
    ):
        bd = _compute_stage_breakdown(profile_records)
        assert bd.bottleneck == "reranking"

    def test_compute_stage_breakdown_identifies_llm_bottleneck(self):
        records = [
            make_profile_record(
                retrieval=50.0, reranking=100.0, llm_ttft=2000.0, rag_ttft=2200.0
            )
            for _ in range(5)
        ]
        bd = _compute_stage_breakdown(records)
        assert bd.bottleneck == "llm"

    def test_compute_stage_breakdown_empty_records(self):
        bd = _compute_stage_breakdown([])
        assert bd.rag_ttft_ms is None
        assert bd.bottleneck == "unknown"

    def test_compute_stage_breakdown_missing_server_metrics(self):
        records = [
            ProfileRecord(
                query="q",
                client_ttft_ms=100.0,
                client_e2e_ms=200.0,
                output_tokens=0,
                input_tokens=0,
                server_metrics={
                    "rag_ttft_ms": None,
                    "llm_ttft_ms": None,
                    "retrieval_time_ms": None,
                    "reranking_time_ms": None,
                    "llm_generation_time_ms": None,
                },
                citation_count=0,
                citation_scores=[],
                citations_raw=[],
            )
        ]
        bd = _compute_stage_breakdown(records)
        assert bd.rag_ttft_ms is None

    # ── Citation quality ──────────────────────────────────────────────────────

    def test_compute_citation_quality_mean_score(self, profile_records):
        cq = _compute_citation_quality(profile_records)
        assert cq.mean_score == pytest.approx(0.727, abs=0.05)

    def test_compute_citation_quality_mean_count(self, profile_records):
        cq = _compute_citation_quality(profile_records)
        assert cq.mean_count == pytest.approx(3.0, abs=0.1)

    def test_compute_citation_quality_percentiles(self, profile_records):
        cq = _compute_citation_quality(profile_records)
        assert cq.p50_score is not None
        assert cq.p90_score is not None
        assert cq.p50_score <= cq.p90_score

    def test_compute_citation_quality_empty(self):
        cq = _compute_citation_quality([])
        assert cq.mean_score is None
        assert cq.mean_count is None

    # ── compute_from_profiler ─────────────────────────────────────────────────

    def test_compute_from_profiler_returns_summary(self, profile_records):
        summary = compute_from_profiler(profile_records)
        assert isinstance(summary, RagMetricsSummary)
        assert summary.stage_breakdown.bottleneck == "reranking"
        assert summary.citation_quality.mean_score is not None
        assert summary.profile_client_ttft_p50_ms is not None

    def test_compute_from_profiler_all_failed(self):
        failed = [make_profile_record(error="timeout") for _ in range(5)]
        summary = compute_from_profiler(failed)
        assert summary.stage_breakdown.rag_ttft_ms is None
        assert summary.citation_quality.mean_score is None

    def test_compute_from_profiler_mixed_success_failure(
        self, profile_records_with_errors
    ):
        summary = compute_from_profiler(profile_records_with_errors)
        assert summary.stage_breakdown.rag_ttft_ms is not None

    # ── enrich_with_aiperf ────────────────────────────────────────────────────

    def test_enrich_with_aiperf_fills_load_metrics(self, profile_records):
        summary = compute_from_profiler(profile_records)
        aiperf_data = fake_aiperf_json(ttft_p50=3200.0, ttft_p99=12000.0)

        summary = enrich_with_aiperf(
            summary, aiperf_data, concurrency=8, total_requests=200
        )

        assert summary.concurrency == 8
        assert summary.total_requests == 200
        assert summary.load_ttft_p50_ms == pytest.approx(3200.0)
        assert summary.load_ttft_p99_ms == pytest.approx(12000.0)
        assert summary.load_throughput_tok_s == pytest.approx(28.4)
        assert summary.load_request_throughput == pytest.approx(0.95)
        assert summary.load_error_rate == pytest.approx(0.01)

    def test_enrich_with_empty_aiperf_output(self, profile_records):
        summary = compute_from_profiler(profile_records)
        summary = enrich_with_aiperf(summary, {}, concurrency=4, total_requests=50)
        assert summary.load_ttft_p50_ms is None

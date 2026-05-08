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

"""
Output side of rag-perf: shared dataclasses, metric aggregation, and reporting.

Dataclasses
-----------
  ``ProfileRecord``      — one profiled request (timing + server metrics + citations).
  ``ProfileResult``      — collection of records; .successful / .failed properties.
  ``StageBreakdown``     — mean per-stage timings and the inferred bottleneck.
  ``CitationQuality``    — aggregate citation-score statistics.
  ``RagMetricsSummary``  — unified result combining profiling + load-test data.

Logic
-----
  ``MetricsAggregator``  — derives RagMetricsSummary from profiler records and
                           the aiperf JSON output.
  ``Reporter``           — Rich tables, Markdown report, JSON / CSV exports.
"""

from __future__ import annotations

import csv
import json
import statistics
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

_console = Console()


# ────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ProfileRecord:
    """Timing and metadata for a single profiled request."""

    query: str
    client_ttft_ms: float
    client_e2e_ms: float
    output_tokens: int
    input_tokens: int
    server_metrics: dict[str, float | None]
    """
    Server-reported pipeline stage timing::

        {
            "rag_ttft_ms":           <float | None>,
            "llm_ttft_ms":           <float | None>,
            "retrieval_time_ms":     <float | None>,
            "reranking_time_ms":     <float | None>,
            "llm_generation_time_ms": <float | None>,
        }
    """
    citation_count: int
    citation_scores: list[float]
    citations_raw: list[dict[str, Any]]
    error: str | None = None


@dataclass
class ProfileResult:
    """Aggregated result from a profiler run."""

    records: list[ProfileRecord] = field(default_factory=list)
    metadata_jsonl_path: str | None = None

    @property
    def successful(self) -> list[ProfileRecord]:
        return [r for r in self.records if r.error is None]

    @property
    def failed(self) -> list[ProfileRecord]:
        return [r for r in self.records if r.error is not None]


@dataclass
class StageBreakdown:
    """Mean time spent in each RAG pipeline stage, plus fractional shares."""

    retrieval_ms: float | None = None
    reranking_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_generation_ms: float | None = None
    rag_ttft_ms: float | None = None

    retrieval_frac: float | None = None
    reranking_frac: float | None = None
    llm_frac: float | None = None

    bottleneck: str = "unknown"
    """One of: ``"retrieval"``, ``"reranking"``, ``"llm"``, ``"unknown"``."""


@dataclass
class CitationQuality:
    """Aggregate relevance-score statistics for retrieved citations."""

    mean_score: float | None = None
    p50_score: float | None = None
    p90_score: float | None = None
    mean_count: float | None = None


@dataclass
class RagMetricsSummary:
    """Unified benchmark summary combining profiler data and aiperf load-test data."""

    # From the profiling pass
    stage_breakdown: StageBreakdown = field(default_factory=StageBreakdown)
    citation_quality: CitationQuality = field(default_factory=CitationQuality)
    profile_client_ttft_p50_ms: float | None = None
    profile_client_ttft_p90_ms: float | None = None
    profile_client_e2e_p50_ms: float | None = None

    # From aiperf load test
    load_ttft_mean_ms: float | None = None
    load_ttft_p50_ms: float | None = None
    load_ttft_p90_ms: float | None = None
    load_ttft_p99_ms: float | None = None
    load_e2e_mean_ms: float | None = None
    load_e2e_p90_ms: float | None = None
    load_e2e_p99_ms: float | None = None
    load_throughput_tok_s: float | None = None
    load_request_throughput: float | None = None
    load_error_rate: float | None = None

    # Run metadata
    concurrency: int | None = None
    total_requests: int | None = None
    collection_names: list[str] = field(default_factory=list)
    vdb_top_k: int | None = None
    reranker_top_k: int | None = None
    profile_requests_failed: int = 0
    profile_requests_total: int = 0


# ────────────────────────────────────────────────────────────────────────────
# MetricsAggregator
# ────────────────────────────────────────────────────────────────────────────


class MetricsAggregator:
    """
    RAG-specific metrics computation and aggregation.

    Consumes ``ProfileRecord`` objects from the direct profiling pass and the
    aiperf JSON output dict, and produces a unified ``RagMetricsSummary``.
    """

    @staticmethod
    def from_profiler(records: list[ProfileRecord]) -> RagMetricsSummary:
        """
        Build a ``RagMetricsSummary`` from profiler records alone.

        Used when only the direct profiling pass was run (e.g., ``rag-perf profile``),
        without a full aiperf load test.
        """
        summary = RagMetricsSummary()
        good = [r for r in records if r.error is None]
        if not good:
            return summary

        summary.stage_breakdown = MetricsAggregator._compute_stage_breakdown(good)
        summary.citation_quality = MetricsAggregator._compute_citation_quality(good)

        ttfts = [r.client_ttft_ms for r in good]
        e2es = [r.client_e2e_ms for r in good]
        summary.profile_client_ttft_p50_ms = MetricsAggregator._percentile(ttfts, 50)
        summary.profile_client_ttft_p90_ms = MetricsAggregator._percentile(ttfts, 90)
        summary.profile_client_e2e_p50_ms = MetricsAggregator._percentile(e2es, 50)

        return summary

    @staticmethod
    def enrich_with_aiperf(
        summary: RagMetricsSummary,
        aiperf_result: dict[str, Any],
        concurrency: int,
        total_requests: int,
    ) -> RagMetricsSummary:
        """
        Merge aiperf load-test metrics into an existing ``RagMetricsSummary``.

        Args:
            summary:         Summary already populated by ``from_profiler``.
            aiperf_result:   Dict loaded from aiperf's ``profile_export_aiperf.json``.
            concurrency:     Concurrency level used for the aiperf run.
            total_requests:  Request count used for the aiperf run.
        """
        summary.concurrency = concurrency
        summary.total_requests = total_requests

        ttft = aiperf_result.get("time_to_first_token") or {}
        summary.load_ttft_mean_ms = MetricsAggregator._ms(ttft.get("avg"))
        summary.load_ttft_p50_ms = MetricsAggregator._ms(ttft.get("p50"))
        summary.load_ttft_p90_ms = MetricsAggregator._ms(ttft.get("p90"))
        summary.load_ttft_p99_ms = MetricsAggregator._ms(ttft.get("p99"))

        e2e = aiperf_result.get("request_latency") or {}
        summary.load_e2e_mean_ms = MetricsAggregator._ms(e2e.get("avg"))
        summary.load_e2e_p90_ms = MetricsAggregator._ms(e2e.get("p90"))
        summary.load_e2e_p99_ms = MetricsAggregator._ms(e2e.get("p99"))

        tput = aiperf_result.get("output_token_throughput") or {}
        summary.load_throughput_tok_s = tput.get("avg")

        req_tput = aiperf_result.get("request_throughput") or {}
        summary.load_request_throughput = req_tput.get("avg")

        err_count = (aiperf_result.get("error_request_count") or {}).get("avg") or 0.0
        req_count = (aiperf_result.get("request_count") or {}).get("avg") or 0.0
        summary.load_error_rate = (err_count / req_count) if req_count > 0 else None

        return summary

    @staticmethod
    def _compute_stage_breakdown(records: list[ProfileRecord]) -> StageBreakdown:
        """Compute mean stage timings and derive fractional shares."""
        bd = StageBreakdown()

        def _avg(key: str) -> float | None:
            vals = [
                r.server_metrics[key]
                for r in records
                if r.server_metrics.get(key) is not None
            ]
            return statistics.mean(vals) if vals else None

        bd.rag_ttft_ms = _avg("rag_ttft_ms")
        bd.llm_ttft_ms = _avg("llm_ttft_ms")
        bd.retrieval_ms = _avg("retrieval_time_ms")
        bd.reranking_ms = _avg("reranking_time_ms")
        bd.llm_generation_ms = _avg("llm_generation_time_ms")

        total = bd.rag_ttft_ms
        if total and total > 0:
            bd.retrieval_frac = (bd.retrieval_ms or 0) / total
            bd.reranking_frac = (bd.reranking_ms or 0) / total
            bd.llm_frac = (bd.llm_ttft_ms or 0) / total

            fractions = {
                "retrieval": bd.retrieval_frac,
                "reranking": bd.reranking_frac,
                "llm": bd.llm_frac,
            }
            bd.bottleneck = max(fractions, key=lambda k: fractions[k] or 0.0)

        return bd

    @staticmethod
    def _compute_citation_quality(records: list[ProfileRecord]) -> CitationQuality:
        """Aggregate citation count and relevance scores."""
        cq = CitationQuality()
        all_scores: list[float] = []
        counts: list[int] = []

        for rec in records:
            counts.append(rec.citation_count)
            all_scores.extend(rec.citation_scores)

        if counts:
            cq.mean_count = statistics.mean(counts)
        if all_scores:
            cq.mean_score = statistics.mean(all_scores)
            cq.p50_score = MetricsAggregator._percentile(all_scores, 50)
            cq.p90_score = MetricsAggregator._percentile(all_scores, 90)

        return cq

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float | None:
        """Compute the Nth percentile of a list.  Returns None for empty lists."""
        if not values:
            return None
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * pct / 100
        lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

    @staticmethod
    def _ms(value: Any) -> float | None:
        """
        Pass a value through as float milliseconds; convert ns if the value is huge.

        aiperf exports all latency metrics in **milliseconds**, so values like
        3200.0 are returned as-is.  Values > 1_000_000 are treated as nanoseconds.
        """
        if value is None:
            return None
        fv = float(value)
        if fv > 1_000_000:
            return fv / 1_000_000  # ns → ms
        return fv


# ────────────────────────────────────────────────────────────────────────────
# Reporter
# ────────────────────────────────────────────────────────────────────────────


class Reporter:
    """
    Rich-based terminal reporting and Markdown report generation.

    Provides single-run tables, sweep comparison tables, Markdown reports,
    and JSON/CSV export.
    """

    @staticmethod
    def print_summary(
        summary: RagMetricsSummary, title: str = "RAG-Perf Results"
    ) -> None:
        """Print a formatted Rich table for a single ``RagMetricsSummary``."""
        t = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
        )
        t.add_column("Metric", style="bold", min_width=38)
        t.add_column("Value", justify="right", min_width=16)

        def row(label: str, value: Any, unit: str = "", style: str = "") -> None:
            if value is None:
                t.add_row(label, "[dim]N/A[/dim]")
                return
            text = (
                f"{value:,.1f} {unit}".strip()
                if isinstance(value, float)
                else str(value)
            )
            t.add_row(label, Text(text, style=style))

        t.add_section()
        t.add_row("[bold]─── Profiling Pass (server-side)[/bold]", "")

        sb: StageBreakdown = summary.stage_breakdown

        row("  Server RAG TTFT (mean)", sb.rag_ttft_ms, "ms")
        Reporter._stage_row(t, "  └ Retrieval", sb.retrieval_ms, sb.retrieval_frac)
        Reporter._stage_row(
            t,
            "  └ Reranking",
            sb.reranking_ms,
            sb.reranking_frac,
            highlight=sb.bottleneck == "reranking",
        )
        Reporter._stage_row(
            t,
            "  └ LLM TTFT",
            sb.llm_ttft_ms,
            sb.llm_frac,
            highlight=sb.bottleneck == "llm",
        )
        row("  Client TTFT p50", summary.profile_client_ttft_p50_ms, "ms")
        row("  Client TTFT p90", summary.profile_client_ttft_p90_ms, "ms")

        cq = summary.citation_quality
        row("  Citation count (mean)", cq.mean_count, "chunks")
        row("  Citation relevance score (mean)", cq.mean_score)
        row("  Citation relevance score (p90)", cq.p90_score)

        if sb.bottleneck != "unknown":
            t.add_row(
                "  Bottleneck",
                Text(sb.bottleneck.upper(), style="bold yellow"),
            )

        if summary.load_ttft_p50_ms is not None:
            t.add_section()
            t.add_row(
                f"[bold]─── Load Test (concurrency={summary.concurrency})[/bold]", ""
            )
            row("  TTFT mean", summary.load_ttft_mean_ms, "ms")
            row("  TTFT p50", summary.load_ttft_p50_ms, "ms")
            row("  TTFT p90", summary.load_ttft_p90_ms, "ms")
            row("  TTFT p99", summary.load_ttft_p99_ms, "ms", style="bold red")
            row("  E2E latency p90", summary.load_e2e_p90_ms, "ms")
            row("  E2E latency p99", summary.load_e2e_p99_ms, "ms")
            row("  Token throughput", summary.load_throughput_tok_s, "tok/s")
            row("  Request throughput", summary.load_request_throughput, "req/s")
            err = summary.load_error_rate
            style = "bold red" if err and err > 0.01 else ""
            row("  Error rate", f"{(err or 0)*100:.1f}%", style=style)

        _console.print(t)

    @staticmethod
    def _stage_row(
        table: Table,
        label: str,
        ms: float | None,
        frac: float | None,
        highlight: bool = False,
    ) -> None:
        """Add one pipeline-stage row with a mini ASCII progress bar."""
        if ms is None:
            table.add_row(label, "[dim]N/A[/dim]")
            return
        bar = Reporter._mini_bar(frac or 0.0)
        pct = f"{(frac or 0)*100:.0f}%"
        style = "bold yellow" if highlight else ""
        val_str = f"{ms:,.1f} ms  {bar} {pct}"
        table.add_row(label, Text(val_str, style=style))

    @staticmethod
    def _mini_bar(frac: float, width: int = 12) -> str:
        """Render a simple block-character progress bar."""
        filled = round(frac * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def print_sweep_table(
        summaries: list[RagMetricsSummary],
        sweep_param: str | None = None,
    ) -> None:
        """Print a side-by-side comparison table for a parameter sweep.

        Column labels auto-detect the varying axes (concurrency, vdb_top_k,
        reranker_top_k). If multiple axes vary, the label combines them
        (e.g. ``conc=4 vdb=20 rr=4``). When ``sweep_param`` is given it
        overrides the auto-detection — useful for pinning the label.

        Load-test rows are suppressed when no summary carries aiperf data
        (i.e. profile-only runs).
        """
        if not summaries:
            return

        # Detect which axes vary across the summaries.
        axes = [
            ("concurrency", "conc"),
            ("vdb_top_k", "vdb"),
            ("reranker_top_k", "rr"),
        ]
        varying = (
            [(attr, prefix) for attr, prefix in axes
             if len({getattr(s, attr) for s in summaries}) > 1]
        )

        def label_for(s: RagMetricsSummary) -> str:
            if sweep_param is not None:
                return str(getattr(s, sweep_param, "?"))
            if not varying:
                # Nothing varies (e.g. n_times>1 with single point) — label by
                # iteration index to disambiguate columns.
                return f"iter#{summaries.index(s) + 1}"
            if len(varying) == 1:
                attr, _ = varying[0]
                return str(getattr(s, attr, "?"))
            return " ".join(
                f"{prefix}={getattr(s, attr)}" for attr, prefix in varying
            )

        title_axes = (
            sweep_param
            or (", ".join(a for a, _ in varying) if varying else "iterations")
        )
        has_aiperf = any(s.load_ttft_p50_ms is not None for s in summaries)

        t = Table(
            title=f"RAG-Perf Sweep — {title_axes}",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
        )

        t.add_column("Metric", style="bold", min_width=32)
        for s in summaries:
            t.add_column(label_for(s), justify="right", min_width=12)

        def sweep_row(metric_label: str, attr: str, fmt: str = "{:.0f}") -> None:
            values = []
            for s in summaries:
                obj: Any = s
                for part in attr.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                values.append(fmt.format(obj) if obj is not None else "N/A")
            t.add_row(metric_label, *values)

        if has_aiperf:
            sweep_row("TTFT p50 (ms)", "load_ttft_p50_ms")
            sweep_row("TTFT p99 (ms)", "load_ttft_p99_ms")
            sweep_row("E2E p99 (ms)", "load_e2e_p99_ms")
            sweep_row("Token throughput (tok/s)", "load_throughput_tok_s", "{:.1f}")
            sweep_row("Request throughput (req/s)", "load_request_throughput", "{:.2f}")
            sweep_row("Error rate (%)", "load_error_rate", "{:.1%}")
            t.add_section()
        sweep_row("Server RAG TTFT (ms)", "stage_breakdown.rag_ttft_ms")
        sweep_row("  Retrieval (ms)", "stage_breakdown.retrieval_ms")
        sweep_row("  Reranking (ms)", "stage_breakdown.reranking_ms")
        sweep_row("  LLM TTFT (ms)", "stage_breakdown.llm_ttft_ms")
        t.add_section()
        sweep_row("Citation count (mean)", "citation_quality.mean_count", "{:.1f}")
        sweep_row("Citation score (mean)", "citation_quality.mean_score", "{:.3f}")

        _console.print(t)

        # The "optimal throughput" / "max concurrency under p99" footer is
        # only meaningful when at least one summary carries aiperf data.
        if not has_aiperf:
            return

        best_tput_idx = max(
            range(len(summaries)),
            key=lambda i: summaries[i].load_request_throughput or 0,
            default=0,
        )
        best_p99_idx = min(
            (
                i
                for i, s in enumerate(summaries)
                if (s.load_ttft_p99_ms or 1e9) < 30_000
            ),
            default=None,
        )

        # Describe the optimal point using the auto-detected varying axes
        # (or fall back to concurrency if nothing varies).
        def describe_point(s: RagMetricsSummary) -> str:
            if not varying:
                return f"concurrency={s.concurrency}"
            return " ".join(
                f"{prefix}={getattr(s, attr)}" for attr, prefix in varying
            )

        best_tput_val = summaries[best_tput_idx].load_request_throughput
        tput_str = f"{best_tput_val:.2f} req/s" if best_tput_val is not None else "N/A"
        _console.print(
            f"\n[bold green]Optimal throughput:[/bold green] "
            f"{describe_point(summaries[best_tput_idx])}  ({tput_str})"
        )
        if best_p99_idx is not None:
            _console.print(
                f"[bold green]Best p99 TTFT < 30s:[/bold green] "
                f"{describe_point(summaries[best_p99_idx])}"
            )

    @staticmethod
    def write_markdown_report(
        summary: RagMetricsSummary,
        output_path: str | Path,
        config_yaml: str | None = None,
        sweep_summaries: list[RagMetricsSummary] | None = None,
    ) -> None:
        """Write a self-contained Markdown benchmark report."""
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# RAG-Perf Benchmark Report",
            "",
            f"Generated: {ts}  ",
            f"Collection: {', '.join(summary.collection_names) or 'N/A'}  ",
            f"vdb_top_k: {summary.vdb_top_k}  |  reranker_top_k: {summary.reranker_top_k}  ",
            "",
            "---",
            "",
            "## Pipeline Stage Breakdown (Profiling Pass)",
            "",
            "| Stage | Mean (ms) | % of TTFT |",
            "|---|---|---|",
        ]

        sb = summary.stage_breakdown
        for label, ms, frac in [
            ("Retrieval", sb.retrieval_ms, sb.retrieval_frac),
            ("Reranking", sb.reranking_ms, sb.reranking_frac),
            ("LLM TTFT", sb.llm_ttft_ms, sb.llm_frac),
        ]:
            ms_str = f"{ms:.1f}" if ms else "N/A"
            pct_str = f"{(frac or 0)*100:.0f}%" if frac else "N/A"
            botmark = (
                " ⚠ **bottleneck**" if label.lower().startswith(sb.bottleneck) else ""
            )
            lines.append(f"| {label} | {ms_str} | {pct_str}{botmark} |")

        lines += [
            "",
            f"**Server RAG TTFT (mean):** {sb.rag_ttft_ms:.1f} ms"
            if sb.rag_ttft_ms
            else "",
            f"**Bottleneck:** {sb.bottleneck.upper()}",
            "",
        ]

        if summary.load_ttft_p50_ms is not None:
            lines += [
                f"## Load-Test Results (concurrency={summary.concurrency})",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| TTFT p50 | {summary.load_ttft_p50_ms:.0f} ms |",
                f"| TTFT p99 | {summary.load_ttft_p99_ms:.0f} ms |",
                f"| E2E p99 | {summary.load_e2e_p99_ms:.0f} ms |",
                f"| Token throughput | {summary.load_throughput_tok_s:.1f} tok/s |",
                f"| Request throughput | {summary.load_request_throughput:.2f} req/s |",
                f"| Error rate | {(summary.load_error_rate or 0)*100:.1f}% |",
                "",
            ]

        if sweep_summaries:
            lines += ["## Concurrency Sweep", ""]
            concs = [str(s.concurrency) for s in sweep_summaries]
            lines.append("| Metric | " + " | ".join(concs) + " |")
            lines.append("|---" * (len(concs) + 1) + "|")
            for attr, label in [
                ("load_ttft_p50_ms", "TTFT p50 (ms)"),
                ("load_ttft_p99_ms", "TTFT p99 (ms)"),
                ("load_request_throughput", "req/s"),
            ]:
                vals = []
                for s in sweep_summaries:
                    v = getattr(s, attr, None)
                    vals.append(f"{v:.1f}" if v else "N/A")
                lines.append(f"| {label} | " + " | ".join(vals) + " |")
            lines.append("")

        if config_yaml:
            lines += [
                "## Configuration",
                "",
                "```yaml",
                textwrap.indent(config_yaml, ""),
                "```",
            ]

        Path(output_path).write_text("\n".join(lines))
        _console.print(f"[dim]Report written to {output_path}[/dim]")

    @staticmethod
    def export_json(summary: RagMetricsSummary, path: str | Path) -> None:
        """Export the summary as a structured JSON file."""

        def _as_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _as_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_as_dict(item) for item in obj]
            return obj

        Path(path).write_text(json.dumps(_as_dict(summary), indent=2, default=str))
        _console.print(f"[dim]JSON results written to {path}[/dim]")

    @staticmethod
    def export_csv(summaries: list[RagMetricsSummary], path: str | Path) -> None:
        """Export one or more summaries as a flat CSV file."""
        rows = []
        for s in summaries:
            rows.append(
                {
                    "concurrency": s.concurrency,
                    "total_requests": s.total_requests,
                    "collection_names": ",".join(s.collection_names),
                    "vdb_top_k": s.vdb_top_k,
                    "reranker_top_k": s.reranker_top_k,
                    "server_rag_ttft_ms": s.stage_breakdown.rag_ttft_ms,
                    "retrieval_ms": s.stage_breakdown.retrieval_ms,
                    "reranking_ms": s.stage_breakdown.reranking_ms,
                    "llm_ttft_ms": s.stage_breakdown.llm_ttft_ms,
                    "bottleneck": s.stage_breakdown.bottleneck,
                    "citation_count_mean": s.citation_quality.mean_count,
                    "citation_score_mean": s.citation_quality.mean_score,
                    "load_ttft_p50_ms": s.load_ttft_p50_ms,
                    "load_ttft_p90_ms": s.load_ttft_p90_ms,
                    "load_ttft_p99_ms": s.load_ttft_p99_ms,
                    "load_throughput_tok_s": s.load_throughput_tok_s,
                    "load_request_throughput": s.load_request_throughput,
                    "load_error_rate": s.load_error_rate,
                }
            )

        if not rows:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _console.print(f"[dim]CSV results written to {path}[/dim]")

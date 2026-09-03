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
Execution side of rag-perf: profiler, aiperf wrapper, and the orchestrator.

Three service classes:

- ``RagProfiler``       Async httpx + SSE parser for the profiling pass.
                         Captures per-request server-side stage timings and
                         citation metadata that aiperf cannot access.
- ``AiperfRunner``      Subprocess wrapper around aiperf for the load-test
                         phase. Translates a RunConfig into the right
                         ``aiperf profile`` invocation and reads back the
                         JSON artefact.
- ``BenchmarkRunner``   Top-level orchestrator. ``run`` computes the grid
                         from config and dispatches one point at a time. Loads
                         queries → profiling pass → aiperf load test →
                         aggregation → reporting.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from rag_perf.config import RunConfig
from rag_perf.query import QueryLoader
from rag_perf.reporting import (
    MetricsAggregator,
    ProfileRecord,
    ProfileResult,
    RagMetricsSummary,
    Reporter,
)

_console = Console()
logger = logging.getLogger(__name__)


class RagProfiler:
    """
    Direct async profiler for the RAG server.

    Sends a small number of requests via httpx and captures server-side
    pipeline stage timing, citation quality, and client-side timing that
    aiperf cannot provide on its own.
    """

    @staticmethod
    async def run(
        config: RunConfig,
        requests: list[dict[str, Any]],
        concurrency: int = 1,
        save_metadata: bool = True,
        output_dir: str | None = None,
    ) -> ProfileResult:
        """
        Send ``requests`` to the RAG server and collect per-request metadata.

        Args:
            config:        Full run configuration (target URL, timeout, etc.).
            requests:      List of fully-resolved request dicts from ``QueryLoader.load``.
            concurrency:   Number of simultaneous in-flight requests.
            save_metadata: If True, write per-request records to a JSONL file.
            output_dir:    Directory for the metadata JSONL file.

        Returns:
            ``ProfileResult`` with per-request records and optional JSONL path.
        """
        url = f"{config.target.url.rstrip('/')}/v1/generate"
        timeout = httpx.Timeout(config.target.timeout_s, connect=10.0)

        result = ProfileResult()
        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            tasks = [
                RagProfiler._send_one(client, url, req, semaphore) for req in requests
            ]
            result.records = await asyncio.gather(*tasks)

        if save_metadata and result.successful:
            out_dir = Path(output_dir or config.output.dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = out_dir / "profiler_records.jsonl"
            RagProfiler._write_records_jsonl(result.records, jsonl_path)
            result.metadata_jsonl_path = str(jsonl_path)

        return result

    @staticmethod
    async def _send_one(
        client: httpx.AsyncClient,
        url: str,
        request_body: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> ProfileRecord:
        """Send a single request and collect all metrics from the SSE stream."""
        query = next(
            (
                m["content"]
                for m in request_body.get("messages", [])
                if m["role"] == "user"
            ),
            "",
        )

        async with semaphore:
            t_send = time.perf_counter()
            ttft_ms: float | None = None
            output_tokens = 0
            input_tokens = 0
            server_metrics: dict[str, float | None] = {}
            citation_count = 0
            citation_scores: list[float] = []
            citations_raw: list[dict[str, Any]] = []
            full_text: list[str] = []

            try:
                async with client.stream(
                    "POST",
                    url,
                    json=request_body,
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.startswith("data:"):
                            continue
                        data_str = raw_line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")

                        if content and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_send) * 1000.0

                        if content:
                            full_text.append(content)

                        # Citations land on the first chunk that carries them
                        # (server attaches them alongside the initial empty
                        # content delta). Latch on the first non-empty
                        # citations payload so it survives all subsequent chunks.
                        if citation_count == 0:
                            cites_obj = chunk.get("citations") or {}
                            cites_list = cites_obj.get("results") or []
                            if cites_list:
                                citation_count = len(cites_list)
                                citation_scores = [
                                    r["score"]
                                    for r in cites_list
                                    if isinstance(r, dict) and "score" in r
                                ]
                                citations_raw = cites_list

                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason is not None:
                            usage = chunk.get("usage") or {}
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)

                            raw_sm = chunk.get("metrics") or {}
                            server_metrics = {
                                "rag_ttft_ms": raw_sm.get("rag_ttft_ms"),
                                "llm_ttft_ms": raw_sm.get("llm_ttft_ms"),
                                "retrieval_time_ms": raw_sm.get("retrieval_time_ms"),
                                "reranking_time_ms": raw_sm.get(
                                    "context_reranker_time_ms"
                                ),
                                "llm_generation_time_ms": raw_sm.get(
                                    "llm_generation_time_ms"
                                ),
                            }

                e2e_ms = (time.perf_counter() - t_send) * 1000.0

                return ProfileRecord(
                    query=query,
                    client_ttft_ms=ttft_ms or e2e_ms,
                    client_e2e_ms=e2e_ms,
                    output_tokens=output_tokens,
                    input_tokens=input_tokens,
                    server_metrics=server_metrics,
                    citation_count=citation_count,
                    citation_scores=citation_scores,
                    citations_raw=citations_raw,
                )

            except Exception as exc:
                e2e_ms = (time.perf_counter() - t_send) * 1000.0
                return ProfileRecord(
                    query=query,
                    client_ttft_ms=e2e_ms,
                    client_e2e_ms=e2e_ms,
                    output_tokens=0,
                    input_tokens=0,
                    server_metrics={},
                    citation_count=0,
                    citation_scores=[],
                    citations_raw=[],
                    error=str(exc),
                )

    @staticmethod
    def _write_records_jsonl(records: list[ProfileRecord], path: Path) -> None:
        """Serialise profiler records to a JSONL file."""
        with path.open("w") as fh:
            for rec in records:
                obj = {
                    "query": rec.query,
                    "client_ttft_ms": rec.client_ttft_ms,
                    "client_e2e_ms": rec.client_e2e_ms,
                    "output_tokens": rec.output_tokens,
                    "input_tokens": rec.input_tokens,
                    "server_metrics": rec.server_metrics,
                    "citation_count": rec.citation_count,
                    "citation_scores": rec.citation_scores,
                    "citations": rec.citations_raw,
                    "error": rec.error,
                }
                fh.write(json.dumps(obj) + "\n")


class AiperfRunner:
    """
    Thin subprocess wrapper around aiperf for the load-test phase.

    Translates a ``RunConfig`` into the correct ``aiperf profile`` CLI arguments,
    invokes aiperf as a subprocess, and reads back its structured JSON output.
    """

    _ENDPOINT_TYPE = "nvidia_rag"
    _ENDPOINT_FALLBACK = "rag_perf.plugin.nvidia_rag:NvidiaRagEndpoint"

    @staticmethod
    def run_rag_on(
        config: RunConfig,
        queries_jsonl: str,
        artifact_dir: str,
        concurrency: int | None = None,
        total_requests: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG-On load test via aiperf.

        Args:
            config:          Full run configuration.
            queries_jsonl:   Path to the ShareGPT-format JSONL query file.
            artifact_dir:    Directory where aiperf writes its output artefacts.
            concurrency:     Override ``config.load.concurrency`` for this run.
            total_requests:  Override ``config.load.total_requests`` for this run.

        Returns:
            The contents of aiperf's ``profile_export_aiperf.json`` as a dict,
            or an empty dict if the file was not produced.
        """
        conc = concurrency or config.load.concurrency
        n_req = total_requests or config.load.total_requests
        url = config.target.url.rstrip("/")

        cmd = AiperfRunner._base_aiperf_cmd(
            endpoint_type=AiperfRunner._ENDPOINT_TYPE,
            url=url,
            model=config.model_name,
            concurrency=conc,
            total_requests=n_req,
            warmup_requests=config.load.warmup_requests,
            artifact_dir=artifact_dir,
            queries_jsonl=queries_jsonl,
            timeout_s=config.target.timeout_s,
            tokenizer=config.tokenizer,
        )

        rag = config.rag
        gen = config.generation
        cmd += [
            "--extra-inputs",
            f"collection_names:{json.dumps(rag.collection_names)}",
            "--extra-inputs",
            f"vdb_top_k:{rag.vdb_top_k}",
            "--extra-inputs",
            f"reranker_top_k:{rag.reranker_top_k}",
            "--extra-inputs",
            f"use_knowledge_base:{str(rag.use_knowledge_base).lower()}",
            "--extra-inputs",
            f"enable_reranker:{str(rag.enable_reranker).lower()}",
            "--extra-inputs",
            f"enable_citations:{str(rag.enable_citations).lower()}",
            "--extra-inputs",
            f"confidence_threshold:{rag.confidence_threshold}",
            "--extra-inputs",
            f"max_tokens:{gen.max_tokens}",
            "--extra-inputs",
            f"temperature:{gen.temperature}",
        ]
        if gen.min_tokens is not None:
            cmd += ["--extra-inputs", f"min_tokens:{gen.min_tokens}"]
        if gen.ignore_eos:
            cmd += ["--extra-inputs", "ignore_eos:true"]

        AiperfRunner._run(cmd)
        return AiperfRunner._read_aiperf_json(artifact_dir)

    @staticmethod
    def _base_aiperf_cmd(
        endpoint_type: str,
        url: str,
        model: str,
        concurrency: int,
        total_requests: int,
        warmup_requests: int,
        artifact_dir: str,
        queries_jsonl: str,
        timeout_s: int,
        tokenizer: str = "",
    ) -> list[str]:
        """Build the common aiperf command prefix for RAG-On runs."""
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "aiperf",
            "profile",
            "-m",
            model,
            "--endpoint-type",
            endpoint_type,
            "--streaming",
            "-u",
            url,
            "--concurrency",
            str(concurrency),
            "--request-count",
            str(total_requests),
            "--warmup-request-count",
            str(warmup_requests),
            "--artifact-dir",
            artifact_dir,
            "--custom-dataset-type",
            "single-turn",
            "--input-file",
            queries_jsonl,
            "--ui",
            "simple",
            "--max-workers",
            str(concurrency),
        ]

        if tokenizer:
            cmd += ["--tokenizer", tokenizer]
        else:
            cmd += ["--use-server-token-count"]

        return cmd

    @staticmethod
    def _run(cmd: list[str]) -> None:
        """Execute an aiperf command, streaming its output to the user's terminal."""
        env = os.environ.copy()
        existing = env.get("PYTHONWARNINGS", "")
        suppress = "ignore::UserWarning:pydantic.main"
        env["PYTHONWARNINGS"] = f"{existing},{suppress}" if existing else suppress

        cmd_str = shlex.join(cmd)
        logger.debug("aiperf command: %s", cmd_str)
        _console.print(f"\n  [dim]$ {cmd_str}[/dim]\n")

        proc = subprocess.run(cmd, env=env, check=True)
        _ = proc

    @staticmethod
    def _read_aiperf_json(artifact_dir: str) -> dict[str, Any]:
        """Read aiperf's JSON results file from the artifact directory."""
        json_path = Path(artifact_dir) / "profile_export_aiperf.json"
        if not json_path.exists():
            return {}
        with json_path.open() as fh:
            return json.load(fh)


class BenchmarkRunner:
    """
    Declarative benchmark orchestrator.

    Computes the grid implied by the config (Cartesian of ``load.concurrency``
    × ``sweep.vdb_top_k`` × ``sweep.reranker_top_k``), then for each grid point
    runs a profiling pass and — unless ``aiperf.enabled`` is False — an aiperf
    load test. Output layout adapts to the grid size:

    - **single point, n_times = 1** → flat ``run_<ts>/{profiling,aiperf_rag_on}/``
      with ``report.md`` / ``results.json`` / ``results.csv`` directly under
      the run directory.
    - **multiple points or n_times > 1** → nested ``run_<ts>/iter_<i>/<point>/…``
      with the same per-point subdirs and aggregate report files at the run root.

    Profile-only mode (``aiperf.enabled = false``) skips the aiperf phase and
    writes ``profile_report.md`` / ``profile_results.json`` instead.
    """

    @staticmethod
    def run(config: RunConfig) -> list[RagMetricsSummary]:
        """Run the benchmark described by *config*. Returns one summary per grid point.

        The CLI calls this; programmatic callers can do the same. The returned
        list is ordered by iteration then by grid point (concurrency-major).
        """
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(config.output.dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        points = list(BenchmarkRunner._iter_grid_points(config))
        n_points = len(points)
        n_times = config.load.iterations
        is_flat = n_points == 1 and n_times == 1

        if not is_flat:
            _console.rule(
                f"[bold cyan]RAG-Perf — {n_points} point(s) × {n_times} iteration(s) "
                f"= {n_points * n_times} total[/bold cyan]"
            )

        # Load enough queries to feed whichever pass needs more.
        # The profiling pass takes load.profile_requests; aiperf takes
        # total_requests + warmup_requests.
        query_count = max(
            config.load.profile_requests,
            config.load.total_requests + config.load.warmup_requests,
        )
        queries = QueryLoader.load(
            config.input,
            config.rag,
            config.generation,
            count=query_count,
        )
        shared_queries_jsonl = (
            QueryLoader.make_temp(queries) if config.aiperf.enabled else ""
        )

        all_summaries: list[RagMetricsSummary] = []

        for it in range(1, n_times + 1):
            if is_flat:
                iter_dir = run_dir
            else:
                iter_dir = run_dir / f"iter_{it}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                if n_times > 1:
                    _console.rule(f"[bold cyan]Iteration {it}/{n_times}[/bold cyan]")

            iter_summaries: list[RagMetricsSummary] = []
            for idx, point_config in enumerate(points, 1):
                conc = point_config.load.concurrency
                vdb_k = point_config.rag.vdb_top_k
                rr_k = point_config.rag.reranker_top_k

                if is_flat:
                    point_dir = iter_dir
                else:
                    # ISL is meaningful only for synthetic queries (file-based
                    # queries have varying lengths); use "var" as the label
                    # in the point-dir name when input is file-driven.
                    isl: int | str = (
                        point_config.input.synthetic.min_query_tokens
                        if point_config.input.synthetic is not None
                        else "var"
                    )
                    osl = point_config.generation.max_tokens
                    model_clean = point_config.model_name.replace("/", "-")
                    out = point_config.output
                    point_name = (
                        f"CR:{conc}"
                        f"_ISL:{isl}"
                        f"_OSL:{osl}"
                        f"_VDB-K:{vdb_k}"
                        f"_RERANKER-K:{rr_k}"
                        f"_Model:{model_clean}"
                    )
                    if out.cluster:
                        point_name += f"_Cluster:{out.cluster}"
                    if out.gpu:
                        point_name += f"_GPU:{out.gpu}"
                    if out.experiment_name:
                        point_name += f"_Experiment:{out.experiment_name}"
                    point_dir = iter_dir / point_name
                    point_dir.mkdir(parents=True, exist_ok=True)
                    _console.rule(
                        f"[bold]Point {idx}/{n_points}: "
                        f"conc={conc}  vdb_top_k={vdb_k}  rr_top_k={rr_k}[/bold]"
                    )

                summary = BenchmarkRunner._run_point(
                    config=point_config,
                    queries=queries,
                    shared_queries_jsonl=shared_queries_jsonl,
                    point_dir=str(point_dir),
                )
                summary.collection_names = list(point_config.rag.collection_names)
                summary.vdb_top_k = vdb_k
                summary.reranker_top_k = rr_k
                iter_summaries.append(summary)

                if (
                    config.load.sleep_between_points_s > 0
                    and idx < n_points
                ):
                    _console.print(
                        f"  [dim]Sleeping {config.load.sleep_between_points_s}s before next point…[/dim]"
                    )
                    time.sleep(config.load.sleep_between_points_s)

            # Per-iteration table is suppressed when n_times>1 because the cli
            # prints a single aggregate table across all iterations at the end.
            # Showing the per-iter table here would duplicate information.
            if not is_flat and len(iter_summaries) > 1 and n_times == 1:
                Reporter.print_sweep_table(iter_summaries)
            all_summaries.extend(iter_summaries)

        if not is_flat:
            _console.rule("[bold cyan]Run Complete[/bold cyan]")

        BenchmarkRunner._write_aggregate_outputs(
            config=config, run_dir=run_dir, all_summaries=all_summaries, is_flat=is_flat
        )
        return all_summaries

    @staticmethod
    def _write_aggregate_outputs(
        config: RunConfig,
        run_dir: Path,
        all_summaries: list[RagMetricsSummary],
        is_flat: bool,
    ) -> None:
        """Write top-level CSV / JSON / Markdown for the run.

        Naming differs by mode:
        - aiperf disabled → ``profile_results.{json,csv}`` + ``profile_report.md``
        - aiperf enabled, single point → ``results.{json,csv}`` + ``report.md``
        - aiperf enabled, multi-point  → ``results.{json,csv}`` + ``report.md``
          (CSV holds one row per point)
        """
        if not all_summaries:
            return

        prefix = "profile_" if not config.aiperf.enabled else ""

        if "csv" in config.output.formats:
            Reporter.export_csv(all_summaries, run_dir / f"{prefix}results.csv")
        if "json" in config.output.formats:
            payload = (
                all_summaries[0] if len(all_summaries) == 1 else all_summaries
            )
            Reporter.export_json(payload, run_dir / f"{prefix}results.json")  # type: ignore[arg-type]
        if config.output.markdown_report:
            Reporter.write_markdown_report(
                all_summaries[0],
                run_dir / f"{prefix}report.md",
                config_yaml=config.to_yaml_str(),
                sweep_summaries=None if is_flat else all_summaries,
            )

    @staticmethod
    def _run_point(
        config: RunConfig,
        queries: list[dict],
        shared_queries_jsonl: str,
        point_dir: str,
    ) -> RagMetricsSummary:
        """Execute one benchmark point: profiling pass, optionally + aiperf load test."""
        profiling_dir = Path(point_dir) / "profiling"
        aiperf_rag_dir = Path(point_dir) / "aiperf_rag_on"

        _console.print(
            "[cyan]→ Running profiling pass (collecting server-side metrics)...[/cyan]"
        )

        # _iter_grid_points always emits scalar; assert that invariant.
        assert isinstance(config.load.concurrency, int), (
            f"_run_point expected scalar concurrency, got {type(config.load.concurrency).__name__}"
        )
        point_concurrency = config.load.concurrency

        profile_queries = queries[: config.load.profile_requests]
        profile_result = asyncio.run(
            RagProfiler.run(
                config=config,
                requests=profile_queries,
                concurrency=min(2, point_concurrency),
                save_metadata=True,
                output_dir=str(profiling_dir),
            )
        )

        if profile_result.failed:
            _console.print(
                f"[yellow]⚠ {len(profile_result.failed)} profiling requests failed[/yellow]"
            )

        summary = MetricsAggregator.from_profiler(profile_result.successful)
        # Stamp the point's identifying axis values onto the summary regardless
        # of whether aiperf runs — this is what the comparison table reads as
        # column labels.
        summary.concurrency = point_concurrency
        summary.profile_requests_failed = len(profile_result.failed)
        summary.profile_requests_total = len(profile_queries)

        if not config.aiperf.enabled:
            return summary

        _console.print(
            f"[cyan]→ Running aiperf load test "
            f"(concurrency={point_concurrency}, "
            f"requests={config.load.total_requests})...[/cyan]"
        )

        aiperf_result = AiperfRunner.run_rag_on(
            config=config,
            queries_jsonl=shared_queries_jsonl,
            artifact_dir=str(aiperf_rag_dir),
            concurrency=point_concurrency,
            total_requests=config.load.total_requests,
        )

        summary = MetricsAggregator.enrich_with_aiperf(
            summary,
            aiperf_result,
            concurrency=point_concurrency,
            total_requests=config.load.total_requests,
        )
        return summary

    @staticmethod
    def _iter_grid_points(config: RunConfig) -> Iterator[RunConfig]:
        """Yield one ``RunConfig`` per grid point in the Cartesian product of all axes.

        Each axis (``load.concurrency``, ``rag.vdb_top_k``, ``rag.reranker_top_k``)
        is scalar-or-list — the helpers ``concurrency_list`` /
        ``vdb_top_k_list`` / ``reranker_top_k_list`` normalise to a list so
        ``itertools.product`` can treat them uniformly.
        """
        for conc, vdb_k, rr_k in itertools.product(
            config.load.concurrency_list,
            config.rag.vdb_top_k_list,
            config.rag.reranker_top_k_list,
        ):
            yield config.with_overrides(
                load__concurrency=conc,
                rag__vdb_top_k=vdb_k,
                rag__reranker_top_k=rr_k,
            )

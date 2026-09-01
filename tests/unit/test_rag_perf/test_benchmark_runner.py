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

"""Tests for ``rag_perf.runner.BenchmarkRunner.run`` — single-point, sweep, and profile-only flows."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from rag_perf.config import (
    InputConfig,
    InputSource,
    LoadConfig,
    OutputConfig,
    RagParams,
    RunConfig,
    TargetConfig,
)
from rag_perf.reporting import ProfileResult, RagMetricsSummary
from rag_perf.runner import BenchmarkRunner


def _make_config(output_dir: Path, **overrides) -> RunConfig:
    """Build a minimal RunConfig that BenchmarkRunner can drive."""
    return RunConfig(
        target=TargetConfig(url="http://rag:8081"),
        load=LoadConfig(concurrency=4, total_requests=20, warmup_requests=2),
        rag=RagParams(
            collection_names=["multimodal_data"], vdb_top_k=10, reranker_top_k=2
        ),
        input=InputConfig(file="examples/queries.jsonl"),
        output=OutputConfig(
            dir=str(output_dir),
            formats=["json", "csv"],
            markdown_report=True,
        ),
        **overrides,
    )


def _enter_runtime_patches(stack: ExitStack) -> None:
    """Patch the IO/network seams so ``BenchmarkRunner.run`` executes purely in-memory.

    Reporter methods are patched separately by callers that want to inspect
    or assert on them; everything else (query loading, profiler, aiperf) is
    blanket-stubbed here.
    """
    stack.enter_context(patch("rag_perf.runner.QueryLoader.load", return_value=[]))
    stack.enter_context(
        patch("rag_perf.runner.QueryLoader.make_temp", return_value="/tmp/q.jsonl")
    )
    stack.enter_context(patch("rag_perf.runner.RagProfiler.run", return_value=None))
    stack.enter_context(
        patch("rag_perf.runner.asyncio.run", return_value=ProfileResult(records=[]))
    )
    stack.enter_context(
        patch("rag_perf.runner.AiperfRunner.run_rag_on", return_value={})
    )
    stack.enter_context(
        patch(
            "rag_perf.runner.MetricsAggregator.from_profiler",
            return_value=RagMetricsSummary(),
        )
    )
    stack.enter_context(
        patch(
            "rag_perf.runner.MetricsAggregator.enrich_with_aiperf",
            side_effect=lambda s, *a, **k: s,
        )
    )
    stack.enter_context(patch("rag_perf.runner.Reporter.print_sweep_table"))


class TestBenchmarkRunner:
    """Tests for ``rag_perf.runner.BenchmarkRunner.run``."""

    # ── Single-point (flat layout) ───────────────────────────────────────────

    def test_run_single_point_creates_flat_run_dir(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            summaries = BenchmarkRunner.run(cfg)

        run_dirs = [
            d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("run_")
        ]
        assert len(run_dirs) == 1
        # No iter_<i>/ subdirectory in the flat layout.
        assert not any(c.name.startswith("iter_") for c in run_dirs[0].iterdir())
        assert len(summaries) == 1

    def test_run_single_threads_collection_metadata_into_summary(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            summaries = BenchmarkRunner.run(cfg)

        assert summaries[0].collection_names == ["multimodal_data"]
        assert summaries[0].vdb_top_k == 10
        assert summaries[0].reranker_top_k == 2

    def test_run_single_writes_csv_only_when_format_requested(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(
            output__formats=["csv"], output__markdown_report=False
        )
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            mock_csv = stack.enter_context(
                patch("rag_perf.runner.Reporter.export_csv")
            )
            mock_json = stack.enter_context(
                patch("rag_perf.runner.Reporter.export_json")
            )
            mock_md = stack.enter_context(
                patch("rag_perf.runner.Reporter.write_markdown_report")
            )
            BenchmarkRunner.run(cfg)

        mock_csv.assert_called_once()
        mock_json.assert_not_called()
        mock_md.assert_not_called()

    # ── Profile-only mode ────────────────────────────────────────────────────

    def test_profile_only_skips_aiperf_and_uses_profile_filenames(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(aiperf__enabled=False)
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            mock_json = stack.enter_context(
                patch("rag_perf.runner.Reporter.export_json")
            )
            mock_md = stack.enter_context(
                patch("rag_perf.runner.Reporter.write_markdown_report")
            )
            mock_aiperf = stack.enter_context(
                patch("rag_perf.runner.AiperfRunner.run_rag_on")
            )
            BenchmarkRunner.run(cfg)

        # aiperf.enabled=False → AiperfRunner is never called.
        mock_aiperf.assert_not_called()
        # Output files are prefixed with profile_.
        assert Path(mock_json.call_args[0][1]).name == "profile_results.json"
        assert Path(mock_md.call_args[0][1]).name == "profile_report.md"

    # ── _iter_grid_points ────────────────────────────────────────────────────

    def test_iter_grid_points_full_cartesian(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(
            load__concurrency=[1, 4],
            rag__vdb_top_k=[10, 20],
            rag__reranker_top_k=[2, 4],
        )
        points = list(BenchmarkRunner._iter_grid_points(cfg))
        # 2 × 2 × 2 = 8 points.
        assert len(points) == 8
        seen = {
            (p.load.concurrency, p.rag.vdb_top_k, p.rag.reranker_top_k) for p in points
        }
        assert len(seen) == 8

    def test_iter_grid_points_uses_rag_defaults_when_axis_unset(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(load__concurrency=[1, 4, 8])
        points = list(BenchmarkRunner._iter_grid_points(cfg))
        assert len(points) == 3
        for p in points:
            assert p.rag.vdb_top_k == 10
            assert p.rag.reranker_top_k == 2

    def test_iter_grid_points_scalar_concurrency_yields_one(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(load__concurrency=8)
        points = list(BenchmarkRunner._iter_grid_points(cfg))
        assert len(points) == 1
        assert points[0].load.concurrency == 8

    # ── Sweep (nested layout) ────────────────────────────────────────────────

    def test_sweep_writes_one_summary_per_point(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(load__concurrency=[1, 4, 8])
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            mock_csv = stack.enter_context(
                patch("rag_perf.runner.Reporter.export_csv")
            )
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            summaries = BenchmarkRunner.run(cfg)

        assert len(summaries) == 3
        # Aggregate CSV is written once with all three summaries.
        mock_csv.assert_called_once()
        args, _ = mock_csv.call_args
        assert len(args[0]) == 3

    def test_sweep_uses_nested_iter_directory(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(load__concurrency=[1, 4])
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            BenchmarkRunner.run(cfg)

        run_dir = next(
            d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("run_")
        )
        iter_dirs = [d for d in run_dir.iterdir() if d.name.startswith("iter_")]
        assert [d.name for d in iter_dirs] == ["iter_1"]

    def test_sweep_n_times_creates_iter_per_repetition(self, tmp_path: Path):
        cfg = _make_config(tmp_path).with_overrides(
            load__concurrency=[1, 4], load__iterations=3
        )
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            summaries = BenchmarkRunner.run(cfg)

        # 2 concurrencies × 3 iterations = 6 summaries.
        assert len(summaries) == 6
        run_dir = next(d for d in tmp_path.iterdir() if d.name.startswith("run_"))
        iter_dirs = sorted(
            d.name for d in run_dir.iterdir() if d.name.startswith("iter_")
        )
        assert iter_dirs == ["iter_1", "iter_2", "iter_3"]

    def test_sweep_point_dir_naming_includes_metadata(self, tmp_path: Path):
        # Use n_times=2 to force nested layout (single-point + n_times=1 is flat).
        cfg = _make_config(tmp_path).with_overrides(
            load__concurrency=[8],
            rag__vdb_top_k=[20],
            load__iterations=2,
            output__cluster="dgxa100",
            output__gpu="A100",
            output__experiment_name="baseline",
        )
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            stack.enter_context(patch("rag_perf.runner.Reporter.export_csv"))
            stack.enter_context(patch("rag_perf.runner.Reporter.export_json"))
            stack.enter_context(patch("rag_perf.runner.Reporter.write_markdown_report"))
            BenchmarkRunner.run(cfg)

        run_dir = next(d for d in tmp_path.iterdir() if d.name.startswith("run_"))
        iter_dir = run_dir / "iter_1"
        point_dirs = [d.name for d in iter_dir.iterdir()]
        assert len(point_dirs) == 1
        name = point_dirs[0]
        assert "CR:8" in name
        assert "VDB-K:20" in name
        assert "Cluster:dgxa100" in name
        assert "GPU:A100" in name
        assert "Experiment:baseline" in name

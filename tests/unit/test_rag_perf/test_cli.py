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

"""Tests for ``rag_perf.cli`` — config-only CLI surface."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from rag_perf.cli import _load_config, main
from rag_perf.reporting import RagMetricsSummary


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid RunConfig YAML and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """\
        target:
          url: "http://test:8081"
        load:
          concurrency: 4
          total_requests: 50
          warmup_requests: 5
        rag:
          collection_names: ["multimodal_data"]
          vdb_top_k: 10
          reranker_top_k: 2
        input:
          file: "examples/queries.jsonl"
        output:
          dir: "./out"
          formats: [json]
          markdown_report: false
        """
        )
    )
    return path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ────────────────────────────────────────────────────────────────────────────
# Help / version / required flags
# ────────────────────────────────────────────────────────────────────────────


class TestCLIHelp:
    """Tests for ``rag-perf --help``."""

    def test_help_shows_only_config_and_help(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # Only --config, --help, --version should be present.
        assert "--config" in result.output
        assert "--help" in result.output
        assert "--version" in result.output
        # Removed flags must not appear.
        for stale in (
            "--url",
            "--collection",
            "--vdb-top-k",
            "--reranker-top-k",
            "--max-tokens",
            "--queries",
            "--output-dir",
            "--concurrency",
            "--requests",
            "--warmup",
            "--profile-only",
        ):
            assert stale not in result.output, (
                f"Unexpected stale flag {stale!r} in --help output"
            )

    def test_missing_required_config_exits_nonzero(self, runner: CliRunner):
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "config" in result.output.lower()


# ────────────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────────────


class TestLoadConfig:
    """Tests for ``_load_config`` — pure YAML loader (no overrides)."""

    def test_load_config_returns_runconfig(self, minimal_config_yaml: Path):
        cfg = _load_config(str(minimal_config_yaml))
        assert cfg.target.url == "http://test:8081"
        assert cfg.load.concurrency == 4
        assert cfg.rag.collection_names == ["multimodal_data"]

    def test_load_config_missing_file_aborts(self, tmp_path: Path):
        with pytest.raises(click.Abort):
            _load_config(str(tmp_path / "nope.yaml"))

    def test_load_config_invalid_yaml_aborts(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("rag:\n  vdb_top_k: 9999\n")  # out of range
        with pytest.raises(click.Abort):
            _load_config(str(bad))


# ────────────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────────────


class TestCLIDispatch:
    """End-to-end Click invocations against the unified ``main`` command."""

    def test_basic_invocation_calls_run(
        self, runner: CliRunner, minimal_config_yaml: Path
    ):
        with (
            patch(
                "rag_perf.cli.BenchmarkRunner.run",
                return_value=[RagMetricsSummary()],
            ) as mock_run,
            patch("rag_perf.cli.Reporter.print_summary"),
        ):
            result = runner.invoke(main, ["-c", str(minimal_config_yaml)])

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()

    def test_missing_config_exits_nonzero(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(main, ["-c", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

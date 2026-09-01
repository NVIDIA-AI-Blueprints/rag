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

"""Tests for ``rag_perf.runner.AiperfRunner``: command construction and JSON read-back."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_perf.config import (
    GenerationParams,
    LoadConfig,
    RagParams,
    RunConfig,
    TargetConfig,
)
from rag_perf.runner import AiperfRunner


class TestAiperfRunner:
    """Tests for ``rag_perf.runner.AiperfRunner``."""

    # ── _base_aiperf_cmd: command construction ───────────────────────────────

    def test_base_aiperf_cmd_includes_required_flags(self, tmp_path: Path):
        cmd = AiperfRunner._base_aiperf_cmd(
            endpoint_type="nvidia_rag",
            url="http://rag:8081",
            model="nvidia/test-model",
            concurrency=8,
            total_requests=200,
            warmup_requests=10,
            artifact_dir=str(tmp_path),
            queries_jsonl="/tmp/queries.jsonl",
            timeout_s=300,
        )
        assert "-m" in cmd and "nvidia/test-model" in cmd
        assert "--endpoint-type" in cmd and "nvidia_rag" in cmd
        assert "-u" in cmd and "http://rag:8081" in cmd
        assert "--concurrency" in cmd and "8" in cmd
        assert "--request-count" in cmd and "200" in cmd
        assert "--warmup-request-count" in cmd and "10" in cmd
        assert "--input-file" in cmd and "/tmp/queries.jsonl" in cmd
        assert "--streaming" in cmd

    def test_base_aiperf_cmd_uses_tokenizer_when_provided(self, tmp_path: Path):
        cmd = AiperfRunner._base_aiperf_cmd(
            endpoint_type="nvidia_rag",
            url="http://x",
            model="m",
            concurrency=1,
            total_requests=10,
            warmup_requests=0,
            artifact_dir=str(tmp_path),
            queries_jsonl="/tmp/q.jsonl",
            timeout_s=60,
            tokenizer="hf/llama-3",
        )
        assert "--tokenizer" in cmd
        assert "hf/llama-3" in cmd
        assert "--use-server-token-count" not in cmd

    def test_base_aiperf_cmd_falls_back_to_server_token_count(self, tmp_path: Path):
        cmd = AiperfRunner._base_aiperf_cmd(
            endpoint_type="nvidia_rag",
            url="http://x",
            model="m",
            concurrency=1,
            total_requests=10,
            warmup_requests=0,
            artifact_dir=str(tmp_path),
            queries_jsonl="/tmp/q.jsonl",
            timeout_s=60,
            tokenizer="",
        )
        assert "--use-server-token-count" in cmd
        assert "--tokenizer" not in cmd

    def test_base_aiperf_cmd_creates_artifact_dir(self, tmp_path: Path):
        artifact_dir = tmp_path / "artifacts" / "subdir"
        AiperfRunner._base_aiperf_cmd(
            endpoint_type="nvidia_rag",
            url="http://x",
            model="m",
            concurrency=1,
            total_requests=10,
            warmup_requests=0,
            artifact_dir=str(artifact_dir),
            queries_jsonl="/tmp/q.jsonl",
            timeout_s=60,
        )
        assert artifact_dir.exists()

    # ── run_rag_on: full command assembly + extra-inputs ─────────────────────

    def test_run_rag_on_passes_rag_params_as_extra_inputs(self, tmp_path: Path):
        config = RunConfig(
            target=TargetConfig(url="http://rag:8081"),
            load=LoadConfig(concurrency=4, total_requests=50, warmup_requests=5),
            rag=RagParams(
                collection_names=["multimodal_data"],
                vdb_top_k=20,
                reranker_top_k=4,
                use_knowledge_base=True,
                enable_reranker=True,
                enable_citations=False,
                confidence_threshold=0.3,
            ),
            generation=GenerationParams(max_tokens=128, temperature=0.0),
        )

        captured: dict = {}

        def fake_run(cmd):
            captured["cmd"] = cmd

        # Pre-write the JSON artefact aiperf would normally produce so the read-back
        # has something to load. Real aiperf is mocked at the _run boundary.
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        (artifact_dir / "profile_export_aiperf.json").write_text(
            json.dumps({"time_to_first_token": {"avg": 100.0}})
        )

        with patch.object(AiperfRunner, "_run", new=fake_run):
            result = AiperfRunner.run_rag_on(
                config=config,
                queries_jsonl="/tmp/q.jsonl",
                artifact_dir=str(artifact_dir),
            )

        cmd = captured["cmd"]
        joined = " ".join(cmd)

        # extra-inputs for RAG fields must all appear
        assert "collection_names:" in joined
        assert '["multimodal_data"]' in joined
        assert "vdb_top_k:20" in joined
        assert "reranker_top_k:4" in joined
        assert "use_knowledge_base:true" in joined
        assert "enable_citations:false" in joined
        assert "confidence_threshold:0.3" in joined
        assert "max_tokens:128" in joined
        assert "temperature:0.0" in joined

        # Concurrency and request-count come from LoadConfig.
        assert "--concurrency" in cmd
        assert "4" in cmd

        # JSON was read back.
        assert result["time_to_first_token"]["avg"] == 100.0

    def test_run_rag_on_omits_min_tokens_when_none(self, tmp_path: Path):
        config = RunConfig(
            target=TargetConfig(url="http://x"),
            generation=GenerationParams(
                max_tokens=64, min_tokens=None, ignore_eos=False
            ),
        )
        captured: dict = {}

        def fake_run(cmd):
            captured["cmd"] = cmd

        artifact_dir = tmp_path / "art"
        artifact_dir.mkdir()
        (artifact_dir / "profile_export_aiperf.json").write_text("{}")

        with patch.object(AiperfRunner, "_run", new=fake_run):
            AiperfRunner.run_rag_on(config, "/tmp/q.jsonl", str(artifact_dir))

        joined = " ".join(captured["cmd"])
        assert "min_tokens:" not in joined
        assert "ignore_eos:" not in joined

    def test_run_rag_on_includes_min_tokens_and_ignore_eos(self, tmp_path: Path):
        config = RunConfig(
            target=TargetConfig(url="http://x"),
            generation=GenerationParams(
                max_tokens=128, min_tokens=128, ignore_eos=True
            ),
        )
        captured: dict = {}

        def fake_run(cmd):
            captured["cmd"] = cmd

        artifact_dir = tmp_path / "art"
        artifact_dir.mkdir()
        (artifact_dir / "profile_export_aiperf.json").write_text("{}")

        with patch.object(AiperfRunner, "_run", new=fake_run):
            AiperfRunner.run_rag_on(config, "/tmp/q.jsonl", str(artifact_dir))

        joined = " ".join(captured["cmd"])
        assert "min_tokens:128" in joined
        assert "ignore_eos:true" in joined

    def test_run_rag_on_concurrency_override_takes_precedence(self, tmp_path: Path):
        config = RunConfig(
            target=TargetConfig(url="http://x"),
            load=LoadConfig(concurrency=4, total_requests=50),
        )
        captured: dict = {}

        def fake_run(cmd):
            captured["cmd"] = cmd

        artifact_dir = tmp_path / "art"
        artifact_dir.mkdir()
        (artifact_dir / "profile_export_aiperf.json").write_text("{}")

        with patch.object(AiperfRunner, "_run", new=fake_run):
            AiperfRunner.run_rag_on(
                config,
                "/tmp/q.jsonl",
                str(artifact_dir),
                concurrency=32,
                total_requests=500,
            )

        cmd = captured["cmd"]
        # Passed-in overrides win over config values.
        idx = cmd.index("--concurrency")
        assert cmd[idx + 1] == "32"
        idx = cmd.index("--request-count")
        assert cmd[idx + 1] == "500"

    # ── _read_aiperf_json: artefact loading ──────────────────────────────────

    def test_read_aiperf_json_returns_empty_when_missing(self, tmp_path: Path):
        result = AiperfRunner._read_aiperf_json(str(tmp_path))
        assert result == {}

    def test_read_aiperf_json_loads_from_artifact_dir(self, tmp_path: Path):
        payload = {"time_to_first_token": {"avg": 3200.0, "p99": 12000.0}}
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(payload))
        assert AiperfRunner._read_aiperf_json(str(tmp_path)) == payload

    # ── _run: subprocess invocation ──────────────────────────────────────────

    def test_run_invokes_subprocess_with_command(self):
        with patch("rag_perf.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            AiperfRunner._run(["echo", "hello"])
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0] == ["echo", "hello"]
            assert kwargs.get("check") is True

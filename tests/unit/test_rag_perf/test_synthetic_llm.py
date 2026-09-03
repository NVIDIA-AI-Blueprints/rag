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

"""Tests for ``rag_perf.query.SyntheticQueryGenerator``: prompt loading, LLM dispatch, dataset resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_perf.config import SyntheticInputConfig, SyntheticMode
from rag_perf.query import PromptTemplates, SyntheticQueryGenerator


def _mock_chat_completion(content: str = "What is NVIDIA?") -> MagicMock:
    """Build an httpx.Response mock that returns a chat-completion payload."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"content": content}}],
        }
    )
    return resp


@pytest.fixture
def prompts_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid prompts YAML covering both modes."""
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "random:\n"
        "  system: |\n"
        "    Generate queries of about {word_target} words.\n"
        "  user: |\n"
        "    Generate query #{index}.\n"
        "dataset_based:\n"
        "  system: |\n"
        "    Reword the question in about {word_target} words.\n"
        "  user: |\n"
        "    #{index}: rephrase: {ref}\n"
    )
    return path


class TestSyntheticQueryGenerator:
    """Tests for ``rag_perf.query.SyntheticQueryGenerator``."""

    # ── _load_prompts ────────────────────────────────────────────────────────

    def test_load_prompts_explicit_file(self, prompts_yaml: Path):
        prompts = SyntheticQueryGenerator._load_prompts("random", str(prompts_yaml))
        assert isinstance(prompts, PromptTemplates)
        assert "{word_target}" in prompts.system
        assert "{index}" in prompts.user

    def test_load_prompts_missing_explicit_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Prompts file not found"):
            SyntheticQueryGenerator._load_prompts("random", str(tmp_path / "nope.yaml"))

    def test_load_prompts_unknown_mode_raises(self, prompts_yaml: Path):
        with pytest.raises(KeyError, match="Mode 'bogus'"):
            SyntheticQueryGenerator._load_prompts("bogus", str(prompts_yaml))

    # ── _word_target ─────────────────────────────────────────────────────────

    def test_word_target_token_to_word_ratio(self):
        # 1 token ≈ 0.75 words, with a floor of 10.
        assert SyntheticQueryGenerator._word_target(100) == 75
        assert SyntheticQueryGenerator._word_target(1) == 10
        assert SyntheticQueryGenerator._word_target(50) == 37

    # ── _resolve_model ───────────────────────────────────────────────────────

    def test_resolve_model_uses_explicit_value(self):
        cfg = SyntheticInputConfig(
            llm_model="my/model", llm_url="http://x:9000/v1/chat/completions"
        )
        assert SyntheticQueryGenerator._resolve_model(cfg) == "my/model"

    def test_resolve_model_auto_discovers_from_endpoint(self):
        cfg = SyntheticInputConfig(
            llm_model="", llm_url="http://x:9000/v1/chat/completions"
        )
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)
        resp.json = MagicMock(return_value={"data": [{"id": "discovered/model"}]})
        with patch("rag_perf.query.httpx.get", return_value=resp):
            assert SyntheticQueryGenerator._resolve_model(cfg) == "discovered/model"

    def test_resolve_model_falls_back_when_endpoint_unreachable(self):
        cfg = SyntheticInputConfig(
            llm_model="", llm_url="http://x:9000/v1/chat/completions"
        )
        with patch("rag_perf.query.httpx.get", side_effect=ConnectionError("nope")):
            assert SyntheticQueryGenerator._resolve_model(cfg) == "local"

    # ── _call_llm ────────────────────────────────────────────────────────────

    def test_call_llm_strips_content(self):
        with patch(
            "rag_perf.query.httpx.post",
            return_value=_mock_chat_completion("  hello  \n"),
        ):
            out = SyntheticQueryGenerator._call_llm(
                "http://x", "m", [{"role": "user", "content": "hi"}]
            )
        assert out == "hello"

    def test_call_llm_empty_content_raises(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)
        resp.json = MagicMock(return_value={"choices": [{"message": {"content": ""}}]})
        with patch("rag_perf.query.httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="empty content"):
                SyntheticQueryGenerator._call_llm("http://x", "m", [])

    def test_call_llm_does_not_fall_back_to_reasoning_content(self):
        # Reasoning models (Nemotron Omni etc.) emit chain-of-thought into
        # `reasoning_content` and the actual answer into `content`. When
        # `content` is empty we MUST raise — silently substituting reasoning
        # for the answer pollutes the synthetic JSONL with "let me think..."
        # text.
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)
        resp.json = MagicMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "from reasoning",
                        }
                    }
                ],
            }
        )
        with patch("rag_perf.query.httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="reasoning_content was populated"):
                SyntheticQueryGenerator._call_llm("http://x", "m", [])

    def test_call_llm_passes_extra_body_to_payload(self):
        captured: dict = {}

        def fake_post(url, json, timeout):  # noqa: A002
            captured["payload"] = json
            return _mock_chat_completion("ok")

        with patch("rag_perf.query.httpx.post", side_effect=fake_post):
            SyntheticQueryGenerator._call_llm(
                "http://x",
                "m",
                [{"role": "user", "content": "hi"}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}

    # ── _generate_random ─────────────────────────────────────────────────────

    def test_generate_random_returns_n_queries(self, prompts_yaml: Path):
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.RANDOM,
            num_queries=3,
            min_query_tokens=20,
            llm_url="http://x:9000/v1/chat/completions",
            llm_model="m",
            prompts_file=str(prompts_yaml),
        )
        with patch(
            "rag_perf.query.httpx.post",
            return_value=_mock_chat_completion("query body"),
        ):
            queries = SyntheticQueryGenerator.generate(cfg)
        assert len(queries) == 3
        for q in queries:
            assert "query body" in q

    def test_generate_random_strips_numbered_prefixes(self, prompts_yaml: Path):
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.RANDOM,
            num_queries=1,
            llm_url="http://x",
            llm_model="m",
            prompts_file=str(prompts_yaml),
        )
        with patch(
            "rag_perf.query.httpx.post",
            return_value=_mock_chat_completion("1. real query"),
        ):
            queries = SyntheticQueryGenerator.generate(cfg)
        assert queries[0] == "real query"

    def test_generate_random_propagates_llm_failure(self, prompts_yaml: Path):
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.RANDOM,
            num_queries=2,
            llm_url="http://x",
            llm_model="m",
            prompts_file=str(prompts_yaml),
        )
        with patch("rag_perf.query.httpx.post", side_effect=RuntimeError("503")):
            with pytest.raises(
                RuntimeError, match="Random synthetic query generation failed"
            ):
                SyntheticQueryGenerator.generate(cfg)

    # ── _resolve_dataset_file ────────────────────────────────────────────────

    def test_resolve_dataset_file_explicit_path_exists(self, tmp_path: Path):
        ds = tmp_path / "queries.json"
        ds.write_text("[]")
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.DATASET_BASED,
            dataset_file=str(ds),
        )
        assert SyntheticQueryGenerator._resolve_dataset_file(cfg) == str(ds)

    def test_resolve_dataset_file_explicit_path_missing(self, tmp_path: Path):
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.DATASET_BASED,
            dataset_file=str(tmp_path / "nope.json"),
        )
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            SyntheticQueryGenerator._resolve_dataset_file(cfg)

    def test_resolve_dataset_file_auto_lookup_by_name(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "datasets" / "my-set").mkdir(parents=True)
        train = tmp_path / "datasets" / "my-set" / "train.json"
        train.write_text("[]")
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.DATASET_BASED,
            dataset_name="my-set",
        )
        # Auto-lookup returns a CWD-relative path; resolve both before comparing.
        result = SyntheticQueryGenerator._resolve_dataset_file(cfg)
        assert Path(result).resolve() == train.resolve()

    # ── _extract_questions ───────────────────────────────────────────────────

    def test_extract_questions_from_list_of_dicts(self):
        data = [{"question": "Q1?"}, {"question": "Q2?"}, {"text": "no-question key"}]
        out = SyntheticQueryGenerator._extract_questions(data, "test")
        assert out == ["Q1?", "Q2?", "no-question key"]

    def test_extract_questions_from_nested_data_key(self):
        data = {"data": [{"question": "Q1?"}, {"query": "Q2?"}]}
        out = SyntheticQueryGenerator._extract_questions(data, "test")
        assert out == ["Q1?", "Q2?"]

    def test_extract_questions_skips_non_string_values(self):
        data = [{"question": 42}, {"question": "ok"}, {"question": ""}]
        out = SyntheticQueryGenerator._extract_questions(data, "test")
        assert out == ["ok"]

    # ── _generate_dataset_based ──────────────────────────────────────────────

    def test_generate_dataset_based_seeds_with_reference_questions(
        self, tmp_path: Path, prompts_yaml: Path
    ):
        ds = tmp_path / "queries.json"
        ds.write_text(json.dumps([{"question": "Original?"}]))
        cfg = SyntheticInputConfig(
            mode=SyntheticMode.DATASET_BASED,
            dataset_file=str(ds),
            num_queries=2,
            llm_url="http://x",
            llm_model="m",
            prompts_file=str(prompts_yaml),
        )
        captured_messages: list = []

        def post(url, json, timeout):  # noqa: ARG001, A002
            captured_messages.append(json["messages"])
            return _mock_chat_completion("rephrased query")

        with patch("rag_perf.query.httpx.post", side_effect=post):
            queries = SyntheticQueryGenerator.generate(cfg)

        assert len(queries) == 2
        # The user message must reference the seed question text.
        for msgs in captured_messages:
            user = next(m for m in msgs if m["role"] == "user")
            assert "Original?" in user["content"]

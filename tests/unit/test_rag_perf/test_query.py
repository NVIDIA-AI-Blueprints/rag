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

"""Tests for ``rag_perf.query``: JSONL/CSV loading, request building, sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rag_perf.config import (
    GenerationParams,
    InputConfig,
    InputSource,
    RagParams,
)
from rag_perf.query import QueryLoader

# Bind the QueryLoader static methods to module-level names so the tests below
# can call them as plain functions.
load_queries = QueryLoader.load
write_aiperf_sharegpt = QueryLoader.write_sharegpt
_load_jsonl = QueryLoader._load_jsonl
_load_csv = QueryLoader._load_csv
_build_request = QueryLoader._build_request
_sample = QueryLoader._sample


class TestQueryLoader:
    """Tests for ``rag_perf.query.QueryLoader`` and related helpers."""

    # ── JSONL loading ─────────────────────────────────────────────────────────

    def test_load_jsonl_basic(self, queries_jsonl: Path):
        records = _load_jsonl(str(queries_jsonl))
        assert len(records) == 3
        assert records[0]["query"] == "What was NVIDIA revenue in Q3 2024?"

    def test_load_jsonl_preserves_overrides(self, queries_jsonl: Path):
        records = _load_jsonl(str(queries_jsonl))
        assert records[2]["collection_names"] == ["custom-collection"]
        assert records[2]["vdb_top_k"] == 50

    def test_load_jsonl_missing_query_key(self, tmp_path: Path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"text": "no query key here"}\n')
        with pytest.raises(ValueError, match="missing required 'query' key"):
            _load_jsonl(str(bad))

    def test_load_jsonl_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            _load_jsonl(str(bad))

    def test_load_jsonl_empty_file_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="No queries found"):
            _load_jsonl(str(empty))

    def test_load_jsonl_skips_comments(self, tmp_path: Path):
        content = (
            "# This is a comment\n" '{"query": "Real query"}\n' "# Another comment\n"
        )
        path = tmp_path / "with_comments.jsonl"
        path.write_text(content)
        records = _load_jsonl(str(path))
        assert len(records) == 1
        assert records[0]["query"] == "Real query"

    # ── CSV loading ───────────────────────────────────────────────────────────

    def test_load_csv_basic(self, queries_csv: Path):
        records = _load_csv(str(queries_csv))
        assert len(records) == 2
        assert records[0]["query"] == "What is NVIDIA data center revenue?"

    def test_load_csv_numeric_overrides_parsed(self, queries_csv: Path):
        records = _load_csv(str(queries_csv))
        assert records[0]["max_tokens"] == 512

    def test_load_csv_missing_query_column(self, tmp_path: Path):
        bad = tmp_path / "no_query.csv"
        bad.write_text("text,tokens\nsome text,100\n")
        with pytest.raises(ValueError, match="'query' column"):
            _load_csv(str(bad))

    # ── Request building ──────────────────────────────────────────────────────

    def test_build_request_uses_rag_defaults(
        self, rag_params: RagParams, gen_params: GenerationParams
    ):
        raw = {"query": "Test question?"}
        req = _build_request(raw, rag_params, gen_params)

        assert req["collection_names"] == ["test-collection"]
        assert req["vdb_top_k"] == 20
        assert req["reranker_top_k"] == 4
        assert req["max_tokens"] == 128
        assert req["temperature"] == 0.0
        assert req["stream"] is True

    def test_build_request_per_query_override_takes_precedence(
        self, rag_params: RagParams, gen_params: GenerationParams
    ):
        raw = {
            "query": "Test?",
            "collection_names": ["override-collection"],
            "vdb_top_k": 99,
            "max_tokens": 1024,
        }
        req = _build_request(raw, rag_params, gen_params)

        assert req["collection_names"] == ["override-collection"]
        assert req["vdb_top_k"] == 99
        assert req["max_tokens"] == 1024
        assert req["reranker_top_k"] == 4

    def test_build_request_messages_format(
        self, rag_params: RagParams, gen_params: GenerationParams
    ):
        raw = {"query": "What is 2+2?"}
        req = _build_request(raw, rag_params, gen_params)

        assert "messages" in req
        assert len(req["messages"]) == 1
        msg = req["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "What is 2+2?"

    def test_build_request_stream_always_true(
        self, rag_params: RagParams, gen_params: GenerationParams
    ):
        raw = {"query": "Test?"}
        req = _build_request(raw, rag_params, gen_params)
        assert req["stream"] is True

    # ── Sampling strategies ───────────────────────────────────────────────────

    def test_sample_sequential_cycles(self):
        queries = [{"query": f"q{i}"} for i in range(3)]
        sampled = _sample(queries, 7, "sequential", seed=42)
        assert len(sampled) == 7
        assert [q["query"] for q in sampled] == [
            "q0",
            "q1",
            "q2",
            "q0",
            "q1",
            "q2",
            "q0",
        ]

    def test_sample_random_uses_seed_for_reproducibility(self):
        queries = [{"query": f"q{i}"} for i in range(10)]
        s1 = _sample(queries, 5, "random", seed=42)
        s2 = _sample(queries, 5, "random", seed=42)
        assert s1 == s2

    def test_sample_random_different_seeds_differ(self):
        queries = [{"query": f"q{i}"} for i in range(20)]
        s1 = _sample(queries, 10, "random", seed=1)
        s2 = _sample(queries, 10, "random", seed=99)
        assert s1 != s2

    def test_sample_shuffle_once_covers_all(self):
        queries = [{"query": f"q{i}"} for i in range(5)]
        sampled = _sample(queries, 5, "shuffle-once", seed=42)
        assert sorted(q["query"] for q in sampled) == sorted(
            q["query"] for q in queries
        )

    def test_sample_exact_count(self, rag_params, gen_params):
        queries = [{"query": f"q{i}"} for i in range(3)]
        for count in [1, 3, 7, 20]:
            result = _sample(queries, count, "random", seed=0)
            assert len(result) == count

    # ── load_queries() integration ────────────────────────────────────────────

    def test_load_queries_from_jsonl(
        self, queries_jsonl: Path, rag_params: RagParams, gen_params: GenerationParams
    ):
        cfg = InputConfig(file=str(queries_jsonl))
        result = load_queries(cfg, rag_params, gen_params, count=6)

        assert len(result) == 6
        for req in result:
            assert "messages" in req
            assert req["stream"] is True

    # ── ShareGPT conversion ───────────────────────────────────────────────────

    def test_write_aiperf_sharegpt_format(
        self, tmp_path: Path, rag_params: RagParams, gen_params: GenerationParams
    ):
        queries = [
            _build_request({"query": "NVIDIA revenue?"}, rag_params, gen_params),
            _build_request({"query": "Intel plans?"}, rag_params, gen_params),
        ]
        out_path = tmp_path / "sharegpt.jsonl"
        write_aiperf_sharegpt(queries, out_path)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj
            assert isinstance(obj["text"], str)
            assert len(obj["text"]) > 0

    def test_write_aiperf_sharegpt_query_text_preserved(
        self, tmp_path: Path, rag_params: RagParams, gen_params: GenerationParams
    ):
        queries = [
            _build_request({"query": "Exact query text here?"}, rag_params, gen_params)
        ]
        out_path = tmp_path / "out.jsonl"
        write_aiperf_sharegpt(queries, out_path)

        obj = json.loads(out_path.read_text())
        assert obj["text"] == "Exact query text here?"

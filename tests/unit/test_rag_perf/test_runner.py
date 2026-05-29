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

"""Tests for ``rag_perf.runner.RagProfiler``: SSE parsing, concurrency, JSONL output."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_perf.config import InputConfig, InputSource, RunConfig, TargetConfig
from rag_perf.query import QueryLoader
from rag_perf.runner import RagProfiler

from tests.unit.test_rag_perf.utils import build_sse_lines, make_stream_mock

# Bind for terse test bodies.
run_profiler = RagProfiler.run
_build_request = QueryLoader._build_request


class TestRagProfiler:
    """Tests for ``rag_perf.runner.RagProfiler``."""

    # ── Class-local fixtures ──────────────────────────────────────────────────

    @pytest.fixture
    def profiler_config(self) -> RunConfig:
        return RunConfig(
            target=TargetConfig(url="http://test-server:8081", timeout_s=30),
            input=InputConfig(file="examples/queries.jsonl"),
        )

    @pytest.fixture
    def one_request(self, rag_params, gen_params) -> list[dict]:
        return [
            _build_request(
                {"query": "What was NVIDIA revenue?"}, rag_params, gen_params
            )
        ]

    # ── Core SSE parsing ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profiler_extracts_server_metrics(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(
            server_metrics={
                "rag_ttft_ms": 3240.0,
                "llm_ttft_ms": 915.0,
                "retrieval_time_ms": 145.0,
                "context_reranker_time_ms": 2180.0,
                "llm_generation_time_ms": 4200.0,
            }
        )
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        assert len(result.successful) == 1
        rec = result.successful[0]
        assert rec.server_metrics["rag_ttft_ms"] == pytest.approx(3240.0)
        assert rec.server_metrics["llm_ttft_ms"] == pytest.approx(915.0)
        assert rec.server_metrics["retrieval_time_ms"] == pytest.approx(145.0)
        assert rec.server_metrics["reranking_time_ms"] == pytest.approx(2180.0)
        assert rec.server_metrics["llm_generation_time_ms"] == pytest.approx(4200.0)

    @pytest.mark.asyncio
    async def test_profiler_extracts_citation_scores(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(citation_scores=[0.92, 0.75, 0.61, 0.50])
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        rec = result.successful[0]
        assert rec.citation_count == 4
        assert rec.citation_scores == pytest.approx([0.92, 0.75, 0.61, 0.50])

    @pytest.mark.asyncio
    async def test_profiler_extracts_token_counts(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(usage={"prompt_tokens": 384, "completion_tokens": 96})
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        rec = result.successful[0]
        assert rec.input_tokens == 384
        assert rec.output_tokens == 96

    @pytest.mark.asyncio
    async def test_profiler_ttft_is_positive(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(tokens=["Hello", " world"])
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        rec = result.successful[0]
        assert rec.client_ttft_ms > 0
        assert rec.client_e2e_ms >= rec.client_ttft_ms

    @pytest.mark.asyncio
    async def test_profiler_query_text_preserved(
        self, profiler_config, rag_params, gen_params, tmp_path
    ):
        req = [
            _build_request(
                {"query": "Unique query text 12345?"}, rag_params, gen_params
            )
        ]
        lines = build_sse_lines()
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                req,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        assert result.successful[0].query == "Unique query text 12345?"

    # ── Citation raw content ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profiler_citations_raw_are_collected(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(citation_scores=[0.9, 0.8])
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        rec = result.successful[0]
        assert len(rec.citations_raw) == 2
        for cite in rec.citations_raw:
            assert "content" in cite
            assert "score" in cite
            assert "document_name" in cite

    # ── Error handling ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profiler_handles_http_error(
        self, profiler_config, one_request, tmp_path
    ):
        import httpx

        @asynccontextmanager
        async def _error_stream(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=MagicMock(),
                response=MagicMock(status_code=503),
            )
            yield resp

        with patch("httpx.AsyncClient.stream", new=_error_stream):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        assert len(result.failed) == 1
        assert "503" in result.failed[0].error
        assert len(result.successful) == 0

    @pytest.mark.asyncio
    async def test_profiler_handles_malformed_sse(
        self, profiler_config, one_request, tmp_path
    ):
        lines = [
            "data: not-json",
            ": keep-alive comment",
            build_sse_lines()[0],
            *build_sse_lines()[-2:],
        ]
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        assert len(result.records) == 1

    # ── Multiple requests and concurrency ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profiler_multiple_requests(
        self, profiler_config, rag_params, gen_params, tmp_path
    ):
        reqs = [
            _build_request({"query": f"Query {i}?"}, rag_params, gen_params)
            for i in range(5)
        ]
        lines = build_sse_lines()

        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                reqs,
                concurrency=3,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        assert len(result.records) == 5
        assert len(result.successful) == 5
        assert len(result.failed) == 0

    # ── JSONL metadata output ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profiler_writes_jsonl_metadata(
        self, profiler_config, one_request, tmp_path
    ):
        lines = build_sse_lines(citation_scores=[0.9, 0.7])
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                one_request,
                concurrency=1,
                save_metadata=True,
                output_dir=str(tmp_path),
            )

        assert result.metadata_jsonl_path is not None
        jsonl_path = Path(result.metadata_jsonl_path)
        assert jsonl_path.exists()

        records = [
            json.loads(line) for line in jsonl_path.read_text().strip().split("\n")
        ]
        assert len(records) == 1
        rec = records[0]
        assert "query" in rec
        assert "server_metrics" in rec
        assert "citations" in rec
        assert "citation_scores" in rec
        assert rec["citation_count"] == 2
        assert "rag_ttft_ms" in rec["server_metrics"]

    @pytest.mark.asyncio
    async def test_profiler_jsonl_no_citation_scores_when_empty(
        self, profiler_config, rag_params, gen_params, tmp_path
    ):
        final_chunk = {
            "id": "test",
            "choices": [{"delta": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            "citations": {"results": []},
            "metrics": {
                "rag_ttft_ms": 1000.0,
                "llm_ttft_ms": 800.0,
                "retrieval_time_ms": 100.0,
                "context_reranker_time_ms": 0.0,
                "llm_generation_time_ms": 1500.0,
            },
        }
        lines = [
            'data: {"id":"t","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            f"data: {json.dumps(final_chunk)}",
            "data: [DONE]",
        ]

        req = [_build_request({"query": "Test?"}, rag_params, gen_params)]
        with make_stream_mock(lines):
            result = await run_profiler(
                profiler_config,
                req,
                concurrency=1,
                save_metadata=False,
                output_dir=str(tmp_path),
            )

        rec = result.successful[0]
        assert rec.citation_count == 0
        assert rec.citation_scores == []

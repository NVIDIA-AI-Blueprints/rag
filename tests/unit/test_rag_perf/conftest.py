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

"""Shared pytest fixtures for the rag-perf test suite.

The rag-perf package itself lives at ``scripts/rag-perf/rag_perf/`` (it is a
self-contained CLI driver, not part of the RAG server runtime). To let
``from rag_perf …`` resolve when these tests run from the repo root, this
conftest prepends that directory to ``sys.path`` at import time.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/rag-perf/ to sys.path so the rag_perf package is importable.
_RAG_PERF_DIR = Path(__file__).resolve().parents[3] / "scripts" / "rag-perf"
if str(_RAG_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_PERF_DIR))

from rag_perf.config import (  # noqa: E402
    GenerationParams,
    InputConfig,
    InputSource,
    RagParams,
    RunConfig,
    TargetConfig,
)
from rag_perf.reporting import ProfileRecord  # noqa: E402

from tests.unit.test_rag_perf.utils import (  # noqa: E402
    build_sse_lines,
    make_profile_record,
)


@pytest.fixture
def basic_config() -> RunConfig:
    """Minimal RunConfig with safe defaults for unit testing."""
    return RunConfig(
        target=TargetConfig(url="http://test-rag-server:8081", timeout_s=30),
        input=InputConfig(file="examples/queries.jsonl"),
    )


@pytest.fixture
def rag_params() -> RagParams:
    return RagParams(
        collection_names=["test-collection"],
        vdb_top_k=20,
        reranker_top_k=4,
        enable_reranker=True,
        enable_citations=True,
    )


@pytest.fixture
def gen_params() -> GenerationParams:
    return GenerationParams(max_tokens=128, temperature=0.0)


@pytest.fixture
def queries_jsonl(tmp_path: Path) -> Path:
    """Write a small JSONL query file and return its path."""
    lines = [
        {"query": "What was NVIDIA revenue in Q3 2024?"},
        {"query": "Summarize the key risks in the 10-K.", "max_tokens": 256},
        {
            "query": "Compare AAPL and MSFT margins.",
            "collection_names": ["custom-collection"],
            "vdb_top_k": 50,
        },
    ]
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in lines))
    return path


@pytest.fixture
def queries_csv(tmp_path: Path) -> Path:
    """Write a small CSV query file and return its path."""
    content = textwrap.dedent(
        """\
        query,max_tokens
        What is NVIDIA data center revenue?,512
        Describe Intel foundry plans.,256
    """
    )
    path = tmp_path / "queries.csv"
    path.write_text(content)
    return path


@pytest.fixture
def profile_records() -> list[ProfileRecord]:
    """Ten realistic ProfileRecord objects with reranking as the bottleneck."""
    return [
        make_profile_record(
            query=f"Query {i}?",
            rag_ttft=3200.0 + i * 40,
            llm_ttft=900.0 + i * 15,
            retrieval=140.0 + i * 2,
            reranking=2150.0 + i * 30,
            llm_gen=4100.0 + i * 50,
        )
        for i in range(10)
    ]


@pytest.fixture
def profile_records_with_errors(
    profile_records: list[ProfileRecord],
) -> list[ProfileRecord]:
    """Profile records with 2 failed requests mixed in."""
    failed = [
        make_profile_record(query="Failed query 1?", error="Connection timeout"),
        make_profile_record(query="Failed query 2?", error="HTTP 503"),
    ]
    return profile_records[:8] + failed


@pytest.fixture
def sse_lines() -> list[str]:
    """Standard SSE response lines for a successful RAG request."""
    return build_sse_lines()


@pytest.fixture
def mock_rag_client(sse_lines: list[str]):
    """Patch httpx.AsyncClient.stream to return a fake SSE response."""

    @asynccontextmanager
    async def _mock_stream(*args, **kwargs) -> AsyncIterator:
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)
        resp.status_code = 200

        async def _aiter_lines():
            for line in sse_lines:
                yield line

        resp.aiter_lines = _aiter_lines
        yield resp

    with patch("httpx.AsyncClient.stream", new=_mock_stream):
        yield


@pytest.fixture
def mock_request_info():
    """Minimal duck-typed stand-in for an aiperf RequestInfo object."""
    from types import SimpleNamespace

    text_media = SimpleNamespace(contents=["What was NVIDIA revenue in Q3 2024?"])
    turn = SimpleNamespace(texts=[text_media], max_tokens=None)
    endpoint = SimpleNamespace(
        extra=[
            ("collection_names", ["test-collection"]),
            ("vdb_top_k", 20),
            ("reranker_top_k", 4),
            ("use_knowledge_base", True),
            ("enable_citations", True),
        ]
    )
    model_endpoint = SimpleNamespace(endpoint=endpoint)
    return SimpleNamespace(
        turns=[turn],
        model_endpoint=model_endpoint,
        system_message=None,
        model_config={},
    )


@pytest.fixture
def mock_sse_response():
    """Factory that builds a mock InferenceServerResponse for a given JSON payload."""

    class FakeResponse:
        def __init__(self, json_data: dict, perf_ns: int = 1_000_000):
            self._json = json_data
            self.perf_ns = perf_ns

        def get_json(self):
            return self._json

        def get_text(self):
            return json.dumps(self._json)

        def get_raw(self):
            return self._json

    return FakeResponse

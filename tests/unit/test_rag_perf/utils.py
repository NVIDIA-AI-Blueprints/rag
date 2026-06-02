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
Shared non-fixture helpers for the rag-perf test suite.

Pytest fixtures live in ``conftest.py`` (auto-discovered); this module holds
plain functions used to build test inputs (profile records, SSE streams,
fake aiperf output, plugin chunks).
"""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from rag_perf.reporting import ProfileRecord


def make_profile_record(
    query: str = "Test query?",
    rag_ttft: float = 3240.0,
    llm_ttft: float = 915.0,
    retrieval: float = 145.0,
    reranking: float = 2180.0,
    llm_gen: float = 4200.0,
    citation_scores: list[float] | None = None,
    citations_raw: list[dict] | None = None,
    error: str | None = None,
) -> ProfileRecord:
    """Factory for realistic ProfileRecord test objects."""
    scores = citation_scores or [0.85, 0.72, 0.61]
    raw = citations_raw or [
        {
            "content": base64.b64encode(f"Chunk {i+1} text content.".encode()).decode(),
            "score": scores[i] if i < len(scores) else 0.5,
            "document_name": f"doc_{i+1}.pdf",
            "document_type": "text",
        }
        for i in range(len(scores))
    ]
    return ProfileRecord(
        query=query,
        client_ttft_ms=rag_ttft + 50,
        client_e2e_ms=rag_ttft + llm_gen + 50,
        output_tokens=64,
        input_tokens=512,
        server_metrics={
            "rag_ttft_ms": rag_ttft,
            "llm_ttft_ms": llm_ttft,
            "retrieval_time_ms": retrieval,
            "reranking_time_ms": reranking,
            "llm_generation_time_ms": llm_gen,
        },
        citation_count=len(scores),
        citation_scores=scores,
        citations_raw=raw,
        error=error,
    )


def build_sse_lines(
    tokens: list[str] | None = None,
    server_metrics: dict | None = None,
    citation_scores: list[float] | None = None,
    usage: dict | None = None,
) -> list[str]:
    """
    Build a list of SSE data lines as the RAG server would emit them.

    Structure: N token chunks → 1 final chunk (metrics, citations, usage) → [DONE]
    """
    tokens = tokens or ["The", " revenue", " was", " $10B", "."]
    metrics = server_metrics or {
        "rag_ttft_ms": 3240.0,
        "llm_ttft_ms": 915.0,
        "retrieval_time_ms": 145.0,
        "context_reranker_time_ms": 2180.0,
        "llm_generation_time_ms": 4200.0,
    }
    scores = citation_scores or [0.85, 0.72, 0.61]
    usage_data = usage or {"prompt_tokens": 512, "completion_tokens": len(tokens)}

    lines = []
    for token in tokens:
        chunk = {
            "id": "chatcmpl-test",
            "choices": [{"delta": {"content": token}, "finish_reason": None}],
            "citations": None,
            "metrics": None,
        }
        lines.append(f"data: {json.dumps(chunk)}")

    citations_obj = {
        "results": [
            {
                "content": base64.b64encode(f"Chunk {i+1} text.".encode()).decode(),
                "score": score,
                "document_name": f"doc_{i+1}.pdf",
                "document_type": "text",
            }
            for i, score in enumerate(scores)
        ]
    }
    final_chunk = {
        "id": "chatcmpl-test",
        "choices": [{"delta": None, "finish_reason": "stop"}],
        "usage": usage_data,
        "citations": citations_obj,
        "metrics": metrics,
    }
    lines.append(f"data: {json.dumps(final_chunk)}")
    lines.append("data: [DONE]")
    return lines


def make_stream_mock(sse_lines: list[str]):
    """Patch ``httpx.AsyncClient.stream`` to yield fake SSE lines."""

    @asynccontextmanager
    async def _mock_stream(*args, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)

        async def _aiter_lines():
            for line in sse_lines:
                yield line

        resp.aiter_lines = _aiter_lines
        yield resp

    return patch("httpx.AsyncClient.stream", new=_mock_stream)


def fake_aiperf_json(ttft_p50: float = 3200.0, ttft_p99: float = 12000.0) -> dict:
    """Build a synthetic ``profile_export_aiperf.json``-shaped dict."""
    return {
        "time_to_first_token": {
            "avg": ttft_p50 * 1.1,
            "p50": ttft_p50,
            "p90": ttft_p50 * 1.5,
            "p99": ttft_p99,
        },
        "request_latency": {
            "avg": ttft_p50 * 3,
            "p90": ttft_p50 * 4,
            "p99": ttft_p50 * 6,
        },
        "output_token_throughput": {"avg": 28.4},
        "request_throughput": {"avg": 0.95},
        "request_count": {"avg": 200.0},
        "error_request_count": {"avg": 2.0},
    }


def plugin_token_chunk(content: str) -> dict:
    """Build a single mid-stream chat-completion chunk for plugin tests."""
    return {
        "id": "test",
        "choices": [{"delta": {"content": content}, "finish_reason": None}],
        "citations": None,
        "metrics": None,
    }


def plugin_final_chunk(
    server_metrics: dict | None = None,
    citation_scores: list[float] | None = None,
    usage: dict | None = None,
) -> dict:
    """Build a final-chunk payload (metrics + citations + usage) for plugin tests."""
    scores = citation_scores or [0.85, 0.72]
    metrics = server_metrics or {
        "rag_ttft_ms": 3240.0,
        "llm_ttft_ms": 915.0,
        "retrieval_time_ms": 145.0,
        "context_reranker_time_ms": 2180.0,
        "llm_generation_time_ms": 4200.0,
    }
    return {
        "id": "test",
        "choices": [{"delta": None, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 512, "completion_tokens": 64},
        "citations": {
            "results": [
                {
                    "content": base64.b64encode(f"Chunk {i+1}".encode()).decode(),
                    "score": score,
                    "document_name": f"doc_{i+1}.pdf",
                }
                for i, score in enumerate(scores)
            ]
        },
        "metrics": metrics,
    }

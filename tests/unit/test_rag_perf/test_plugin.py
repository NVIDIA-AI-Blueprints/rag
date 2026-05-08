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

"""Tests for ``rag_perf.plugin.nvidia_rag`` — payload formatting and SSE parsing."""

from __future__ import annotations

import pytest

pytest.importorskip("aiperf", reason="aiperf must be installed to run plugin tests")

from tests.unit.test_rag_perf.utils import (  # noqa: E402
    plugin_final_chunk,
    plugin_token_chunk,
)


class TestPlugin:
    """Tests for ``rag_perf.plugin.nvidia_rag.NvidiaRagEndpoint``."""

    @pytest.fixture
    def plugin(self):
        from rag_perf.plugin.nvidia_rag import NvidiaRagEndpoint

        try:
            return NvidiaRagEndpoint()
        except TypeError:
            from unittest.mock import MagicMock

            return NvidiaRagEndpoint(model_endpoint=MagicMock())

    # ── format_payload tests ──────────────────────────────────────────────────

    def test_format_payload_builds_messages(self, plugin, mock_request_info):
        payload = plugin.format_payload(mock_request_info)

        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) >= 1

        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert "NVIDIA" in user_msg["content"]

    def test_format_payload_includes_rag_fields(self, plugin, mock_request_info):
        payload = plugin.format_payload(mock_request_info)

        assert payload["collection_names"] == ["test-collection"]
        assert payload["vdb_top_k"] == 20
        assert payload["reranker_top_k"] == 4
        assert payload["use_knowledge_base"] is True
        assert payload["enable_citations"] is True

    def test_format_payload_stream_always_true(self, plugin, mock_request_info):
        payload = plugin.format_payload(mock_request_info)
        assert payload["stream"] is True

    def test_format_payload_default_enable_citations(self, plugin, mock_request_info):
        mock_request_info.model_endpoint.endpoint.extra = [
            item
            for item in mock_request_info.model_endpoint.endpoint.extra
            if item[0] != "enable_citations"
        ]
        payload = plugin.format_payload(mock_request_info)
        assert payload.get("enable_citations") is True

    def test_format_payload_includes_system_message(self, plugin, mock_request_info):
        mock_request_info.model_config["arbitrary_types_allowed"] = True
        object.__setattr__(
            mock_request_info, "system_message", "You are a financial expert."
        )
        payload = plugin.format_payload(mock_request_info)

        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a financial expert."

    # ── parse_response tests ──────────────────────────────────────────────────

    def test_parse_response_returns_none_for_done(self, plugin, mock_sse_response):
        resp = mock_sse_response(None)
        assert plugin.parse_response(resp) is None

    def test_parse_response_returns_none_for_empty_choices(
        self, plugin, mock_sse_response
    ):
        resp = mock_sse_response({"id": "test", "choices": []})
        assert plugin.parse_response(resp) is None

    def test_parse_response_token_chunk_returns_text(self, plugin, mock_sse_response):
        resp = mock_sse_response(plugin_token_chunk("revenue"))
        result = plugin.parse_response(resp)

        assert result is not None
        assert result.data is not None
        assert result.data.text == "revenue"

    def test_parse_response_final_chunk_extracts_server_metrics(
        self, plugin, mock_sse_response
    ):
        resp = mock_sse_response(
            plugin_final_chunk(
                server_metrics={
                    "rag_ttft_ms": 3240.0,
                    "llm_ttft_ms": 915.0,
                    "retrieval_time_ms": 145.0,
                    "context_reranker_time_ms": 2180.0,
                    "llm_generation_time_ms": 4200.0,
                }
            )
        )
        result = plugin.parse_response(resp)

        assert result is not None
        assert "server_metrics" in result.metadata
        sm = result.metadata["server_metrics"]
        assert sm["rag_ttft_ms"] == pytest.approx(3240.0)
        assert sm["llm_ttft_ms"] == pytest.approx(915.0)
        assert sm["retrieval_time_ms"] == pytest.approx(145.0)
        assert sm["reranking_time_ms"] == pytest.approx(2180.0)
        assert sm["llm_generation_time_ms"] == pytest.approx(4200.0)

    def test_parse_response_final_chunk_extracts_citation_scores(
        self, plugin, mock_sse_response
    ):
        resp = mock_sse_response(plugin_final_chunk(citation_scores=[0.92, 0.75, 0.61]))
        result = plugin.parse_response(resp)

        assert result is not None
        assert result.metadata.get("citation_count") == 3
        assert result.metadata.get("citation_scores") == pytest.approx(
            [0.92, 0.75, 0.61]
        )

    def test_parse_response_final_chunk_stores_raw_citations(
        self, plugin, mock_sse_response
    ):
        resp = mock_sse_response(plugin_final_chunk(citation_scores=[0.9]))
        result = plugin.parse_response(resp)

        assert "citations" in result.metadata
        assert len(result.metadata["citations"]) == 1
        assert "content" in result.metadata["citations"][0]
        assert "score" in result.metadata["citations"][0]

    def test_parse_response_final_chunk_captures_usage(self, plugin, mock_sse_response):
        usage = {"prompt_tokens": 512, "completion_tokens": 64}
        resp = mock_sse_response(plugin_final_chunk(usage=usage))
        result = plugin.parse_response(resp)
        assert result.usage == usage

    def test_parse_response_handles_missing_metrics_gracefully(
        self, plugin, mock_sse_response
    ):
        chunk = {
            "id": "test",
            "choices": [{"delta": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = mock_sse_response(chunk)
        result = plugin.parse_response(resp)

        assert result is not None
        assert "server_metrics" not in result.metadata
        assert "citation_count" not in result.metadata

    def test_parse_response_mid_stream_metadata_is_empty(
        self, plugin, mock_sse_response
    ):
        resp = mock_sse_response(plugin_token_chunk("hello"))
        result = plugin.parse_response(resp)
        assert result.metadata == {}

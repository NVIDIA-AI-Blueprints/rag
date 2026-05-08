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
aiperf endpoint plugin for the NVIDIA RAG Blueprint ``/v1/generate`` API.

aiperf discovers this endpoint via the ``aiperf.plugins`` entry point declared
in ``pyproject.toml``. After ``pip install -e scripts/rag-perf``, aiperf can
resolve ``--endpoint-type nvidia_rag``.

Two classes:
  - ``NvidiaRagEndpoint``  Subclass of ``aiperf.endpoints.base_endpoint.BaseEndpoint``.
                            Implements ``format_payload`` (build the request body)
                            and ``parse_response`` (extract token deltas + server-side
                            stage metrics + citation scores from the SSE stream).
  - ``PluginRegistry``     Fallback in-process registration for environments where
                            entry-point auto-discovery has not yet taken effect.
"""

from __future__ import annotations

import contextlib
import json as _json
from typing import Any

from aiperf.common.models import (  # type: ignore[import]
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint  # type: ignore[import]

PLUGIN_NAME = "nvidia_rag"
PLUGIN_CLASS_PATH = "rag_perf.plugin.nvidia_rag:NvidiaRagEndpoint"

# Fields the RAG server accepts beyond the standard OpenAI schema.
# Supplied via --extra-input flags to aiperf, e.g. ``--extra-input vdb_top_k:100``.
_RAG_SPECIFIC_FIELDS: frozenset[str] = frozenset(
    {
        "collection_names",
        "vdb_top_k",
        "reranker_top_k",
        "use_knowledge_base",
        "enable_reranker",
        "enable_citations",
        "confidence_threshold",
        "fetch_full_page_context",
        "fetch_neighboring_pages",
        "vdb_endpoint",
        "enable_query_rewriting",
        "enable_guardrails",
        "filter_expr",
    }
)


class NvidiaRagEndpoint(BaseEndpoint):
    """
    aiperf endpoint plugin for the NVIDIA RAG Blueprint ``/v1/generate`` API.

    Inherits from ``BaseEndpoint`` and implements the two required abstract
    methods:

    * ``format_payload``  — builds the JSON body for each POST request.
    * ``parse_response``  — parses each SSE chunk; extracts server-side
                            stage timing and citation scores from the
                            final chunk.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """
        Build the JSON payload for ``POST /v1/generate``.

        Sources of data (in priority order, highest first):

        1. **Per-query overrides** — embedded in ``turn.texts`` as a JSON
           prefix (handled by ``inputs.py`` → aiperf ShareGPT conversion).
           Currently not used; all per-query fields come from the turns' text.

        2. **Extra inputs** — supplied to aiperf via ``--extra-input key:value``
           flags; available in ``request_info.model_endpoint.endpoint.extra``.
           These apply to *every* request in the run.

        3. **Built-in defaults** — safe defaults for required RAG fields
           (``use_knowledge_base=True``, ``enable_citations=True``) when not
           supplied via extra inputs.

        Args:
            request_info: Populated by aiperf; contains the user query in
                          ``request_info.turns[-1].texts`` and run-level
                          ``extra`` parameters from ``--extra-input`` flags.

        Returns:
            A JSON-serialisable dict ready to POST to ``/v1/generate``.
        """
        turn = request_info.turns[-1]

        query_parts = [
            content
            for text_media in turn.texts
            for content in text_media.contents
            if content
        ]
        query_text = " ".join(query_parts).strip()

        messages: list[dict[str, str]] = []
        if request_info.system_message:
            messages.append({"role": "system", "content": request_info.system_message})
        messages.append({"role": "user", "content": query_text})

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "max_tokens": turn.max_tokens or 512,
            "temperature": 0.0,
        }

        extra_raw = request_info.model_endpoint.endpoint.extra or []
        extra: dict[str, Any] = (
            dict(extra_raw) if isinstance(extra_raw, list) else extra_raw
        )

        for k in list(extra):
            if isinstance(extra[k], str):
                with contextlib.suppress(_json.JSONDecodeError, ValueError):
                    extra[k] = _json.loads(extra[k])

        for field in _RAG_SPECIFIC_FIELDS:
            if field in extra:
                payload[field] = extra[field]

        payload.setdefault("use_knowledge_base", True)
        payload.setdefault("enable_citations", True)

        self.trace(lambda: f"RAG payload: {list(payload.keys())}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """
        Parse one SSE chunk from the RAG server's streaming response.

        The RAG server sends standard OpenAI-style ``chat.completion.chunk``
        objects.  Most chunks carry only a token delta.  The **final** chunk
        (identified by ``finish_reason != null``) additionally carries:

        * ``metrics``   — server-side pipeline stage timing (5 fields)
        * ``citations`` — retrieved document chunks with relevance scores
        * ``usage``     — token counts

        All three are captured when present and stored in
        ``ParsedResponse.metadata`` for downstream aggregation by
        ``rag_perf.metrics``.

        Args:
            response: One SSE message from the server; ``response.get_json()``
                      returns the parsed JSON dict.

        Returns:
            ``ParsedResponse`` with the token delta text and any metadata,
            or ``None`` to skip this chunk (e.g., ``[DONE]`` sentinel).
        """
        json_obj = response.get_json()
        if json_obj is None:
            return None

        choices = json_obj.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta") or {}
        content: str | None = delta.get("content")
        finish_reason: str | None = choice.get("finish_reason")

        metadata: dict[str, Any] = {}

        if finish_reason is not None:
            server_metrics = json_obj.get("metrics")
            if isinstance(server_metrics, dict):
                metadata["server_metrics"] = {
                    "rag_ttft_ms": server_metrics.get("rag_ttft_ms"),
                    "llm_ttft_ms": server_metrics.get("llm_ttft_ms"),
                    "retrieval_time_ms": server_metrics.get("retrieval_time_ms"),
                    "reranking_time_ms": server_metrics.get("context_reranker_time_ms"),
                    "llm_generation_time_ms": server_metrics.get(
                        "llm_generation_time_ms"
                    ),
                }

            citations_obj = json_obj.get("citations")
            if isinstance(citations_obj, dict):
                results = citations_obj.get("results") or []
                scores = [
                    r["score"] for r in results if isinstance(r, dict) and "score" in r
                ]
                metadata["citation_count"] = len(results)
                if scores:
                    metadata["citation_scores"] = scores
                metadata["citations"] = results

            self.trace(
                lambda: (
                    f"Final chunk: finish_reason={finish_reason}, "
                    f"server_metrics={metadata.get('server_metrics')}, "
                    f"citations={metadata.get('citation_count', 0)}"
                )
            )

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(content),
            usage=json_obj.get("usage"),
            metadata=metadata,
        )


class PluginRegistry:
    """Programmatic registration of the nvidia_rag endpoint with aiperf."""

    @staticmethod
    def register() -> None:
        """
        Register the ``nvidia_rag`` endpoint plugin with aiperf's in-memory
        plugin registry.

        This is a fallback for environments where entry-point auto-discovery
        has not yet taken effect (e.g., during development without a full
        ``pip install``).  Safe to call multiple times — subsequent calls are
        no-ops.
        """
        try:
            from aiperf.plugin.enums import PluginType  # type: ignore[import]
            from aiperf.plugin.plugin import (
                PluginRegistry as _AiperfRegistry,  # type: ignore[import]
            )

            registry: _AiperfRegistry = _AiperfRegistry.instance()
            if not registry.has(PluginType.ENDPOINT, PLUGIN_NAME):
                registry.register(
                    category=PluginType.ENDPOINT,
                    name=PLUGIN_NAME,
                    cls=NvidiaRagEndpoint,
                    metadata={
                        "endpoint_path": "/v1/generate",
                        "supports_streaming": True,
                        "produces_tokens": True,
                        "tokenizes_input": False,
                        "supports_audio": False,
                        "supports_images": False,
                        "supports_videos": False,
                        "metrics_title": "NVIDIA RAG Blueprint",
                    },
                )
        except Exception:
            pass

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

import pytest


class DummyMetrics:
    def __init__(self):
        self.updates = []

    def update_latency_metrics(self, payload: dict):
        self.updates.append(payload)


def make_sync_generator(chunks):
    def gen():
        yield from chunks

    return gen()


@pytest.mark.asyncio
async def test_generate_answer_updates_metrics(monkeypatch):
    from nvidia_rag.rag_server.response_generator import generate_answer

    metrics_client = DummyMetrics()

    # single chunk then end
    chunks = ["Hello"]
    gen = make_sync_generator(chunks)

    outputs = []
    for out in generate_answer(
        generator=gen,
        contexts=[],
        model="test-model",
        collection_name="",
        enable_citations=False,
        context_reranker_time_ms=5.0,
        retrieval_time_ms=10.0,
        rag_start_time_sec=None,
        otel_metrics_client=metrics_client,
    ):
        outputs.append(out)

    # Should yield len(chunks) + 1 (final [DONE])
    assert len(outputs) == len(chunks) + 1

    # Verify metrics updated with expected keys (rag_ttft_ms may be None)
    assert len(metrics_client.updates) == 1
    payload = metrics_client.updates[0]
    assert "llm_ttft_ms" in payload
    assert payload["context_reranker_time_ms"] == 5.0
    assert payload["retrieval_time_ms"] == 10.0
    assert "llm_generation_time_ms" in payload


@pytest.mark.asyncio
async def test_generate_answer_skips_metrics_when_none():
    from nvidia_rag.rag_server.response_generator import generate_answer

    gen = make_sync_generator(["Hello"])

    # Do not pass metrics client
    outputs = []
    for out in generate_answer(
        generator=gen,
        contexts=[],
        model="test-model",
        collection_name="",
        enable_citations=False,
    ):
        outputs.append(out)

    assert len(outputs) == 2  # one token + final [DONE]
    # Nothing else to assert; just ensures no exception without metrics client

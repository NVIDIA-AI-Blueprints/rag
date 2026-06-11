# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for agentic RAG helpers (no live LLM or network)."""

from __future__ import annotations

import inspect
import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import nvidia_rag.rag_server.agentic_rag as agentic_rag_pkg
from nvidia_rag.rag_server.agentic_rag import (
    AgenticRAGGraphState,
    AgenticSearchParams,
    build_agentic_rag_agent,
    make_retriever_fn,
    run_agentic_pipeline,
)
from nvidia_rag.rag_server.agentic_rag import runner as agentic_runner
from nvidia_rag.rag_server.agentic_rag.agentic_rag import AgenticRag
from nvidia_rag.rag_server.agentic_rag.builder import (
    AgenticLLMOverrides,
    _agentic_all_citations,
    _agentic_llm_overrides,
    _agentic_search_params,
)
from nvidia_rag.rag_server.agentic_rag.prompt import build_prompts
from nvidia_rag.rag_server.agentic_rag.tracing import (
    AgentMetrics,
    LLMCallRecord,
    QueryTrace,
    RetrievalRecord,
    get_current_trace,
)
from nvidia_rag.rag_server.response_generator import (
    Citations,
    ErrorCodeMapping,
    SourceMetadata,
    SourceResult,
)
from nvidia_rag.utils.agentic_rag_config import AgenticRAGConfig


def _minimal_agent() -> AgenticRag:
    return AgenticRag(
        planner_llm=MagicMock(),
        task_llm=MagicMock(),
        seed_gen_llm=MagicMock(),
        synthesis_llm=MagicMock(),
        retriever_fn=AsyncMock(return_value=None),
        log_level="WARNING",
        concurrency_limit=2,
    )


class TestAgenticRagPackageExports:
    def test_public_api_matches_all(self) -> None:
        expected = {
            "AgenticRag",
            "AgenticRAGGraphState",
            "AgenticSearchParams",
            "_agentic_search_params",
            "build_agentic_rag_agent",
            "make_retriever_fn",
            "run_agentic_pipeline",
        }
        assert set(agentic_rag_pkg.__all__) == expected
        assert AgenticRag is agentic_rag_pkg.AgenticRag
        assert run_agentic_pipeline is agentic_runner.run_agentic_pipeline


class TestAgenticRAGGraphState:
    def test_defaults(self) -> None:
        state = AgenticRAGGraphState(user_query="hello")
        assert state.user_query == "hello"
        assert state.initial_context == []
        assert state.retrieval_plan == {}
        assert state.task_results == {}
        assert state.final_answer == ""
        assert state.verification_round == 0


class TestAgenticSearchParams:
    def test_dataclass_defaults(self) -> None:
        p = AgenticSearchParams()
        assert p.collection_names is None
        assert p.vdb_top_k is None
        assert p.vdb_auth_token == ""
        assert p.filter_expr == ""


class TestAgenticRagStaticHelpers:
    def test_filter_think_tokens_no_tags(self) -> None:
        text = '{"a": 1}'
        assert AgenticRag._filter_think_tokens(text) is text

    def test_filter_think_tokens_strips_closed_block(self) -> None:
        raw = '<think>ignore</think>{"scope_only": false}'
        assert AgenticRag._filter_think_tokens(raw) == '{"scope_only": false}'

    def test_filter_think_tokens_truncated_block(self) -> None:
        assert AgenticRag._filter_think_tokens("<think>no close") == ""

    def test_rebuild_result_text_vs_chart(self) -> None:
        text_chunk = {
            "doc_name": "a.pdf",
            "content": "body",
            "score": 0.9,
            "document_type": "text",
        }
        r_text = AgenticRag._rebuild_result(text_chunk)
        assert r_text["document_name"] == "a.pdf"
        assert r_text["content"] == "body"
        assert "metadata" not in r_text

        chart_chunk = {
            "doc_name": "c.png",
            "content": "chart desc",
            "document_type": "chart",
        }
        r_chart = AgenticRag._rebuild_result(chart_chunk)
        assert r_chart["metadata"]["description"] == "chart desc"

    def test_get_text_content_uses_metadata_description_for_images(self) -> None:
        desc = "  alt text  "
        assert (
            AgenticRag._get_text_content(
                {
                    "document_type": "image",
                    "content": "",
                    "metadata": {"description": desc},
                }
            )
            == desc
        )

    def test_clean_answer_strips_markdown_headers(self) -> None:
        out = AgenticRag._clean_answer("## Title\n\nplain")
        assert "Title" in out
        assert "##" not in out


class TestAgenticRagInstanceHelpers:
    def test_extract_chunks_from_model_dump_shape(self) -> None:
        agent = _minimal_agent()
        dumped = {
            "results": [
                {
                    "document_name": "doc.txt",
                    "content": "hello world",
                    "score": 0.5,
                    "document_type": "text",
                }
            ],
            "total_results": 1,
        }
        chunks = agent._extract_chunks(dumped)
        assert len(chunks) == 1
        assert chunks[0]["doc_name"] == "doc.txt"
        assert chunks[0]["content"] == "hello world"

    def test_extract_chunks_skips_empty_image_without_description(self) -> None:
        agent = _minimal_agent()
        dumped = {
            "results": [
                {
                    "document_name": "x.png",
                    "content": "   ",
                    "document_type": "image",
                    "metadata": {},
                }
            ],
            "total_results": 1,
        }
        assert agent._extract_chunks(dumped) == []

    def test_format_chunks_for_prompt_sorts_by_score(self) -> None:
        agent = _minimal_agent()
        chunks = [
            {"doc_name": "low", "content": "b", "score": 0.1, "document_type": "text"},
            {"doc_name": "high", "content": "a", "score": 0.9, "document_type": "text"},
        ]
        text = agent._format_chunks_for_prompt(chunks, max_tokens=100_000)
        assert text.index("high") < text.index("low")

    def test_parse_task_answer_json_and_plain(self) -> None:
        agent = _minimal_agent()
        j = '{"completeness": "partial", "answer": "**x**", "missing": "y"}'
        parsed = agent._parse_task_answer(j)
        assert parsed["completeness"] == "partial"
        assert parsed["answer"] == "x"
        assert parsed["missing"] == "y"

        plain = agent._parse_task_answer("  direct answer  ")
        assert plain["completeness"] == "complete"
        assert plain["answer"] == "direct answer"

    def test_build_execution_levels_adds_ids(self) -> None:
        agent = _minimal_agent()
        tasks = [{"question": "q1", "query": "r1"}]
        levels = agent._build_execution_levels(tasks)
        assert len(levels) == 1
        assert levels[0][0]["id"] == "auto_1"


class TestMakeRetrieverFn:
    @pytest.mark.asyncio
    async def test_forwards_search_kwargs_from_context(self) -> None:
        mock_rag = MagicMock()
        mock_rag.search = AsyncMock(
            return_value=Citations(
                total_results=0,
                results=[],
            )
        )
        retriever = make_retriever_fn(mock_rag, default_reranker_top_k=7)
        params = AgenticSearchParams(
            collection_names=["col_a"],
            vdb_top_k=12,
            reranker_top_k=None,
        )
        token = _agentic_search_params.set(params)
        acc_token = _agentic_all_citations.set(OrderedDict())
        try:
            out = await retriever("search me", stage="execute")
            assert out is not None
            mock_rag.search.assert_awaited_once()
            call_kw = mock_rag.search.await_args.kwargs
            assert call_kw["query"] == "search me"
            assert call_kw["collection_names"] == ["col_a"]
            assert call_kw["vdb_top_k"] == 12
            assert call_kw["reranker_top_k"] == 7
            assert call_kw["stage"] == "execute"
        finally:
            _agentic_search_params.reset(token)
            _agentic_all_citations.reset(acc_token)

    @pytest.mark.asyncio
    async def test_accumulates_citations_when_context_set(self) -> None:
        mock_rag = MagicMock()
        sr = SourceResult(
            document_name="n",
            content="c",
            metadata=SourceMetadata(),
            stage="execute",
        )
        mock_rag.search = AsyncMock(
            return_value=Citations(total_results=1, results=[sr])
        )
        retriever = make_retriever_fn(mock_rag, default_reranker_top_k=5)
        acc: OrderedDict[str, list[SourceResult]] = OrderedDict()
        p_token = _agentic_search_params.set(AgenticSearchParams())
        c_token = _agentic_all_citations.set(acc)
        try:
            await retriever("q", stage="execute")
            assert "execute" in acc
            assert len(acc["execute"]) == 1
            assert acc["execute"][0].document_name == "n"
        finally:
            _agentic_search_params.reset(p_token)
            _agentic_all_citations.reset(c_token)

    @pytest.mark.asyncio
    async def test_returns_none_on_search_exception(self) -> None:
        mock_rag = MagicMock()
        mock_rag.search = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = make_retriever_fn(mock_rag, default_reranker_top_k=3)
        p_token = _agentic_search_params.set(AgenticSearchParams())
        c_token = _agentic_all_citations.set(OrderedDict())
        try:
            assert await retriever("q") is None
        finally:
            _agentic_search_params.reset(p_token)
            _agentic_all_citations.reset(c_token)


class TestBuildPrompts:
    def test_injects_limits_and_returns_expected_keys(self) -> None:
        prompts = build_prompts(max_plan_tasks=4, max_verification_tasks=2)
        assert set(prompts.keys()) == {
            "planner_prompt",
            "task_answer_prompt",
            "seed_gen_prompt",
            "synthesis_prompt",
            "verification_prompt",
            "planner_replan_instruction",
        }
        planner_msgs = prompts["planner_prompt"].messages
        system_tpl = planner_msgs[0].prompt.template
        assert "Maximum 4 tasks." in system_tpl
        ver_msgs = prompts["verification_prompt"].messages
        ver_system = ver_msgs[0].prompt.template
        assert "Maximum 2 tasks." in ver_system


class TestTracing:
    def test_query_trace_record_and_totals(self) -> None:
        t = QueryTrace(query_text="q")
        t.record_llm_call("step", 10, 20, 5.0)
        t.finalize()
        assert t.total_llm_calls == 1
        assert t.total_input_tokens == 10
        assert t.total_output_tokens == 20
        d = t.to_dict()
        assert d["query_text"] == "q"
        assert d["total_llm_calls"] == 1
        assert "query_id=" in t.one_line_summary()

    def test_agent_metrics_summary_empty_and_nonempty(self) -> None:
        m = AgentMetrics()
        assert m.summary() == {"total_queries": 0}
        tr = QueryTrace(query_text="z")
        tr.llm_calls.append(LLMCallRecord("a", 1, 1, 1.0))
        tr.finalize()
        m.update(tr)
        s = m.summary()
        assert s["total_queries"] == 1
        assert "llm_calls" in s

    def test_get_current_trace_default_none(self) -> None:
        assert get_current_trace() is None

    def test_query_trace_serializes_retrieval_calls(self) -> None:
        tr = QueryTrace(query_text="z")
        tr.retrieval_calls.append(
            RetrievalRecord(
                stage="initial_retrieval",
                chunks=3,
                duration_ms=12.34,
                error=False,
            )
        )
        data = tr.to_dict()
        assert data["retrieval_calls"] == [
            {
                "stage": "initial_retrieval",
                "chunks": 3,
                "duration_ms": 12.3,
                "error": False,
            }
        ]


class _DummyOtelMetrics:
    def __init__(self) -> None:
        self.agentic_calls = []
        self.latency_updates = []

    def record_agentic_query_trace(
        self, trace, *, status: str, verification_enabled: bool
    ) -> None:
        self.agentic_calls.append(
            {
                "trace": trace,
                "status": status,
                "verification_enabled": verification_enabled,
            }
        )

    def update_latency_metrics(self, payload: dict) -> None:
        self.latency_updates.append(payload)


class TestRunAgenticPipeline:
    @pytest.mark.asyncio
    async def test_success_wraps_final_answer(self) -> None:
        # Non-streaming path: uses graph.ainvoke and wraps the answer in a
        # single SSE chunk via generate_answer_async.
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"final_answer": "synthesized"})

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="user q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=False,
        )
        assert resp.status_code == ErrorCodeMapping.SUCCESS
        blob = "".join([c async for c in resp.generator])
        assert "synthesized" in blob

    @pytest.mark.asyncio
    async def test_failure_returns_server_error_message(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=ValueError("graph broke"))

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=False,
        )
        assert resp.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        text = "".join([c async for c in resp.generator])
        assert "error" in text.lower() or "encountered" in text.lower()

    @pytest.mark.asyncio
    async def test_non_streaming_records_agentic_metrics(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"final_answer": "synthesized"})
        metrics = _DummyOtelMetrics()

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=True)

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="user q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=False,
            metrics=metrics,
        )

        assert resp.status_code == ErrorCodeMapping.SUCCESS
        assert len(metrics.agentic_calls) == 1
        assert metrics.agentic_calls[0]["status"] == "success"
        assert metrics.agentic_calls[0]["verification_enabled"] is True

    @pytest.mark.asyncio
    async def test_streaming_records_agentic_metrics(self) -> None:
        class _FakeChunk:
            def __init__(self, content: str = "") -> None:
                self.content = content
                self.additional_kwargs = {}

        async def _fake_astream(_state, *, config, stream_mode):  # noqa: ARG001
            yield (
                "messages",
                (_FakeChunk(content="done"), {"langgraph_node": "synthesize"}),
            )

        graph = MagicMock()
        graph.astream = _fake_astream
        metrics = _DummyOtelMetrics()

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=False)

        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=AgenticRAGConfig(),
            search_params=AgenticSearchParams(),
            enable_streaming=True,
            metrics=metrics,
        )
        chunks = [c async for c in resp.generator]

        assert chunks
        assert len(metrics.agentic_calls) == 1
        assert metrics.agentic_calls[0]["status"] == "success"
        assert metrics.agentic_calls[0]["verification_enabled"] is False

    @pytest.mark.asyncio
    async def test_streaming_error_records_agentic_metrics(self) -> None:
        async def _fake_astream(_state, *, config, stream_mode):  # noqa: ARG001
            raise RuntimeError("boom")
            yield  # pragma: no cover

        graph = MagicMock()
        graph.astream = _fake_astream
        metrics = _DummyOtelMetrics()

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=False)

        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=AgenticRAGConfig(),
            search_params=AgenticSearchParams(),
            enable_streaming=True,
            metrics=metrics,
        )
        chunks = [json.loads(c.removeprefix("data: ")) async for c in resp.generator]

        assert chunks[-1]["event_type"] == "error"
        assert len(metrics.agentic_calls) == 1
        assert metrics.agentic_calls[0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_streaming_emits_stage_and_final_chunks(self) -> None:
        """Verify-disabled streaming path: synthesize tokens stream live as
        ``final_answer`` chunks; stage_start/stage_end labels come from
        USER_FACING_LABELS."""

        class _FakeChunk:
            def __init__(self, content: str = "", reasoning: str | None = None) -> None:
                self.content = content
                self.additional_kwargs = (
                    {"reasoning_content": reasoning} if reasoning else {}
                )
                self.usage_metadata = None

        async def _fake_astream(_state, *, config, stream_mode):  # noqa: ARG001
            # 1) plan node starts (intermediate)
            yield (
                "custom",
                {
                    "node": "plan",
                    "event": "stage_start",
                    "key": "plan.start.scope",
                    "params": {},
                },
            )
            yield (
                "messages",
                (
                    _FakeChunk(reasoning="thinking about plan"),
                    {"langgraph_node": "plan"},
                ),
            )
            yield (
                "custom",
                {
                    "node": "plan",
                    "event": "stage_end",
                    "key": "plan.end.with_tasks",
                    "params": {"task_count": 1},
                },
            )
            # 2) synthesize streams tokens — labeled final_answer because the
            # agent has verification disabled (no draft/revision dance).
            yield (
                "messages",
                (_FakeChunk(content="Hello "), {"langgraph_node": "synthesize"}),
            )
            yield (
                "messages",
                (_FakeChunk(content="world"), {"langgraph_node": "synthesize"}),
            )

        graph = MagicMock()
        graph.astream = _fake_astream

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=False)

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=True,
        )
        assert resp.status_code == ErrorCodeMapping.SUCCESS
        chunks = [c async for c in resp.generator]
        assert chunks, "expected at least one SSE chunk"

        # Parse the SSE-encoded ChainResponse JSONs.
        parsed = [json.loads(c.removeprefix("data: ").strip()) for c in chunks]
        event_types = [p.get("event_type") for p in parsed]
        # Expected ordering: stage_start, intermediate_reasoning, stage_end,
        # final_answer*, then the trailing finish chunk (event_type=None).
        assert "stage_start" in event_types
        assert "intermediate_reasoning" in event_types
        assert "stage_end" in event_types
        assert event_types.count("final_answer") == 2

        # The final-answer chunks carry tokens in delta.content.
        final_tokens = [
            p["choices"][0]["delta"]["content"]
            for p in parsed
            if p.get("event_type") == "final_answer"
        ]
        assert "".join(final_tokens) == "Hello world"

        # Trailing finish chunk has finish_reason="stop".
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"

        # Stage_start / stage_end labels come from USER_FACING_LABELS, not from
        # the writer payload — assert the dict is the source of truth.
        from nvidia_rag.rag_server.agentic_rag.streaming import USER_FACING_LABELS

        stage_start_labels = [
            p["choices"][0]["message"]["reasoning_content"]
            for p in parsed
            if p.get("event_type") == "stage_start"
        ]
        assert USER_FACING_LABELS["plan.start.scope"] in stage_start_labels

        stage_end_labels = [
            p["choices"][0]["message"]["reasoning_content"]
            for p in parsed
            if p.get("event_type") == "stage_end"
        ]
        assert any("1" in lbl for lbl in stage_end_labels), (
            f"expected substituted task_count, got: {stage_end_labels}"
        )

    @pytest.mark.asyncio
    async def test_streaming_buffers_parallel_tokens_per_run(self) -> None:
        """Tokens from parallel-task nodes (execute, verify_execute) must be
        buffered per run_id and emitted as one consolidated chunk per task at
        the node's stage_end — preserving information without jumbling.
        Synthesize-node tokens still stream live (single LLM call)."""

        class _FakeChunk:
            def __init__(
                self,
                content: str = "",
                reasoning: str | None = None,
                id: str | None = None,
            ) -> None:
                self.content = content
                self.id = id
                self.additional_kwargs = (
                    {"reasoning_content": reasoning} if reasoning else {}
                )
                self.usage_metadata = None

        async def _fake_astream(_state, *, config, stream_mode):  # noqa: ARG001
            yield (
                "custom",
                {
                    "node": "execute",
                    "event": "stage_start",
                    "key": "execute.start.answer",
                    "params": {"task_count": 2},
                },
            )
            # Two parallel tasks emitting interleaved tokens — distinguished
            # by chunk.id (each LangChain LLM invocation gets its own UUID).
            # Task A produces reasoning + an answer; Task B does the same.
            # Tokens are interleaved character-by-character to simulate the
            # real wire jumble.
            yield (
                "messages",
                (
                    _FakeChunk(reasoning="lion ", id="run-A"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(reasoning="hammer ", id="run-B"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(reasoning="activity", id="run-A"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(reasoning="cost", id="run-B"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(content='{"answer": "sun', id="run-A"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(content='{"answer": "$2', id="run-B"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(content='screen"}', id="run-A"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "messages",
                (
                    _FakeChunk(content='0.00"}', id="run-B"),
                    {"langgraph_node": "execute"},
                ),
            )
            yield (
                "custom",
                {
                    "node": "execute",
                    "event": "stage_end",
                    "key": "execute.end.done",
                    "params": {"task_count": 2, "answered": 2},
                },
            )
            # Synthesize: single-LLM-call node — tokens stream live.  Verify
            # is disabled in this test, so synthesize tokens go straight to
            # ``final_answer`` rather than being held back for end-of-stream.
            yield (
                "messages",
                (
                    _FakeChunk(content="final answer", id="run-S"),
                    {"langgraph_node": "synthesize"},
                ),
            )

        graph = MagicMock()
        graph.astream = _fake_astream

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=False)

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=True,
        )
        assert resp.status_code == ErrorCodeMapping.SUCCESS
        chunks = [c async for c in resp.generator]
        parsed = [json.loads(c.removeprefix("data: ").strip()) for c in chunks]

        # Exactly 2 intermediate_reasoning + 2 intermediate_output chunks
        # from "execute" (one consolidated chunk per run_id).
        execute_reasoning = [
            p
            for p in parsed
            if p.get("event_type") == "intermediate_reasoning"
            and p.get("stage") == "execute"
        ]
        execute_output = [
            p
            for p in parsed
            if p.get("event_type") == "intermediate_output"
            and p.get("stage") == "execute"
        ]
        assert len(execute_reasoning) == 2, (
            f"expected 2 consolidated reasoning chunks, got {len(execute_reasoning)}: "
            f"{[p['choices'][0]['message']['reasoning_content'] for p in execute_reasoning]}"
        )
        assert len(execute_output) == 2, (
            f"expected 2 consolidated output chunks, got {len(execute_output)}: "
            f"{[p['choices'][0]['message']['reasoning_content'] for p in execute_output]}"
        )

        # Each consolidated chunk holds ONE task's full text — no cross-task
        # mixing.  Run A's reasoning is "lion activity"; Run B's is "hammer
        # cost".  Order between run A and run B follows insertion order
        # (run A first, since it produced the first chunk).
        reasoning_texts = [
            p["choices"][0]["message"]["reasoning_content"] for p in execute_reasoning
        ]
        assert reasoning_texts == ["lion activity", "hammer cost"]

        output_texts = [
            p["choices"][0]["message"]["reasoning_content"] for p in execute_output
        ]
        assert output_texts == ['{"answer": "sunscreen"}', '{"answer": "$20.00"}']

        # The consolidated chunks land BEFORE the corresponding stage_end.
        execute_stage_end_idx = next(
            i
            for i, p in enumerate(parsed)
            if p.get("event_type") == "stage_end" and p.get("stage") == "execute"
        )
        for p in execute_reasoning + execute_output:
            assert parsed.index(p) < execute_stage_end_idx, (
                "consolidated chunks must precede stage_end"
            )

        # Synthesize tokens DID pass through (single-LLM-call node, lives on).
        final_answer_chunks = [
            p for p in parsed if p.get("event_type") == "final_answer"
        ]
        assert final_answer_chunks, "expected final_answer to pass through"

    @pytest.mark.asyncio
    async def test_streaming_verify_enabled_emits_single_final_chunk(self) -> None:
        """Verify-enabled streaming path: synthesize tokens stream as
        ``intermediate_*`` and the user-facing answer is delivered as a single
        ``final_answer`` chunk at stream end, sourced from ``state.final_answer``
        captured via ``stream_mode="values"``."""

        class _FakeChunk:
            def __init__(self, content: str = "") -> None:
                self.content = content
                self.additional_kwargs: dict[str, str] = {}
                self.usage_metadata = None

        async def _fake_astream(_state, *, config, stream_mode):  # noqa: ARG001
            assert "values" in stream_mode, (
                "translator must subscribe to values mode to track state.final_answer"
            )
            # Initial state snapshot — empty answer, must be ignored.
            yield ("values", {"final_answer": ""})
            # First synthesize pass — tokens here would have been labeled as
            # final_answer in the legacy code, but with verification on they
            # must surface as intermediate_output (delta.reasoning_content).
            yield (
                "messages",
                (_FakeChunk(content="draft "), {"langgraph_node": "synthesize"}),
            )
            yield (
                "messages",
                (_FakeChunk(content="answer"), {"langgraph_node": "synthesize"}),
            )
            yield ("values", {"final_answer": "draft answer"})
            # Verification finds gaps -> re-synthesize.  Same intermediate
            # treatment for these tokens; only state.final_answer matters
            # for the final emission.
            yield (
                "messages",
                (_FakeChunk(content="revised "), {"langgraph_node": "synthesize"}),
            )
            yield (
                "messages",
                (_FakeChunk(content="answer"), {"langgraph_node": "synthesize"}),
            )
            yield ("values", {"final_answer": "revised answer"})

        graph = MagicMock()
        graph.astream = _fake_astream

        class _Agent:
            def __init__(self) -> None:
                self.metrics = AgentMetrics()
                self.verification_cfg = SimpleNamespace(enabled=True)

        cfg = AgenticRAGConfig()
        resp = await run_agentic_pipeline(
            agent=_Agent(),
            graph=graph,
            query="q",
            cfg=cfg,
            search_params=AgenticSearchParams(),
            enable_streaming=True,
        )
        assert resp.status_code == ErrorCodeMapping.SUCCESS
        chunks = [c async for c in resp.generator]
        parsed = [json.loads(c.removeprefix("data: ").strip()) for c in chunks]

        # No live final_answer tokens during synthesize — only the single
        # end-of-stream chunk.
        final_answer_chunks = [
            p for p in parsed if p.get("event_type") == "final_answer"
        ]
        assert len(final_answer_chunks) == 1, (
            f"expected exactly one final_answer chunk, got {len(final_answer_chunks)}"
        )
        assert final_answer_chunks[0]["choices"][0]["delta"]["content"] == (
            "revised answer"
        )

        # Synthesize tokens were re-routed to intermediate_output, with the
        # content riding in delta.reasoning_content (per the agentic protocol).
        synth_intermediate = [
            p
            for p in parsed
            if p.get("event_type") == "intermediate_output"
            and p.get("stage") == "synthesize"
        ]
        assert len(synth_intermediate) == 4, (
            f"expected 4 intermediate_output chunks (2 per synthesize pass), "
            f"got {len(synth_intermediate)}"
        )
        assert (
            "".join(
                p["choices"][0]["delta"]["reasoning_content"]
                for p in synth_intermediate
            )
            == "draft answerrevised answer"
        )

        # The single final_answer chunk lands AFTER the last intermediate one.
        last_intermediate_idx = parsed.index(synth_intermediate[-1])
        final_idx = parsed.index(final_answer_chunks[0])
        assert final_idx > last_intermediate_idx, (
            "final_answer must follow all intermediate synthesize chunks"
        )

        # Trailing finish chunk still terminates the stream.
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


class TestAgenticLLMOverrides:
    """Runtime LLM override (FR-1527): model / llm_endpoint passed in the
    /generate request flow into all 4 agentic role properties via the
    ``_agentic_llm_overrides`` ContextVar — preserving the cached default
    LLMs when no override is set and isolating concurrent requests."""

    def test_no_override_returns_cached_default_llms(self) -> None:
        agent = _minimal_agent()
        # All 4 role properties return the per-role cached defaults.
        assert agent.planner_llm is agent._default_planner_llm
        assert agent.task_llm is agent._default_task_llm
        assert agent.seed_gen_llm is agent._default_seed_gen_llm
        assert agent.synthesis_llm is agent._default_synthesis_llm

    def test_override_replaces_all_four_role_llms_with_same_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub get_llm so we don't touch real LangChain / NIM endpoints.
        built = MagicMock(name="override-llm")
        get_llm_calls: list[dict] = []

        def fake_get_llm(**kwargs):
            get_llm_calls.append(kwargs)
            return built

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        agent = _minimal_agent()
        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(
                model="custom/model-x",
                llm_endpoint="https://endpoint-x:8000",
                api_key="key-abc",
            )
        )
        try:
            # All four role properties yield the override-built client …
            assert agent.planner_llm is built
            assert agent.task_llm is built
            assert agent.seed_gen_llm is built
            assert agent.synthesis_llm is built
        finally:
            _agentic_llm_overrides.reset(token)

        # … and the override client is built exactly once per request even
        # though four roles asked for it (per-request cache on the dataclass).
        assert len(get_llm_calls) == 1
        assert get_llm_calls[0]["model"] == "custom/model-x"
        assert get_llm_calls[0]["llm_endpoint"] == "https://endpoint-x:8000"
        assert get_llm_calls[0]["api_key"] == "key-abc"

    def test_override_with_empty_fields_falls_back_to_defaults(self) -> None:
        agent = _minimal_agent()
        # AgenticLLMOverrides with both fields empty/None is treated as "no
        # override" — defaults still apply.  Defensive guard for callers that
        # set the ContextVar unconditionally.
        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(model=None, llm_endpoint=None, api_key="key")
        )
        try:
            assert agent.planner_llm is agent._default_planner_llm
            assert agent.task_llm is agent._default_task_llm
        finally:
            _agentic_llm_overrides.reset(token)

    @pytest.mark.asyncio
    async def test_concurrent_requests_see_independent_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two coroutines on the same event loop, each with its own override,
        must observe their own client — proves no ContextVar leakage."""
        import asyncio

        # Per-request, get_llm is called once and returns a unique stub.  We
        # key the stub by the model name so each coroutine can assert it got
        # the right one.
        def fake_get_llm(**kwargs):
            stub = MagicMock(name=f"llm-{kwargs['model']}")
            stub.model_name = kwargs["model"]
            return stub

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        agent = _minimal_agent()

        async def one_request(model_name: str) -> str:
            token = _agentic_llm_overrides.set(
                AgenticLLMOverrides(model=model_name, llm_endpoint="https://x")
            )
            try:
                # Yield control so the other coroutine can run with its own
                # override set — this is the real concurrency stress.
                await asyncio.sleep(0)
                # All four roles must see this coroutine's model.
                assert agent.planner_llm.model_name == model_name
                assert agent.task_llm.model_name == model_name
                assert agent.seed_gen_llm.model_name == model_name
                assert agent.synthesis_llm.model_name == model_name
                return agent.planner_llm.model_name
            finally:
                _agentic_llm_overrides.reset(token)

        results = await asyncio.gather(
            one_request("model-A"),
            one_request("model-B"),
        )
        assert sorted(results) == ["model-A", "model-B"]


class TestBuildAgenticRagAgentSignature:
    """Ensure the factory remains awaitable (import-time contract)."""

    def test_build_agentic_rag_agent_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(build_agentic_rag_agent)


class TestAgenticRoleGenerationParamFallback:
    """Per-role generation-param fallback in builder._make_role_llm.

    Covers Case 2 of the FR (no /generate override): each role's LLM is built
    with temperature / top_p / max_tokens resolved via
        role_cfg → planner_cfg → rag_config.llm.parameters.
    """

    def _build_rag_config(
        self,
        *,
        main_temperature: float | None = None,
        main_top_p: float | None = None,
        main_max_tokens: int = 32768,
        planner_overrides: dict | None = None,
        task_overrides: dict | None = None,
    ):
        """Construct a minimal NvidiaRAGConfig and patch in role overrides."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        cfg = NvidiaRAGConfig()
        # Force main-LLM generation params explicitly (cls defaults read env).
        cfg.llm.parameters.temperature = main_temperature
        cfg.llm.parameters.top_p = main_top_p
        cfg.llm.parameters.max_tokens = main_max_tokens

        for field_name, value in (planner_overrides or {}).items():
            setattr(cfg.agentic_rag.planner_llm, field_name, value)
        for field_name, value in (task_overrides or {}).items():
            setattr(cfg.agentic_rag.task_llm, field_name, value)
        return cfg

    def test_role_value_wins_over_planner_and_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_temperature=0.9,
            main_top_p=0.99,
            main_max_tokens=8000,
            planner_overrides={
                "model_name": "planner/model",
                "temperature": 0.5,
                "top_p": 0.7,
                "max_tokens": 4000,
            },
            task_overrides={
                "model_name": "task/model",
                "temperature": 0.1,
                "top_p": 0.3,
                "max_tokens": 1000,
            },
        )

        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        assert calls, "get_llm must have been invoked"
        kw = calls[-1]
        assert kw["temperature"] == 0.1
        assert kw["top_p"] == 0.3
        assert kw["max_tokens"] == 1000
        assert kw["model"] == "task/model"

    def test_planner_value_wins_when_role_gen_params_explicitly_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a role's gen-param fields are explicitly set to None (rare —
        the role defaults are non-None — but valid via the typing), the
        fallback chain hops to the planner role's values."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_temperature=0.9,
            main_top_p=0.99,
            main_max_tokens=8000,
            planner_overrides={
                "model_name": "planner/model",
                "temperature": 0.5,
                "top_p": 0.7,
                "max_tokens": 4000,
            },
            # task role leaves model_name empty AND explicitly nulls gen params
            # so the fallback to planner fires.
            task_overrides={
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
            },
        )

        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        # Per-role model_name is empty → falls back to planner's model.
        assert kw["model"] == "planner/model"
        # Generation params fall back to planner's values.
        assert kw["temperature"] == 0.5
        assert kw["top_p"] == 0.7
        assert kw["max_tokens"] == 4000

    def test_main_rag_config_value_wins_when_role_and_planner_gen_params_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When BOTH the role and the planner have None gen params, the chain
        falls all the way through to ``rag_config.llm.parameters``."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        # Planner has a model_name (so a model can be resolved) but nulled
        # gen params; task likewise nulled.
        cfg = self._build_rag_config(
            main_temperature=0.42,
            main_top_p=0.55,
            main_max_tokens=12345,
            planner_overrides={
                "model_name": "planner/model",
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
            },
            task_overrides={
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
            },
        )

        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        # All gen params fall through to rag_config.llm.parameters.
        assert kw["temperature"] == 0.42
        assert kw["top_p"] == 0.55
        assert kw["max_tokens"] == 12345

    def test_role_defaults_used_when_no_overrides_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each role's built-in defaults (planner/seed_gen=0.1, task/synthesis=0.0,
        top_p=1.0, max_tokens=32768) are applied when no env vars are set."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = NvidiaRAGConfig()
        # Give planner a model_name so a model can be resolved; leave gen
        # params at their built-in defaults.
        cfg.agentic_rag.planner_llm.model_name = "planner/m"

        for role_name, expected_temp in [
            ("planner_llm", 0.1),
            ("task_llm", 0.0),
            ("seed_gen_llm", 0.1),
            ("synthesis_llm", 0.0),
        ]:
            calls.clear()
            role_cfg = getattr(cfg.agentic_rag, role_name)
            _make_role_llm(role_cfg, cfg.agentic_rag.planner_llm, cfg)
            kw = calls[-1]
            assert kw["temperature"] == expected_temp, (
                f"{role_name} expected temperature={expected_temp}, got "
                f"{kw['temperature']}"
            )
            assert kw["top_p"] == 1.0, (
                f"{role_name} expected top_p=1.0, got {kw['top_p']}"
            )
            assert kw["max_tokens"] == 32768, (
                f"{role_name} expected max_tokens=32768, got {kw['max_tokens']}"
            )


class TestAgenticRuntimeGenerationParamOverrides:
    """Per-request /generate overrides for temperature / top_p / max_tokens.

    Covers Case 1 of the FR: values passed at request time take precedence
    over every env-var level and are applied to every role's LLM.
    """

    def test_temperature_only_override_keeps_role_models_but_applies_temp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No model/endpoint override → each role keeps its configured model,
        but the request-provided temperature is applied to all roles.
        """
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            stub = MagicMock(name=f"role-{kwargs.get('model')}")
            stub.model = kwargs.get("model")
            stub.temperature = kwargs.get("temperature")
            calls.append(kwargs)
            return stub

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        cfg = NvidiaRAGConfig()
        cfg.agentic_rag.planner_llm.model_name = "planner/m"
        cfg.agentic_rag.task_llm.model_name = "task/m"
        cfg.agentic_rag.seed_gen_llm.model_name = "seed/m"
        cfg.agentic_rag.synthesis_llm.model_name = "synth/m"

        agent = AgenticRag(
            planner_llm=MagicMock(),
            task_llm=MagicMock(),
            seed_gen_llm=MagicMock(),
            synthesis_llm=MagicMock(),
            retriever_fn=AsyncMock(return_value=None),
            log_level="WARNING",
            concurrency_limit=2,
            rag_config=cfg,
        )

        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(temperature=0.77)
        )
        try:
            # Each role builds its own client because model/endpoint are NOT
            # overridden — generation-param-only override doesn't collapse roles.
            llms = [
                agent.planner_llm,
                agent.task_llm,
                agent.seed_gen_llm,
                agent.synthesis_llm,
            ]
        finally:
            _agentic_llm_overrides.reset(token)

        # 4 distinct clients, one per role.
        assert len({id(x) for x in llms}) == 4
        assert len(calls) == 4
        # All four carry the override temperature.
        assert all(c["temperature"] == 0.77 for c in calls)
        # Each role still uses its own configured model.
        models = sorted(c["model"] for c in calls)
        assert models == ["planner/m", "seed/m", "synth/m", "task/m"]

    def test_model_override_collapses_all_roles_to_one_client_with_gen_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model + temperature overridden together → exactly one shared client
        is built, used by all 4 roles, with the request temperature applied.
        """
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []
        built = MagicMock(name="shared-override-llm")

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return built

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        cfg = NvidiaRAGConfig()
        cfg.agentic_rag.planner_llm.model_name = "planner/m"

        agent = AgenticRag(
            planner_llm=MagicMock(),
            task_llm=MagicMock(),
            seed_gen_llm=MagicMock(),
            synthesis_llm=MagicMock(),
            retriever_fn=AsyncMock(return_value=None),
            log_level="WARNING",
            concurrency_limit=2,
            rag_config=cfg,
        )

        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(
                model="runtime/model",
                llm_endpoint="https://runtime",
                temperature=0.33,
                top_p=0.66,
                max_tokens=999,
            )
        )
        try:
            assert agent.planner_llm is built
            assert agent.task_llm is built
            assert agent.seed_gen_llm is built
            assert agent.synthesis_llm is built
        finally:
            _agentic_llm_overrides.reset(token)

        # Single shared client → get_llm called exactly once.
        assert len(calls) == 1
        kw = calls[0]
        assert kw["model"] == "runtime/model"
        assert kw["llm_endpoint"] == "https://runtime"
        assert kw["temperature"] == 0.33
        assert kw["top_p"] == 0.66
        assert kw["max_tokens"] == 999

    def test_partial_gen_param_override_falls_back_per_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only temperature passed at request time → top_p uses task role's
        configured value; max_tokens falls all the way through to the main RAG
        LLM (per-field fallback)."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        cfg = NvidiaRAGConfig()
        cfg.llm.parameters.temperature = 0.99
        cfg.llm.parameters.top_p = 0.88
        cfg.llm.parameters.max_tokens = 7777
        cfg.agentic_rag.planner_llm.model_name = "planner/m"
        # Null out planner gen params so the chain can fall through them.
        cfg.agentic_rag.planner_llm.temperature = None
        cfg.agentic_rag.planner_llm.top_p = None
        cfg.agentic_rag.planner_llm.max_tokens = None
        # task_llm role has top_p set but not temperature/max_tokens.
        cfg.agentic_rag.task_llm.model_name = "task/m"
        cfg.agentic_rag.task_llm.temperature = None
        cfg.agentic_rag.task_llm.top_p = 0.42
        cfg.agentic_rag.task_llm.max_tokens = None

        agent = AgenticRag(
            planner_llm=MagicMock(),
            task_llm=MagicMock(),
            seed_gen_llm=MagicMock(),
            synthesis_llm=MagicMock(),
            retriever_fn=AsyncMock(return_value=None),
            log_level="WARNING",
            concurrency_limit=2,
            rag_config=cfg,
        )

        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(temperature=0.05)
        )
        try:
            _ = agent.task_llm
        finally:
            _agentic_llm_overrides.reset(token)

        # The single task-role build call should carry:
        # temperature → override value
        # top_p → task role's configured value (0.42)
        # max_tokens → main RAG LLM value (7777, not set on task or planner)
        task_calls = [c for c in calls if c.get("model") == "task/m"]
        assert task_calls, "expected get_llm call with task/m model"
        kw = task_calls[0]
        assert kw["temperature"] == 0.05
        assert kw["top_p"] == 0.42
        assert kw["max_tokens"] == 7777

    def test_runtime_overrides_take_precedence_over_role_env_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the per-role agentic env vars are set AND request-time
        gen-param overrides are also passed, the request values win."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        cfg = NvidiaRAGConfig()
        cfg.agentic_rag.planner_llm.model_name = "planner/m"
        cfg.agentic_rag.task_llm.model_name = "task/m"
        cfg.agentic_rag.task_llm.temperature = 0.1
        cfg.agentic_rag.task_llm.top_p = 0.2
        cfg.agentic_rag.task_llm.max_tokens = 1000

        agent = AgenticRag(
            planner_llm=MagicMock(),
            task_llm=MagicMock(),
            seed_gen_llm=MagicMock(),
            synthesis_llm=MagicMock(),
            retriever_fn=AsyncMock(return_value=None),
            log_level="WARNING",
            concurrency_limit=2,
            rag_config=cfg,
        )

        token = _agentic_llm_overrides.set(
            AgenticLLMOverrides(
                temperature=0.9,
                top_p=0.95,
                max_tokens=5000,
            )
        )
        try:
            _ = agent.task_llm
        finally:
            _agentic_llm_overrides.reset(token)

        task_calls = [c for c in calls if c.get("model") == "task/m"]
        assert task_calls
        kw = task_calls[0]
        assert kw["temperature"] == 0.9
        assert kw["top_p"] == 0.95
        assert kw["max_tokens"] == 5000

    def test_thinking_params_resolved_from_role_config_when_override_path_triggered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a temperature override triggers the _resolve_role_llm override path,
        enable_thinking / reasoning_budget / low_effort must come from the per-role
        agentic config — NOT from the global LLM_ENABLE_THINKING config.

        Regression test for the bug where the override path called get_llm() without
        passing thinking params, causing it to fall back to the global config.
        """
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)

        cfg = NvidiaRAGConfig()
        cfg.llm.parameters.enable_thinking = True
        cfg.llm.parameters.reasoning_budget = 256
        cfg.llm.parameters.low_effort = True
        cfg.agentic_rag.planner_llm.model_name = "planner/m"
        # Task role explicitly disables thinking (the defaults).
        cfg.agentic_rag.task_llm.model_name = "task/m"
        cfg.agentic_rag.task_llm.enable_thinking = False
        cfg.agentic_rag.task_llm.reasoning_budget = 0
        cfg.agentic_rag.task_llm.low_effort = False

        agent = AgenticRag(
            planner_llm=MagicMock(),
            task_llm=MagicMock(),
            seed_gen_llm=MagicMock(),
            synthesis_llm=MagicMock(),
            retriever_fn=AsyncMock(return_value=None),
            log_level="WARNING",
            concurrency_limit=2,
            rag_config=cfg,
        )

        # A temperature override triggers the override path.
        token = _agentic_llm_overrides.set(AgenticLLMOverrides(temperature=0.5))
        try:
            _ = agent.task_llm
        finally:
            _agentic_llm_overrides.reset(token)

        task_calls = [c for c in calls if c.get("model") == "task/m"]
        assert task_calls, "expected get_llm call with task/m model"
        kw = task_calls[0]
        # Thinking params must come from task role config, NOT global config.
        assert kw["enable_thinking"] is False, "task role thinking must be False even when global is True"
        assert kw["reasoning_budget"] == 0, "task role budget must be 0 even when global is 256"
        assert kw["low_effort"] is False, "task role low_effort must be False even when global is True"


class TestAgenticRoleThinkingParamFallback:
    """Per-role thinking-param fallback in builder._make_role_llm.

    Covers enable_thinking / reasoning_budget / low_effort resolved via
        role_cfg → planner_cfg → rag_config.llm.parameters.
    Default for all role thinking fields is None (inherit from main config).
    """

    def _build_rag_config(
        self,
        *,
        main_enable_thinking: bool = False,
        main_reasoning_budget: int = 0,
        main_low_effort: bool = False,
        planner_overrides: dict | None = None,
        task_overrides: dict | None = None,
        synthesis_overrides: dict | None = None,
    ):
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        cfg = NvidiaRAGConfig()
        cfg.llm.parameters.enable_thinking = main_enable_thinking
        cfg.llm.parameters.reasoning_budget = main_reasoning_budget
        cfg.llm.parameters.low_effort = main_low_effort
        cfg.agentic_rag.planner_llm.model_name = "planner/model"

        for field_name, value in (planner_overrides or {}).items():
            setattr(cfg.agentic_rag.planner_llm, field_name, value)
        for field_name, value in (task_overrides or {}).items():
            setattr(cfg.agentic_rag.task_llm, field_name, value)
        for field_name, value in (synthesis_overrides or {}).items():
            setattr(cfg.agentic_rag.synthesis_llm, field_name, value)
        return cfg

    def test_thinking_defaults_are_false(self) -> None:
        """All four role configs default thinking to off (false/0/false), independent of LLM_ENABLE_THINKING."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        cfg = NvidiaRAGConfig()
        for role_name in ("planner_llm", "task_llm", "seed_gen_llm", "synthesis_llm"):
            role_cfg = getattr(cfg.agentic_rag, role_name)
            assert role_cfg.enable_thinking is False, f"{role_name}.enable_thinking should default to False"
            assert role_cfg.reasoning_budget == 0, f"{role_name}.reasoning_budget should default to 0"
            assert role_cfg.low_effort is False, f"{role_name}.low_effort should default to False"

    def test_thinking_off_by_default_regardless_of_main_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Agentic roles default to thinking=false even when LLM_ENABLE_THINKING=true in main config."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_enable_thinking=True,
            main_reasoning_budget=4096,
            main_low_effort=True,
        )

        # Default role thinking fields are False/0/False, so they short-circuit
        # before reaching main config — thinking stays off.
        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        assert kw["enable_thinking"] is False
        assert kw["reasoning_budget"] == 0
        assert kw["low_effort"] is False

    def test_role_thinking_value_wins_over_main_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Role-specific thinking params override main config values."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_enable_thinking=True,
            main_reasoning_budget=4096,
            main_low_effort=True,
            task_overrides={
                "model_name": "task/model",
                "enable_thinking": False,
                "reasoning_budget": 512,
                "low_effort": False,
            },
        )

        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        assert kw["enable_thinking"] is False
        assert kw["reasoning_budget"] == 512
        assert kw["low_effort"] is False

    def test_planner_thinking_value_used_as_fallback_when_role_thinking_explicitly_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planner thinking params propagate to other roles only when those roles
        explicitly set their thinking fields to None (opt-in fallback)."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_enable_thinking=False,
            main_reasoning_budget=0,
            planner_overrides={
                "enable_thinking": True,
                "reasoning_budget": 2048,
                "low_effort": True,
            },
        )

        # Explicitly set task thinking fields to None to opt into fallback.
        cfg.agentic_rag.task_llm.enable_thinking = None
        cfg.agentic_rag.task_llm.reasoning_budget = None
        cfg.agentic_rag.task_llm.low_effort = None

        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        assert kw["enable_thinking"] is True
        assert kw["reasoning_budget"] == 2048
        assert kw["low_effort"] is True

    def test_role_thinking_default_false_does_not_inherit_planner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Roles with default thinking (False) do NOT inherit from planner even when planner has thinking enabled."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            planner_overrides={
                "enable_thinking": True,
                "reasoning_budget": 2048,
            },
        )

        # task uses default enable_thinking=False, which short-circuits the fallback.
        _make_role_llm(cfg.agentic_rag.task_llm, cfg.agentic_rag.planner_llm, cfg)
        kw = calls[-1]
        assert kw["enable_thinking"] is False
        assert kw["reasoning_budget"] == 0

    def test_per_role_thinking_independence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different roles can have independent thinking configurations."""
        calls: list[dict] = []

        def fake_get_llm(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("nvidia_rag.utils.llm.get_llm", fake_get_llm)
        from nvidia_rag.rag_server.agentic_rag.builder import _make_role_llm

        cfg = self._build_rag_config(
            main_enable_thinking=False,
            planner_overrides={
                "reasoning_budget": 256,
                "enable_thinking": True,
            },
            synthesis_overrides={
                "model_name": "synthesis/model",
                "reasoning_budget": 8192,
                "enable_thinking": True,
                "low_effort": False,
            },
        )

        calls.clear()
        _make_role_llm(cfg.agentic_rag.planner_llm, cfg.agentic_rag.planner_llm, cfg)
        planner_kw = calls[-1]

        calls.clear()
        _make_role_llm(cfg.agentic_rag.synthesis_llm, cfg.agentic_rag.planner_llm, cfg)
        synthesis_kw = calls[-1]

        assert planner_kw["reasoning_budget"] == 256
        assert synthesis_kw["reasoning_budget"] == 8192
        assert planner_kw["enable_thinking"] is True
        assert synthesis_kw["enable_thinking"] is True

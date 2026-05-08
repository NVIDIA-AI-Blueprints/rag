# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    _agentic_all_citations,
    _agentic_search_params,
)
from nvidia_rag.rag_server.agentic_rag.prompt import build_prompts
from nvidia_rag.rag_server.agentic_rag.tracing import (
    AgentMetrics,
    LLMCallRecord,
    QueryTrace,
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

    def test_sanitize_json_string_escapes_newlines_in_strings(self) -> None:
        dirty = '{"x": "line1\nline2"}'
        clean = AgenticRag._sanitize_json_string(dirty)
        assert "\n" not in clean.split('"x":')[1]
        assert json.loads(clean)["x"] == "line1\nline2"

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
    def test_parse_json_response_direct_and_embedded(self) -> None:
        agent = _minimal_agent()
        assert agent._parse_json_response('{"k": 1}') == {"k": 1}
        wrapped = 'prefix {"k": 2} suffix'
        assert agent._parse_json_response(wrapped) == {"k": 2}

    def test_parse_json_response_invalid_returns_error_dict(self) -> None:
        agent = _minimal_agent()
        out = agent._parse_json_response("not json at all")
        assert out.get("error") == "Failed to parse JSON"

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
    async def test_streaming_emits_stage_and_final_chunks(self) -> None:
        """Streaming path: graph.astream events are translated into SSE chunks
        with the expected ``event_type``/``stage`` fields."""

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
            # 2) synthesize: mark final, then stream tokens
            yield (
                "custom",
                {"node": "synthesize", "event": "final_stage_marker", "is_final": True},
            )
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
            # Synthesize: single-LLM-call node — tokens stream live.
            yield (
                "custom",
                {"node": "synthesize", "event": "final_stage_marker", "is_final": True},
            )
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


class TestBuildAgenticRagAgentSignature:
    """Ensure the factory remains awaitable (import-time contract)."""

    def test_build_agentic_rag_agent_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(build_agentic_rag_agent)

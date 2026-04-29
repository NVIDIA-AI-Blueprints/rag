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

"""Graph-Based Agentic RAG Agent.

Architecture: plan-and-execute with two-phase planning + verification gate.

  initial_retrieval → plan → execute ──(scope_only?)──→ plan (replan) → execute
                                      └─(otherwise)──→ synthesize → verify
                                                        └─(pass)──→ END
                                                        └─(fail)──→ verify_execute → synthesize → END

Phase 1 — scope discovery:
  When the query needs comprehensive coverage or has unresolved ambiguity, the
  planner creates scope-discovery tasks.  After they execute the planner is
  called again with the results.

Phase 2 — data retrieval:
  The planner creates targeted answer tasks.  Each task is a mini-agent with
  smart retry logic (seed-query generation on partial/none results).

Verification:
  After first synthesis a verification LLM checks the answer for gaps.  If
  issues are found, targeted re-retrieval tasks run and synthesis is repeated.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_P = "[AGENTIC_RAG]"


# ---------------------------------------------------------------------------
# Local fallback config dataclasses
# Used when no config objects are passed to AgenticRag.__init__.
# Production code passes AgenticRAGConfig sub-objects from
# nvidia_rag.utils.configuration, which expose the same attribute names.
# ---------------------------------------------------------------------------


@dataclass
class _PlannerConfig:
    max_plan_tasks: int = 5
    max_scope_rounds: int = 2
    max_attempts: int = 3


@dataclass
class _TaskExecutionConfig:
    scope_max_retries: int = 1
    answer_max_retries: int = 3


@dataclass
class _LLMTransportConfig:
    call_timeout: int = 300
    max_retries: int = 4


@dataclass
class _VerificationConfig:
    enabled: bool = True
    max_tasks: int = 3


@dataclass
class _ContextConfig:
    max_tokens: int = 100000


# =============================================================================
# STATE
# =============================================================================


class AgenticRAGGraphState(BaseModel):
    """LangGraph state passed between nodes."""

    user_query: str = Field(default="")
    initial_context: list[Any] = Field(default_factory=list)
    retrieval_plan: dict = Field(default_factory=dict)
    task_results: dict[str, dict] = Field(default_factory=dict)
    scope_results: dict[str, str] = Field(default_factory=dict)
    needs_replan: bool = Field(default=False)
    scope_rounds: int = Field(default=0)
    final_answer: str = Field(default="")
    verification_tasks: list[dict] = Field(default_factory=list)
    verification_results: dict[str, dict] = Field(default_factory=dict)
    verification_issues: list[str] = Field(default_factory=list)
    verification_round: int = Field(default=0)


# =============================================================================
# AGENT
# =============================================================================


class AgenticRag:
    """Plan-and-execute RAG agent with scope discovery and verification.

    Graph topology::

        retrieve → plan → execute → [replan?] → synthesize → [verify?] → END
    """

    _TEXT_DOCUMENT_TYPES = frozenset({"text", "audio"})
    _CHARS_PER_TOKEN_ESTIMATE = 2.5

    @staticmethod
    def _rebuild_result(chunk: dict) -> dict:
        """Reconstruct a retrieval-result dict from a stored chunk."""
        doc_type = chunk.get("document_type", "text")
        result: dict = {
            "document_name": chunk["doc_name"],
            "content": chunk["content"],
            "score": chunk.get("score", 0.0),
            "document_type": doc_type,
        }
        if doc_type not in AgenticRag._TEXT_DOCUMENT_TYPES:
            result["metadata"] = {"description": chunk["content"]}
        return result

    def __init__(
        self,
        planner_llm: BaseChatModel,
        task_llm: BaseChatModel,
        seed_gen_llm: BaseChatModel,
        synthesis_llm: BaseChatModel,
        retriever_fn: Callable[..., Any],
        log_level: str = "INFO",
        concurrency_limit: int = 3,
        planner_config=None,
        task_execution_config=None,
        llm_config=None,
        verification_config=None,
        context_config=None,
    ):
        self.planner_llm = planner_llm
        self.task_llm = task_llm
        self.seed_gen_llm = seed_gen_llm
        self.synthesis_llm = synthesis_llm
        self.retriever_fn = retriever_fn
        self.graph = None

        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)

        self.planner_cfg = planner_config or _PlannerConfig()
        self.task_exec_cfg = task_execution_config or _TaskExecutionConfig()
        self.llm_cfg = llm_config or _LLMTransportConfig()
        self.verification_cfg = verification_config or _VerificationConfig()
        self.context_cfg = context_config or _ContextConfig()

        self._tokenizer = None
        self._semaphore = asyncio.Semaphore(concurrency_limit)

        from nvidia_rag.rag_server.agentic_rag.tracing import AgentMetrics

        self.metrics = AgentMetrics()

        from nvidia_rag.rag_server.agentic_rag.prompt import build_prompts

        prompts = build_prompts(
            max_plan_tasks=self.planner_cfg.max_plan_tasks,
            max_verification_tasks=self.verification_cfg.max_tasks,
        )
        self.planner_prompt = prompts["planner_prompt"]
        self.task_answer_prompt = prompts["task_answer_prompt"]
        self.seed_gen_prompt = prompts["seed_gen_prompt"]
        self.synthesis_prompt = prompts["synthesis_prompt"]
        self.verification_prompt = prompts["verification_prompt"]
        self.planner_replan_instruction = prompts["planner_replan_instruction"]

        self._otel = otel_trace.get_tracer("agentic_rag")

        logger.info("%s Agent initialized (log_level=%s)", _P, log_level.upper())

    @contextmanager
    def _otel_and_trace(self, span_name: str, span_kind: str = "CHAIN"):
        """Open an OTel span and record QueryTrace node timing together."""
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        trace = get_current_trace()
        with self._otel.start_as_current_span(span_name) as span:
            span.set_attribute("openinference.span.kind", span_kind)
            if trace:
                with trace.trace_node(span_name):
                    yield span
            else:
                yield span

    # =========================================================================
    # TOKEN COUNTING
    # =========================================================================

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            return self._tokenizer
        except Exception:
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Fast approximate token count for budgeting (context truncation, etc.).

        Uses tiktoken ``cl100k_base`` as a reasonable cross-model estimator.
        NOT used for metrics — actual API-reported counts are used there.
        """
        try:
            tok = self._get_tokenizer()
            if tok is not None:
                return len(tok.encode(text))
            return len(text) // 4
        except Exception:
            return len(text) // 4

    # =========================================================================
    # LLM CALL (retry + timeout + rate-limit)
    # =========================================================================

    @staticmethod
    def _filter_think_tokens(content: str) -> str:
        """Strip <think>…</think> blocks from LLM output."""
        if not content or "<think>" not in content:
            return content
        if "</think>" in content:
            return content[content.rfind("</think>") + len("</think>") :].strip()
        logger.warning("%s Truncated <think> block (no closing tag), discarding", _P)
        return ""

    @staticmethod
    def _sanitize_json_string(raw: str) -> str:
        """Escape unescaped control characters inside JSON string values."""
        out: list[str] = []
        in_string = False
        i = 0
        length = len(raw)
        while i < length:
            ch = raw[i]
            if ch == "\\" and in_string:
                out.append(ch)
                if i + 1 < length:
                    i += 1
                    out.append(raw[i])
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                i += 1
                continue
            if in_string:
                if ch == "\n":
                    out.append("\\n")
                    i += 1
                    continue
                if ch == "\r":
                    out.append("\\r")
                    i += 1
                    continue
                if ch == "\t":
                    out.append("\\t")
                    i += 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    async def _call_llm(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        inputs: dict[str, Any],
        step_name: str = "LLM Call",
        json_mode: bool = False,
    ) -> str:
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        if json_mode:
            model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""
            if not any(kw in model_name.lower() for kw in ("claude", "anthropic", "gpt", "gemini")):
                llm = llm.bind(response_format={"type": "json_object"})

        chain = prompt | llm
        max_retries = self.llm_cfg.max_retries
        llm_timeout = self.llm_cfg.call_timeout

        with self._otel.start_as_current_span(f"llm:{step_name}") as ospan:
            ospan.set_attribute("openinference.span.kind", "LLM")
            ospan.set_attribute("input.value", json.dumps(inputs, indent=2, default=str))
            ospan.set_attribute("input.mime_type", "application/json")
            t0 = time.perf_counter()
            final_attempt = 1

            for attempt in range(max_retries + 1):
                final_attempt = attempt + 1
                try:
                    async with self._semaphore:
                        response = await asyncio.wait_for(
                            chain.ainvoke(inputs), timeout=llm_timeout
                        )
                    break
                except TimeoutError:
                    logger.warning(
                        "%s [%s] Timeout after %ds (attempt %d/%d)",
                        _P,
                        step_name,
                        llm_timeout,
                        attempt + 1,
                        max_retries + 1,
                    )
                    if attempt < max_retries:
                        continue
                    raise TimeoutError(
                        f"[{step_name}] LLM timed out after {max_retries + 1} attempts"
                    ) from None
                except Exception as ex:
                    ex_str = str(ex)
                    is_retryable = any(
                        code in ex_str
                        for code in (
                            "429",
                            "500",
                            "502",
                            "503",
                            "504",
                            "timeout",
                            "Timeout",
                        )
                    )
                    if is_retryable and attempt < max_retries:
                        wait = 2**attempt * 5
                        logger.warning(
                            "%s [%s] Transient error (attempt %d/%d), retry in %ds — %s",
                            _P,
                            step_name,
                            attempt + 1,
                            max_retries + 1,
                            wait,
                            ex_str[:200],
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise

            reasoning = getattr(response, "reasoning_content", None)
            if not reasoning and hasattr(response, "additional_kwargs"):
                reasoning = (response.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                preview = str(reasoning)[:300].replace("\n", " ")
                logger.debug(
                    "%s [%s] Reasoning: %s%s",
                    _P,
                    step_name,
                    preview,
                    "..." if len(str(reasoning)) > 300 else "",
                )

            response_content = self._filter_think_tokens(str(response.content))

            # Prefer API-reported token counts; fall back to tiktoken estimate.
            # usage_metadata can be a dict (ChatNVIDIA) or an object (other providers).
            usage = getattr(response, "usage_metadata", None)
            if usage:
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
            else:
                input_tokens = 0
                output_tokens = self._estimate_tokens(response_content)
                if reasoning:
                    output_tokens += self._estimate_tokens(str(reasoning))

            duration_ms = (time.perf_counter() - t0) * 1000
            trace = get_current_trace()
            if trace:
                trace.record_llm_call(
                    step_name=step_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    attempt_number=final_attempt,
                )

            ospan.set_attribute("output.value", response_content)
            ospan.set_attribute("output.mime_type", "text/plain")
            ospan.set_attribute("llm.token_count.prompt", input_tokens)
            ospan.set_attribute("llm.token_count.completion", output_tokens)
            ospan.set_attribute("llm.token_count.total", input_tokens + output_tokens)
            meta = {
                "latency_ms": round(duration_ms, 1),
                "attempts": final_attempt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if reasoning:
                meta["reasoning"] = str(reasoning)
            ospan.set_attribute("metadata", json.dumps(meta))

            logger.debug(
                "%s [%s] Response — in: %d tok, out: %d tok%s, %.0fms",
                _P,
                step_name,
                input_tokens,
                output_tokens,
                " (incl. reasoning)" if reasoning else "",
                duration_ms,
            )

            return response_content

    # =========================================================================
    # JSON PARSING
    # =========================================================================

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse a JSON object from an LLM response, with fallback sanitization."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end <= start:
            logger.warning("%s No JSON object found in response: %.200s", _P, response)
            return {"error": "Failed to parse JSON", "raw_response": response}

        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(self._sanitize_json_string(json_str))
        except json.JSONDecodeError:
            pass

        logger.warning("%s JSON parse failed: %.200s", _P, response)
        return {"error": "Failed to parse JSON", "raw_response": response}

    # =========================================================================
    # CONTENT HELPERS
    # =========================================================================

    @staticmethod
    def _get_text_content(doc: dict) -> str:
        """Return usable text from a retrieval result."""
        doc_type = doc.get("document_type", "text")
        if doc_type in AgenticRag._TEXT_DOCUMENT_TYPES:
            content = doc.get("content", "")
            if content and content.strip():
                return content
        description = doc.get("metadata", {}).get("description", "")
        if description and description.strip():
            return description
        return ""

    def _extract_chunks(self, raw_context: Any) -> list[dict[str, str]]:
        """Flatten raw retrieval results into a list of text chunks."""
        chunks = []
        discarded = 0
        items = raw_context if isinstance(raw_context, list) else [raw_context]
        for item in items:
            if isinstance(item, dict):
                results = item.get("results", [item])
                for doc in results:
                    doc_type = doc.get("document_type", "text")
                    text = self._get_text_content(doc)
                    if not text or not text.strip():
                        discarded += 1
                        continue
                    chunks.append(
                        {
                            "doc_name": doc.get("document_name", "Unknown"),
                            "content": text,
                            "score": doc.get("score", 0.0),
                            "document_type": doc_type,
                        }
                    )
        if discarded:
            logger.debug(
                "%s Discarded %d chunks (image/chart with no description)",
                _P,
                discarded,
            )
        return chunks

    def _format_chunks_for_prompt(self, chunks: list[dict], max_tokens: int | None = None) -> str:
        """Format chunks as a numbered list sorted by relevance, truncated to token budget."""
        if not chunks:
            return "(no documents)"

        if max_tokens is None:
            max_tokens = self.context_cfg.max_tokens

        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

        parts = []
        current_tokens = 0
        for idx, c in enumerate(sorted_chunks, 1):
            dtype = c.get("document_type", "text")
            label = {"table": "Table", "chart": "Chart"}.get(dtype, "Chunk")
            doc_name = c.get("doc_name", "")
            header = f"[{label} {idx}] File: {doc_name}" if doc_name else f"[{label} {idx}]"
            section = f"{header}\n{c['content']}"

            section_tokens = self._estimate_tokens(section)
            if current_tokens + section_tokens > max_tokens:
                remaining = max_tokens - current_tokens
                if remaining > 100:
                    chars = int(remaining * self._CHARS_PER_TOKEN_ESTIMATE)
                    parts.append(section[:chars] + "\n... [truncated]")
                break
            parts.append(section)
            current_tokens += section_tokens

        return "\n\n".join(parts)

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Strip markdown formatting artifacts from LLM output."""
        if not text:
            return text

        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        lines = text.split("\n")
        result_lines: list[str] = []
        bullet_buffer: list[str] = []

        def flush_bullets():
            if bullet_buffer:
                result_lines.append("; ".join(bullet_buffer) + ".")
                bullet_buffer.clear()

        for line in lines:
            stripped = line.strip()
            if re.match(r"^(?:[-*+•]\s|\d+[.)]\s)", stripped):
                content = re.sub(r"^(?:[-*+•]\s|\d+[.)]\s)\s*", "", stripped).rstrip(".;,")
                if content:
                    bullet_buffer.append(content)
            else:
                flush_bullets()
                result_lines.append(line)

        flush_bullets()
        text = "\n".join(result_lines)
        text = re.sub(r":\s*;\s*", ": ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # =========================================================================
    # TASK EXECUTION ENGINE
    # =========================================================================

    def _build_execution_levels(self, tasks: list[dict]) -> list[list[dict]]:
        """Return tasks grouped into parallel execution levels (currently one level)."""
        if not tasks:
            return []
        for i, t in enumerate(tasks):
            if "id" not in t:
                t["id"] = f"auto_{i + 1}"
        return [tasks]

    def _parse_task_answer(self, raw_answer: str) -> dict:
        """Parse structured task answer JSON; falls back to plain-text heuristic."""
        if not raw_answer:
            return {"completeness": "none", "answer": "[NO DATA]", "missing": ""}

        parsed = self._parse_json_response(raw_answer)
        if parsed and "completeness" in parsed:
            return {
                "completeness": parsed.get("completeness", "complete"),
                "answer": self._clean_answer(parsed.get("answer", "") or ""),
                "missing": parsed.get("missing", ""),
            }

        text = self._clean_answer(raw_answer.strip())
        if not text or text == "[NO DATA]":
            return {"completeness": "none", "answer": "[NO DATA]", "missing": ""}
        return {"completeness": "complete", "answer": text, "missing": ""}

    async def _execute_single_task(
        self,
        task: dict,
        is_scope: bool = False,
        stage: str = "execute",
    ) -> dict:
        """Execute one task as a mini-agent with seed-query retries and partial-answer merging."""
        effective_max_retries = (
            self.task_exec_cfg.scope_max_retries
            if is_scope
            else self.task_exec_cfg.answer_max_retries
        )
        question = task["question"]
        query = task["query"]
        tid = task["id"]

        with self._otel.start_as_current_span(f"task:{tid}") as tspan:
            tspan.set_attribute("openinference.span.kind", "CHAIN")
            tspan.set_attribute("input.value", question)

            def _finish_task(res: dict) -> dict:
                tspan.set_attribute("output.value", res["answer"])
                tspan.set_attribute(
                    "metadata",
                    json.dumps(
                        {
                            "question": question,
                            "query": query,
                            "status": res["status"],
                            "attempts": res["attempts"],
                        }
                    ),
                )
                return res

            attempt_log = []
            accumulated_answer = ""

            for attempt in range(effective_max_retries):
                current_query = query if attempt == 0 else None

                if attempt > 0:
                    tried_summary = "\n".join(
                        f'  {i + 1}. Query: "{e["query"]}" → {e["outcome"]}'
                        for i, e in enumerate(attempt_log)
                    )
                    seed_response = await self._call_llm(
                        self.seed_gen_llm,
                        self.seed_gen_prompt,
                        {"question": question, "tried_queries": tried_summary},
                        step_name=f"Task {tid} seed gen (attempt {attempt + 1})",
                        json_mode=False,
                    )
                    seed_result = self._parse_json_response(seed_response)

                    if seed_result.get("stop", False):
                        logger.debug(
                            "%s Task %s stopping — %s",
                            _P,
                            tid,
                            seed_result.get("reasoning", "exhausted"),
                        )
                        result = (
                            {
                                "answer": accumulated_answer,
                                "status": "answered",
                                "attempts": attempt + 1,
                            }
                            if accumulated_answer
                            else {
                                "answer": "[NO DATA]",
                                "status": "no_data",
                                "attempts": attempt + 1,
                            }
                        )
                        return _finish_task(result)

                    current_query = seed_result.get("seed_query") or query

                logger.debug(
                    "%s Task %s attempt %d — query: %s", _P, tid, attempt + 1, current_query
                )

                with self._otel.start_as_current_span(
                    f"retrieve:task_{tid}_att{attempt + 1}"
                ) as rspan:
                    rspan.set_attribute("openinference.span.kind", "RETRIEVER")
                    rspan.set_attribute("input.value", current_query or "")
                    try:
                        raw_context = await self.retriever_fn(current_query, stage)
                        if raw_context is None:
                            raw_context = []
                    except Exception as ex:
                        logger.warning("%s Task %s retrieval failed: %s", _P, tid, ex)
                        rspan.set_attribute("metadata", json.dumps({"error": str(ex)[:200]}))
                        raw_context = []

                chunks = self._extract_chunks(
                    raw_context if isinstance(raw_context, list) else [raw_context]
                )

                if not chunks:
                    logger.debug("%s Task %s attempt %d — no chunks returned", _P, tid, attempt + 1)
                    result = (
                        {
                            "answer": accumulated_answer,
                            "status": "answered",
                            "attempts": attempt + 1,
                        }
                        if accumulated_answer
                        else {"answer": "[NO DATA]", "status": "no_data", "attempts": attempt + 1}
                    )
                    return _finish_task(result)

                n_chunks = len(chunks)
                docs_str = self._format_chunks_for_prompt(chunks)

                task_question = question
                if accumulated_answer:
                    task_question = (
                        f"{question}\n\n"
                        f"IMPORTANT: A prior retrieval already found this partial answer:\n"
                        f"{accumulated_answer}\n\n"
                        f"Your job: find the MISSING information and produce a COMPLETE answer "
                        f"that merges the prior data with any new data you find in these documents."
                    )

                raw_answer = await self._call_llm(
                    self.task_llm,
                    self.task_answer_prompt,
                    {"question": task_question, "documents": docs_str},
                    step_name=f"Task {tid} answer (attempt {attempt + 1})",
                )
                parsed = self._parse_task_answer(raw_answer)

                if parsed["completeness"] == "complete":
                    logger.debug("%s Task %s — complete answer found", _P, tid)
                    result = {
                        "answer": parsed["answer"],
                        "status": "answered",
                        "attempts": attempt + 1,
                    }
                    return _finish_task(result)

                if parsed["completeness"] == "partial":
                    accumulated_answer = parsed["answer"]
                    attempt_log.append(
                        {
                            "query": current_query,
                            "outcome": f"PARTIAL ({n_chunks} docs) — found: {parsed['answer'][:80]}... | still missing: {parsed['missing']}",
                        }
                    )
                    logger.debug(
                        "%s Task %s — partial (%d/%d), missing: %.100s",
                        _P,
                        tid,
                        attempt + 1,
                        effective_max_retries,
                        parsed["missing"],
                    )
                    continue

                logger.debug("%s Task %s — [NO DATA] from %d docs", _P, tid, n_chunks)
                result = (
                    {"answer": accumulated_answer, "status": "answered", "attempts": attempt + 1}
                    if accumulated_answer
                    else {"answer": "[NO DATA]", "status": "no_data", "attempts": attempt + 1}
                )
                return _finish_task(result)

            result = (
                {
                    "answer": accumulated_answer,
                    "status": "answered",
                    "attempts": effective_max_retries,
                }
                if accumulated_answer
                else {"answer": "[NO DATA]", "status": "no_data", "attempts": effective_max_retries}
            )
            return _finish_task(result)

    async def _execute_plan(
        self,
        tasks: list[dict],
        is_scope: bool = False,
        stage: str = "execute",
    ) -> dict[str, dict]:
        """Execute all tasks in parallel."""
        levels = self._build_execution_levels(tasks)
        all_results: dict[str, dict] = {}

        for level_tasks in levels:
            logger.debug(
                "%s Executing %d tasks: %s",
                _P,
                len(level_tasks),
                [t["id"] for t in level_tasks],
            )

            coros = [self._execute_single_task(task, is_scope=is_scope, stage=stage) for task in level_tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(level_tasks, results, strict=True):
                if isinstance(result, Exception):
                    logger.error("%s Task %s failed: %s", _P, task["id"], result)
                    result = {"answer": "[NO DATA]", "status": "error", "attempts": 0}
                all_results[task["id"]] = result

        return all_results

    # =========================================================================
    # GRAPH NODES
    # =========================================================================

    async def initial_retrieval_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Single-query retrieval using the user's original query."""
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        with self._otel_and_trace("initial_retrieval") as ospan:
            ospan.set_attribute("input.value", state.user_query)
            logger.info("%s Initial retrieval started", _P)

            with self._otel.start_as_current_span("retrieve:original") as rspan:
                rspan.set_attribute("openinference.span.kind", "RETRIEVER")
                rspan.set_attribute("input.value", state.user_query)
                try:
                    raw = await self.retriever_fn(state.user_query, "initial_retrieval")
                    if raw is None:
                        raw = []
                    result = raw if isinstance(raw, list) else [raw]
                    chunks = self._extract_chunks(result)
                    for i, c in enumerate(chunks):
                        rspan.set_attribute(
                            f"retrieval.documents.{i}.document.content",
                            c.get("content", ""),
                        )
                        rspan.set_attribute(
                            f"retrieval.documents.{i}.document.score",
                            c.get("score", 0.0),
                        )
                        rspan.set_attribute(
                            f"retrieval.documents.{i}.document.metadata",
                            json.dumps({"doc_name": c.get("doc_name", "")}),
                        )
                    rspan.set_attribute(
                        "metadata",
                        json.dumps({"document_count": len(chunks)}),
                    )
                except Exception as ex:
                    logger.warning("%s Retrieval failed: %s", _P, ex)
                    rspan.set_attribute("metadata", json.dumps({"error": str(ex)[:200]}))
                    chunks = []

            logger.info("%s Retrieval complete: %d chunks", _P, len(chunks))

            stats = {"chunks": len(chunks)}
            ospan.set_attribute("output.value", json.dumps(stats, indent=2))
            ospan.set_attribute("output.mime_type", "application/json")
            ospan.set_attribute("metadata", json.dumps(stats))

            trace = get_current_trace()
            if trace:
                trace.retrieval_stats = stats

            rebuilt_context = [{"results": [self._rebuild_result(c) for c in chunks]}]

            return {"initial_context": rebuilt_context}

    async def plan_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Create or refine the retrieval plan (called once or twice for scope discovery)."""
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        is_replan = bool(state.scope_results)
        phase = "phase 2 (answer plan)" if is_replan else "phase 1 (scope discovery)"
        with self._otel_and_trace(f"plan ({phase})") as ospan:
            current_round = state.scope_rounds
            ospan.set_attribute("input.value", state.user_query)
            logger.info("%s Planning (%s, scope_round=%d)", _P, phase, current_round)

            chunks = self._extract_chunks(state.initial_context)
            initial_content = (
                self._format_chunks_for_prompt(chunks) if chunks else "(no documents retrieved)"
            )

            scope_section = ""
            if is_replan:
                scope_parts = [f"[{tid}]: {answer}" for tid, answer in state.scope_results.items()]
                scope_section = (
                    "\nScope Discovery Results (what actually exists in the database):\n"
                    + "\n".join(scope_parts)
                    + "\n"
                    + self.planner_replan_instruction
                )

            planner_max_attempts = self.planner_cfg.max_attempts
            try:
                plan = None
                for planner_attempt in range(planner_max_attempts):
                    response = await self._call_llm(
                        self.planner_llm,
                        self.planner_prompt,
                        {
                            "query": state.user_query,
                            "initial_context": initial_content,
                            "scope_section": scope_section,
                        },
                        step_name=f"Planner ({phase})",
                        json_mode=False,
                    )
                    plan = self._parse_json_response(response)

                    if "error" in plan:
                        logger.warning(
                            "%s Plan parse failed (attempt %d/%d)",
                            _P,
                            planner_attempt + 1,
                            planner_max_attempts,
                        )
                        if planner_attempt < planner_max_attempts - 1:
                            continue
                        plan = {
                            "scope_only": False,
                            "scope_resolution": "Plan parsing failed — answering from initial retrieval",
                            "resolved_query": state.user_query,
                            "tasks": [],
                            "synthesis_instruction": "Answer directly from initial retrieval context",
                        }
                        break

                    is_degenerate = "resolved_query" not in plan and "scope_only" not in plan
                    if is_degenerate and planner_attempt < planner_max_attempts - 1:
                        logger.warning(
                            "%s Degenerate plan (missing key fields), retrying (%d/%d)",
                            _P,
                            planner_attempt + 1,
                            planner_max_attempts,
                        )
                        continue

                    tasks_list = plan.get("tasks", [])
                    if isinstance(tasks_list, list) and tasks_list:
                        required_keys = {"id", "question", "query"}
                        malformed_keys = [
                            i
                            for i, t in enumerate(tasks_list)
                            if not isinstance(t, dict) or not required_keys.issubset(t.keys())
                        ]
                        if malformed_keys and planner_attempt < planner_max_attempts - 1:
                            issues = []
                            for i in malformed_keys:
                                t = tasks_list[i]
                                if not isinstance(t, dict):
                                    issues.append(f"task[{i}] not a dict")
                                else:
                                    missing = required_keys - set(t.keys())
                                    if missing:
                                        issues.append(f"task[{i}] missing {missing}")
                            logger.warning(
                                "%s Malformed tasks (%s), retrying (%d/%d)",
                                _P,
                                ", ".join(issues),
                                planner_attempt + 1,
                                planner_max_attempts,
                            )
                            continue
                        if malformed_keys:
                            for i, t in enumerate(tasks_list):
                                if isinstance(t, dict) and "id" not in t:
                                    t["id"] = f"auto_{i + 1}"
                    break

                if not isinstance(plan.get("tasks"), list):
                    plan["tasks"] = []
                max_plan_tasks = self.planner_cfg.max_plan_tasks
                plan["tasks"] = plan["tasks"][:max_plan_tasks]

                if plan.get("scope_only", False) and not plan["tasks"]:
                    plan["scope_only"] = False

                if plan.get("scope_only", False) or not plan["tasks"]:
                    plan["synthesis_instruction"] = ""

                max_scope_rounds = self.planner_cfg.max_scope_rounds
                if is_replan and current_round >= max_scope_rounds:
                    if plan.get("scope_only", False):
                        logger.warning(
                            "%s Max scope rounds (%d) reached — forcing answer phase",
                            _P,
                            max_scope_rounds,
                        )
                        plan["scope_only"] = False

                new_round = current_round + 1 if plan.get("scope_only", False) else current_round

                logger.info(
                    "%s Plan: scope_only=%s, %d tasks, resolved_query=%.80s",
                    _P,
                    plan.get("scope_only", False),
                    len(plan.get("tasks", [])),
                    plan.get("resolved_query", "")[:80],
                )
                logger.debug("%s Plan detail: %s", _P, json.dumps(plan, indent=2)[:2000])

                task_ids = [t.get("id", "") for t in plan.get("tasks", [])]
                plan_json = json.dumps(plan, indent=2)
                ospan.set_attribute("output.value", plan_json)
                ospan.set_attribute("output.mime_type", "application/json")
                ospan.set_attribute(
                    "metadata",
                    json.dumps(
                        {
                            "phase": phase,
                            "scope_only": plan.get("scope_only", False),
                            "task_count": len(task_ids),
                            "task_ids": task_ids,
                            "resolved_query": plan.get("resolved_query", ""),
                        }
                    ),
                )

                trace = get_current_trace()
                if trace:
                    trace.plan_summary = {
                        "scope_only": plan.get("scope_only", False),
                        "task_count": len(task_ids),
                        "scope_rounds": new_round,
                        "resolved_query": plan.get("resolved_query", ""),
                    }

                return {
                    "retrieval_plan": plan,
                    "needs_replan": False,
                    "scope_rounds": new_round,
                }

            except Exception as ex:
                logger.exception("%s Planning failed: %s", _P, ex)
                return {
                    "retrieval_plan": {
                        "scope_only": False,
                        "scope_resolution": f"Planning failed: {ex}",
                        "resolved_query": state.user_query,
                        "tasks": [],
                        "synthesis_instruction": "Answer directly from initial retrieval context",
                    },
                    "needs_replan": False,
                }

    async def execute_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Execute all tasks from the current plan."""
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        tasks = state.retrieval_plan.get("tasks", [])
        is_scope_only = state.retrieval_plan.get("scope_only", False)
        exec_label = "scope discovery" if is_scope_only else "answer"
        with self._otel_and_trace(f"execute ({exec_label})") as ospan:
            task_ids = [t.get("id", "") for t in tasks]
            ospan.set_attribute("input.value", json.dumps(task_ids))
            ospan.set_attribute("input.mime_type", "application/json")

            if not tasks:
                logger.info("%s Execute: no tasks — synthesizing from initial retrieval", _P)
                return {"task_results": {}, "needs_replan": False}

            logger.info("%s Executing %d %s tasks", _P, len(tasks), exec_label)

            exec_stage = "execute_scope" if is_scope_only else "execute"
            results = await self._execute_plan(tasks, is_scope=is_scope_only, stage=exec_stage)

            statuses = [r.get("status", "unknown") for r in results.values()]
            answered = statuses.count("answered")
            no_data = statuses.count("no_data")
            exec_output = {
                tid: {"status": r.get("status", "unknown"), "answer": r.get("answer", "")}
                for tid, r in results.items()
            }
            ospan.set_attribute("output.value", json.dumps(exec_output, indent=2))
            ospan.set_attribute("output.mime_type", "application/json")
            ospan.set_attribute(
                "metadata",
                json.dumps(
                    {
                        "mode": exec_label,
                        "task_count": len(tasks),
                        "answered": answered,
                        "no_data": no_data,
                    }
                ),
            )
            logger.info("%s Execution complete: %d answered, %d no_data", _P, answered, no_data)
            logger.debug(
                "%s Execution detail: %s",
                _P,
                json.dumps(
                    {tid: r.get("status", "unknown") for tid, r in results.items()},
                    indent=2,
                ),
            )

            trace = get_current_trace()
            if trace:
                trace.task_results_summary = {
                    tid: {
                        "status": r.get("status", "unknown"),
                        "attempts": r.get("attempts", 0),
                    }
                    for tid, r in results.items()
                }

            if is_scope_only:
                scope_answers = dict(state.scope_results) if state.scope_results else {}
                for tid, r in results.items():
                    ans = r.get("answer", "[NO DATA]")
                    if ans and ans.strip() != "[NO DATA]":
                        scope_answers[tid] = ans

                if scope_answers:
                    ospan.set_attribute(
                        "metadata",
                        json.dumps(
                            {
                                "mode": exec_label,
                                "task_count": len(tasks),
                                "answered": answered,
                                "no_data": no_data,
                                "replan_decision": f"REPLAN: {len(scope_answers)} of {len(results)} scope tasks returned data",
                            }
                        ),
                    )
                    logger.info(
                        "%s Scope: %d/%d tasks returned data — re-planning",
                        _P,
                        len(scope_answers),
                        len(results),
                    )
                    return {
                        "task_results": results,
                        "scope_results": scope_answers,
                        "needs_replan": True,
                    }
                else:
                    ospan.set_attribute(
                        "metadata",
                        json.dumps(
                            {
                                "mode": exec_label,
                                "task_count": len(tasks),
                                "answered": answered,
                                "no_data": no_data,
                                "replan_decision": "SKIP REPLAN: all scope tasks returned [NO DATA] — synthesizing from initial retrieval",
                            }
                        ),
                    )
                    logger.info("%s Scope: all tasks returned [NO DATA] — no replan", _P)
                    return {"task_results": results, "needs_replan": False}

            return {"task_results": results, "needs_replan": False}

    async def synthesize_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Combine task sub-answers (and optionally verification data) into a final answer."""
        is_verification_pass = state.verification_round > 0
        synth_label = (
            f"synthesize (verification pass {state.verification_round})"
            if is_verification_pass
            else "synthesize"
        )
        with self._otel_and_trace(synth_label) as ospan:
            pass_label = synth_label
            ospan.set_attribute("input.value", state.user_query)
            logger.info("%s %s started", _P, pass_label)

            tasks = state.retrieval_plan.get("tasks", [])
            task_results = state.task_results
            synthesis_instruction = (
                state.retrieval_plan.get("synthesis_instruction", "")
                or "Answer the question directly"
            )

            has_verification_data = (
                any(
                    r.get("answer", "").strip() not in ("", "[NO DATA]")
                    for r in state.verification_results.values()
                )
                if state.verification_results
                else False
            )

            if is_verification_pass and not has_verification_data:
                logger.info(
                    "%s %s — all verification tasks returned [NO DATA], keeping original answer",
                    _P,
                    pass_label,
                )
                return {"final_answer": state.final_answer}

            chunks = self._extract_chunks(state.initial_context)

            effective_tasks = [
                t
                for t in tasks
                if task_results.get(t["id"], {}).get("answer", "[NO DATA]") != "[NO DATA]"
            ]

            if not effective_tasks and not state.verification_results:
                if not chunks:
                    return {
                        "final_answer": "No relevant information found in the available documents."
                    }

                docs_str = self._format_chunks_for_prompt(chunks)
                sub_answers = (
                    f"[Source documents — synthesize the answer directly from these]\n{docs_str}"
                )
            else:
                answer_parts = []

                if is_verification_pass and has_verification_data:
                    if state.final_answer:
                        answer_parts.append(f"[PREVIOUS ANSWER]: {state.final_answer}")
                    if state.verification_issues:
                        issues_text = "\n".join(f"- {issue}" for issue in state.verification_issues)
                        answer_parts.append(f"[GAPS IDENTIFIED IN PREVIOUS ANSWER]:\n{issues_text}")
                    synthesis_instruction = (
                        "IMPORTANT: The [PREVIOUS ANSWER] had gaps listed under [GAPS IDENTIFIED IN PREVIOUS ANSWER]. "
                        "Keep the [PREVIOUS ANSWER] as the base. "
                        "Fill the gaps using [VERIFICATION — NEW DATA] and the supplementary context. "
                        "Do not rewrite parts that were already correct. "
                        "Underlying retrieval goal: " + synthesis_instruction
                    )

                for task in effective_tasks:
                    tid = task["id"]
                    result = task_results.get(tid, {})
                    answer = result.get("answer", "[NO DATA]")
                    answer_parts.append(f"Task {tid} ({task.get('question', '')}): {answer}")

                if state.verification_results:
                    v_question_map = {
                        t["id"]: t.get("question", "") for t in state.verification_tasks
                    }
                    for tid, result in state.verification_results.items():
                        answer = result.get("answer", "[NO DATA]")
                        v_question = v_question_map.get(tid, "")
                        if answer and answer.strip() != "[NO DATA]":
                            v_label = (
                                f"[VERIFICATION — NEW DATA] {tid} ({v_question})"
                                if v_question
                                else f"[VERIFICATION — NEW DATA] {tid}"
                            )
                            answer_parts.append(f"{v_label}: {answer}")

                sub_answers = "\n\n".join(answer_parts)

                if chunks:
                    initial_docs = self._format_chunks_for_prompt(chunks)
                    sub_answers += (
                        "\n\n--- Supplementary Context (from initial retrieval) ---\n"
                        + initial_docs
                    )

            resolved_query = (
                state.retrieval_plan.get("resolved_query", state.user_query) or state.user_query
            )
            resolved_section = ""
            if resolved_query.strip().lower() != state.user_query.strip().lower():
                resolved_section = (
                    f"\nClarified question (an implicit or ambiguous reference in the question above "
                    f"has been resolved from the documents; the User Question remains authoritative): "
                    f"{resolved_query}\n"
                )

            try:
                response = await self._call_llm(
                    self.synthesis_llm,
                    self.synthesis_prompt,
                    {
                        "query": state.user_query,
                        "resolved_section": resolved_section,
                        "synthesis_instruction": synthesis_instruction,
                        "sub_answers": sub_answers,
                    },
                    step_name=pass_label,
                )
                answer = self._clean_answer(response) if response else ""

                if not answer:
                    answer = "No relevant information found in the available documents."

                ospan.set_attribute("output.value", answer)
                ospan.set_attribute(
                    "metadata",
                    json.dumps(
                        {
                            "is_verification_pass": is_verification_pass,
                            "answer_length": len(answer),
                        }
                    ),
                )
                logger.info("%s %s complete", _P, pass_label)
                logger.debug("%s Answer preview: %.300s", _P, answer)
                return {"final_answer": answer}

            except Exception as ex:
                logger.exception("%s Synthesis failed: %s", _P, ex)
                return {"final_answer": "I encountered an error generating the answer."}

    def _build_resolved_query_section(self, state: AgenticRAGGraphState) -> str:
        """Build the resolved-query header for the verification prompt."""
        resolved = state.retrieval_plan.get("resolved_query", "") or ""
        if resolved and resolved.strip().lower() != state.user_query.strip().lower():
            return f"\nDisambiguated Question (implicit or ambiguous references in the original question were resolved from the documents): {resolved}\n"
        return ""

    async def verify_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Check answer completeness and identify retrieval gaps."""
        from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace

        with self._otel_and_trace("verify") as ospan:
            ospan.set_attribute("input.value", state.final_answer)
            logger.info("%s Verification started (round %d)", _P, state.verification_round)

            tasks = state.retrieval_plan.get("tasks", [])
            task_summary_parts = []
            for task in tasks:
                tid = task["id"]
                result_entry = state.task_results.get(tid, {})
                status = result_entry.get("status", "unknown")
                answer = result_entry.get("answer", "")
                answer_preview = (answer[:150] + "...") if len(answer) > 150 else answer
                task_summary_parts.append(
                    f"- {tid}: {task.get('question', '')} → {status}: {answer_preview}"
                )
            task_summary = (
                "\n".join(task_summary_parts) if task_summary_parts else "(no tasks were executed)"
            )

            max_v_tasks = self.verification_cfg.max_tasks

            try:
                response = await self._call_llm(
                    self.planner_llm,
                    self.verification_prompt,
                    {
                        "query": state.user_query,
                        "resolved_query_section": self._build_resolved_query_section(state),
                        "answer": state.final_answer,
                        "task_summary": task_summary,
                    },
                    step_name="Verification",
                    json_mode=False,
                )
                result = self._parse_json_response(response)

                passed = result.get("status") == "pass"
                issues = result.get("issues", [])

                if passed:
                    ospan.set_attribute("output.value", json.dumps(result, indent=2))
                    ospan.set_attribute("output.mime_type", "application/json")
                    ospan.set_attribute(
                        "metadata",
                        json.dumps({"passed": True, "issues": 0, "follow_up_tasks": 0}),
                    )
                    logger.info("%s Verification PASSED", _P)
                    logger.debug(
                        "%s Verification reasoning: %s",
                        _P,
                        result.get("reasoning", ""),
                    )
                    trace = get_current_trace()
                    if trace:
                        trace.verification_outcome = {
                            "passed": True,
                            "issues": [],
                            "follow_up_tasks": 0,
                        }
                    return {
                        "verification_tasks": [],
                        "verification_round": state.verification_round + 1,
                    }

                v_tasks = result.get("tasks", [])
                if not isinstance(v_tasks, list):
                    v_tasks = []
                required_keys = {"id", "question", "query"}
                v_tasks = [
                    t for t in v_tasks if isinstance(t, dict) and required_keys.issubset(t.keys())
                ][:max_v_tasks]

                ospan.set_attribute("output.value", json.dumps(result, indent=2))
                ospan.set_attribute("output.mime_type", "application/json")
                ospan.set_attribute(
                    "metadata",
                    json.dumps(
                        {
                            "passed": False,
                            "issues": len(issues),
                            "follow_up_tasks": len(v_tasks),
                        }
                    ),
                )

                logger.info(
                    "%s Verification FAILED: %d issues, %d follow-up tasks",
                    _P,
                    len(issues),
                    len(v_tasks),
                )
                logger.debug("%s Verification issues: %s", _P, issues)
                logger.debug(
                    "%s Verification tasks: %s",
                    _P,
                    [t.get("id", "") for t in v_tasks],
                )

                trace = get_current_trace()
                if trace:
                    trace.verification_outcome = {
                        "passed": False,
                        "issues": issues if isinstance(issues, list) else [],
                        "follow_up_tasks": len(v_tasks),
                    }

                return {
                    "verification_tasks": v_tasks,
                    "verification_issues": issues if isinstance(issues, list) else [],
                    "verification_round": state.verification_round + 1,
                }

            except Exception as ex:
                logger.exception("%s Verification failed: %s", _P, ex)
                return {
                    "verification_tasks": [],
                    "verification_round": state.verification_round + 1,
                }

    async def verify_execute_node(self, state: AgenticRAGGraphState) -> dict[str, Any]:
        """Execute verification follow-up tasks (targeted re-retrieval)."""
        with self._otel_and_trace("verify_execute") as ospan:
            v_tasks = state.verification_tasks
            ospan.set_attribute(
                "input.value", json.dumps([t.get("id", "") for t in (v_tasks or [])])
            )
            if not v_tasks:
                return {"verification_results": {}}

            logger.info("%s Verify-execute: running %d follow-up tasks", _P, len(v_tasks))
            results = await self._execute_plan(v_tasks, stage="verify_execute")

            statuses = [r.get("status", "unknown") for r in results.values()]
            ospan.set_attribute(
                "metadata",
                json.dumps(
                    {
                        "task_count": len(v_tasks) if v_tasks else 0,
                        "answered": statuses.count("answered"),
                        "no_data": statuses.count("no_data"),
                    }
                ),
            )
            logger.info(
                "%s Verify-execute complete: %d answered, %d no_data",
                _P,
                statuses.count("answered"),
                statuses.count("no_data"),
            )
            return {"verification_results": results}

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    @staticmethod
    def _route_after_execute(state: AgenticRAGGraphState) -> str:
        """Route to replan (scope discovery follow-up) or synthesize."""
        if state.needs_replan:
            return "plan"
        return "synthesize"

    def _route_after_synthesize(self, state: AgenticRAGGraphState) -> str:
        """Route to verification (if enabled and first pass) or end."""
        if self.verification_cfg.enabled and state.verification_round == 0:
            return "verify"
        return END

    @staticmethod
    def _route_after_verify(state: AgenticRAGGraphState) -> str:
        """Route to verify-execute (if gaps found) or end."""
        if state.verification_tasks:
            return "verify_execute"
        return END

    async def build_graph(self):
        """Compile the LangGraph state machine."""
        graph = StateGraph(AgenticRAGGraphState)

        graph.add_node("initial_retrieval", self.initial_retrieval_node)
        graph.add_node("plan", self.plan_node)
        graph.add_node("execute", self.execute_node)
        graph.add_node("synthesize", self.synthesize_node)
        graph.add_node("verify", self.verify_node)
        graph.add_node("verify_execute", self.verify_execute_node)

        graph.set_entry_point("initial_retrieval")
        graph.add_edge("initial_retrieval", "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "plan": "plan",
                "synthesize": "synthesize",
            },
        )
        graph.add_conditional_edges(
            "synthesize",
            self._route_after_synthesize,
            {
                "verify": "verify",
                END: END,
            },
        )
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {
                "verify_execute": "verify_execute",
                END: END,
            },
        )
        graph.add_edge("verify_execute", "synthesize")

        self.graph = graph.compile()
        logger.debug("%s Graph compiled", _P)
        return self.graph

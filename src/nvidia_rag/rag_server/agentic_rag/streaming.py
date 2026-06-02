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

"""Streaming infrastructure for the agentic RAG pipeline.

This module isolates everything needed to translate LangGraph's combined
``stream_mode=["messages", "custom", "debug", "values"]`` event stream into
the ``ChainResponse`` SSE chunks that the rag-server emits.  Keeping it
here means ``agentic_rag.py`` and ``runner.py`` stay focused on RAG logic.

Public surface
--------------
* ``EventType``                     — enum of all event_type values emitted on the wire
* ``USER_FACING_LABELS``            — single source of truth for every UI-visible string
                                       emitted as a stage_start / stage_end / agent_event
                                       (writers pass only ``key`` + ``params``; the
                                       translator does the lookup and substitution)
* ``make_stage_start``              — writer payload for node entry (key + params)
* ``make_stage_end``                — writer payload for node exit (key + params)
* ``make_agent_event``              — writer payload for mid-stage updates (key + params)
                                       — surfaces on the wire as ``event_type='agent_event'``
* ``translate_graph_stream``        — async generator that consumes ``graph.astream(...)``
                                       and yields SSE-formatted ``ChainResponse`` strings

Design notes
------------
* The translator alone formats SSE strings; nodes only emit semantic events via ``writer``.
* Whether synthesize tokens are labeled as ``final_*`` or ``intermediate_*`` is decided
  statically from the agent's ``verification_cfg.enabled`` flag, which the runner passes
  to :func:`translate_graph_stream`:

    - **verification disabled**: synthesize is the only LLM-emitting terminal node;
      its content tokens stream live as ``final_answer`` (in ``delta.content``) and its
      reasoning tokens as ``final_reasoning``.
    - **verification enabled**: synthesize may run twice (draft + post-verification
      revision) and we don't know up front which pass is terminal.  Both passes therefore
      stream as ``intermediate_output``/``intermediate_reasoning`` so the user sees the
      agent "thinking", and at stream end we emit ONE ``final_answer`` chunk whose
      ``delta.content`` is taken from the authoritative ``state.final_answer`` value
      (captured via ``stream_mode="values"``).

* Citations attach to the first ``final_answer`` chunk — that's the live token chunk when
  verification is disabled, or the single end-of-stream chunk when verification is enabled.
* RAG/LLM TTFT is measured to the first ``final_answer`` content chunk in both modes.
  When verification is enabled this corresponds to "time until the answer is shown".
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum
from typing import Any
from uuid import uuid4

from nvidia_rag.rag_server.agentic_rag.tracing import get_current_trace
from nvidia_rag.rag_server.response_generator import (
    ChainResponse,
    ChainResponseChoices,
    Citations,
    Message,
    Metrics,
    Usage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CHUNK TYPES
# =============================================================================


class EventType(StrEnum):
    """Enumerates the values the ``event_type`` field on ChainResponse can take.

    ``StrEnum`` makes each member a real ``str`` so ``EventType.STAGE_START ==
    "stage_start"`` and JSON serialization Just Works without custom encoders.
    """

    # Lifecycle events emitted by nodes via ``writer({...})``.
    STAGE_START = "stage_start"
    """Node has started; ``reasoning_content`` carries a friendly one-liner."""

    STAGE_END = "stage_end"
    """Node has finished; ``reasoning_content`` carries a short summary."""

    # LLM token chunks from the messages stream.  Single-LLM-call nodes
    # (``plan``, ``verify``) emit these as live token deltas.  Parallel-task
    # nodes (``execute``, ``verify_execute``) emit them as ONE consolidated
    # chunk per concurrent task, deferred until the node's ``stage_end`` —
    # see PARALLEL_TASK_NODES for the rationale.
    INTERMEDIATE_REASONING = "intermediate_reasoning"
    """Reasoning-content from a non-final node.  Streamed live for
    single-LLM-call nodes; one consolidated chunk per task for parallel
    nodes."""

    INTERMEDIATE_OUTPUT = "intermediate_output"
    """Visible-content output from a non-final node (planner / verifier JSON,
    per-task answers, etc.).  Streamed live for single-LLM-call nodes; one
    consolidated chunk per task for parallel nodes."""

    FINAL_REASONING = "final_reasoning"
    """Reasoning-content delta from the final synthesize call."""

    FINAL_ANSWER = "final_answer"
    """User-facing answer, mirrored into ``Message.content`` (delta+message) and
    carrying the run's citations.  When verification is disabled this is a
    stream of token deltas from the (only) synthesize call.  When verification
    is enabled this is a single chunk emitted at stream end whose content is
    the authoritative ``state.final_answer`` value."""

    # Structured semantic update from any node, plus forwarded LangGraph
    # debug-mode events.  The ``stage`` field discriminates which node
    # produced the update (``plan``, ``execute``, ``verify``,
    # ``initial_retrieval``, ``verify_execute``, etc.).  The detail string is
    # carried in ``reasoning_content``.
    AGENT_EVENT = "agent_event"
    """Structured semantic update from a graph node.  Covers what previously
    used dedicated chunk types (retrieval summaries, plan summaries,
    verification outcomes) plus LangGraph debug events when
    ``AgenticRAGConfig.enable_debug_stream=True``.  The ``stage`` field
    identifies the source node."""

    ERROR = "error"
    """Pipeline-level error.  ``Message.content`` carries the user-visible error
    message; ``reasoning_content`` carries the error detail when available."""


# =============================================================================
# PARALLEL-TASK NODES — PER-RUN BUFFERING
# =============================================================================
# Graph nodes that fan out to N concurrent ``_execute_single_task`` coroutines
# via ``asyncio.gather``.  When tokens from those nested LLM calls reach the
# ``stream_mode="messages"`` channel, they arrive interleaved across tasks and
# are unreadable if forwarded as-is.
#
# Why filter at the translator (not at the LLM call site): LangGraph
# propagates its messages-mode callback handler via async context variables,
# so any chat model invocation inside a node — including ``ainvoke`` — fires
# ``on_llm_new_token`` callbacks regardless of whether the explicit ``config``
# argument is passed through.  Translator-side handling is the reliable
# choke point.
#
# Strategy: instead of dropping these tokens, we buffer them per-run-id (the
# UUID LangChain attaches to each ``BaseChatModel`` invocation) for the
# duration of the parallel phase.  When the node's ``stage_end`` event
# arrives — which only fires after ``asyncio.gather`` returns, i.e. all
# tasks have completed — we drain the buffer and emit ONE consolidated
# ``intermediate_reasoning`` chunk and ONE ``intermediate_output`` chunk per
# run_id.  This preserves all information, eliminates token jumbling, and
# isolates each task in its own pair of chunks.
#
# Trade-off: clients see no live tokens during the parallel phase — they
# get a burst of consolidated chunks at stage_end instead.  Single-LLM-call
# nodes (``plan``, ``verify``, ``synthesize``) still stream live as before.

PARALLEL_TASK_NODES: frozenset[str] = frozenset({"execute", "verify_execute"})


# =============================================================================
# USER-FACING LABELS
# =============================================================================
# Single source of truth for every string we surface on the wire as a
# stage_start / stage_end / agent_event ``reasoning_content``.  Writer call
# sites in nodes pass only a ``key`` (and any ``{placeholder}`` params); the
# translator looks up the template here and formats it.  Keeping the strings
# in one place means UX wording can be reviewed/changed without grepping
# through the agentic_rag pipeline code.
#
# Convention for keys: ``{stage}.{phase}[.{variant}]`` — e.g.
# ``plan.start.scope`` / ``execute.end.no_data`` / ``verify.end.passed``.
# Templates can use ``{placeholder}`` substitutions; values come from the
# ``params`` kwargs passed to the make_* helpers.


USER_FACING_LABELS: dict[str, str] = {
    # --- initial_retrieval ----------------------------------------------------
    "initial_retrieval.start": "Searching the knowledge base for relevant information…",
    "initial_retrieval.end": "Found {chunks} relevant passage(s).",
    # --- plan: phase 1 (scope discovery) -------------------------------------
    "plan.start.scope": "Exploring what's available in the knowledge base…",
    "plan.end.scope": "Identified {task_count} topic(s) to explore further.",
    # --- plan: phase 2 (answer plan) -----------------------------------------
    "plan.start.answer": "Planning the next retrieval steps…",
    "plan.end.with_tasks": "Created {task_count} targeted retrieval task(s).",
    "plan.end.no_tasks": "Initial context is sufficient — no further retrieval needed.",
    "plan.end.failed": "Couldn't produce a plan — proceeding with available context.",
    # --- execute --------------------------------------------------------------
    "execute.start.answer": "Executing {task_count} retrieval task(s) in parallel…",
    "execute.start.scope": "Running scope-discovery searches in parallel…",
    "execute.end.done": "Completed retrieval ({answered} of {task_count} task(s) answered).",
    "execute.end.no_tasks": "Skipping retrieval — using the initial context.",
    "execute.end.replan": "Found enough information to refine the plan.",
    "execute.end.no_data": "No additional information found — using the initial context.",
    # --- synthesize -----------------------------------------------------------
    "synthesize.start": "Composing the answer…",
    "synthesize.start.refining": "Refining the answer with new findings…",
    "synthesize.end": "Answer ready.",
    "synthesize.end.unchanged": "Keeping the previous answer (no new information).",
    "synthesize.end.failed": "Encountered an error while generating the answer.",
    # --- verify ---------------------------------------------------------------
    "verify.start": "Reviewing the answer for completeness…",
    "verify.end.passed": "Answer looks complete.",
    "verify.end.failed": "Identified {issues} potential gap(s) — running follow-up retrieval.",
    "verify.end.failed_unspecified": "Need additional information — running follow-up retrieval.",
    "verify.end.error": "Could not verify the answer — using the current draft.",
    # --- verify_execute -------------------------------------------------------
    "verify_execute.start": "Filling in the identified gaps…",
    "verify_execute.end": "Filled {answered} additional gap(s).",
    "verify_execute.end.no_tasks": "No follow-up retrieval needed.",
    # --- generic / fallback ---------------------------------------------------
    # Used when callers don't pass a specific key.
    "default.start": "Working…",
    "default.end": "Done.",
}


def _format_label(key: str, params: dict[str, Any] | None = None) -> str:
    """Look up ``key`` in :data:`USER_FACING_LABELS` and substitute placeholders.

    Falls back to a humanized version of the key if no template is registered
    (and logs a warning so missing keys are easy to spot).  This keeps an
    unknown key from breaking the stream — at worst the client sees a
    slightly less polished string.
    """
    template = USER_FACING_LABELS.get(key)
    if template is None:
        logger.warning("[stream] no user-facing label registered for key %r", key)
        return key.replace("_", " ").replace(".", " — ").strip().capitalize() or "—"
    if not params:
        return template
    try:
        return template.format(**params)
    except (KeyError, IndexError) as ex:
        logger.warning(
            "[stream] label format error for key %r (params=%s): %s", key, params, ex
        )
        return template


# =============================================================================
# CUSTOM-EVENT PAYLOAD BUILDERS
# =============================================================================
# Nodes call ``writer(make_*())`` to push semantic events into the custom
# stream.  Payloads carry a ``key`` (looked up in USER_FACING_LABELS by the
# translator) plus structured ``params`` — never an inline rendered string.


def make_stage_start(
    stage: str, *, key: str | None = None, **params: Any
) -> dict[str, Any]:
    """Build a stage_start payload.

    Args:
        stage:  Graph node name (used as the ``stage`` field on the wire).
        key:    Label key in :data:`USER_FACING_LABELS`.  Defaults to
                ``f"{stage}.start"`` so simple cases need no key argument.
        **params: Substitutions for ``{placeholder}`` slots in the template.
    """
    return {
        "node": stage,
        "event": "stage_start",
        "key": key or f"{stage}.start",
        "params": params,
    }


def make_stage_end(
    stage: str, *, key: str | None = None, **params: Any
) -> dict[str, Any]:
    """Build a stage_end payload (see :func:`make_stage_start` for arg semantics)."""
    return {
        "node": stage,
        "event": "stage_end",
        "key": key or f"{stage}.end",
        "params": params,
    }


def make_agent_event(stage: str, *, key: str, **params: Any) -> dict[str, Any]:
    """Build a generic ``agent_event`` payload.

    Use this for mid-stage updates that aren't a node start/end (e.g. a
    per-task retrieval summary inside a long-running parallel execute).
    """
    return {
        "node": stage,
        "event": "agent_event",
        "key": key,
        "params": params,
    }


# =============================================================================
# SSE CHUNK BUILDERS
# =============================================================================


def _now() -> int:
    return int(time.time())


def _build_chunk(
    *,
    resp_id: str,
    model: str,
    event_type: str | None,
    stage: str | None,
    content: str = "",
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    citations: Citations | None = None,
    metrics: Metrics | None = None,
    usage: Usage | None = None,
) -> str:
    """Build a single SSE-encoded ChainResponse line."""
    chain_response = ChainResponse(
        id=resp_id,
        model=model,
        object="chat.completion.chunk",
        created=_now(),
        event_type=event_type,
        stage=stage,
    )
    chain_response.choices.append(
        ChainResponseChoices(
            index=0,
            message=Message(
                role="assistant",
                content=content,
                reasoning_content=reasoning_content,
            ),
            delta=Message(
                role=None,
                content=content,
                reasoning_content=reasoning_content,
            ),
            finish_reason=finish_reason,
        )
    )
    if citations is not None:
        chain_response.citations = citations
    if metrics is not None:
        chain_response.metrics = metrics
    if usage is not None:
        chain_response.usage = usage

    return "data: " + chain_response.model_dump_json() + "\n\n"


# =============================================================================
# DEBUG-EVENT FORMATTING
# =============================================================================


def _format_debug_event(payload: dict[str, Any]) -> str:
    """Compact JSON-string summary of a LangGraph debug event."""
    event_type = payload.get("type", "?")
    node_payload = payload.get("payload", {}) or {}
    summary: dict[str, Any] = {"type": event_type, "step": payload.get("step")}
    if event_type == "task":
        summary["node"] = node_payload.get("name")
    elif event_type == "task_result":
        summary["node"] = node_payload.get("name")
        summary["wrote"] = [w[0] for w in node_payload.get("writes", []) if w]
    return json.dumps(summary, default=str)


# =============================================================================
# PARALLEL-TASK BUFFER DRAIN
# =============================================================================


def _drain_parallel_buffer(
    parallel_buffers: dict[str, dict[str, dict[str, list[str]]]],
    node: str,
    *,
    resp_id: str,
    model: str,
) -> list[str]:
    """Build SSE chunks from any tokens buffered for ``node`` and clear them.

    ``parallel_buffers`` is shaped ``{node: {run_id: {"content": [...],
    "reasoning": [...]}}}``.  For each ``run_id`` in insertion order (which
    corresponds to first-token-arrival order across the concurrent tasks),
    emits at most two consolidated chunks:

    * one ``intermediate_reasoning`` if the run produced any reasoning tokens
    * one ``intermediate_output`` if the run produced any visible-content
      tokens

    Returns an ordered list of SSE-encoded ``ChainResponse`` strings.  The
    node's entry is removed from ``parallel_buffers`` so a subsequent stage
    on the same node (e.g. scope-discovery → answer phase) starts fresh.
    """
    out: list[str] = []
    runs = parallel_buffers.pop(node, None)
    if not runs:
        return out
    for parts in runs.values():
        reasoning = "".join(parts.get("reasoning", []))
        if reasoning:
            out.append(
                _build_chunk(
                    resp_id=resp_id,
                    model=model,
                    event_type=EventType.INTERMEDIATE_REASONING.value,
                    stage=node,
                    reasoning_content=reasoning,
                )
            )
        content = "".join(parts.get("content", []))
        if content:
            out.append(
                _build_chunk(
                    resp_id=resp_id,
                    model=model,
                    event_type=EventType.INTERMEDIATE_OUTPUT.value,
                    stage=node,
                    reasoning_content=content,
                )
            )
    return out


def _resolve_run_id(chunk: Any, metadata: dict[str, Any] | None) -> str:
    """Best-effort discriminator for grouping tokens from one LLM invocation.

    LangChain stamps every ``BaseChatModel`` invocation with a UUID; chunks
    from the same astream share that id.  We look at ``chunk.id`` first
    (most reliable across providers), then ``metadata["run_id"]``, and fall
    back to ``"_unknown"``.  An ``"_unknown"`` bucket means concurrent
    tasks would collapse into one consolidated chunk on this provider —
    not ideal, but never crashes.
    """
    candidate = getattr(chunk, "id", None)
    if not candidate and metadata:
        candidate = metadata.get("run_id")
    return str(candidate) if candidate else "_unknown"


# =============================================================================
# THE TRANSLATOR
# =============================================================================


async def translate_graph_stream(
    graph: Any,
    initial_state: Any,
    *,
    model: str,
    recursion_limit: int,
    verification_enabled: bool,
    citations_provider: Callable[[], Citations | None] | None = None,
    enable_debug_stream: bool = False,
    rag_start_time_sec: float | None = None,
    on_complete: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    """Consume ``graph.astream(...)`` and yield SSE-formatted ``ChainResponse``
    strings ready to be streamed to the HTTP client.

    Args:
        graph:                Compiled LangGraph (must support ``astream``).
        initial_state:        Initial state for the graph.
        model:                Model name to stamp on each ChainResponse.
        recursion_limit:      LangGraph recursion limit.
        verification_enabled: Whether the agent's verification stage is on for
                              this run.  Controls how synthesize tokens are
                              labeled — see the module docstring for details.
        citations_provider:   A zero-arg callable that returns the current
                              ``Citations | None`` object built up from the
                              retriever bridge.  Called when the FIRST
                              ``final_answer`` chunk is about to be emitted.
                              Pass ``None`` to skip citations.
        enable_debug_stream:  When True, debug-mode events are forwarded to the
                              client as ``agent_event`` chunks; otherwise they
                              are only logged server-side.
        rag_start_time_sec:   Wall-clock start of the request (for RAG TTFT).
        on_complete:          Optional async callable invoked after the
                              final-answer stream ends.  Receives the final
                              answer string for trace/metrics bookkeeping.
    """
    resp_id = str(uuid4())
    request_start = time.time()

    # Per-stream state ---------------------------------------------------------
    first_final_token_seen = False
    first_token_seen = False  # any kind of token; for logging only
    rag_ttft_ms: float | None = None
    llm_ttft_ms: float | None = None

    # Live-streamed final-answer tokens (verify-disabled path only).  When
    # verification is enabled, synthesize tokens stream as intermediate and
    # this stays empty — we use ``latest_final_answer`` from the values stream
    # instead.
    accumulated_final_answer: list[str] = []

    # Latest authoritative ``state.final_answer`` observed via the LangGraph
    # values stream.  Updated each time a node returns; at stream end this
    # holds the answer that the synthesize node committed to state — which
    # is what we emit as the single ``final_answer`` chunk in verify-enabled
    # runs, and what we report to ``on_complete``.
    latest_final_answer: str = ""

    # Per-run token buffer for parallel-task nodes.  See PARALLEL_TASK_NODES
    # for the rationale.  Shape: {node: {run_id: {"content": [parts],
    # "reasoning": [parts]}}}.  Drained at the corresponding stage_end (and
    # again at stream end as a safety net for graphs that didn't emit one).
    parallel_buffers: dict[str, dict[str, dict[str, list[str]]]] = {}

    try:
        async for mode, data in graph.astream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            stream_mode=["messages", "custom", "debug", "values"],
        ):
            # -----------------------------------------------------------------
            # CUSTOM mode — semantic events emitted by node ``writer({...})``.
            # -----------------------------------------------------------------
            if mode == "custom":
                event = data.get("event", "")
                node = data.get("node", "")

                if event in ("stage_start", "stage_end", "agent_event"):
                    # When a parallel-task node closes, drain its per-run
                    # token buffer first so the consolidated intermediate_*
                    # chunks land BEFORE the stage_end event on the wire.
                    if event == "stage_end" and node in PARALLEL_TASK_NODES:
                        for sse in _drain_parallel_buffer(
                            parallel_buffers, node, resp_id=resp_id, model=model
                        ):
                            yield sse

                    label = _format_label(
                        data.get("key", ""), data.get("params") or {}
                    )
                    if event == "stage_start":
                        wire_event_type = EventType.STAGE_START.value
                    elif event == "stage_end":
                        wire_event_type = EventType.STAGE_END.value
                    else:
                        wire_event_type = EventType.AGENT_EVENT.value
                    yield _build_chunk(
                        resp_id=resp_id,
                        model=model,
                        event_type=wire_event_type,
                        stage=node,
                        reasoning_content=label,
                    )
                    continue

                # Unknown custom event — log and skip rather than 500.
                logger.debug("[stream] unknown custom event: %s", data)
                continue

            # -----------------------------------------------------------------
            # MESSAGES mode — LLM token deltas.  Tuple of (chunk, metadata).
            # -----------------------------------------------------------------
            if mode == "messages":
                chunk, metadata = data
                node = (metadata or {}).get("langgraph_node", "") or ""

                content_delta = getattr(chunk, "content", "") or ""
                if content_delta and not isinstance(content_delta, str):
                    content_delta = str(content_delta)

                kw = getattr(chunk, "additional_kwargs", None) or {}
                reasoning_delta = kw.get("reasoning_content")
                if reasoning_delta and not isinstance(reasoning_delta, str):
                    reasoning_delta = str(reasoning_delta)

                if not content_delta and not reasoning_delta:
                    continue

                # Parallel-task nodes — buffer per-run-id; the stage_end
                # handler will drain the buffer and emit one consolidated
                # intermediate_reasoning + intermediate_output per task.
                if node in PARALLEL_TASK_NODES:
                    run_id = _resolve_run_id(chunk, metadata)
                    run_buf = parallel_buffers.setdefault(node, {}).setdefault(
                        run_id, {"content": [], "reasoning": []}
                    )
                    if reasoning_delta:
                        run_buf["reasoning"].append(reasoning_delta)
                    if content_delta:
                        run_buf["content"].append(content_delta)
                    continue

                # synthesize streams as ``final_*`` only when verification is
                # disabled (it's the sole terminal LLM node in that path).
                # When verification is enabled, both the draft and any
                # post-verification revision stream as ``intermediate_*``; the
                # authoritative answer is emitted as a single ``final_answer``
                # chunk at stream end (see end-of-stream handler below).
                is_final_node = node == "synthesize" and not verification_enabled

                if not first_token_seen:
                    first_token_seen = True
                    logger.debug(
                        "[stream] first LLM token from node=%s (final=%s)",
                        node,
                        is_final_node,
                    )

                # Reasoning delta — emit as own chunk so clients can render it
                # in a separate panel without splitting on content boundaries.
                if reasoning_delta:
                    yield _build_chunk(
                        resp_id=resp_id,
                        model=model,
                        event_type=(
                            EventType.FINAL_REASONING.value
                            if is_final_node
                            else EventType.INTERMEDIATE_REASONING.value
                        ),
                        stage=node,
                        reasoning_content=reasoning_delta,
                    )

                if content_delta:
                    if is_final_node:
                        # Mirror response_generator.generate_answer_async:
                        # citations + TTFT attach to the FIRST final-answer chunk.
                        citations: Citations | None = None
                        if not first_final_token_seen:
                            first_final_token_seen = True
                            llm_ttft_ms = (time.time() - request_start) * 1000
                            if rag_start_time_sec is not None:
                                rag_ttft_ms = (time.time() - rag_start_time_sec) * 1000
                            logger.info(
                                "    == LLM Time to First Token (TTFT): %.2f ms ==",
                                llm_ttft_ms,
                            )
                            if rag_ttft_ms is not None:
                                logger.info(
                                    "    == RAG Time to First Token (TTFT): %.2f ms ==",
                                    rag_ttft_ms,
                                )
                            if citations_provider is not None:
                                try:
                                    citations = citations_provider()
                                except Exception as cex:  # noqa: BLE001
                                    logger.warning(
                                        "[stream] citations provider failed: %s", cex
                                    )

                        accumulated_final_answer.append(content_delta)
                        yield _build_chunk(
                            resp_id=resp_id,
                            model=model,
                            event_type=EventType.FINAL_ANSWER.value,
                            stage=node,
                            content=content_delta,
                            citations=citations,
                        )
                    else:
                        yield _build_chunk(
                            resp_id=resp_id,
                            model=model,
                            event_type=EventType.INTERMEDIATE_OUTPUT.value,
                            stage=node,
                            reasoning_content=content_delta,
                        )
                continue

            # -----------------------------------------------------------------
            # DEBUG mode — internal LangGraph lifecycle events.
            # -----------------------------------------------------------------
            if mode == "debug":
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[stream] debug event: %s", _format_debug_event(data))
                if enable_debug_stream:
                    yield _build_chunk(
                        resp_id=resp_id,
                        model=model,
                        event_type=EventType.AGENT_EVENT.value,
                        stage=(data.get("payload") or {}).get("name"),
                        reasoning_content=_format_debug_event(data),
                    )
                continue

            # -----------------------------------------------------------------
            # VALUES mode — full state snapshot after each node update.  We
            # only need ``final_answer`` from it; tracking the latest value
            # here avoids buffering tokens or emitting extra writer events
            # from the synthesize node.
            # -----------------------------------------------------------------
            if mode == "values":
                if isinstance(data, dict):
                    val = data.get("final_answer")
                    if isinstance(val, str) and val:
                        latest_final_answer = val
                continue

        # ---------------------------------------------------------------------
        # Safety net — flush any parallel-task buffers that didn't get drained
        # by a stage_end (graph crashed mid-execute, node was cancelled, etc.).
        # ---------------------------------------------------------------------
        for leftover_node in list(parallel_buffers.keys()):
            for sse in _drain_parallel_buffer(
                parallel_buffers, leftover_node, resp_id=resp_id, model=model
            ):
                yield sse

        # ---------------------------------------------------------------------
        # Verify-enabled path — emit the single ``final_answer`` chunk now.
        # Synthesize tokens streamed earlier as ``intermediate_*``; this is
        # where the user actually sees the answer.  Citations and TTFT attach
        # here because this is the first (and only) ``final_answer`` chunk in
        # this mode.  Skip if no answer was produced (e.g. graph crashed
        # before synthesize wrote state.final_answer).
        # ---------------------------------------------------------------------
        if verification_enabled and latest_final_answer:
            citations: Citations | None = None
            llm_ttft_ms = (time.time() - request_start) * 1000
            if rag_start_time_sec is not None:
                rag_ttft_ms = (time.time() - rag_start_time_sec) * 1000
            logger.info(
                "    == LLM Time to First Token (TTFT): %.2f ms ==", llm_ttft_ms
            )
            if rag_ttft_ms is not None:
                logger.info(
                    "    == RAG Time to First Token (TTFT): %.2f ms ==", rag_ttft_ms
                )
            if citations_provider is not None:
                try:
                    citations = citations_provider()
                except Exception as cex:  # noqa: BLE001
                    logger.warning("[stream] citations provider failed: %s", cex)
            yield _build_chunk(
                resp_id=resp_id,
                model=model,
                event_type=EventType.FINAL_ANSWER.value,
                stage="synthesize",
                content=latest_final_answer,
                citations=citations,
            )

        # ---------------------------------------------------------------------
        # Stream finished — emit the trailing finish chunk with metrics, just
        # as generate_answer_async does for the regular pipeline.
        # ---------------------------------------------------------------------
        llm_generation_time_ms = (time.time() - request_start) * 1000
        final_metrics = Metrics(
            rag_ttft_ms=rag_ttft_ms,
            llm_ttft_ms=llm_ttft_ms,
            llm_generation_time_ms=llm_generation_time_ms,
        )

        finish_response = ChainResponse(
            id=resp_id,
            model=model,
            object="chat.completion.chunk",
            created=_now(),
            metrics=final_metrics,
        )
        finish_response.choices.append(
            ChainResponseChoices(
                index=0,
                message=Message(role="assistant", content=""),
                delta=Message(role=None, content=""),
                finish_reason="stop",
            )
        )
        yield "data: " + finish_response.model_dump_json() + "\n\n"

        if on_complete is not None:
            # Verify-enabled streams emit the answer as a single chunk drawn
            # from state, not as token deltas, so the accumulator is empty in
            # that mode — fall back to the values-stream snapshot.
            shown_final_answer = (
                latest_final_answer
                if verification_enabled
                else "".join(accumulated_final_answer)
            )
            try:
                await on_complete(shown_final_answer)
            except Exception as cex:  # noqa: BLE001
                logger.warning("[stream] on_complete hook failed: %s", cex)

    except asyncio.CancelledError:
        logger.info("[stream] client disconnected — cancelling graph stream")
        raise
    except Exception as ex:  # noqa: BLE001
        logger.exception("[stream] graph stream failed: %s", ex)
        trace = get_current_trace()
        if trace is not None and trace.end_time is None:
            trace.error = str(ex)[:500]
            trace.finalize()
        # Surface the error to the client so it doesn't hang on a half-stream.
        yield _build_chunk(
            resp_id=resp_id,
            model=model,
            event_type=EventType.ERROR.value,
            stage=None,
            content="I encountered an error while processing your request via agentic RAG.",
            reasoning_content=str(ex)[:500],
            finish_reason="stop",
        )

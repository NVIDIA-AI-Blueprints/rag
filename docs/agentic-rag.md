<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Agentic RAG for NVIDIA RAG Blueprint

## Overview

Standard Retrieval-Augmented Generation answers a query in a single shot: embed the query, retrieve top-k chunks, and ask an LLM to generate an answer from them. This works well for direct factual questions, but struggles when the query is ambiguous, spans multiple documents, requires combining several facts, or asks for information that needs to be located precisely inside a large or noisy corpus.

Agentic RAG addresses these cases by treating the query as a problem the system must reason about rather than a single retrieval call. Instead of one retrieve-then-generate pass, an LLM-driven agent plans a short sequence of focused sub-questions, executes each one against the retriever, evaluates the partial answers, retries with reformulated queries when results are incomplete, and finally synthesizes everything into a coherent response. An optional verification step inspects the synthesized answer for gaps and triggers targeted re-retrieval when needed.

The [NVIDIA RAG Blueprint](readme.md) implements Agentic RAG as a LangGraph plan-and-execute pipeline that sits alongside the standard RAG chain. It combines:

- **Two-phase planning** — an initial scope-discovery phase explores what the corpus actually contains for ambiguous queries, followed by a targeted answer-planning phase that produces concrete retrieval tasks.
- **Mini-agent task execution** — each task runs as a small retrieve-answer-retry loop, where a seed-query generator LLM reformulates the search whenever the partial answer indicates missing information.
- **Synthesis** — task sub-answers and the initial retrieval context are merged into a single final answer.
- **Optional verification** — a post-synthesis quality gate that detects coverage gaps, vague claims, and wrong-subject drift, then re-retrieves to fill them.

The pipeline is disabled by default, since Agentic RAG trades latency and LLM-call count for accuracy. It is best suited to multi-hop questions, ambiguous queries, queries that span multiple documents, and queries that require numeric extraction from tables or charts. It can be enabled globally for a deployment, or selectively per request — see [Enable Agentic RAG](#enable-agentic-rag).

## Key Benefits

- **No dataset-specific configuration.** Auto-adapts to any document collection through scope discovery; no per-corpus rules required.
- **Resolves ambiguous queries.** Scope discovery explores what data exists in the vector database before planning, so under-specified questions are disambiguated from the corpus itself.
- **Adaptive cost.** Simple queries are answered directly from the initial retrieval (minimal LLM calls); complex queries get full planning, retries, and verification.
- **Parallel task execution.** Independent tasks in a plan execute concurrently, minimizing wall time.
- **Verification gate.** A post-synthesis quality check catches incomplete coverage, vague answers, false negatives, and wrong-subject drift, then re-retrieves to fill gaps.

## Limitations

- Latency and LLM-call count are materially higher than the standard chain. Use the per-request override (see [Enable per request](#enable-per-request)) to apply it selectively rather than globally if you have latency-sensitive paths.
- The following features are not applied on the agentic path: NeMo Guardrails, Self-Reflection, Query Decomposition, and VLM Inference. Query rewriting, multi-turn chat history, multi-collection retrieval, citations, filter generation, and reranking are supported.
- Verification is single-pass — there is no nested verification loop.
- Tasks within a plan execute in a single parallel level; there is no DAG / depends-on construct.

## Architecture Overview

The pipeline is a LangGraph state machine with five components:

1. **Initial Retrieval** — runs the user query through the standard `/search` path (vector DB + reranker → top-k chunks) so planning is grounded in what the corpus actually contains.
2. **Planner (two-phase).** A single LLM decides between three plan shapes:
   - *Scope discovery plan* — 2–3 discovery tasks that explore what exists in the corpus when the query is ambiguous; the planner is then re-invoked with the discovery results.
   - *Answer plan* — targeted answer tasks built around what was found.
   - *Empty plan* — no tasks; the initial retrieval is sufficient and the pipeline goes directly to synthesis (the cheap path for simple queries).
3. **Task Execute** — each task runs as a mini-agent: retrieve → answer → if partial, the seed-query generator produces a follow-up query targeting what is still missing, and retries. Tasks within a plan execute concurrently.
4. **Synthesis** — combines task sub-answers, the initial retrieval context, and the resolved query into one coherent answer. Falls back to the initial context if all tasks return `[NO DATA]`.
5. **Verification (optional)** — inspects the answer for gaps. On `pass`, the answer is final. On `fail`, follow-up tasks run through the same task-execute engine and synthesis is repeated with the gap data.

## Enable Agentic RAG

### Enable per request (API) - Recommended

The preferred way to enable Agentic RAG is per request via the `agentic` field in the `/v1/generate` request body.
The server-level `ENABLE_AGENTIC_RAG` env var controls only the default behavior when `agentic` is omitted.

```jsonc
{
  "messages": [{"role": "user", "content": "..."}],
  "use_knowledge_base": true,
  "agentic": true,
  "collection_names": ["..."]
}
```

When `agentic` is omitted or `null`, the server falls back to `ENABLE_AGENTIC_RAG`. Agentic RAG is only applied when `use_knowledge_base=true`. The agentic path also honors `enable_streaming`: when `true` (default), the agent streams stage events and final-answer tokens as Server-Sent Events; when `false`, the graph runs to completion and the full answer is returned in a single chunk. The standard RAG chain always streams.

### Change the deployment default (environment variable)

Use this when you want to change the default behavior for all requests that do not explicitly set `agentic`.

### Docker Deployment

Follow the deployment guide for [Self-Hosted Models](deploy-docker-self-hosted.md) or [NVIDIA-Hosted Models](deploy-docker-nvidia-hosted.md). The reference compose env file (`deploy/compose/nvdev.env`) already contains the agentic LLM settings; only the enable flag needs to be flipped.

```bash
export ENABLE_AGENTIC_RAG=true
```

Then restart the RAG server:

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Helm Deployment

Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml):

```yaml
envVars:
  # ... existing configurations ...
  ENABLE_AGENTIC_RAG: "true"

# Optional — per-role API keys (only required when overriding NVIDIA_API_KEY).
envSecrets:
  agenticPlannerLlmApiKey: ""
  agenticTaskLlmApiKey: ""
  agenticSeedGenLlmApiKey: ""
  agenticSynthesisLlmApiKey: ""
```

Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

## Configuration

All agentic RAG behavior is controlled through environment variables exposed by `deploy/compose/docker-compose-rag-server.yaml` and the corresponding Helm `values.yaml`.

### Top-level

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_AGENTIC_RAG` | `false` | Route knowledge-base queries through the agentic pipeline. Can be overridden per request via the `agentic` field. |
| `AGENTIC_LOG_LEVEL` | `INFO` | Log level for the agent (`DEBUG` / `INFO` / `WARNING` / `ERROR`). |
| `AGENTIC_VERIFICATION_ENABLED` | `false` | Run the verification gate after first synthesis. Enable for higher accuracy at extra LLM cost. |
| `AGENTIC_CONTEXT_MAX_TOKENS` | `100000` | Token budget for chunk context inside agent prompts; chunks beyond this are truncated. |

### Per-role LLMs

Each LLM role has its own env var prefix. If a role's `MODEL` is left empty, the builder falls back to the planner LLM, so a minimal deployment only needs the four `AGENTIC_PLANNER_LLM_*` variables.

| Role | Used for | Server URL | Model | API Key |
| --- | --- | --- | --- | --- |
| Planner | Scope resolution + task creation + verification | `AGENTIC_PLANNER_LLM_SERVERURL` | `AGENTIC_PLANNER_LLM_MODEL` | `AGENTIC_PLANNER_LLM_APIKEY` |
| Task | Answering individual sub-questions | `AGENTIC_TASK_LLM_SERVERURL` | `AGENTIC_TASK_LLM_MODEL` | `AGENTIC_TASK_LLM_APIKEY` |
| Seed-gen | Generating retry follow-up queries | `AGENTIC_SEED_GEN_LLM_SERVERURL` | `AGENTIC_SEED_GEN_LLM_MODEL` | `AGENTIC_SEED_GEN_LLM_APIKEY` |
| Synthesis | Final answer generation | `AGENTIC_SYNTHESIS_LLM_SERVERURL` | `AGENTIC_SYNTHESIS_LLM_MODEL` | `AGENTIC_SYNTHESIS_LLM_APIKEY` |

The default `SERVERURL` is `nim-llm:8000` and the default `MODEL` is `nvidia/nemotron-3-super-120b-a12b`. Setting `SERVERURL=""` routes the role to the NVIDIA-hosted API; the `APIKEY` is optional and inherits `NVIDIA_API_KEY` when unset.

## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md)
- [Customize Prompts](prompt-customization.md)
- [Query Decomposition](query_decomposition.md)
- [Self-Reflection](self-reflection.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

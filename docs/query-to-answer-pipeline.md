<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Query-to-Answer Pipeline

This document explains the end‑to‑end flow from when a user submits a query until the RAG system returns an answer, and how to analyze the time spent in each stage of the pipeline.

## Pipeline Overview: Query → Answer

A single RAG request passes through the following stages in sequence, and optional stages are included only when their corresponding features are enabled.

1. **Query rewriter** (optional) – When [multi-turn conversation](multiturn.md) and query rewriting are enabled, the last user message is rewritten into a standalone, context-rich query using the conversation history, improving retrieval for follow-up questions.

2. **Retriever** – The (optionally rewritten) query is embedded and used to search the vector store, returning top‑k candidate chunks (configurable via `vdb_top_k`) from the chosen collection or collections.

3. **Context reranker** (optional) – When [reranking](accuracy_perf.md) is enabled, a reranker model scores the retrieved chunks for relevance to the query and returns the top‑k chunks (configurable via `reranker_top_k`) to use as context.

4. **LLM generation (llm-stream)** – The query and selected context are sent to the LLM, which generates the answer and streams it back to the client. [Guardrails](nemo-guardrails.md), [citations](user-interface.md), and [VLM inference](vlm.md) are applied during this stage when enabled.

Additional optional logic (for example, [query decomposition](query_decomposition.md) or [self-reflection](self-reflection.md)) may run around or within these stages, but the core flow is the sequence described above.

## How to Study Time Spent in the Pipeline

You can analyze where time is spent in two ways: **distributed traces** (per-request, per-stage) and **aggregate metrics** (histograms over many requests).

### Zipkin  traces

When [observability is enabled](observability.md), each request is traced. In Zipkin:

1. Open **http://localhost:9411** and find a trace for a RAG request (for example,  a call to `/v1/generate`).
2. The trace shows **spans** for each workflow: `query-rewriter`, `retriever`, `context-reranker`, `llm-stream`; each span's **duration** is the time spent in that stage.
3. To view a stage's inputs and outputs, click the span and look for `traceloop.entity.input` and `traceloop.entity.output` in the span details.

Span durations let you compare slow and fast requests and identify which stage contributes most to latency (for example, retrieval versus LLM generation).

### Prometheus / Grafana metrics

The RAG server exports latency metrics (in milliseconds) that correspond to each stage of the pipeline:

| Metric name | Description | Pipeline stage |
|-------------|-------------|----------------|
| `retrieval_time_ms` | Time to fetch and return candidate chunks from the vector store | Retriever |
| `context_reranker_time_ms` | Time to rerank chunks (when reranking is enabled) | Context reranker |
| `llm_ttft_ms` | Time from sending the request to receiving the first token from the LLM | Start of LLM generation |
| `llm_generation_time_ms` | Total time for the LLM to produce the complete response | LLM generation (llm-stream) |
| `rag_ttft_ms` | Time from request start until the client receives the first token (entire pipeline before the first token) | End-to-end (query → first token) |

These metrics are available at **http://localhost:8889/metrics** when you run the default Docker observability stack. In Grafana, configure a Prometheus data source that points to this Prometheus instance, then build dashboards (for example, histograms or percentile panels) to analyze time spent in retrieval, reranking, and LLM generation over many requests.

### Quick checklist for studying latency

- **Slow first token** – Check `rag_ttft_ms` and the trace: compare the time spent in retriever, reranker, and the atart of LLM generation; long retriever or reranker spans often indicate the cause.
- **Slow full response** – Check `llm_generation_time_ms` and the `llm-stream` span; adjust `max_tokens`, model choice, or generation settings if this stage dominates.
- **Retrieval heavy latentcy** – Compare `retrieval_time_ms` and `context_reranker_time_ms`; if they are high, consider `vdb_top_k` / `reranker_top_k` or revising your indexing strategy.

## Related Topics

- [Observability](observability.md) – Enable tracing and metrics (Zipkin, Grafana).
- [RAG Pipeline Debugging Guide](debugging.md) – Verify deployment and troubleshoot failures.
- [Best Practices for Common Settings](accuracy_perf.md) – Tune retrieval and generation (e.g. reranker, top-k).

# Pull Request Summary: Query-to-Answer Pipeline Documentation

## Summary

Adds documentation explaining what happens from query → answer and how to study time spent in each part of the RAG pipeline. Addresses stakeholder request for more visibility into pipeline flow and latency analysis.

## Motivation

- No single doc described the end-to-end query-to-answer flow (stage order and purpose).
- Observability docs covered Zipkin/Grafana setup but did not map pipeline stages to traces/metrics or explain how to interpret timing.

## Changes

### New

- **`docs/query-to-answer-pipeline.md`**
  - **Pipeline overview:** Ordered stages (query rewriter → retriever → context reranker → LLM generation) with short descriptions and when optional stages run.
  - **Studying time:** How to use Zipkin span durations and Prometheus/Grafana metrics to see where latency is spent.
  - **Metrics table:** Maps `retrieval_time_ms`, `context_reranker_time_ms`, `llm_ttft_ms`, `llm_generation_time_ms`, `rag_ttft_ms` to pipeline stages.
  - **Latency checklist:** Guidance for slow first token, slow full response, and retrieval-heavy cases.

### Updated

- **`docs/observability.md`**
  - New section: “Query-to-Answer Pipeline and Studying Time Spent” with summary and link to the new doc.
  - Related Topics: link to Query-to-Answer Pipeline.

- **`docs/index.md`** and **`docs/readme.md`**
  - Observability and Telemetry: added [Query-to-Answer Pipeline](query-to-answer-pipeline.md) to the list and (index only) toctree.

- **`docs/debugging.md`**
  - Retrieval pipeline section: intro sentence and link to Query-to-Answer Pipeline for stages and timing.
  - Related Topics: link to Query-to-Answer Pipeline.

## Testing

- Documentation-only change; no code or tests modified.
- Links and paths verified relative to `rag/docs/`.

## Related

- Stakeholder request: “Can we add more documentation and explanation on what transpires from query -> answer with information on how to study time spent in various parts of the pipeline?”

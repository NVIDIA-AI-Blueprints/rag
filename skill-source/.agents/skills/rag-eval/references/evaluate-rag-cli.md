# `evaluate_rag.py` CLI reference


Use when:
- The user wants to measure RAG pipeline quality.
- The user is asking about accuracy, relevancy or groundedness.
- The user wants to run the filesystem benchmark evaluator (`scripts/eval/evaluate_rag.py`) with `corpus/` plus `train.json`.



Use `scripts/eval/evaluate_rag.py` and its `argparse` definitions as the source of truth for flags, default values, optional behaviors, and full command examples.

## Prerequisites

1. `NVIDIA_API_KEY` in the environment (script raises if missing) — RAGAS judge LLM.
2. `RAG_EVAL_JUDGE_MODEL` (optional) — model id for the RAGAS judge (`ChatNVIDIA`). When unset or empty, defaults to `mistralai/mixtral-8x22b-instruct-v0.1`.
3. RAG server at `http://<host>:<port>/v1/generate` (streaming).
4. Ingestor at `--ingestor_server_url` (default `http://localhost:8082`); code appends `/v1/` via `urljoin` — pass base URL without trailing `/v1`.
5. From repo root: `uv sync --project scripts/eval` (evaluator dependencies are in `scripts/eval/pyproject.toml`; `uv run --project scripts/eval` uses that environment).

When preparing a dataset root from an external benchmark, the `corpus/` directory should use PDF files when possible so runs align with `--file-type pdf` defaults and production-style document ingestion. If sources are bare links or lack a file extension, prefer materializing PDFs rather than plain text; see `scripts/eval/README.md`.

## Arguments

| Argument | Required | Default / notes |
|----------|----------|------------------|
| `--dataset-paths` | Yes (one or more paths) | Each root must contain `corpus/` and `train.json`. |
| `--host` | Yes | RAG server host. |
| `--port` | Yes | RAG server port. |
| `--file-type` | No | Default `pdf`. Prefer PDF-based `corpus/` when converting benchmarks; if the value contains `pdf`, PDF page counts are used for ingestion metrics. |
| `--verbose` | No | Flag. |
| `--thread` | No | Default `4`; number of parallel workers for queries and related work. |
| `--output_dir` | No | Default `results`; each dataset gets a subdirectory named after the dataset basename. |
| `--retries` | No | Default `3`; currently unused in the code path, kept for CLI compatibility. |
| `--batch_size` | No | Default `1000`; ingestion batch size. |
| `--top_k` | No | If set, sent as `reranker_top_k`; if omitted, it is not sent. |
| `--vdb_top_k` | No | If set, sent as `vdb_top_k`; if omitted, it is not sent. |
| `--ingestor_server_url` | No | Default `http://localhost:8082`. |
| `--skip_ingestion` | No | Flag; run query and scoring only (collection must already exist). |
| `--skip_evaluation` | No | Flag; perform ingestion only. |
| `--delete_collection` | No | Flag; delete the collection after the run for that dataset. |
| `--force_ingestion` | No | Flag; delete the collection first, then re-ingest. |
| `--collection` | No | Override collection name (default is the dataset folder basename). |
| `--model` | No | If set, passed to `generate` as `model`; otherwise omitted so the server default is used. |
| `--llm_endpoint` | No | If set, passed as `llm_endpoint`; otherwise omitted. |
| `--timeout` | No | Default `180` seconds for RAG HTTP requests. |

## `train.json` — `contexts`

Each element includes `filename` (corpus basename without extension) and `text` (snippet or chunk):

```json
"contexts": [
  { "filename": "DOCUMENT_STEM", "text": "..." }
]
```

See `scripts/eval/README.md` for the full contract (including optional plain-string arrays).

## Fixed behavior (not CLI flags)

- The evaluator does not send `vdb_endpoint`, embedding dimension, or related overrides to the ingestor or `/v1/generate`; services use their configured defaults (environment / server config).
- Ingestion uploads always use `blocking: true` for a synchronous ingestor response.
- The client does not send `split_options` on document upload, so chunk size and overlap are controlled by the ingestor server configuration.
- RAG queries always use `POST /v1/generate` with a single user turn per benchmark row, and `enable_filter_generator` is set to `false` in the generate payloads.

## Examples

Single dataset, full ingest + eval (from repository root):

```bash
export NVIDIA_API_KEY="nvapi-..."

uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths ./datasets/sample_eval \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --output_dir results
```

Collection already populated:

```bash
uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths ./datasets/sample_eval \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --skip_ingestion \
  --output_dir results
```

## Outputs (under `--output_dir/<dataset_basename>/`)

- `rag_<label>_evaluation_summary.json` — mean metrics plus a token usage summary when available.
- `rag_<label>_evaluation_results.json` — per-metric lists and per-sample usage details when available.
- `rag_<label>_evaluation_metrics.json` — ingestion KPIs, evaluation means, and the token usage structure.
- `rag_<label>_evaluation_data.json` is written before scoring.

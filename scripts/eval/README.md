# RAG evaluation scripts

Scripts in this folder benchmark a deployed RAG stack. They load a local dataset into the ingestor, send each question to the RAG server’s generate API, and evaluate the responses using RAGAS with [NVIDIA metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/).

## Prerequisites

- The RAG server and ingestor server must be accessible on the network (for example, after completing the [Quickstart: self-hosted Docker](../../docs/deploy-docker-self-hosted.md)).
- Set `NVIDIA_API_KEY` in the environment; it is required for `langchain_nvidia_ai_endpoints` to run the RAGAS judge.
- `RAG_EVAL_JUDGE_MODEL` (optional) — LLM id passed to `ChatNVIDIA` for RAGAS scoring; defaults to `mistralai/mixtral-8x22b-instruct-v0.1` when unset or empty.

## Install (this repository)

From the repository root, sync and run with uv’s `--project` flag pointing at this folder:

```bash
uv sync --project scripts/eval
uv run --project scripts/eval python scripts/eval/evaluate_rag.py --help
```

Or work inside `scripts/eval` (creates `.venv` next to `pyproject.toml`):

```bash
cd scripts/eval
uv sync
uv run python evaluate_rag.py --help
```

Use the same `--project scripts/eval` / `cd scripts/eval` pattern for every command below when invoking `evaluate_rag.py`.

## Accepted dataset format

Use `evaluate_rag.py` as the driver. Invoke it with one or more dataset root directories via `--dataset-paths`. It does not support dataset names, only filesystem paths.

`evaluate_rag.py` validates each dataset root with `validate_dataset_roots():`

- The path must be a directory.
- A `corpus\` directory must exist (documents to ingest are discovered recursively under it).
- A `train.json` file must exist, be a regular file, and contain UTF-8 JSON (see shapes below).

If ingestion is not skipped, every file under corpus/ that is not already marked as ingested for the target collection will be uploaded. Use the same collection name that the eval run will query (by default, the dataset directory’s basename, unless you pass `--collection`).

### Converting external benchmarks into this layout

When importing a dataset from elsewhere (hosted catalogs, JSONL, CSV, APIs, etc.), materialize `corpus/` as PDF whenever possible—export or print sources to PDF so the bundle matches common document RAG and the default `--file-type pdf` (including PDF page metrics during ingest).

### Directory layout (summary)

```text
my_dataset/                 ← dataset root (pass this to --dataset-paths)
  corpus/                   ← required; source documents (nested dirs allowed)
    doc_a.pdf               
    notes.pdf               
  train.json                ← required; eval questions and answers
```

### `train.json`

Use a JSON array of objects. The script reads these fields from each object:

| Field              | Required                  | Used by evaluator                                                      |
|--------------------|---------------------------|-------------------------------------------------------------------------|
| `question`         | Yes                       | Sent to the RAG server as the user message for each item.              |
| `answer`           | Yes (for meaningful scores) | Used as the reference for RAGAS judge metrics (`evaluate_result`).   |
| `id` or `query_id` | No                        | If present, stored on the saved eval row for traceability.             |
| `contexts`         | No                        | Optional; format below. |
| Other keys (for example, `is_impossible`) | No | Ignored by the current driver unless you extend it.                    |

Context relevance and response groundedness compare the model answer to the contexts retrieved from the RAG server. E2E accuracy uses `question`, `answer`, and the model’s answer.

Each item in `contexts` should include the `filename` and `text` fields:

- `filename` — stem of the source file under `corpus/` (no extension), e.g. `COMPANY_2020_10K` for `COMPANY_2020_10K.pdf`.
- `text` — the span tied to that file (page, snippet, or chunk text).

```json
"contexts": [
  {
    "filename": "COMPANY_2020_10K",
    "text": "…"
  }
]
```

Multiple objects are allowed when several spans or files apply. A legacy shape of plain strings (`["…", "…"]`) is also valid for simple bundles where a file stem is not needed.

Minimal example:

```json
[
  {
    "id": "q1",
    "question": "What is the corporate tax rate in the United States?",
    "answer": "21%"
  }
]
```

Corpus files should be the documents you want indexed; prefer PDF when building the bundle from external data, especially when upstream references omit a file extension. Naming must stay consistent with how your ingestor stores `document_name` (citation parsing matches streamed citation results using basenames).

### Ingestion: `--file-type`

The default value is `pdf`. If the converted corpus is mostly PDFs (recommended when preparing benchmarks, including when you materialize sources from links without explicit extensions), leave defaults or pass `--file-type pdf`; the substring `pdf` enables PDF page counts in ingestion metrics. For non-PDF corpora, use values such as `txt` or `txt,html` so they match what is under `corpus/`.

### Checklist

- The dataset path is a directory containing `corpus/` and `train.json`.
- `train.json` is valid JSON: array of objects (top-level dict / multi-turn bundles are rejected).
- Every turn / row has `question` and `answer` where you need judge scores.
- `corpus/` holds the files you intend to retrieve against.
- `NVIDIA_API_KEY` is set for cloud judge models.


## What gets measured

- Ingestion: time, file counts, and (for PDFs) pages per second — appended to `rag_<label>_evaluation_metrics.json`.
- Quality (RAGAS): [NVIDIA metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/) from `ragas.metrics` — `AnswerAccuracy` (column `nv_accuracy`), `ContextRelevance` (`nv_context_relevance`), `ResponseGroundedness` (`nv_response_groundedness`); the latter two require non-empty retrieved contexts from the RAG response.
- Token usage: aggregated from streaming generate responses when the server sends `usage` chunks.

The Judge model for RAGAS is read from `RAG_EVAL_JUDGE_MODEL` (default `mistralai/mixtral-8x22b-instruct-v0.1`); see `evaluate_rag.py`.

## Main entrypoint

`evaluate_rag.py` — orchestrates collection check/create, blocking bulk upload to the ingestor (no async task polling in this client), parallel queries to `/v1/generate`, then RAGAS evaluation and JSON exports.

### Example: single dataset, local services

Adjust host, ports, and ingestor URL to match your stack.

```bash
export NVIDIA_API_KEY=your_key_here

uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths /path/to/my_dataset \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --output_dir results
```

### Example: skip ingestion (collection already populated)

```bash
uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths /path/to/my_dataset \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --skip_ingestion \
  --output_dir results
```

### Useful flags

| Flag | Notes |
|------|--------|
| `--collection` | Override collection name (default: dataset folder name). |
| `--skip_ingestion` / `--skip_evaluation` | Partial runs. |
| `--force_ingestion` | Deletes the collection first, then re-ingests. |
| `--delete_collection` | Deletes the collection after the run. |
| `--top_k` | Sent as `reranker_top_k` when set. |
| `--vdb_top_k` | Vector DB candidate pool when set. |
| `--model` / `--llm_endpoint` | Optional overrides passed to generate (omit to use server defaults). |
| `--batch_size` | Ingestion batch size (server max applies). |
| `--thread` | Parallelism for queries (and related work). |
| `--timeout` | RAG HTTP timeout seconds (default 180). |

Document uploads use `blocking: true`. The client does not send `split_options`; chunk size and overlap follow the ingestor server configuration. The ingestor URL you pass is normalized to include the `v1` API prefix internally.

## Outputs

Under `--output_dir` (default `results`), each dataset gets a subfolder named after the dataset directory:

| File | Content |
|------|---------|
| `rag_<label>_evaluation_data.json` | Per-query model outputs and contexts (written before scoring). |
| `rag_<label>_evaluation_summary.json` | Mean metrics (+ token usage summary when available). |
| `rag_<label>_evaluation_results.json` | Full RAGAS vectors and per-sample usage when present. |
| `rag_<label>_evaluation_metrics.json` | Structured ingestion + evaluation + token KPIs (`RagEvaluationMetrics`). |

---
name: rag-eval
description: >
  Prepares filesystem RAG benchmark roots (`corpus/` + `train.json`), validates `train.json` shape, explains
  optional `contexts` for RAGAS, and debugs `scripts/eval/evaluate_rag.py` runs with uv. Use when the user
  works with on-disk eval bundles, importing external benchmarks into this layout (corpus: prefer PDF, including when sources lack extensions), `train.json` schema,
  `evaluate_rag.py` flags, skip_ingestion, collection naming, NVIDIA_API_KEY or
  RAG_EVAL_JUDGE_MODEL for the RAGAS judge—even if they do not name this skill.
argument-hint: on-disk RAG eval | evaluate_rag | train.json | corpus | contexts | uv run --project scripts/eval
allowed-tools: Read, Grep, Glob, Bash(ls *), Bash(python3 *)
license: Apache-2.0
compatibility: Repository checkout with uv; Python 3.11+; run from repo root; uv sync --project scripts/eval (eval deps live in scripts/eval/pyproject.toml); network to RAG, ingestor, and vdb endpoints; NVIDIA_API_KEY for RAGAS; optional RAG_EVAL_JUDGE_MODEL (default mistralai/mixtral-8x22b-instruct-v0.1).
metadata:
  author: nvidia-rag-team
  version: "2.0"
---

# On-disk RAG evaluation (`corpus/` + `train.json`)

This skill covers the filesystem benchmark contract and how it ties into `evaluate_rag.py`. It does not replace the blueprint’s high-level evaluation guide; use `rag-blueprint` → `references/configure/evaluation.md` for notebooks vs CLI tradeoffs and links to `docs/evaluate.md`.

## Source of truth

| Piece | Location |
|-------|----------|
| Driver | `scripts/eval/evaluate_rag.py` (`CORPUS_DIRECTORY` = `corpus`, `EVAL_DATA` = `train.json`) |
| Human README | `scripts/eval/README.md` |
| Full CLI (flags, defaults, examples) | [`references/evaluate-rag-cli.md`](references/evaluate-rag-cli.md) |

Work from repository root (imports and paths assume this).

## Dataset layout

Each `--dataset-paths` entry is a directory containing:

1. `corpus/` — files indexed recursively for ingestion (see corpus format below).
2. `train.json` — evaluation questions and answers (see below).

## `train.json` schema

The driver accepts a **top-level JSON array** of objects only. Required per row: `question`, `answer`. Optional: `id` or `query_id`.

Optional `contexts`: JSON array of objects. Each object must include `filename` (corpus file stem, no extension) and `text` (the span for that file).

```json
"contexts": [
  { "filename": "DOCUMENT_STEM", "text": "..." }
]
```

Multiple entries per row are allowed. Plain strings (`["...", "..."]`) remain acceptable for minimal bundles without per-file tagging.

### Quick validation

```bash
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d, list) and all(isinstance(x, dict) for x in d), 'train.json must be a list of objects'" train.json
```

Pass the path to `train.json` as the only argument (example above uses `train.json` in the current directory).

## Corpus format when converting external benchmarks

When you map another benchmark into this layout, prefer putting sources in `corpus/` as PDF whenever you can (render or export HTML, Office, markdown, etc. to PDF). That matches typical production RAG on documents, aligns with the evaluator default `--file-type pdf`, and unlocks PDF page counts in ingestion metrics.

If the upstream artifact only gives URLs or document pointers with no filename or extension (common in published benchmarks), assume PDF as the target format for materialized files under `corpus/` instead of defaulting to plain text. Use other formats (for example `.txt`, `.html`) only when converting to PDF is impractical; then set `--file-type` to match what dominates under `corpus/`.

Keep `train.json` `filename` fields aligned with corpus basenames without an extension (for example `Report_2023` for `Report_2023.pdf`).

## Bringing external data into this layout

Benchmarks packaged elsewhere (CSV, JSONL, parquet, archives, APIs, annotation exports, etc.) are not consumed directly by the CLI. Convert them so each eval root has `corpus/` documents and a `train.json` that follows the schema above—map source fields to `question`, `answer`, and optionally `id` / `query_id` and `contexts`. Implement conversion in whichever way fits your pipeline (notebook, ad hoc script, or reusable importer). Prefer PDFs in `corpus/` when converting, including when sources are links or lack an extension. Keep `corpus/` filenames consistent with how your ingestor and citations surface `document_name` so retrieval and scoring align with runtime behavior.

## Running the benchmark

Export `NVIDIA_API_KEY` (and optionally `RAG_EVAL_JUDGE_MODEL` for the RAGAS judge LLM id), then from repo root use `uv run --project scripts/eval python scripts/eval/evaluate_rag.py` with `--dataset-paths`, `--host`, `--port`, and ingestor URL flags as needed. The script does not pass vector DB URL or embedding dimension—those come from ingestor/RAG server configuration (see [`references/evaluate-rag-cli.md`](references/evaluate-rag-cli.md)).

## Gotchas

- Wrong working directory: run from the repo root so `scripts/eval/evaluate_rag.py` runs with correct paths.
- `--ingestor_server_url`: use `http://host:port` without `/v1`, because the code appends `/v1/` automatically.
- Vector DB URL and embedding behavior are not set by this client; configure the deployed ingestor and RAG server (for example `APP_VECTORSTORE_URL` and embedding model settings).

## Pre-flight checklist

1. Each dataset root: `corpus/` + `train.json` (`corpus/` preferably PDF when converted from external sources, including URL-only or extension-less references).
2. `train.json`: top-level array of rows (dict-shaped `train.json` is rejected).
3. Rows include `question` and `answer` for meaningful RAGAS scores.
4. `NVIDIA_API_KEY` exported before invoking the script (optional `RAG_EVAL_JUDGE_MODEL` if not using the default judge id).

# Stage 4 — Run ingestion + evaluation (in the background)

Goal: run `evaluate_rag.py` for the chosen datasets, in agentic (or standard) mode, as a long-running
**background** process. It does both ingestion and evaluation in one pass.

## 4.1 Ask which datasets to run (QA)

Use `AskUserQuestion` (multi-select) to let the user pick. Default selection if they have no preference:
**`google_frames`, `financebench`, `kg_rag`, `hotpotqa`**. One command can take several datasets
(`--datasets` is space-separated), but each runs sequentially and a big dataset can take hours — consider
one process per dataset so a single failure doesn't block the rest, and so each can be timestamped
independently (Stage 5). Confirm only datasets already downloaded in Stage 3.

Also confirm the **mode**: agentic (default) or standard. This decides whether `--agentic` is on the
command (and whether `ENABLE_AGENTIC_RAG=true` was set in Stage 1).

## 4.2 Read the argparse — build the command from the script, not the example

Before constructing anything:

```bash
cd "$SCRIPTS_DIR" && python3 evaluate_rag.py --help
```

Key flags (from `evaluate_rag.py`):

| Flag | Meaning / value for this skill |
|------|--------------------------------|
| `--datasets` | **Plural**, space-separated list (e.g. `--datasets financebench kg_rag`). Names must be in `ALLOWED_DATASETS`. |
| `--host` / `--port` | rag-server reachable from where the script runs — usually `localhost` / `8081`. **Client-side.** |
| `--ingestor_server_url` | rag ingestor base URL, **no `/v1` suffix** (code appends it) — usually `http://localhost:8082`. **Client-side.** |
| `--rag_api_version` | `2` for the current RAG (v2 schema, metrics file, citations). Use `2`. |
| `--vdb_endpoint` | Vector DB endpoint **as the servers see it** (Docker-internal). Elasticsearch: `http://elasticsearch:9200`; Milvus: `http://milvus:19530`. **Server-side.** |
| `--llm_endpoint` | LLM endpoint **as the rag-server sees it** (e.g. `nim-llm:8000` on-prem). Omit to use the server's configured LLM. **Server-side.** |
| `--model` | LLM model name. Match the deployed model (e.g. `nvidia/nemotron-3-super-120b-a12b`), or omit to use the server default. |
| `--agentic` | **Bare flag.** Present ⇒ agentic run (`agentic:true` per request). Omit for standard RAG. |
| `--top_k` | Reranker top-k for generation (e.g. `10`). |
| `--output_dir` | Default `results`. Relative to CWD ⇒ lands in `scripts/results/`. |
| `--thread` | Concurrency for ingestion + generation (e.g. `16`–`32`). |
| `--embedding_dimension` | Match the embedding model (default `2048`). |
| `--force_ingestion` | Deletes + re-ingests the collection. **Destructive** — confirm with the user. |
| `--skip_ingestion` / `--skip_evaluation` | Reuse an already-ingested collection / ingest only. |
| `--chunk_size` / `--chunk_overlap` / `--batch_size` | Ingestion params; defaults usually fine. |

> **Endpoint gotcha (important):** `--host/--port` and `--ingestor_server_url` are called *by the eval
> client*, so use `localhost` when the script runs on the host. `--vdb_endpoint` and `--llm_endpoint` are
> forwarded into the request and resolved *by the rag-server/ingestor inside the Docker network*, so they
> use **container hostnames** (`elasticsearch`, `milvus`, `nim-llm`). Read the actual values from the
> active env file (`APP_VECTORSTORE_URL`, `APP_LLM_SERVERURL`, `APP_LLM_MODELNAME`) rather than guessing;
> mismatches cause empty contexts or connection errors.

## 4.3 Example command (agentic)

Adapt to what Stage 1 detected. Agentic financebench run on a host with Elasticsearch + on-prem LLM:

```bash
cd "$SCRIPTS_DIR"
source .venv/bin/activate
export NVIDIA_API_KEY="$NVIDIA_API_KEY"

PYTHONUNBUFFERED=1 python3 evaluate_rag.py \
  --datasets financebench \
  --host localhost --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint nim-llm:8000 \
  --model nvidia/nemotron-3-super-120b-a12b \
  --top_k 10 \
  --thread 16 \
  --output_dir results \
  --agentic
```

For **NVIDIA-hosted (cloud)**: omit `--llm_endpoint` (and often `--model`) to use the server's configured
hosted LLM; `--vdb_endpoint` still points at the deployed vector DB container.

For **standard RAG**: drop `--agentic` (and ensure `ENABLE_AGENTIC_RAG` is not forcing agentic on the
server). Everything else is identical — this is how the skill serves standard RAG.

## 4.4 Launch in the background

This is a 1–5 hour process. Launch it with the Bash tool using **`run_in_background: true`** and capture a
log. Then **stop** — do not sleep/poll on a timer. The harness re-invokes you when the process exits; only
then proceed to Stage 5.

Pattern (the tool's background mode keeps it running across turns; the `tee`'d log lets you triage later):

```bash
cd "$SCRIPTS_DIR" && source .venv/bin/activate && export NVIDIA_API_KEY="$NVIDIA_API_KEY" && \
PYTHONUNBUFFERED=1 python3 evaluate_rag.py <flags from 4.3> 2>&1 | tee "eval_<dataset>_run.log"
```

Guidance:
- One background invocation per dataset (or per command) so failures and timestamps stay isolated.
- If you must set a fallback wakeup, make it long (≥1200 s) — the exit notification is the real signal.
- Note the log path for each run so Stage 5 can read errors if it fails.

## 4.5 What "done" looks like

On a successful exit you will see the `RAG Evaluation` banner, ingestion metrics, `EVALUATION RESULTS`
(End-2-End Accuracy, context_relevance, response_groundedness, recall, token usage), and
`Evaluation complete. Results stored in directory: results`. Proceed to Stage 5 to verify the files on
disk regardless of what the console said.

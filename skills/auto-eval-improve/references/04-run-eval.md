# Stage 4 — Run ingestion + evaluation (in the background)

Goal: run `evaluate_rag.py` for the chosen datasets, in agentic (or standard) mode, as a long-running
**background** process. It does both ingestion and evaluation in one pass.

## 4.0 Create the experiment folder (do this first, once per invocation)

Every artifact for this campaign — logs, result snapshots, intermediate state, the final report — lives
under a single **experiment folder** inside `$SCRIPTS_DIR/experiments/`. One experiment = one invocation
of this skill (baseline + up to 3 improve cycles, across all datasets you run this turn). Create it once,
before launching any eval, and reuse `$EXP_DIR` for every later stage.

```bash
cd "$SCRIPTS_DIR"
MODE=agentic                                    # agentic | standard (the mode confirmed in 4.1)
EXP_TS=$(date +%Y%m%d_%H%M%S)
EXP_DIR="experiments/exp_${EXP_TS}_${MODE}"
mkdir -p "$EXP_DIR"
echo "Experiment: $EXP_DIR"
```

Seed `state.md` — the experiment's **manifest + live progress checkpoint** (replaces the old ad-hoc
`AUTO_EVAL_STATE.md`). Fill the header with the real, detected values (mode, datasets, deployment
endpoints / model / key `AGENTIC_*` env from the active env file). Stage 5.5 appends one row per cycle;
treat it as the resume/checkpoint doc:

```markdown
# Experiment exp_<ts>_<mode>

- **Mode:** agentic | standard
- **Datasets:** financebench, kg_rag, …
- **rag-server / ingestor:** localhost:8081 / http://localhost:8082
- **vdb / llm endpoint / model:** http://elasticsearch:9200 / nim-llm:8000 / nvidia/nemotron-3-super-120b-a12b
- **Key env:** ENABLE_AGENTIC_RAG=true, AGENTIC_VERIFICATION_ENABLED=false, …
- **Started:** <ts>

## Progress
| Dataset | Cycle | Snapshot | Change applied | e2e | ctx_rel | grounded | recall@1 | p50_latency_s | ttft_s | Verdict |
|---------|-------|----------|----------------|-----|---------|----------|----------|---------------|--------|---------|

Record **latency alongside accuracy** every cycle (p50 end-to-end + mean/p50 TTFT, from the `latency` block
in `rag_<dataset>_evaluation_metrics.json`). The verdict must weigh both: an accuracy-neutral change that
cuts p50 latency is a *win*; an accuracy gain that costs latency is a *trade-off* to call out explicitly.

## Observations (fill in during Stage 6.3 — the evidence base for cross-dataset recommendations)

Record findings **per dataset** so code-change recommendations can be judged across the WHOLE experiment,
not from one dataset in isolation. A code/flow change affects every dataset, so it must be justified by
evidence spanning all of them (see reference 06, 6.4 and D.2). Keep adding rows as you analyze each run.

### Per-dataset failure analysis
| Dataset | n queries | n hard-fails (e2e<0.5) | Dominant failure mode(s) | Example failing indices | Notes |
|---------|-----------|------------------------|--------------------------|-------------------------|-------|

### Cross-dataset failure-mode rollup (which modes recur across datasets)
| Failure mode | Datasets affected | Approx. share of fails | Where in pipeline (retrieval / planner / task-answer / synthesis / eval) |
|--------------|-------------------|------------------------|--------------------------------------------------------------------------|

### Candidate code-change hypotheses (each must be assessed against ALL datasets before recommending)
| Hypothesis (what + where in `rag/`) | Targets which failure mode | Datasets it should HELP | Datasets it could HURT / risk | Cross-dataset verdict |
|-------------------------------------|----------------------------|-------------------------|-------------------------------|-----------------------|
```

The resulting tree (filled in by Stages 4–6):

```
experiments/exp_<ts>_<mode>/
├── state.md                       # manifest + live cycle log (this file)
├── report.md                      # final consolidated report (Stage 6.8)
└── <dataset>/                     # financebench/, kg_rag/, …
    ├── baseline_<TS>/             # baseline | cycle1 | cycle2 | cycle3
    │   ├── rag_<dataset>_evaluation_{metrics,results,data,summary}.json
    │   ├── failure.txt            # only if >50% queries failed
    │   ├── eval.log               # this run's tee'd log
    │   └── REPRODUCE.md           # self-contained rerun steps (Stage 5.4)
    └── cycle1_<TS>/ …
```

> `--output_dir` stays at its default `results` — the script writes `results/<dataset>/` as **transient
> scratch**, and each stage copies that snapshot into `$EXP_DIR/<dataset>/<cycle>_<TS>/` (Stage 5.3). Do
> not point `--output_dir` at the experiment folder (the script would append a second `/<dataset>`).

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
| `--llm_endpoint` | LLM endpoint **as the rag-server sees it**. Accepts a **full URL** for cloud (e.g. `https://inference-api.nvidia.com/v1`) **or** a `host:port` for on-prem (e.g. `nim-llm:8000`). **Always pass it**, matching the deployment's `APP_LLM_SERVERURL`. **Server-side.** |
| `--model` | LLM model name. **Always pass it**, matching the deployment's `APP_LLM_MODELNAME` (e.g. `nvidia/nvidia/nemotron-3-super-v3`). **Do NOT omit it** — there is no "use server default": argparse defaults `--model` to `nvdev/meta/llama-3.3-70b-instruct` and sends that in every request, silently overriding the deployed model. |
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
> forwarded into the request and resolved *by the rag-server* — so `--vdb_endpoint` uses the **container
> hostname** (`elasticsearch`, `milvus`), while `--llm_endpoint` is whatever the rag-server uses to reach
> the LLM: a **container `host:port`** for on-prem NIM (`nim-llm:8000`) or a **full external URL** for
> cloud (`https://inference-api.nvidia.com/v1`). Read the actual values from the active env file
> (`APP_VECTORSTORE_URL`, `APP_LLM_SERVERURL`, `APP_LLM_MODELNAME`) rather than guessing; mismatches cause
> empty contexts or connection errors.

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

**Always pass `--model` and `--llm_endpoint`** matching the deployment's `APP_LLM_MODELNAME` /
`APP_LLM_SERVERURL` — never rely on omission (see the flag table). `--llm_endpoint` takes a full URL for
cloud or a `host:port` for on-prem:

```bash
# NVIDIA-hosted / cloud LLM (full URL):
python3 evaluate_rag.py --port 8081 --host localhost --datasets kg_rag --top_k 10 --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint https://inference-api.nvidia.com/v1 --model nvidia/nvidia/nemotron-3-super-v3

# On-prem NIM LLM (host:port):
python3 evaluate_rag.py --port 8081 --host localhost --datasets kg_rag --top_k 10 --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint nim-llm:8000 --model nvidia/nvidia/nemotron-3-super-v3
```

For **standard RAG**: drop `--agentic` (and ensure `ENABLE_AGENTIC_RAG` is not forcing agentic on the
server). Everything else — including passing `--model` and `--llm_endpoint` — is identical.

## 4.4 Launch in the background

This is a 1–5 hour process. Launch it with the Bash tool using **`run_in_background: true`** and capture a
log. Then **stop** — do not sleep/poll on a timer. The harness re-invokes you when the process exits; only
then proceed to Stage 5.

Compute the cycle's snapshot folder **first** and `tee` the log straight into it, so the log lands next to
the results it describes (no more `eval_<dataset>_*.log` scattered at the scripts root). `CYCLE` is
`baseline` for the first run of a dataset, then `cycle1`/`cycle2`/`cycle3` for the Stage 6.7 improve cycles.

```bash
cd "$SCRIPTS_DIR" && source .venv/bin/activate && export NVIDIA_API_KEY="$NVIDIA_API_KEY"
DS=financebench; CYCLE=baseline
TS=$(date +%Y%m%d_%H%M%S)
SNAP="$EXP_DIR/$DS/${CYCLE}_${TS}"; mkdir -p "$SNAP"
PYTHONUNBUFFERED=1 python3 evaluate_rag.py <flags from 4.3> --output_dir results 2>&1 | tee "$SNAP/eval.log"
```

`--output_dir results` writes the script's transient scratch to `results/$DS/`; Stage 5.3 copies the JSONs
into `$SNAP` alongside this log. Note the `$SNAP` path so Stage 5 can read the log if the run fails.

Guidance:
- One background invocation per dataset (or per command) so failures and timestamps stay isolated.
- **Run datasets SEQUENTIALLY, never two evals at once.** Do **not** launch a second dataset's eval while
  another is still running — both hit the same rag-server/ingestor, so concurrent runs multiply the load
  (`--thread` × N), skew metrics, and cause timeouts / transient errors that corrupt results. Wait for the
  current run to finish (and complete Stage 5) before starting the next dataset. One eval process at a time.
- If you must set a fallback wakeup, make it long (≥1200 s) — the exit notification is the real signal.
- Note the `$SNAP/eval.log` path for each run so Stage 5 can read errors if it fails.

## 4.5 What "done" looks like

On a successful exit you will see the `RAG Evaluation` banner, ingestion metrics, `EVALUATION RESULTS`
(End-2-End Accuracy, context_relevance, response_groundedness, recall, token usage, and a **Latency (per
query)** block — samples, mean/p50/p90/p99/min/max total seconds, mean/p90 TTFT), and
`Evaluation complete. Results stored in directory: results`. Proceed to Stage 5 to verify the files on
disk regardless of what the console said.

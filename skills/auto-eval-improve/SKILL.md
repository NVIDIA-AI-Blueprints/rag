---
name: auto-eval-improve
version: "0.1.0"
description: >-
  Automated accuracy + per-query latency eval + improvement loop for Agentic/Standard RAG
  (blueprint-pipeline harness). Tracks RAGAS quality and end-to-end/TTFT latency; not load/throughput
  testing (rag-perf) or scripts/eval (rag-eval).
license: Apache-2.0
compatibility: >-
  NVIDIA RAG Blueprint repo checkout (this repo) for deployment + the blueprint-pipeline repo for the
  eval harness (evaluation/rag-eval/scripts/evaluate_rag.py). Python 3.9–3.12 + uv for the eval env;
  Docker/Compose or Helm for serving; NGC CLI with nv-rag-blueprint org access for datasets.
  Requires NVIDIA_API_KEY (RAGAS judge) and NGC_API_KEY (datasets + on-prem model pulls).
metadata:
  author: "NVIDIA RAG <foundational-rag-dev@exchange.nvidia.com>"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/rag"
  argument-hint: auto eval | agentic eval | evaluate_rag | improve agentic flow | compare results | google_frames financebench kg_rag hotpotqa
  tags:
    - nvidia
    - blueprint
    - rag
    - agentic-rag
    - evaluation
    - ragas
    - benchmarking
  languages:
    - python
    - shell
  frameworks:
    - ragas
    - fastapi
    - langgraph
  domain: ai-ml
allowed-tools: Read Grep Glob Write Edit Agent Bash(echo *) Bash(ls *) Bash(cat *) Bash(pwd *) Bash(git *) Bash(uv *) Bash(python3 *) Bash(pip *) Bash(wget *) Bash(unzip *) Bash(rm *) Bash(mv *) Bash(mkdir *) Bash(find *) Bash(ngc *) Bash(nvidia-smi *) Bash(curl *) Bash(docker *) Bash(kubectl *) Bash(helm *) Bash(jq *) Bash(grep *) Bash(df *) Bash(du *)
---

# Auto-evaluate and improve the (Agentic) RAG workflow

## Purpose

Drive the full automated-evaluation loop for the NVIDIA RAG Blueprint and turn the results into concrete
improvement recommendations. The skill is built for **Agentic RAG** (LangGraph plan-and-execute pipeline)
but transparently supports **Standard RAG** — the only difference is whether the agentic path is enabled.

It optimizes for **two axes at once**: answer/retrieval **quality** (RAGAS) **and per-query latency**
(end-to-end + time-to-first-token), both emitted by the eval harness. Treat them together — a change that
holds accuracy but cuts latency is a win, and a change that buys accuracy must be weighed against its
latency cost (see reference 06, §6.2/§6.4).

It orchestrates six stages in two groups. **Read a stage's reference file only when you actually run that
stage.**

### Part A — Setup (stages 1–3): one-time prerequisites. **Skip any already satisfied.**

These provision the environment. They are **not** re-run on every invocation — detect what already exists
and skip it (see the skip check below). Skip all three when the user says setup is done.

1. **Clone & deploy** — optionally clone this repo (`develop`); deploy rag-server + ingestor + vector DB
   via the **`rag-blueprint`** skill, picking cloud vs on-prem by server capacity, with
   `ENABLE_AGENTIC_RAG=true` for agentic runs. → [`references/01-clone-and-deploy.md`](references/01-clone-and-deploy.md)
2. **Eval repo & env** — locate or clone the `blueprint-pipeline` repo (`develop`) and build the `uv`
   env from `evaluation/rag-eval/scripts/pyproject.toml`. → [`references/02-eval-repo-and-env.md`](references/02-eval-repo-and-env.md)
3. **NGC & datasets** — install NGC CLI if missing, configure it, download + unzip the latest dataset
   versions into `scripts/datasets/`. → [`references/03-ngc-and-datasets.md`](references/03-ngc-and-datasets.md)

**Skip check (run first; skip the stage when its check passes):**

| Stage | Already done if… | Then skip to |
|-------|------------------|--------------|
| 1 | `docker ps` shows healthy `rag-server` + `ingestor-server` and `curl localhost:8081/v1/health` = 200 | stage 2 |
| 2 | `<eval-repo>/evaluation/rag-eval/scripts/.venv` exists and `evaluate_rag.py --help` runs | stage 3 |
| 3 | `scripts/datasets/<dataset>/` exists for the target dataset(s) | stage 4 |

If the user says *"deploy / env / datasets are already done, start from ingestion/eval,"* skip directly to
**stage 4**.

### Part B — Eval & improve loop (stages 4–6): the actual work. **Always run.**

4. **Run eval** — create the per-invocation **experiment folder** (`scripts/experiments/exp_<ts>_<mode>/`),
   ask which datasets to run, build the correct `evaluate_rag.py` command **from the script's argparse**,
   and launch it in the **background** (1–5 h). → [`references/04-run-eval.md`](references/04-run-eval.md)
5. **Results & retry** — verify outputs in the scratch `scripts/results/`, do one diagnosed retry on
   failure (else escalate), and snapshot successful runs into the experiment folder. → [`references/05-results-and-retry.md`](references/05-results-and-retry.md)
6. **Analyze & recommend** — compare runs on **both quality and latency**, find low-accuracy / wrong /
   slow answers, recommend fixes (favoring the accuracy↔latency trade-off — when accuracy ties, prefer the
   faster/cheaper config), then **execute the top fix and auto re-run, iterating up to 3 cycles** (agentic
   or standard flow; eval script only cautiously). Analyze like an agentic-RAG expert and think **beyond
   prompt tweaks** —
   architectural changes (per-stage budgets/top_k, model-per-role, new stages/params) are in scope; see
   [`references/06a-recommendation-ideation.md`](references/06a-recommendation-ideation.md). **Write all
   reports in plain, simple language**, but make recommendations **implementation-ready** (exact change +
   apply/verify/rollback; see the *Writing style* and *Recommendation detail standard* in reference 06).
   → [`references/06-analyze-and-recommend.md`](references/06-analyze-and-recommend.md)

## When NOT to use

- **Throughput / concurrency / load testing** (max QPS, scaling, saturation) → use the **`rag-perf`**
  skill. *This* skill does measure **per-query latency** (end-to-end + TTFT) at the eval's own light load
  — use it for "is this config slower per request"; use `rag-perf` for "how does it behave under load."
- **The in-repo `scripts/eval/evaluate_rag.py` harness** (different script, different flags) → use the
  **`rag-eval`** skill. This skill drives the *blueprint-pipeline* `evaluate_rag.py` harness.
- **Deploy / configure / troubleshoot / shutdown only** (no eval) → use the **`rag-blueprint`** skill.

## Limitations

- **Quality + per-query-latency scope.** Measures answer/retrieval quality (RAGAS: e2e_accuracy,
  context_relevance, response_groundedness, recall) **and per-query latency** (end-to-end +
  time-to-first-token, with p50/p90/p99) emitted by the harness into the metrics/results/data JSONs. It does
  **not** do throughput/concurrency/load testing (→ **`rag-perf`**); not the in-repo `scripts/eval` harness
  (→ **`rag-eval`**) — see "When NOT to use".
- **Latency is measured under the eval's own light load on a shared LLM endpoint**, so absolute wall-clock
  drifts run-to-run (especially against cloud endpoints). Prefer **p50** over mean, and compare configs
  **within the same session/back-to-back**; treat small cross-day deltas as noise. Token usage may be `0`
  for agentic runs (harness limitation) — latency is the more reliable cost signal there.
- **Long-running.** Each eval takes ~1–5 h and must run in the **background**; do not poll on a timer
  (the harness re-invokes you on exit).
- **Sequential only.** Never run two evals concurrently against the same server — they share
  rag-server/ingestor, so concurrent load skews metrics and causes timeouts (Instructions rule 2 / ref 04).
- **Improve loop capped at 3 cycles** per dataset, and stops earlier on a regression or plateau of the
  primary metric (ref 06 §6.7).
- **Source-code changes are excluded from the auto-loop.** The loop applies only env/config and eval-CLI
  changes; rag-server/eval-script code changes require explicit human approval (and an image rebuild for
  most `.py` files) — ref 06 §6.7/§6.8.
- **Keys required.** `NVIDIA_API_KEY` (RAGAS judge, fails at startup if unset) and `NGC_API_KEY` with
  `nv-rag-blueprint` org access (dataset download + on-prem model pulls) — see Prerequisites.
- **Reranker `--top_k` is capped server-side** — values above ~25 return HTTP 422.

## Prerequisites

- This RAG Blueprint repo checked out (you are in it). For deployment, the **`rag-blueprint`** skill is used.
- The **blueprint-pipeline** repo, **a sibling of this `rag` repo** (i.e. checked out next to it in the
  same parent folder, default `../blueprint-pipeline`), branch `develop`. The path is configurable; if
  missing it is cloned — see reference 02.
- `uv` and Python 3.9–3.12 for the eval env.
- **`NVIDIA_API_KEY`** — required by `evaluate_rag.py` at startup (RAGAS judge). **`NGC_API_KEY`** with
  **`nv-rag-blueprint`** org access — required for dataset download (and on-prem model pulls).
- Reachable **rag-server** (default `localhost:8081`) and **ingestor** (default `localhost:8082`).

## Autonomy principles

Inherit the `rag-blueprint` philosophy: **auto-detect everything you can with a command** (GPUs via
`nvidia-smi`, running services via `docker ps`/`kubectl`, ports, repo state, keys present in env, existing
datasets and prior results). **Ask the user only when** an action is theirs to take or a choice is genuinely
open: which datasets to evaluate, whether to (re)deploy, supplying a missing API key, or confirming a
destructive `--force_ingestion` / collection delete. Never invent values — read them from config files.

## Instructions

Run the stages in order (the execution flow), but **skip any Part-A setup stage (1–3) that is already satisfied** (use the skip
check above) — only **Part B (4–6)** runs every time. Read a reference file *before* acting on that stage;
they contain the exact commands, gotchas, and the argparse-derived command builder. Track progress with the
task tools for a long run. Key cross-cutting rules:

1. **Agentic vs standard is a single switch.** Agentic = `ENABLE_AGENTIC_RAG=true` on the server **and**
   `--agentic` on the eval command. Standard = neither. The per-request `--agentic` flag overrides the
   server default, so it must be present for an agentic run regardless of the env var; set both for an
   unambiguous run. Confirm with the user which mode they want (default: **agentic**).
2. **Run the eval in the background** (`run_in_background: true`). It is long-running (~1–5 h). Do **not**
   poll on a timer — the harness re-invokes you when the process exits. **With multiple datasets, run them
   one at a time (sequentially) — never launch two evals concurrently against the same server** (they share
   rag-server/ingestor; concurrent load skews metrics and causes timeouts). See reference 04.
3. **Always derive the command from the live argparse**, not from any example. The script flag is
   `--datasets` (plural, space-separated list) and `--agentic` is a bare flag. Verify with
   `python3 evaluate_rag.py --help` before constructing the command.
4. **Run the script from `evaluation/rag-eval/scripts/`** — `DATASET_BASE_DIR=./datasets/` and the default
   `--output_dir results` are relative to the CWD.
5. **`--vdb_endpoint` and `--llm_endpoint` are resolved server-side** (by rag-server/ingestor inside the
   Docker network), so they use container hostnames (e.g. `http://elasticsearch:9200`, `nim-llm:8000`).
   **`--host`/`--port` and `--ingestor_server_url` are used by the eval client**, so they must be reachable
   from where the script runs (typically `localhost`). See reference 04 — this distinction matters.

## Examples

Typical ways this skill is invoked (it runs end-to-end and asks only for genuinely open choices):

- **"Run an agentic eval on financebench and kg_rag, then recommend improvements."** — full loop:
  deploy-check → eval (agentic) → analyze → execute the top fix → auto re-run (≤3 cycles).
- **"auto eval — datasets google_frames hotpotqa, standard RAG."** — standard-RAG run (drops `--agentic`).
- **"Deploy/env/datasets are already done, start from ingestion and evaluate financebench (agentic)."** —
  skips Part A (stages 1–3) and starts at stage 4.
- **"Compare my last two eval runs for kg_rag and tell me what regressed."** — analysis only (stage 6
  against existing snapshots).
- **"Try `AGENTIC_VERIFICATION_ENABLED=true` on kg_rag and show the delta."** — a focused A/B improve cycle.

## Quick reference — outputs

`evaluate_rag.py` writes per-dataset results to `scripts/<output_dir>/<dataset>/` (default `output_dir`
`results`). **Keep `--output_dir results` — it is transient scratch, overwritten every run.** The skill
then **copies** each run's results into the experiment folder, which is the durable, comparable artifact:

```
scripts/
├── results/<dataset>/                         # transient scratch the script writes; copied out per run
└── experiments/exp_<ts>_<mode>/               # ONE per invocation (mode = agentic | standard); Stage 4.0
    ├── state.md                               # manifest + live cycle-by-cycle progress (was AUTO_EVAL_STATE.md)
    ├── report.md                              # final consolidated report, Stage 6.8 (was AUTO_EVAL_REPORT.md)
    └── <dataset>/                             # financebench/, kg_rag/, …
        └── <cycle>_<TS>/                      # baseline_<ts>, cycle1_<ts>, … — one per run
            ├── rag_<dataset>_evaluation_*.json (4 files, see below)
            ├── failure.txt                    # only when >50% of queries failed — treat as a failed run
            ├── eval.log                       # this run's tee'd log (was eval_<dataset>_*.log at root)
            └── REPRODUCE.md                   # self-contained rerun steps (Stage 5.4)
```

The four JSONs the script writes into each `<dataset>/` (scratch → copied into the snapshot):

| File | Contents |
|------|----------|
| `rag_<dataset>_evaluation_metrics.json` | v2 summary: `evaluation_metrics` (e2e_accuracy, context_relevance, response_groundedness), `ingestion_metrics_list`, `token_usage`, **`latency`** (per-query KPI: `sample_count`, `mean/p50/p90/p99/min/max_total_seconds`, `ttft_sample_count`, `mean/p50/p90/p99_ttft_seconds`) |
| `rag_<dataset>_evaluation_results.json` | Per-metric lists incl. `recall_metrics` (page/document level), `citations`, `token_usage`, **`latency`** (same aggregate) + **`latency_per_sample`** (per-query list) |
| `rag_<dataset>_evaluation_data.json` | Per-query rows: `question`, `answer`, `generated_answer`, `generated_contexts`, `retrieved_docs`, `usage`, **`latency`** (`{total_seconds, ttft_seconds}` for that query) |
| `rag_<dataset>_evaluation_summary.json` | Headline means |
| `failure.txt` | Present only when >50% of queries failed — treat as a failed run |

> **Latency is real per-query timing** the eval *client* records around each `/generate` call: `total_seconds`
> = request start → last chunk (end-to-end); `ttft_seconds` = request start → first content token. The console
> "EVALUATION RESULTS" block also prints these (mean / p50 / p90 / p99 / min / max total + mean/p90 TTFT).

> **Legacy artifacts:** earlier runs left flat artifacts at the scripts root (`results/<dataset>_*`
> snapshots, `eval_*.log`, `AUTO_EVAL_STATE.md`, `AUTO_EVAL_REPORT.md`). The experiment-folder convention
> governs **future** runs only — leave existing flat artifacts as-is (or archive them); do not auto-delete.

## Troubleshooting

A failed run is one that exits non-zero, writes `failure.txt` (>50% of queries failed), produces a
missing/empty metrics file, or reports `e2e_accuracy_mean` `0.0` with empty `generated_answer`s. Diagnose
from the run log (`$SNAP/eval.log`) before retrying — the full signal→cause→fix table is in
**reference 05 §5.2** (and §5.1 covers output validation). Most common cases:

| Error / symptom | Cause | Solution |
|-----------------|-------|----------|
| Immediate exit naming `NVIDIA_API_KEY` | Key unset in the process env | Export `NVIDIA_API_KEY`; relaunch. |
| `Failed to get response from rag-server` / connection refused | rag-server down or wrong `--host`/`--port` | Re-check stage 1 health; fix host/port (client-side, use `localhost`). |
| Ingestor 404 on upload | `--ingestor_server_url` included a `/v1` suffix | Pass the base URL only (`http://host:port`). |
| Empty `generated_contexts` everywhere | Retrieval gap: wrong collection / `--vdb_endpoint` / low `top_k` | Verify the collection exists; set `--vdb_endpoint` to the container hostname; raise `--top_k` (≤25). |
| `failure.txt` present / >50% failed | Flaky LLM endpoint, wrong `--model`, or stale collection | Match `--model`/`--llm_endpoint` to the deployment; lower `--thread`; `--force_ingestion` or a fresh `--collection` (confirm — destructive). |

Apply the fix and retry **once** (background, per stage 4). If it still fails, stop and escalate with the
failing command, the log excerpt, your diagnosis, and what you tried (ref 05 §5.2).

## Source of truth

| Piece | Location |
|-------|----------|
| Eval driver | `blueprint-pipeline/evaluation/rag-eval/scripts/evaluate_rag.py` |
| Eval deps | `…/scripts/pyproject.toml` (poetry-style; build env with `uv`) |
| Eval package | `…/scripts/rag_evaluator/` (must be `pip install -e .`) |
| Real-world invocation reference | `…/scripts/run.sh` (CLI-flag pattern + NGC install) |
| Eval README | `…/scripts/README.md` |
| Deploy / configure / troubleshoot | **`rag-blueprint`** skill + `docs/agentic-rag.md` |
| Agentic config knobs | `skills/rag-blueprint/references/configure/agentic-rag.md`, `src/nvidia_rag/utils/agentic_rag_config.py` |

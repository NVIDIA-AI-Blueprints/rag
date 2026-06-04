---
name: auto-eval-improve
version: "0.1.0"
description: >-
  Run an end-to-end automated RAG accuracy evaluation and recommend improvements. Primarily for
  Agentic RAG (ENABLE_AGENTIC_RAG / per-request agentic=true) but works for Standard RAG too.
  Deploys/uses rag-server + ingestor, sets up the blueprint-pipeline rag-eval harness
  (evaluate_rag.py), downloads NGC datasets, runs ingestion+evaluation in the background, timestamps
  results, compares against the previous run, and proposes fixes to the agentic (or standard) flow.
  Not for latency/throughput benchmarking (use rag-perf) or the in-repo scripts/eval harness (use rag-eval).
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

It orchestrates six stages, each documented in its own reference file so the skill stays maintainable:

1. **Clone & deploy** — optionally clone this repo (`develop`); deploy rag-server + ingestor + vector DB
   via the **`rag-blueprint`** skill, picking cloud vs on-prem by server capacity, with
   `ENABLE_AGENTIC_RAG=true` for agentic runs. → [`references/01-clone-and-deploy.md`](references/01-clone-and-deploy.md)
2. **Eval repo & env** — locate or clone the `blueprint-pipeline` repo (`develop`) and build the `uv`
   env from `evaluation/rag-eval/scripts/pyproject.toml`. → [`references/02-eval-repo-and-env.md`](references/02-eval-repo-and-env.md)
3. **NGC & datasets** — install NGC CLI if missing, configure it, download + unzip the latest dataset
   versions into `scripts/datasets/`. → [`references/03-ngc-and-datasets.md`](references/03-ngc-and-datasets.md)
4. **Run eval** — ask which datasets to run, build the correct `evaluate_rag.py` command **from the
   script's argparse**, and launch it in the **background** (1–5 h). → [`references/04-run-eval.md`](references/04-run-eval.md)
5. **Results & retry** — verify outputs in `scripts/results/`, do one diagnosed retry on failure (else
   escalate), and timestamp successful runs. → [`references/05-results-and-retry.md`](references/05-results-and-retry.md)
6. **Analyze & recommend** — compare current vs previous-latest run, find low-accuracy / wrong answers,
   and recommend fixes in the agentic flow (or standard flow), and cautiously in the eval script. →
   [`references/06-analyze-and-recommend.md`](references/06-analyze-and-recommend.md)

## When NOT to use

- **Latency / throughput / load testing** → use the **`rag-perf`** skill.
- **The in-repo `scripts/eval/evaluate_rag.py` harness** (different script, different flags) → use the
  **`rag-eval`** skill. This skill drives the *blueprint-pipeline* `evaluate_rag.py` harness.
- **Deploy / configure / troubleshoot / shutdown only** (no eval) → use the **`rag-blueprint`** skill.

## Prerequisites

- This RAG Blueprint repo checked out (you are in it). For deployment, the **`rag-blueprint`** skill is used.
- The **blueprint-pipeline** repo (default path `/home/smasurekar/Desktop/Swapnil/gitlab_repos/blueprint-pipeline`,
  branch `develop`). The path is configurable; if missing it is cloned — see reference 02.
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

## Execution flow

Work through the stages in order. Read each reference file *before* acting on that stage; they contain the
exact commands, gotchas, and the argparse-derived command builder. Track progress with the task tools for a
long run. Key cross-cutting rules:

1. **Agentic vs standard is a single switch.** Agentic = `ENABLE_AGENTIC_RAG=true` on the server **and**
   `--agentic` on the eval command. Standard = neither. The per-request `--agentic` flag overrides the
   server default, so it must be present for an agentic run regardless of the env var; set both for an
   unambiguous run. Confirm with the user which mode they want (default: **agentic**).
2. **Run the eval in the background** (`run_in_background: true`). It is long-running (~1–5 h). Do **not**
   poll on a timer — the harness re-invokes you when the process exits. See reference 04.
3. **Always derive the command from the live argparse**, not from any example. The script flag is
   `--datasets` (plural, space-separated list) and `--agentic` is a bare flag. Verify with
   `python3 evaluate_rag.py --help` before constructing the command.
4. **Run the script from `evaluation/rag-eval/scripts/`** — `DATASET_BASE_DIR=./datasets/` and the default
   `--output_dir results` are relative to the CWD.
5. **`--vdb_endpoint` and `--llm_endpoint` are resolved server-side** (by rag-server/ingestor inside the
   Docker network), so they use container hostnames (e.g. `http://elasticsearch:9200`, `nim-llm:8000`).
   **`--host`/`--port` and `--ingestor_server_url` are used by the eval client**, so they must be reachable
   from where the script runs (typically `localhost`). See reference 04 — this distinction matters.

## Quick reference — outputs

For each dataset, `evaluate_rag.py` writes to `scripts/<output_dir>/<dataset>/`:

| File | Contents |
|------|----------|
| `rag_<dataset>_evaluation_metrics.json` | v2 summary: `evaluation_metrics` (e2e_accuracy, context_relevance, response_groundedness), `ingestion_metrics_list`, `token_usage` |
| `rag_<dataset>_evaluation_results.json` | Per-metric lists incl. `recall_metrics` (page/document level), `citations`, `token_usage` |
| `rag_<dataset>_evaluation_data.json` | Per-query rows: `question`, `answer`, `generated_answer`, `generated_contexts`, `retrieved_docs`, `usage` |
| `rag_<dataset>_evaluation_summary.json` | Headline means |
| `failure.txt` | Present only when >50% of queries failed — treat as a failed run |

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

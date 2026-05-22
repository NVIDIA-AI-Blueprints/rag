# BENCHMARK.md — `rag-perf`

This document summarizes how the `rag-perf` skill is evaluated. Format follows the [Skills Publishing Onboarding Guide](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/) (Step 2 — *Reporting: BENCHMARK.md*).

## What this skill is graded on

`rag-perf` routes the agent to the `scripts/rag-perf` tool for latency/throughput benchmarking of a deployed RAG server. Evaluation checks whether the agent loads the right SKILL.md, surfaces the YAML-config + aiperf workflow, names the right perf metrics (TTFT, throughput, concurrency, stage breakdown), and provides actionable triage advice for a high-TTFT scenario.

## Harness

| Item | Value |
|------|-------|
| Eval framework | [Harbor](https://github.com/harbor-eval/harbor) (Tier 3) via [`skill-eval/`](../../skill-eval/) |
| Adapter | `skill-eval/adapters/rag-blueprint/generate.py` (shared, invoked with `--skill-name rag-perf`) |
| Coordinator | `.github/workflows/skills-eval.yml` (PR, nightly cron, manual dispatch) |
| Runner | self-hosted `rag-skill-validator` (cpu profile — GCP `n2d-standard-4`) |
| Agent under test | `claude-code` |
| Agent model | `claude-sonnet-4-6` |
| Judge model | `aws/anthropic/claude-haiku-4-5-v1` |
| Parallelism | up to 4 concurrent checks per task |

## Eval specs

| Spec | Platform | Tasks |
|------|----------|-------|
| [`eval/nvidia_hosted.json`](eval/nvidia_hosted.json) | cpu | 2 |

Tasks exercise:
1. Explaining how to run a performance benchmark via the YAML-driven `rag-perf` command (config shape, metric set named).
2. Triaging high TTFT under load — identifying whether the bottleneck is LLM NIM, embedding NIM, or retrieval using the stage breakdown table.

## Metrics

Each task scores the agent's trajectory against a list of natural-language checks via `skill-eval/verifiers/generic_judge.py`. Reward is mean per-check pass rate.

The five NV-ACES rollup dimensions are not yet computed — see [`skills/PUBLISHING_COMPLIANCE.md`](../PUBLISHING_COMPLIANCE.md).

## Current results

| Spec | Task | With skill | Without skill (baseline) |
|------|------|------------|--------------------------|
| `nvidia_hosted.json` | Perf-benchmark explanation | _populated by Harbor_ | _TODO — pending NV-ACES integration_ |
| `nvidia_hosted.json` | High-TTFT bottleneck triage | _populated by Harbor_ | _TODO — pending NV-ACES integration_ |

## How to reproduce a run locally

```bash
cd skill-eval
python3 adapters/rag-blueprint/generate.py \
  --output-dir datasets/rag-perf-nvidia-hosted \
  --skill-dir ../skills/rag-perf \
  --skill-name rag-perf \
  --spec ../skills/rag-perf/eval/nvidia_hosted.json

uvx harbor run \
  -p datasets/rag-perf-nvidia-hosted/step-1 datasets/rag-perf-nvidia-hosted/step-2 \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model claude-sonnet-4-6 \
  -o jobs -n 1 --yes
```

Required env: `JUDGE_ANTHROPIC_API_KEY`. For synthetic queries, an OpenAI-compatible chat-completions endpoint must be reachable (default `http://localhost:8999/v1/chat/completions`).

## Drift handling

Skill semver pinned to RAG software semver via `scripts/validate_skill_versions.py`. The aiperf endpoint plugin (`scripts/rag-perf` editable install) is exercised by the bundled `nvidia_rag` aiperf endpoint — drift between SKILL.md and the actual `rag-perf` CLI surface will surface in the unit test under `tests/unit/test_skills/`.

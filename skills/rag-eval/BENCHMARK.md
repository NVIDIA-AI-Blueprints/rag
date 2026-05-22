# BENCHMARK.md — `rag-eval`

This document summarizes how the `rag-eval` skill is evaluated. Format follows the [Skills Publishing Onboarding Guide](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/) (Step 2 — *Reporting: BENCHMARK.md*).

## What this skill is graded on

`rag-eval` routes the agent to the `scripts/eval/evaluate_rag.py` CLI for RAGAS-style quality benchmarks against a deployed RAG stack. Evaluation checks whether the agent loads the right SKILL.md, surfaces the right command shape, names the right metrics (faithfulness, context relevancy, answer correctness), and provides actionable triage advice when a metric is low.

## Harness

| Item | Value |
|------|-------|
| Eval framework | [Harbor](https://github.com/harbor-eval/harbor) (Tier 3) via [`skill-eval/`](../../skill-eval/) |
| Adapter | `skill-eval/adapters/rag-blueprint/generate.py` (shared, invoked with `--skill-name rag-eval`) |
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
1. Explaining how to run a RAGAS quality eval against a running RAG stack (command shape, dataset layout, metrics named).
2. Triaging a low-faithfulness result (concept explanation + at least one concrete remediation).

## Metrics

Each task scores the agent's trajectory against a list of natural-language checks via `skill-eval/verifiers/generic_judge.py`. Reward is mean per-check pass rate.

The five NV-ACES rollup dimensions are not yet computed — see [`skills/PUBLISHING_COMPLIANCE.md`](../PUBLISHING_COMPLIANCE.md) for the migration plan.

## Current results

| Spec | Task | With skill | Without skill (baseline) |
|------|------|------------|--------------------------|
| `nvidia_hosted.json` | RAGAS eval explanation | _populated by Harbor_ | _TODO — pending NV-ACES integration_ |
| `nvidia_hosted.json` | Low-faithfulness triage | _populated by Harbor_ | _TODO — pending NV-ACES integration_ |

## How to reproduce a run locally

Generate a Harbor task tree from this skill, then run:

```bash
cd skill-eval
python3 adapters/rag-blueprint/generate.py \
  --output-dir datasets/rag-eval-nvidia-hosted \
  --skill-dir ../skills/rag-eval \
  --skill-name rag-eval \
  --spec ../skills/rag-eval/eval/nvidia_hosted.json

uvx harbor run \
  -p datasets/rag-eval-nvidia-hosted/step-1 datasets/rag-eval-nvidia-hosted/step-2 \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model claude-sonnet-4-6 \
  -o jobs -n 1 --yes
```

Required env: `NVIDIA_API_KEY` (RAGAS judge), `JUDGE_ANTHROPIC_API_KEY` (Harbor verifier).

## Drift handling

Same as `rag-blueprint` — skill semver pinned to RAG software semver via `scripts/validate_skill_versions.py`. The skill references `scripts/eval/` paths directly; any rename of that directory will surface as a drift failure in the unit test under `tests/unit/test_skills/`.

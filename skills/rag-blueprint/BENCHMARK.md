# BENCHMARK.md — `rag-blueprint`

This document summarizes how the `rag-blueprint` skill is evaluated. The format follows the [Skills Publishing Onboarding Guide](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/) (Step 2 — *Self-Check with NV-BASE → Reporting: BENCHMARK.md*).

## What this skill is graded on

The skill drives an AI coding agent end-to-end through a RAG Blueprint deployment scenario (Docker Compose, with self-hosted or NVIDIA-hosted NIMs) and the resulting system must pass a list of natural-language checks evaluated by an LLM judge.

## Harness

| Item | Value |
|------|-------|
| Eval framework | [Harbor](https://github.com/harbor-eval/harbor) (Tier 3) via [`skill-eval/`](../../skill-eval/) |
| Adapter | `skill-eval/adapters/rag-blueprint/generate.py` |
| Coordinator | `.github/workflows/skills-eval.yml` (PR, nightly cron, manual dispatch) |
| Runner | self-hosted `rag-skill-validator` (2× H100 80GB PCIe for `h100.json`; cpu-only for `nvidia_hosted.json`) |
| Agent under test | `claude-code` |
| Agent model | `claude-sonnet-4-6` (current default; CI may override via `JUDGE_MODEL`) |
| Judge model | `aws/anthropic/claude-haiku-4-5-v1` (Anthropic Claude Haiku 4.5 routed through `inference-api.nvidia.com`) |
| Parallelism | up to 4 concurrent checks per task (`JUDGE_PARALLELISM=4`) |

## Eval specs

The skill ships two Harbor specs under `eval/`. Each spec is a `(skills, platforms, resources.platforms, env, expects[])` tuple per the convention in `.github/skill-eval/AGENTS.md`.

| Spec | Platform | Mode | Tasks |
|------|----------|------|-------|
| [`eval/nvidia_hosted.json`](eval/nvidia_hosted.json) | cpu (GCP `n2d-standard-4`) | NVIDIA-hosted (cloud NIMs) | 2 |
| [`eval/h100.json`](eval/h100.json) | gpu (`H100_x2`) | Self-hosted (local NIMs) | 2 |

## Metrics

Per-task scoring is produced by the generic LLM judge in `skill-eval/verifiers/generic_judge.py`. Each task contributes a list of natural-language checks against (a) the agent's trajectory and (b) the live system state via Bash probes. Reward is a float in `[0, 1]` per check; per-task reward is the mean.

The five NV-ACES rollup dimensions (Security / Correctness / Discoverability / Effectiveness / Efficiency) are not yet computed for this skill — Tier 1 NV-BASE is currently disabled in `.github/workflows/skills-nv-base.yml` pending runner install of `nv-base` (see [`skills/PUBLISHING_COMPLIANCE.md`](../PUBLISHING_COMPLIANCE.md)). Migration to NV-ACES `evals.json` schema with the deterministic `skill_execution` / `skill_efficiency` evaluators is tracked there.

## Current results

Populated from the most recent successful run of `.github/workflows/skills-eval.yml` against `develop`. Update on every release.

| Spec | Task | With skill | Without skill (baseline) |
|------|------|------------|--------------------------|
| `nvidia_hosted.json` | Deploy RAG (NVIDIA-hosted Docker) | _populated by Harbor_ | _TODO — baseline pending NV-ACES integration_ |
| `nvidia_hosted.json` | Verify stack health | _populated by Harbor_ | _TODO — baseline pending NV-ACES integration_ |
| `h100.json` | Deploy RAG (self-hosted H100x2 Docker) | _populated by Harbor_ | _TODO — baseline pending NV-ACES integration_ |
| `h100.json` | Verify stack health (local NIMs) | _populated by Harbor_ | _TODO — baseline pending NV-ACES integration_ |

Per the publishing guide: "BENCHMARK.md can be auto-generated from NV-ACES output OR filled in manually after team-owned evaluation." Without-skill baselines will land when NV-ACES integration ships; until then the with-skill column from Harbor is the available signal.

## How to reproduce a run locally

See `skill-eval/README.md` and `skill-eval/CLAUDE.md` for the exact `harbor run` invocation and required environment variables (`NGC_API_KEY`, `JUDGE_ANTHROPIC_API_KEY`).

## Drift handling

SKILL.md is hand-authored and version-pinned to the RAG software version (`pyproject.toml:project.version`). The validator at `scripts/validate_skill_versions.py` (exercised by `tests/unit/test_skills/`) blocks merges where the skill semver drifts from the software semver. This catches the publishing guide's stated drift risk: "NV-BASE catches security and spec issues but does NOT detect SKILL.md drift (skill says X, product now does Y). Teams own product-drift management."

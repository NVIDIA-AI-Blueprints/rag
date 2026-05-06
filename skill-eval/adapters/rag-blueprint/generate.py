#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor task directories for the rag-blueprint skill.

This adapter is named after the skill it serves (`rag-blueprint`). It is
not tied to one eval — it accepts any spec under
`skill-source/.agents/skills/rag-blueprint/eval/<name>.json` and derives
the eval name (used in task names, dataset dir, and print output) from
the spec filename.

Adding a new eval for the rag-blueprint skill:
    1. Drop a new spec, e.g. `helm_deploy.json`, into
       `skill-source/.agents/skills/rag-blueprint/eval/`
    2. Run:
         python3 generate.py \\
             --output-dir ../../datasets/helm-deploy \\
             --skill-dir ../../../skill-source/.agents/skills/rag-blueprint \\
             --spec ../../../skill-source/.agents/skills/rag-blueprint/eval/helm_deploy.json
    3. Run Harbor against `datasets/helm-deploy/step-{1,N}` as usual.

Adding evaluations for a different skill:
    Copy this directory to `adapters/<new-skill-name>/`, edit:
      - SKILL_NAME constant (also drives the SKILL.md copy path)
      - REPO_ROOT constant (cwd pinned for the judge's probes)
      - PREAMBLE / instruction template wording if needed
      - Default spec filename in the CLI
    The shared `envs/local_env.py` and `verifiers/generic_judge.py` work
    unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Skill-specific constants (only this block changes when copying for a
# different skill).
# ---------------------------------------------------------------------------

SKILL_NAME = "rag-blueprint"
TASK_PREFIX = "rag"
REPO_ROOT = "/home/faaranm/dfw/ragbp/rag"
DEFAULT_SPEC = "nvidia_hosted.json"

PREAMBLE = (
    "You are running inside a non-interactive evaluation harness.\n"
    "You are pre-authorized to deploy and configure services autonomously —\n"
    "do not pause to ask for confirmation on any setup action.\n"
    f"Run all commands from the repo root: {REPO_ROOT}/"
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Turn a spec filename like 'nvidia_hosted.json' into 'nvidia-hosted'."""
    stem = Path(name).stem
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "eval"


# ---------------------------------------------------------------------------
# Per-step artifact templates
# ---------------------------------------------------------------------------

def _instruction_md(step: int, total: int, query: str, env: str) -> str:
    return (
        f"{PREAMBLE}\n"
        "\n"
        f"Use the `/{SKILL_NAME}` skill to complete the following task.\n"
        "\n"
        f"## Task {step} of {total}\n"
        "\n"
        f"{query}\n"
        "\n"
        "## Environment notes\n"
        "\n"
        f"{env}\n"
        "\n"
        "Run autonomously without prompting for confirmation.\n"
    )


def _task_toml(step: int, total: int, check_count: int, eval_name: str) -> str:
    return (
        "[task]\n"
        f'name = "{TASK_PREFIX}/{eval_name}-step-{step}"\n'
        f'description = "{eval_name} step {step}/{total}"\n'
        "\n"
        "[environment]\n"
        'skills_dir = "/skills"\n'
        "\n"
        "[metadata]\n"
        f'skill = "{SKILL_NAME}"\n'
        f'eval_name = "{eval_name}"\n'
        f"step_index = {step}\n"
        f"step_count = {total}\n"
        f"check_count = {check_count}\n"
    )


def _test_sh(step: int, spec_name: str, eval_name: str) -> str:
    return (
        "#!/bin/bash\n"
        f"# {SKILL_NAME} verifier ({eval_name} step {step}): "
        "delegates to generic LLM-as-judge.\n"
        "set -uo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"# Pin cwd to the {SKILL_NAME} repo root so the judge's Bash/grep\n"
        "# probes resolve relative paths against the repo, not the harbor\n"
        "# session workdir.\n"
        f'REPO_ROOT="${{REPO_ROOT:-{REPO_ROOT}}}"\n'
        'cd "$REPO_ROOT"\n'
        "\n"
        "# Judge uses JUDGE_ANTHROPIC_API_KEY — separate from Claude Code OAuth.\n"
        '# Export before running: export JUDGE_ANTHROPIC_API_KEY="sk-..."\n'
        'VERIFIER_DIR="${DFW_VERIFIER_DIR:-/logs/verifier}"\n'
        "# inference-api uses fully qualified model names (aws/anthropic/...).\n"
        "# claude CLI validates short names; route via DEFAULT_*_MODEL aliases.\n"
        "# CLAUDE_CODE_DISABLE_THINKING=1 stops claude CLI from sending the\n"
        "# context_management field, which the inference-api proxy rejects.\n"
        'JUDGE_FULL_MODEL="${JUDGE_FULL_MODEL:-aws/anthropic/claude-haiku-4-5-v1}"\n'
        'ANTHROPIC_API_KEY="${JUDGE_ANTHROPIC_API_KEY}" \\\n'
        'ANTHROPIC_BASE_URL="https://inference-api.nvidia.com" \\\n'
        'ANTHROPIC_DEFAULT_HAIKU_MODEL="${JUDGE_FULL_MODEL}" \\\n'
        'ANTHROPIC_DEFAULT_SONNET_MODEL="${JUDGE_FULL_MODEL}" \\\n'
        'ANTHROPIC_DEFAULT_OPUS_MODEL="${JUDGE_FULL_MODEL}" \\\n'
        'CLAUDE_CODE_DISABLE_THINKING=1 \\\n'
        'JUDGE_MODEL="${JUDGE_MODEL:-haiku}" \\\n'
        'uvx --with "anthropic>=0.40.0,claude-agent-sdk" python "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step} \\\n'
        '    --reward-file "$VERIFIER_DIR/reward.txt" \\\n'
        '    --details-file "$VERIFIER_DIR/judge.json"\n'
        "exit 0\n"
    )


def _solve_sh(step: int, eval_name: str) -> str:
    return (
        "#!/bin/bash\n"
        f"# Gold solution stub for {eval_name} step {step}.\n"
        "# Manual verification is required for this deployment-class task.\n"
        'echo "manual verification required"\n'
    )


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(spec: dict, output_root: Path, skill_dir: Path, eval_name: str) -> None:
    expects = spec.get("expects") or []
    spec_name = Path(spec.get("_source_path", DEFAULT_SPEC)).name
    total = len(expects)
    env_note = spec.get("env", "")

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        (step_dir / "instruction.md").write_text(
            _instruction_md(idx, total, expect.get("query", ""), env_note)
        )

        checks = expect.get("checks") or []
        (step_dir / "task.toml").write_text(
            _task_toml(idx, total, len(checks), eval_name)
        )

        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(_test_sh(idx, spec_name, eval_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        else:
            print(f"  WARN  generic_judge.py not found at {GENERIC_JUDGE}", file=sys.stderr)
        spec_src = skill_dir / "eval" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(_solve_sh(idx, eval_name))

        if skill_dir.exists():
            dst = step_dir / "skills" / SKILL_NAME
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(skill_dir, dst)
            # The source SKILL.md restricts Bash to specific patterns
            # (Bash(docker ps *), Bash(curl *), etc.) — fine for interactive
            # use, but it blocks `docker compose up`, `source <env>`, and
            # everything else the deploy needs. Drop the allowed-tools line
            # in the eval copy so the agent has unrestricted Bash. This does
            # NOT modify the source skill; only the per-trial copy.
            skill_md = dst / "SKILL.md"
            if skill_md.exists():
                lines = skill_md.read_text().splitlines(keepends=True)
                stripped = [
                    line for line in lines
                    if not line.startswith("allowed-tools:")
                ]
                skill_md.write_text("".join(stripped))

        print(f"  GEN  {eval_name}/step-{idx}  ({len(checks)} checks)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True,
                        help=f"Dataset output root (e.g. skill-eval/datasets/<eval-name>)")
    parser.add_argument("--skill-dir", required=True,
                        help=f"Path to skill-source/.agents/skills/{SKILL_NAME}")
    parser.add_argument("--spec", default=None,
                        help=f"Path to eval spec JSON "
                             f"(default: <skill-dir>/eval/{DEFAULT_SPEC})")
    parser.add_argument("--eval-name", default=None,
                        help="Override eval name used in task names + output. "
                             "Default: derived from spec filename "
                             "(e.g. nvidia_hosted.json → nvidia-hosted).")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    spec_path = Path(args.spec) if args.spec else (skill_dir / "eval" / DEFAULT_SPEC)

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)
    eval_name = args.eval_name or _slug(spec_path.name)

    print("=== Inputs ===")
    print(f"  skill      : {SKILL_NAME}")
    print(f"  eval_name  : {eval_name}")
    print(f"  output_dir : {output_root}")
    print(f"  skill_dir  : {skill_dir}")
    print(f"  spec       : {spec_path}")
    print(f"  queries    : {len(spec.get('expects', []))}")
    total_checks = sum(len(q.get("checks", [])) for q in spec.get("expects", []))
    print(f"  checks     : {total_checks}")
    print()

    generate(spec, output_root, skill_dir, eval_name)

    print()
    print(f"Generated {len(spec.get('expects', []))} step(s) under {output_root}/")


if __name__ == "__main__":
    main()

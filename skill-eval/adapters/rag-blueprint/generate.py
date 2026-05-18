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

Adding evaluations for a different rag-* skill (e.g. rag-enable-vlm):
    Run with `--skill-name <new-skill> --skill-dir <path-to-its-source>`.
    All references to the skill (slash-command, task.toml metadata, SKILL.md
    copy path, verifier headers) are derived from `--skill-name`. The
    DEFAULT_SKILL_NAME constant is just the default for backwards-compat
    with the rag-blueprint pipeline.

    If the new skill needs a different REPO_ROOT, PREAMBLE wording, or
    PLATFORMS entries, copy this directory to `adapters/<skill-family>/`
    and edit those — but most rag-* skills should share this single
    adapter.
    The shared `envs/local_env.py` and `verifiers/generic_judge.py` work
    unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Skill-specific constants (only this block changes when copying for a
# different skill).
# ---------------------------------------------------------------------------

# Default skill — overridable via `--skill-name` CLI flag. Threaded through
# generate() + helpers so this adapter generates task dirs for any
# rag-* skill (rag-blueprint today, rag-enable-vlm / rag-enable-guardrails
# / etc. when their specs land), not just the original rag-blueprint.
DEFAULT_SKILL_NAME = "rag-blueprint"
TASK_PREFIX = "rag"
# Resolved at generation time. Priority: $RAG_REPO_ROOT > path inferred
# from this file's location. Set RAG_REPO_ROOT in CI to the checkout root
# (e.g. $GITHUB_WORKSPACE) for predictable output across hosts.
REPO_ROOT = os.environ.get("RAG_REPO_ROOT") or str(
    Path(__file__).resolve().parents[3]
)
DEFAULT_SPEC = "nvidia_hosted.json"

# Spec `platforms` field → Brev create flags for the eval-target instance.
# Spec authors write platform NAMES (`cpu`, `L40S`, `H100`); this adapter
# translates them to fields under task.toml [metadata] which BrevEnvironment
# reads at provision time. Mirrors the VSS PLATFORMS dict pattern.
#
# Recognised metadata keys (all optional; absent = no validation):
#   brev_type                — `brev create --type` argument (CLI v0.6.324+)
#   description              — human note; written to [metadata] but unused
#   gpu_type                 — substring matched against brev's gpu_name
#                              (e.g. "H100" matches "H100-SXM-80GB")
#   gpu_count                — minimum count required
#   min_vram_gb_per_gpu      — per-GPU floor (e.g. 80)
#   min_root_disk_gb         — root-fs floor (catches providers that mount
#                              the big disk on /ephemeral, leaving / small)
#   min_gpu_driver_version   — dotted version floor (e.g. "535.0")
#
# BrevEnvironment._check_instance_matches + _check_live_resources read
# these from task.toml [metadata]. CPU evals declare none → validators
# no-op. GPU evals declare the relevant subset → fast-fail on mismatch.
PLATFORMS: dict[str, dict[str, str | int]] = {
    "cpu": {
        "brev_type": "n2d-standard-4",
        "description": "GCP n2d-standard-4 (4 vCPU, 16 GB). Matches the "
                       "runner VM shape — enough for NVIDIA-hosted RAG "
                       "(8 Docker containers, no local inference).",
    },
    # GPU platforms (active — referenced when an eval spec declares one).
    # Hardware floors below come from docs/support-matrix.md (driver 560+,
    # 80 GB VRAM on H100-80GB, etc.) — the same line the human-facing docs
    # commit to. If support-matrix.md changes, update here too.
    #
    # `brev_type` values follow VSS's naming; if you hit "type not found"
    # at provision time, run `brev search gpu --gpu-name H100 --json` on
    # the runner and substitute the actual catalog name.
    "H100_x2": {
        "brev_type": "dmz.h100x2.pcie",
        "description": "2x H100 80 GB PCIe. Default self-hosted Docker "
                       "configuration per docs/support-matrix.md.",
        "gpu_type": "H100",
        "gpu_count": 2,
        "min_vram_gb_per_gpu": 80,
        "min_root_disk_gb": 500,
        "min_gpu_driver_version": "560.0",
    },
    "RTX_PRO_6000_x2": {
        "brev_type": "rtx-pro-6000-x2",   # verify via `brev search gpu`
        "description": "2x RTX PRO 6000. Alternative self-hosted Docker "
                       "configuration per docs/support-matrix.md.",
        "gpu_type": "RTX",
        "gpu_count": 2,
        "min_vram_gb_per_gpu": 48,         # RTX PRO 6000 Blackwell ~48 GB
        "min_root_disk_gb": 500,
        "min_gpu_driver_version": "560.0",
    },
}

PREAMBLE = (
    "You are running inside a non-interactive evaluation harness.\n"
    "You are pre-authorized to deploy and configure services autonomously —\n"
    "do not pause to ask for confirmation on any setup action.\n"
    "First command: `cd \"$RAG_REPO_ROOT\"`. Every subsequent command runs\n"
    "from there. The shell environment already has RAG_REPO_ROOT exported.\n"
    "Do NOT hardcode an absolute path — the repo location differs between\n"
    "LocalEnvironment (runner workspace) and BrevEnvironment ($HOME/rag)."
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

def _instruction_md(step: int, total: int, query: str, env: str,
                    skill_name: str = DEFAULT_SKILL_NAME) -> str:
    return (
        f"{PREAMBLE}\n"
        "\n"
        f"Use the `/{skill_name}` skill to complete the following task.\n"
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


def _task_toml(
    step: int,
    total: int,
    check_count: int,
    eval_name: str,
    platform: str | None,
    platform_meta: dict[str, str],
    skill_name: str = DEFAULT_SKILL_NAME,
) -> str:
    """Generate task.toml for one step.

    `platform` and `platform_meta` are the user-facing platform name (e.g.
    `cpu`) and the resolved hardware mapping (e.g. {"brev_cpu": "4x16"}).
    BrevEnvironment reads these from [metadata] at provision time.
    """
    lines = [
        "[task]",
        f'name = "{TASK_PREFIX}/{eval_name}-step-{step}"',
        f'description = "{eval_name} step {step}/{total}"',
        "",
        "[environment]",
        'skills_dir = "/skills"',
        "",
        "[metadata]",
        f'skill = "{skill_name}"',
        f'eval_name = "{eval_name}"',
        f"step_index = {step}",
        f"step_count = {total}",
        f"check_count = {check_count}",
    ]
    if platform:
        lines.append(f'platform = "{platform}"')
    for key, val in platform_meta.items():
        # description is for humans; everything else (brev_cpu, brev_gpu,
        # etc.) is consumed by BrevEnvironment.
        lines.append(f'{key} = "{val}"')
    return "\n".join(lines) + "\n"


def _test_sh(step: int, spec_name: str, eval_name: str,
             skill_name: str = DEFAULT_SKILL_NAME) -> str:
    return (
        "#!/bin/bash\n"
        f"# {skill_name} verifier ({eval_name} step {step}): "
        "delegates to generic LLM-as-judge.\n"
        "set -uo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"# Pin cwd to the {skill_name} repo root so the judge's Bash/grep\n"
        "# probes resolve relative paths against the repo, not the harbor\n"
        "# session workdir.\n"
        # On BrevEnvironment the runner-side REPO_ROOT path doesn't exist
        # on the target — fall back to $RAG_REPO_ROOT (set in ~/.eval_env
        # by brev_env.py) so the judge resolves probes against the staged
        # repo at $HOME/rag. LocalEnvironment uses the runner path directly.
        f'REPO_ROOT="${{REPO_ROOT:-${{RAG_REPO_ROOT:-{REPO_ROOT}}}}}"\n'
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

def generate(spec: dict, output_root: Path, skill_dir: Path, eval_name: str,
             skill_name: str = DEFAULT_SKILL_NAME) -> None:
    expects = spec.get("expects") or []
    spec_name = Path(spec.get("_source_path", DEFAULT_SPEC)).name
    total = len(expects)
    env_note = spec.get("env", "")

    # Resolve the eval-target platform. Spec carries a list of platform
    # names; for Phase 2 we pick the first one. (Future multi-platform
    # specs will fan out into separate task dirs per platform.)
    platforms = spec.get("platforms") or []
    platform = platforms[0] if platforms else None
    if platform and platform not in PLATFORMS:
        print(
            f"  WARN  platform '{platform}' not in PLATFORMS dict — "
            f"BrevEnvironment will fall back to its default shape",
            file=sys.stderr,
        )
    platform_meta = {
        k: v for k, v in (PLATFORMS.get(platform) or {}).items()
        if k != "description"
    }

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        (step_dir / "instruction.md").write_text(
            _instruction_md(idx, total, expect.get("query", ""), env_note,
                            skill_name)
        )

        checks = expect.get("checks") or []
        (step_dir / "task.toml").write_text(
            _task_toml(idx, total, len(checks), eval_name, platform,
                       platform_meta, skill_name)
        )

        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(_test_sh(idx, spec_name, eval_name, skill_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        else:
            print(f"  WARN  generic_judge.py not found at {GENERIC_JUDGE}", file=sys.stderr)
        # Write the substituted spec (with REPO_ROOT resolved) into the task
        # dir so the judge reads the real path, not the ${RAG_REPO_ROOT}
        # placeholder. _source_path is internal-only and excluded.
        substituted = {k: v for k, v in spec.items() if k != "_source_path"}
        (tests_dir / spec_name).write_text(json.dumps(substituted, indent=2))

        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(_solve_sh(idx, eval_name))

        if skill_dir.exists():
            dst = step_dir / "skills" / skill_name
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
                        help="Dataset output root (e.g. skill-eval/datasets/<eval-name>)")
    parser.add_argument("--skill-name", default=DEFAULT_SKILL_NAME,
                        help=f"Name of the rag-* skill being evaluated. "
                             f"Drives the /<skill-name> slash-command in the "
                             f"agent prompt, the SKILL.md copy path, the "
                             f"skill = ... field in task.toml metadata, and "
                             f"default verifier headers. Default: "
                             f"{DEFAULT_SKILL_NAME}.")
    parser.add_argument("--skill-dir", required=True,
                        help="Path to the source skill folder containing "
                             "SKILL.md (e.g. skill-source/.agents/skills/"
                             "<skill-name>).")
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
    skill_name = args.skill_name
    spec_path = Path(args.spec) if args.spec else (skill_dir / "eval" / DEFAULT_SPEC)

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    # Keep ${RAG_REPO_ROOT} as a literal — the agent's shell expands it at
    # exec-time. For LocalEnvironment this resolves to the runner workspace
    # (script sets RAG_REPO_ROOT); for BrevEnvironment to $HOME/rag (set by
    # brev_env.py's _stage_repo via the target's ~/.eval_env). Earlier
    # versions substituted at generate-time, which baked the runner's path
    # into the prompt and broke remote evals.
    #
    # Backwards-compat: still rewrite the legacy hardcoded /home/faaranm
    # path so older specs work without edits.
    spec_text = spec_path.read_text()
    spec_text = spec_text.replace("/home/faaranm/dfw/ragbp/rag", "$RAG_REPO_ROOT")
    spec = json.loads(spec_text)
    spec["_source_path"] = str(spec_path)
    eval_name = args.eval_name or _slug(spec_path.name)

    print("=== Inputs ===")
    print(f"  skill      : {skill_name}")
    print(f"  eval_name  : {eval_name}")
    print(f"  output_dir : {output_root}")
    print(f"  skill_dir  : {skill_dir}")
    print(f"  spec       : {spec_path}")
    print(f"  queries    : {len(spec.get('expects', []))}")
    total_checks = sum(len(q.get("checks", [])) for q in spec.get("expects", []))
    print(f"  checks     : {total_checks}")
    print()

    generate(spec, output_root, skill_dir, eval_name, skill_name)

    print()
    print(f"Generated {len(spec.get('expects', []))} step(s) under {output_root}/")


if __name__ == "__main__":
    main()

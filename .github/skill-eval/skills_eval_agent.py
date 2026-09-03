#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RAG Skills eval agent — single-shot CI-driven runner.

Called by .github/workflows/skills-eval.yml on push to `pull-request/<N>`
when files under `skills/` (or the harness itself) change. Spawns one
`claude-agent-sdk` agent with `.github/skill-eval/AGENTS.md` as its
system prompt and lets it drive the eval end-to-end: diff →
adapter/dataset → Harbor run → results comment → cleanup.

The agent gets Bash/Read/Edit/Write/Glob/Grep. It is explicitly told
(in AGENTS.md) that it must NOT modify anything under `skills/`.

Env (set by the workflow step):
    PR_NUMBER             PR being evaluated, e.g. "100" (push mode; blank on workflow_dispatch)
    PR_BASE               Base branch, e.g. "develop" (push mode; blank on workflow_dispatch)
    PR_HEAD_SHA           Mirror head SHA (full)
    PR_REPO               "owner/repo"
    GITHUB_RUN_ID         CI run id
    GITHUB_STEP_SUMMARY   Path to markdown file for Actions run summary (manual mode)
    MANUAL_FULL_SWEEP     "1" when workflow_dispatch fired
    MANUAL_SKILLS_FILTER  Single skill name or "*" for all (workflow_dispatch only)
    ANTHROPIC_*           Agent SDK credentials (sourced from coordinator .env)
    GH_TOKEN              PR comment posting (push mode only)
    NGC_API_KEY           For docker login nvcr.io

Exit codes:
    0 - agent completed (eval may still report failures in PR comment)
    1 - setup error (missing env, AGENTS.md not found, sdk install failed)
    2 - agent crashed
    3 - agent hit max_turns without finishing
    4 - agent exited without DONE:/BLOCKED: marker (protocol failure)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# .github/skill-eval/skills_eval_agent.py:
#   parents[0] = .github/skill-eval
#   parents[1] = .github
#   parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = Path(__file__).resolve().parent / "AGENTS.md"

MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "2000"))


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"FATAL: {name} not set in environment", file=sys.stderr)
        sys.exit(1)
    return v


def _ensure_sdk() -> None:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "claude-agent-sdk>=0.0.5"],
            check=False, timeout=180,
        )


def _disable_server_thinking() -> None:
    """NVIDIA Anthropic proxy rejects `context_management` field."""
    if "CLAUDE_CODE_DISABLE_THINKING" not in os.environ:
        os.environ["CLAUDE_CODE_DISABLE_THINKING"] = "1"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent() -> int:
    from claude_agent_sdk import (  # type: ignore
        AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
        ResultMessage, TextBlock, ToolUseBlock,
    )

    manual_sweep = os.environ.get("MANUAL_FULL_SWEEP") == "1"
    pr_head = _require("PR_HEAD_SHA")
    pr_repo = _require("PR_REPO")
    run_id = os.environ.get("GITHUB_RUN_ID", f"local-{int(time.time())}")

    if manual_sweep:
        pr_number = os.environ.get("PR_NUMBER", "") or f"manual-{run_id}"
        pr_base = os.environ.get("PR_BASE", "") or "(manual)"
        skills_filter = (
            os.environ.get("MANUAL_SKILLS_FILTER", "*").strip().splitlines()[0]
            if os.environ.get("MANUAL_SKILLS_FILTER", "").strip()
            else "*"
        )
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    else:
        pr_number = _require("PR_NUMBER")
        pr_base = _require("PR_BASE")
        skills_filter = "*"
        step_summary = ""

    if not AGENTS_MD.exists():
        print(f"FATAL: {AGENTS_MD} not found", file=sys.stderr)
        return 1

    system_prompt = AGENTS_MD.read_text()

    if manual_sweep:
        user_prompt = f"""
**Manual full-sweep run** — `workflow_dispatch` fired (no PR, no diff).

Context:
  repo                = {pr_repo}
  head SHA            = {pr_head}
  workflow run        = {run_id}
  working dir         = {REPO_ROOT}
  skills filter       = {skills_filter}
  GITHUB_STEP_SUMMARY = {step_summary or '(unset — fall back to stdout)'}

Per AGENTS.md § "Manual full-sweep mode":
  - Skip diff. Enumerate skills/*/eval/*.json, keep skill matching the filter.
  - No bot-PR flow (no contributor branch). Record missing adapters as BLOCKED.
  - Write results to $GITHUB_STEP_SUMMARY instead of gh pr comment.

When done, emit `DONE: <n>/<total> specs passed; <m> blockers`.
"""
    else:
        user_prompt = f"""
PR #{pr_number} just pushed new commits touching `skills/` (or eval harness code).

Context:
  repo          = {pr_repo}
  PR number     = {pr_number}
  base branch   = {pr_base}
  mirror head   = {pr_head}
  workflow run  = {run_id}
  working dir   = {REPO_ROOT}

Your workspace is the repo at `{REPO_ROOT}` (already checked out to the mirror head).
The coordinator host is rag-skill-validator; Docker is running.

Process this PR per AGENTS.md: diff → detect changed skills → check adapter →
generate dataset → run Harbor trials → post ONE comment per (PR, spec) batch.

When done, emit a one-line final summary starting with `DONE:`.
On blocker, emit `BLOCKED:` followed by the reason.
"""

    model = os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6"
    print(
        f"[agent] starting · pr={pr_number} base={pr_base} head={pr_head[:8]} "
        f"model={model} max_turns={MAX_TURNS}",
        flush=True,
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=["Bash", "Read", "Edit", "Write", "Glob", "Grep"],
        model=model,
        max_turns=MAX_TURNS,
        permission_mode="bypassPermissions",
        cwd=str(REPO_ROOT),
    )

    final_text: list[str] = []
    total_cost = 0.0
    hit_max_turns = False

    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        print(block.text, flush=True)
                        final_text.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        inp = getattr(block, "input", {}) or {}
                        hint = ""
                        if name == "Bash":
                            hint = str(inp.get("command", ""))[:140].replace("\n", " ")
                        elif name in ("Read", "Edit", "Write"):
                            hint = str(inp.get("file_path", ""))[-140:]
                        elif name in ("Glob", "Grep"):
                            hint = str(inp.get("pattern", ""))[:140]
                        print(f"  [tool] {name} :: {hint}", flush=True)
            elif isinstance(msg, ResultMessage):
                total_cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
                if getattr(msg, "stop_reason", None) == "max_turns":
                    hit_max_turns = True
                break

    print(f"[agent] finished · cost=${total_cost:.2f}", flush=True)

    if hit_max_turns:
        print("[agent] hit max_turns — agent may not have completed", file=sys.stderr)
        return 3

    summary = "\n".join(final_text[-10:])
    if "BLOCKED:" in summary:
        print("[agent] reported blocker", file=sys.stderr)
        return 0
    if "DONE:" in summary:
        return 0

    print(
        "[agent] exited without DONE: or BLOCKED: marker — protocol failure. "
        "Check trial logs in the workflow artifact.",
        file=sys.stderr,
    )
    return 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _disable_server_thinking()
    _ensure_sdk()
    try:
        rc = asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("[agent] interrupted", file=sys.stderr)
        rc = 2
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] crashed: {exc!r}", file=sys.stderr)
        import traceback; traceback.print_exc()
        rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())

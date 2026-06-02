# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A [Harbor](https://github.com/harbor-eval/harbor) evaluation framework that tests the `rag-blueprint` skill end-to-end. The skill deploys NVIDIA RAG Blueprint via Docker Compose (NVIDIA-hosted mode). An LLM-as-judge verifier then evaluates whether the deployment succeeded using natural-language checks.

## Commands

### Run a full evaluation (both steps)

```bash
export NGC_API_KEY="<ngc-pat>"
export JUDGE_ANTHROPIC_API_KEY="<sk-ant-...>"   # separate from agent auth

cd /home/faaranm/dfw/ragbp/rag/skill-eval
uvx harbor run \
  -p datasets/nvidia-hosted/step-1 datasets/nvidia-hosted/step-2 \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model claude-sonnet-4-6 \
  -o jobs -n 1 --yes
```

### Run a single step

Replace `step-1` with `step-2` in the `-p` flag above.

### Regenerate task files from eval spec

Run this after editing the spec or the upstream skill. The eval name is
derived from the spec filename (e.g. `nvidia_hosted.json` → `nvidia-hosted`)
and used in task names + your chosen output directory:

```bash
# Default eval (nvidia_hosted.json → datasets/nvidia-hosted/):
python3 adapters/rag-blueprint/generate.py \
  --output-dir datasets/nvidia-hosted \
  --skill-dir ../skill-source/.agents/skills/rag-blueprint

# Add a new eval — drop a spec under skill-source/.../eval/ and:
python3 adapters/rag-blueprint/generate.py \
  --output-dir datasets/helm-deploy \
  --skill-dir ../skill-source/.agents/skills/rag-blueprint \
  --spec ../skill-source/.agents/skills/rag-blueprint/eval/helm_deploy.json

```

This regenerates `instruction.md`, `task.toml`, `tests/test.sh`, copies the skill, and **strips `allowed-tools:` from the skill's SKILL.md** so the eval agent has unrestricted Bash.

### Adding evals for a different skill

Copy `adapters/rag-blueprint/` to `adapters/<new-skill>/` and edit the
constants block at the top of `generate.py` (`SKILL_NAME`, `TASK_PREFIX`,
`REPO_ROOT`, `DEFAULT_SPEC`). The shared `envs/local_env.py` and
`verifiers/generic_judge.py` need no changes.

### Run the verifier standalone (without Harbor)

Requires the RAG stack to be already deployed:

```bash
cd /home/faaranm/dfw/ragbp/rag
export JUDGE_ANTHROPIC_API_KEY="sk-..."
export DFW_TRAJECTORY_FILE="/path/to/claude-code.txt"   # optional; for trajectory checks
export DFW_VERIFIER_DIR="/tmp/verifier-out"

ANTHROPIC_API_KEY="${JUDGE_ANTHROPIC_API_KEY}" \
ANTHROPIC_BASE_URL="https://inference-api.nvidia.com" \
JUDGE_FULL_MODEL="aws/anthropic/claude-haiku-4-5-v1" \
JUDGE_MODEL="haiku" \
CLAUDE_CODE_DISABLE_THINKING=1 \
uvx --with "anthropic>=0.40.0,claude-agent-sdk" \
  python skill-eval/verifiers/generic_judge.py \
  --spec skill-eval/datasets/deploy-rag/step-1/tests/nvidia_hosted.json \
  --step 1
```

### Lint (parent project ruff config applies)

```bash
cd /home/faaranm/dfw/ragbp/rag
ruff check skill-eval/ --fix && ruff format skill-eval/
```

## Architecture

```
eval spec (nvidia_hosted.json)
       │ adapters/rag-blueprint/generate.py
       ▼
datasets/deploy-rag/step-{1,2}/          ← Harbor task directories
  instruction.md    ← agent prompt (prepends PREAMBLE, injects env notes)
  task.toml         ← Harbor metadata
  tests/test.sh     ← calls generic_judge.py
  tests/nvidia_hosted.json  ← spec copy
  skills/rag-blueprint/     ← skill copy (allowed-tools: stripped)
       │
       │ uvx harbor run ...
       ▼
envs/local_env.py (LocalEnvironment)
  • Maps /logs/, /tests/, /solution/, /skills/ → harbor-workdir/<session_id>/
  • Skips apt-get/npm install fragments (pre-installed on host)
  • For `claude --verbose` invocations: unsets CLAUDE_CONFIG_DIR so SSE auth works
  • Exports DFW_TRAJECTORY_FILE and DFW_VERIFIER_DIR for the judge
       │
       ├─► Agent: claude-code + rag-blueprint skill
       │         trajectory written to harbor-workdir/<id>/logs/agent/claude-code.txt
       │
       └─► Verifier: tests/test.sh → generic_judge.py
                     Runs N checks concurrently (JUDGE_PARALLELISM=4 default)
                     Each check: independent claude-agent-sdk agent with Bash+Read+Grep
                     Writes reward.txt (float 0–1) + judge.json (per-check details)
```

## Key design details

**Eval spec (`nvidia_hosted.json`)** — the single source of truth. Contains:
- `env`: prose describing the deployment environment, injected into the agent prompt
- `expects[]`: list of `{query, checks[]}` pairs; each element becomes a Harbor step

**LLM judge routing** — the judge has no Python-level routing logic. It reads each check in natural language and decides whether to run a live shell probe (Bash), inspect the trajectory (Grep/Read), or both. The system prompt documents the decision rules. Check phrasing matters: backtick-delimited commands are directives; "does NOT include X" checks must use trajectory inspection, not Bash.

**Two API keys** — the agent authenticates via SSE channel (ANTHROPIC_API_KEY="" + CLAUDE_CODE_SSE_PORT from the parent Claude Code session). The judge uses JUDGE_ANTHROPIC_API_KEY (a real sk-ant-... key) routed through `inference-api.nvidia.com` with `CLAUDE_CODE_DISABLE_THINKING=1` (the inference-api proxy rejects the `context_management` field the claude CLI sends when thinking is on).

**Judge model resolution** — `JUDGE_MODEL` env var → `ANTHROPIC_MODEL` → `"claude-sonnet-4-6"`. In CI, `test.sh` sets `JUDGE_MODEL=haiku` and maps all model aliases to `aws/anthropic/claude-haiku-4-5-v1` via `ANTHROPIC_DEFAULT_*_MODEL`.

**Path remapping in LocalEnvironment** — uses regex with a path-boundary lookbehind so `/logs/` only rewrites at path-start positions (not inside already-absolute paths like `/home/.../harbor-workdir/.../logs/...`). Per-session workdir: `harbor-workdir/<session_id>/`.

**`allowed-tools:` stripping** — the source SKILL.md restricts Bash to specific glob patterns (fine for interactive use). The adapter strips that line from the eval copy so the deploy agent can run arbitrary commands. Only the per-trial copy is modified; the source skill is untouched.

**Results** — `eval_result.md` has the most recent run summary. Per-trial details live in `jobs/<timestamp>/` and `harbor-workdir/<session_id>/exec.debug.log`.

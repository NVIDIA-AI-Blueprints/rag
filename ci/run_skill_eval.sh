#!/usr/bin/env bash
# Runs the rag-blueprint skill-eval framework (VSS-style Harbor harness).
#
# Invoked by .github/workflows/run-branch-script.yml on the self-hosted
# rag-skill-validator runner. Mirrors the manual flow in skill-eval/README.md
# so the same command works locally and in CI.
#
# Required env (from the dispatcher workflow):
#   NVIDIA_INFERENCE_KEY    sk-... NV inference proxy key (used as JUDGE_ANTHROPIC_API_KEY)
#   ANTHROPIC_API_KEY       same as above (claude CLI auth)
#   NGC_API_KEY             nvapi-... for docker login nvcr.io
#   CLAUDE_CODE_DISABLE_THINKING=1
#
# Output (uploaded by the workflow as an artifact):
#   skill-eval/jobs/<timestamp>/...    per-trial Harbor results
#   skill-eval/eval_result.md          human-readable summary

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
export RAG_REPO_ROOT="$REPO_ROOT"
SKILL_EVAL_DIR="$REPO_ROOT/skill-eval"
SKILL_DIR="$REPO_ROOT/skill-source/.agents/skills/rag-blueprint"
EVAL_NAME="nvidia-hosted"
DATASETS_DIR="$SKILL_EVAL_DIR/datasets/$EVAL_NAME"

echo "==> Required env check"
: "${NVIDIA_INFERENCE_KEY:?Set NVBASE_INFERENCE_API_KEY secret (sk- inference proxy key)}"
: "${NGC_API_KEY:?Set NGC_API_KEY secret (nvapi-)}"
export JUDGE_ANTHROPIC_API_KEY="${NVIDIA_INFERENCE_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$NVIDIA_INFERENCE_KEY}"
export CLAUDE_CODE_DISABLE_THINKING="${CLAUDE_CODE_DISABLE_THINKING:-1}"
# NVIDIA proxy needs fully-qualified Anthropic model ids.
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://inference-api.nvidia.com}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-aws/anthropic/bedrock-claude-sonnet-4-6}"
export JUDGE_FULL_MODEL="${JUDGE_FULL_MODEL:-aws/anthropic/claude-haiku-4-5-v1}"

echo "==> Install uv (no-op if already present)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> Install Claude Code CLI (no-op if already present)"
if ! command -v claude >/dev/null 2>&1; then
  npm install -g @anthropic-ai/claude-code
fi
claude --version

echo "==> Docker login to nvcr.io"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "==> Clean any leftover Docker state from prior runs"
docker ps -a --format '{{.ID}}' | xargs -r docker stop >/dev/null 2>&1 || true
docker ps -a --format '{{.ID}}' | xargs -r docker rm   >/dev/null 2>&1 || true

echo "==> Generate Harbor task directories from spec"
cd "$SKILL_EVAL_DIR"
rm -rf "$DATASETS_DIR"
python3 adapters/rag-blueprint/generate.py \
  --output-dir "$DATASETS_DIR" \
  --skill-dir "$SKILL_DIR"

echo "==> Run Harbor trials (each step in expects[] becomes a step-N dir)"
mkdir -p jobs
STEP_DIRS=()
while IFS= read -r d; do STEP_DIRS+=("$d"); done < <(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
uvx harbor run \
  -p "${STEP_DIRS[@]}" \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model "$ANTHROPIC_MODEL" \
  --ak api_base="$ANTHROPIC_BASE_URL/v1" \
  --ae CLAUDE_CODE_DISABLE_THINKING=1 \
  -o jobs -n 1 --yes

echo "==> Summarise results into eval_result.md"
python3 - <<'PY'
import json
from pathlib import Path

jobs = sorted(Path("jobs").iterdir(), key=lambda p: p.stat().st_mtime)
if not jobs:
    raise SystemExit("no Harbor jobs produced")
latest = jobs[-1]

lines = ["# Skill-eval results", "", f"Run: `{latest.name}`", ""]
total, passed = 0, 0
for reward_file in sorted(latest.rglob("reward.txt")):
    r = float(reward_file.read_text().strip() or 0)
    judge = reward_file.parent / "judge.json"
    name = reward_file.parents[1].name
    line = f"- **{name}**: reward `{r:.2f}`"
    if judge.exists():
        j = json.loads(judge.read_text())
        passed += j.get("passed", 0)
        total += j.get("total", 0)
        line += f" ({j.get('passed',0)}/{j.get('total',0)} checks)"
    lines.append(line)

lines.insert(2, f"**Overall:** {passed}/{total} checks passed\n")
out = Path("eval_result.md")
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
PY

echo "==> Eval complete"

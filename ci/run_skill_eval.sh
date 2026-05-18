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

# Cleanup runs on EVERY exit path (success, set -e abort, signal). It
# captures target-VM debug state BEFORE tearing down RAG stacks, so the
# uploaded artifact still has enough to post-mortem even when CI fails
# mid-script. The VM itself stays running (warm pool — see Q1 decision).
cleanup() {
  local rc=$?
  set +e   # don't let cleanup steps themselves abort early
  echo "==> Cleanup (rc=$rc, BREV_INSTANCE=${BREV_INSTANCE:-})"
  if [ -n "${BREV_INSTANCE:-}" ] && command -v brev >/dev/null 2>&1; then
    local dbg_dir="$REPO_ROOT/eval-results/debug"
    mkdir -p "$dbg_dir"
    local dbg="$dbg_dir/target-state-$(date +%Y%m%d-%H%M%S).txt"
    {
      echo "=== docker ps -a ==="
      brev exec "$BREV_INSTANCE" "docker ps -a 2>&1"
      echo
      echo "=== docker logs (tail 100 per container) ==="
      brev exec "$BREV_INSTANCE" \
        'for c in $(docker ps -a --format "{{.Names}}"); do echo === $c ===; docker logs --tail 100 $c 2>&1 | head -120; done'
      echo
      echo "=== /logs/agent/setup ==="
      brev exec "$BREV_INSTANCE" "ls -la /logs/agent/setup/ 2>&1; for f in /logs/agent/setup/*; do echo --- \$f ---; cat \"\$f\"; done" 2>&1 | head -200
    } > "$dbg" 2>&1 || true
    echo "Debug dump → $dbg"
    # docker compose down on RAG stacks (keeps image cache, warm pool stays warm).
    for f in \
      deploy/compose/docker-compose-rag-server.yaml \
      deploy/compose/docker-compose-ingestor-server.yaml \
      deploy/compose/vectordb.yaml \
      deploy/compose/nims.yaml \
      deploy/compose/docker-compose-nemo-guardrails.yaml \
      deploy/compose/observability.yaml; do
      brev exec "$BREV_INSTANCE" \
        "[ -f \"\$HOME/rag/$f\" ] && docker compose -f \"\$HOME/rag/$f\" down -v --remove-orphans >/dev/null 2>&1 || true" \
        >/dev/null 2>&1 || true
    done
    echo "VM $BREV_INSTANCE left running (warm pool); containers torn down."
  fi
  exit $rc
}
trap cleanup EXIT

# Branch the Brev target will git-clone (VSS-style fresh tree per run).
# Prefer the locally-checked-out branch — actions/checkout sets HEAD to
# the dispatcher's `ref` input (e.g. feat/skill-eval-ci). $GITHUB_REF_NAME
# is the *workflow's* ref (always 'main' for our dispatcher) so it's the
# wrong source. Final fallback is 'main' for local runs.
export EVAL_TARGET_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
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

# Default eval-target Brev instance. The dispatcher workflow does not
# expose BREV_INSTANCE as an input — set it here so the script alone
# controls Phase 1 vs Phase 2 mode. To force LocalEnvironment for a
# debug run, change this line to `export BREV_INSTANCE=""` or comment
# it out, then push to the feature branch and re-trigger.
export BREV_INSTANCE="${BREV_INSTANCE:-rag-eval-target}"

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

echo "==> Pick Harbor environment (Brev remote target vs local host)"
if [ -n "${BREV_INSTANCE:-}" ]; then
  ENV_IMPORT="envs.brev_env:BrevEnvironment"
  echo "Brev mode: BREV_INSTANCE=$BREV_INSTANCE"
  command -v brev >/dev/null || { echo "brev CLI missing on runner"; exit 1; }
  echo "Brev CLI version: $(brev --version 2>&1 | head -1)"
  # Pre-flight: confirm we can list instances (i.e. authed).
  brev ls >/dev/null 2>&1 || {
    echo "brev not authenticated. Run: brev login --auth nvidia"
    exit 1
  }
  # Warm-pool mode (mirrors VSS): if $BREV_INSTANCE already exists, reuse
  # it so docker image cache (notably nv-ingest, ~11 GB) survives between
  # runs. brev_env.start() handles container/network cleanup on the target
  # without nuking the image cache. Only auto-provision if missing.
  if brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {found=1} END{exit !found}'; then
    echo "Reusing warm $BREV_INSTANCE"
  else
    echo "No existing $BREV_INSTANCE — brev_env will auto-provision"
  fi
else
  ENV_IMPORT="envs.local_env:LocalEnvironment"
  echo "Local mode (BREV_INSTANCE unset): RAG will deploy on this runner VM"
fi

echo "==> Clean leftover Docker state from prior runs (one-shot, before any trial)"
# This runs ONCE per CI run — never between trials — so step-1's deploy
# survives long enough for step-2's judge probes. Targets:
#   - LocalEnvironment: this runner VM is also the deploy host.
#   - BrevEnvironment: tear down the warm-pool target's containers,
#     leaving the docker image cache (nv-ingest ~11 GB) intact.
COMPOSE_FILES=(
  deploy/compose/docker-compose-rag-server.yaml
  deploy/compose/docker-compose-ingestor-server.yaml
  deploy/compose/vectordb.yaml
  deploy/compose/nims.yaml
  deploy/compose/docker-compose-nemo-guardrails.yaml
  deploy/compose/observability.yaml
)
if [ "$ENV_IMPORT" = "envs.local_env:LocalEnvironment" ]; then
  for f in "${COMPOSE_FILES[@]}"; do
    [ -f "$f" ] && docker compose -f "$f" down -v --remove-orphans >/dev/null 2>&1 || true
  done
  docker ps -a --format '{{.Names}}' | \
    grep -E '(rag|milvus|nim|ingest|redis|nemo|grafana|prometheus|embedding|ranking|vlm|ocr|page-elements|graphic-elements|table-structure|nv-ingest)' | \
    xargs -r docker rm -f >/dev/null 2>&1 || true
  sudo rm -rf deploy/compose/volumes 2>/dev/null || true
elif brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {found=1} END{exit !found}'; then
  # Brev mode AND the warm-pool VM exists. Run the same down sequence
  # against the target's $HOME/rag/deploy/compose tree. Repo gets re-
  # staged by brev_env.start() right after this. Image cache preserved.
  for f in "${COMPOSE_FILES[@]}"; do
    brev exec "$BREV_INSTANCE" \
      "[ -f \"\$HOME/rag/$f\" ] && docker compose -f \"\$HOME/rag/$f\" down -v --remove-orphans >/dev/null 2>&1 || true" \
      2>/dev/null || true
  done
  brev exec "$BREV_INSTANCE" \
    "docker network rm nvidia-rag >/dev/null 2>&1 || true; docker ps -a --format '{{.Names}}' | grep -E '(rag|milvus|nim|ingest|redis|nemo)' | xargs -r docker rm -f >/dev/null 2>&1 || true" \
    2>/dev/null || true
fi

echo "==> Generate Harbor task directories from spec"
cd "$SKILL_EVAL_DIR"
rm -rf "$DATASETS_DIR"
python3 adapters/rag-blueprint/generate.py \
  --output-dir "$DATASETS_DIR" \
  --skill-dir "$SKILL_DIR"

echo "==> Run Harbor trials — one invocation per step (Harbor -p takes a single path)"
mkdir -p jobs
# Count harbor invocations that crashed (vs. ran-with-failing-checks).
# `set -e` doesn't propagate from inside a `while` loop body, so we have
# to track this ourselves and decide at the end.
HARBOR_CRASHES=0
while IFS= read -r step_dir; do
  echo "----> harbor run -p $step_dir"
  if ! uvx harbor run \
       -p "$step_dir" \
       --environment-import-path "$ENV_IMPORT" \
       --agent claude-code --model "$ANTHROPIC_MODEL" \
       --ak api_base="$ANTHROPIC_BASE_URL/v1" \
       --ae CLAUDE_CODE_DISABLE_THINKING=1 \
       -o jobs -n 1 --yes; then
    HARBOR_CRASHES=$((HARBOR_CRASHES + 1))
    echo "harbor run exited non-zero for $step_dir"
  fi
done < <(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

echo "==> Summarise results into eval_result.md (walks ALL job dirs)"
python3 - <<'PY'
import json
from pathlib import Path

jobs_root = Path("jobs")
if not jobs_root.exists() or not any(jobs_root.iterdir()):
    raise SystemExit("no Harbor jobs produced")

lines = ["# Skill-eval results", ""]
total, passed = 0, 0
for reward_file in sorted(jobs_root.rglob("reward.txt")):
    r = float(reward_file.read_text().strip() or 0)
    judge = reward_file.parent / "judge.json"
    # parents: reward.txt → verifier → step-N__XXX → <timestamp>
    step_name = reward_file.parents[1].name
    run_name = reward_file.parents[2].name
    line = f"- **{run_name} / {step_name}**: reward `{r:.2f}`"
    if judge.exists():
        j = json.loads(judge.read_text())
        passed += j.get("passed", 0)
        total += j.get("total", 0)
        line += f" ({j.get('passed',0)}/{j.get('total',0)} checks)"
    lines.append(line)

lines.insert(2, f"**Overall:** {passed}/{total} checks passed\n")
out = Path("eval_result.md")
out.write_text("\n".join(lines) + "\n")
# Expose totals to the surrounding shell for the CI exit-code decision.
Path(".eval_total.txt").write_text(f"{total}\n")
Path(".eval_passed.txt").write_text(f"{passed}\n")
print(out.read_text())
PY

EVAL_TOTAL=$(cat "$SKILL_EVAL_DIR/.eval_total.txt" 2>/dev/null || echo 0)
EVAL_PASSED=$(cat "$SKILL_EVAL_DIR/.eval_passed.txt" 2>/dev/null || echo 0)
echo "==> CI exit decision (VSS pattern):"
echo "    HARBOR_CRASHES=$HARBOR_CRASHES  EVAL_TOTAL=$EVAL_TOTAL  EVAL_PASSED=$EVAL_PASSED"

echo "==> Tear down eval target (next CI run starts clean)"
cd "$REPO_ROOT"
# Brev cleanup is handled by the EXIT trap — runs even on script failure.
# LocalEnvironment doesn't get a trap because the runner IS the deploy host
# and we want to leave its state inspectable for debugging.
if [ "$ENV_IMPORT" = "envs.local_env:LocalEnvironment" ]; then
  for f in \
    deploy/compose/docker-compose-rag-server.yaml \
    deploy/compose/docker-compose-ingestor-server.yaml \
    deploy/compose/vectordb.yaml; do
    [ -f "$f" ] && docker compose -f "$f" down -v --remove-orphans >/dev/null 2>&1 || true
  done
  sudo rm -rf deploy/compose/volumes 2>/dev/null || true
fi

echo "==> Stage outputs to eval-results/ for artifact upload"
# The dispatcher workflow's upload-artifact step looks for paths
# `eval-results/`, `**/evals/results/`, `ci-logs/`. The latter glob
# recurses everywhere and chokes on docker-volume dirs owned by root
# (e.g. deploy/compose/volumes/etcd/member → EACCES). Stage our results
# under a clean eval-results/ directory at the repo root so the action
# uploads exactly what we want without needing to crawl docker volumes.
STAGE="$REPO_ROOT/eval-results"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -a "$SKILL_EVAL_DIR/jobs" "$STAGE/jobs"
cp "$SKILL_EVAL_DIR/eval_result.md" "$STAGE/eval_result.md"
echo "Staged artifact tree:"
find "$STAGE" -maxdepth 3 | head -40

echo "==> Eval complete"

# VSS exit-code pattern: CI is red ONLY when the harness itself broke.
# Individual eval-check failures (low reward) stay green — the verdict
# is in the uploaded artifact (eval_result.md + judge.json).
#
# Red signals:
#   - HARBOR_CRASHES > 0  → at least one trial errored (e.g. brev_env,
#     RewardFileNotFoundError) — pipeline didn't run end-to-end
#   - EVAL_TOTAL == 0     → no checks produced — config or harness broken
if [ "$HARBOR_CRASHES" -gt 0 ] || [ "$EVAL_TOTAL" -eq 0 ]; then
  echo "FAIL: pipeline broken — HARBOR_CRASHES=$HARBOR_CRASHES, EVAL_TOTAL=$EVAL_TOTAL"
  exit 1
fi
echo "PASS: pipeline ran end-to-end. Eval verdict: $EVAL_PASSED/$EVAL_TOTAL checks passed (see artifact)."

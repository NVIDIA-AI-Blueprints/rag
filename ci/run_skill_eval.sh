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
# mid-script.
#
# Brev teardown is platform-routed (VSS pattern, memory note
# project-pending-gpu-cleanup): most GPU providers cannot be `brev stop`-ped
# — only `brev delete` ends billing — so the trap reads `brev_type` from
# the generated task.toml and chooses stop / delete / keep accordingly.
# CPU evals run on LocalEnvironment (BREV_INSTANCE unset) so this whole
# block is skipped for them; the runner's docker state is cleaned in the
# success-path teardown lower in the script.
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
    # docker compose down on RAG stacks (target side). Cheap; keeps the VM
    # in a known state for the `action=stop` and `action=keep` paths. On
    # `action=delete` it's redundant but harmless — the VM is about to go.
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

    # Pick teardown action by provider lifecycle. Match against the
    # adapter-emitted `brev_type` in task.toml (any step-N — all share
    # the same value). Lowercase before matching so we tolerate type
    # slugs like "dmz.H100x2.pcie".
    local brev_type=""
    if [ -n "${DATASETS_DIR:-}" ] && [ -d "$DATASETS_DIR" ]; then
      brev_type=$(grep -hoE 'brev_type[[:space:]]*=[[:space:]]*"[^"]+"' \
        "$DATASETS_DIR"/step-*/task.toml 2>/dev/null | head -1 \
        | sed 's/.*"\([^"]*\)".*/\1/' \
        | tr '[:upper:]' '[:lower:]')
    fi
    local action="keep"
    case "$brev_type" in
      *h100*|*massedcompute*|*scaleway*|*hyperstack*|*nebius*|*oci*|*latitude*)
        action="delete" ;;
      *l40s*|*rtx*|*g7e*|*g6e*|*crusoe*)
        action="stop" ;;
    esac
    if [ "$action" != "keep" ]; then
      local cooldown="${COOLDOWN_SEC:-300}"
      echo "==> Brev teardown: $BREV_INSTANCE (type=$brev_type) → $action after ${cooldown}s cooldown"
      sleep "$cooldown"
      brev "$action" "$BREV_INSTANCE" 2>&1 | tail -5 || true
    else
      echo "VM $BREV_INSTANCE left running (no platform match for type=${brev_type:-<unknown>})."
    fi
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
SKILLS_ROOT="$REPO_ROOT/skills"

# Profile routing — filename drives which runner is required:
#   nvidia_hosted.json → LocalEnvironment (cpu, no GPU)
#   h100.json          → BrevEnvironment  (H100_x2, GPU)
# EVAL_PROFILE controls which profile this invocation runs.
# Default: nvidia_hosted (cpu). Set EVAL_PROFILE=h100 for GPU skills.
EVAL_PROFILE="${EVAL_PROFILE:-nvidia_hosted}"

echo "==> Required env check"
: "${NVIDIA_INFERENCE_KEY:?Set NVBASE_INFERENCE_API_KEY secret (sk- inference proxy key)}"
: "${NGC_API_KEY:?Set NGC_API_KEY secret (nvapi-)}"
export JUDGE_ANTHROPIC_API_KEY="${NVIDIA_INFERENCE_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$NVIDIA_INFERENCE_KEY}"
export CLAUDE_CODE_DISABLE_THINKING="${CLAUDE_CODE_DISABLE_THINKING:-1}"
# NVIDIA proxy needs fully-qualified Anthropic model ids.
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://inference-api.nvidia.com}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-aws/anthropic/bedrock-claude-sonnet-4-6}"
# Pin Milvus volumes outside the workspace so docker compose (run from ci/)
# doesn't write root-owned etcd/minio dirs into ci/volumes/ and break the
# artifact upload step (EACCES on scandir ci/volumes/etcd/member).
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-/tmp/milvus-eval}"
export JUDGE_FULL_MODEL="${JUDGE_FULL_MODEL:-aws/anthropic/claude-haiku-4-5-v1}"

# Runtime topology — controlled by whether BREV_INSTANCE is set.
#
#   - CPU evals  (default — most rag-* skills):
#         BREV_INSTANCE unset  →  LocalEnvironment
#         Runner deploys RAG locally on itself (runner == target).
#         No separate Brev VM, no cross-VM `brev exec` plumbing.
#
#   - GPU evals  (rag-enable-vlm, rag-enable-guardrails):
#         Workflow / invoker sets BREV_INSTANCE=rag-eval-gpu-<uuid>
#         →  BrevEnvironment in ephemeral-provisioning mode (Item C)
#         Runner uses `brev create` to spin a fresh GPU VM, drives
#         deploy + judge via `brev exec`, and `brev delete`s the VM
#         in the EXIT trap (Item E).
#
# To force a manual override (e.g. debug a Brev VM end-to-end without
# the CI flow), export BREV_INSTANCE=<name> before invoking the script.
#
# ============================================================================
# >>> GPU TESTING PATCH (Option B) <<<
# This block exists to validate H100×2 self-hosted CI end-to-end via the
# existing dispatcher workflow (no YAML changes needed on main). It's
# production-safe: only fires when EVAL_PROFILE matches a GPU pattern;
# the CPU default (nvidia_hosted) is unaffected.
# Companion files in this patch:
#   ci/run_skill_eval_h100.sh                       (wrapper)
#   skills/rag-deploy-blueprint/eval/h100.json      (deploy spec)
# Keep after validation — generalises to any future GPU profile.
# ============================================================================
# Auto-pick: if EVAL_PROFILE matches a GPU pattern (h100*, l40s*, rtx*,
# gpu_*), generate a fresh BREV_INSTANCE so brev_env enters ephemeral-
# provision mode. CPU profiles (nvidia_hosted) leave BREV_INSTANCE empty
# → LocalEnvironment. Caller can still override BREV_INSTANCE explicitly.
case "$EVAL_PROFILE" in
  h100*|l40s*|rtx*|gpu_*)
    : "${BREV_INSTANCE:=rag-eval-gpu-$(date +%s | tail -c 8)}" ;;
esac
export BREV_INSTANCE="${BREV_INSTANCE:-}"

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
  # ==========================================================================
  # >>> VSS-style pre-provision (Mode-1 pattern) <<<
  # Provision the Brev VM HERE, not inside brev_env. VSS's architecture
  # (vss-feat-skill-eval/.github/skill-eval/skills_eval_agent.py + AGENTS.md)
  # has the coordinator do `brev create` BEFORE invoking harbor, then
  # harbor's brev_env enters Mode 1 (validate-only). We mirror that with
  # bash here.
  #
  # Two reasons this lives in the script, not in brev_env:
  #   1. EOF resilience: when Brev's control plane has transient hiccups
  #      (observed multiple times in run 26142225004:
  #      "unexpected EOF" from brevapi.us-west-2-prod.control-plane.brev.dev),
  #      `brev create` can return non-zero even when the workspace was
  #      actually created server-side. Bash with retries handles this
  #      cleanly; inside brev_env it would cascade-fail 6 trials.
  #   2. Concentration: one create attempt per CI run, not one per
  #      harbor trial — failures are isolated to the orchestrator layer.
  # ==========================================================================

  # If $BREV_INSTANCE is already present (operator pre-provisioned, or
  # leftover from a prior run), reuse it. Otherwise create + wait.
  if brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {found=1} END{exit !found}'; then
    echo "Found existing $BREV_INSTANCE — reusing (skipping create)"
  else
    # Resolve brev_type from the spec JSON (task.toml isn't written yet,
    # we're before the adapter call).
    SPEC_FILE_FOR_TYPE="$SKILLS_ROOT/rag-deploy-blueprint/eval/${EVAL_PROFILE}.json"
    if [ ! -f "$SPEC_FILE_FOR_TYPE" ]; then
      echo "Spec file not found for $EVAL_PROFILE: $SPEC_FILE_FOR_TYPE"
      exit 1
    fi
    BREV_TYPE=$(python3 - <<PY 2>/dev/null || true
import json, sys
spec = json.load(open("$SPEC_FILE_FOR_TYPE"))
plats = (spec.get("resources") or {}).get("platforms") or {}
for p in spec.get("platforms", []):
    cfg = plats.get(p) or {}
    if cfg.get("brev_type"):
        print(cfg["brev_type"]); sys.exit(0)
PY
)
    if [ -z "$BREV_TYPE" ]; then
      echo "Cannot determine brev_type from $SPEC_FILE_FOR_TYPE — set resources.platforms.<name>.brev_type"
      exit 1
    fi
    # Create with EOF-retry loop. Brev's API has been observed to return
    # `unexpected EOF` to `brev create` even after server-side success
    # (run 26142225004's trial-1 traceback). After each attempt, check
    # whether the workspace was actually created — if so, the EOF was a
    # response-truncation, not a real failure.
    echo "==> Pre-provisioning $BREV_INSTANCE type=$BREV_TYPE (up to ${BREV_PROVISION_RETRIES:-5} create attempts)"
    created=0
    for attempt in $(seq 1 "${BREV_PROVISION_RETRIES:-5}"); do
      echo "  attempt $attempt: brev create $BREV_INSTANCE type=$BREV_TYPE"
      echo "$BREV_TYPE" | brev create "$BREV_INSTANCE" --detached 2>&1 | tail -10
      # Whether the CLI reported success or failure, verify reality:
      if brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {found=1} END{exit !found}'; then
        echo "  → workspace $BREV_INSTANCE exists after attempt $attempt"
        created=1
        break
      fi
      echo "  → workspace not present after attempt $attempt; sleeping 15s before retry"
      sleep 15
    done
    if [ "$created" -ne 1 ]; then
      echo "brev create failed permanently after ${BREV_PROVISION_RETRIES:-5} attempts"
      exit 1
    fi
  fi

  # Poll until RUNNING+READY (whether we just created or are reusing).
  # H100 cold-boot on Brev can take 10-15 min. Budget 30 min default.
  echo "==> Waiting for $BREV_INSTANCE to reach RUNNING+READY (up to ${BREV_PROVISION_TIMEOUT:-1800}s)"
  deadline=$(($(date +%s) + ${BREV_PROVISION_TIMEOUT:-1800}))
  last_state=""
  while [ $(date +%s) -lt $deadline ]; do
    state=$(brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" \
      '$1==n {print $2"+"$4; exit}')
    if [ -n "$state" ] && [ "$state" != "$last_state" ]; then
      echo "  $(date -u +%H:%M:%SZ) $BREV_INSTANCE: $state"
      last_state="$state"
    fi
    [ "$state" = "RUNNING+READY" ] && break
    sleep 15
  done
  if [ "$last_state" != "RUNNING+READY" ]; then
    echo "Pre-provision timed out — last state: $last_state"
    exit 1
  fi
  echo "==> $BREV_INSTANCE ready — handing off to harbor (will use Mode 1, no re-provision)"
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
  # Runner is the eval target — clean any leftover docker state from
  # prior CI runs on this same VM. Image cache is preserved (warm pool
  # benefit on the runner itself).
  for f in "${COMPOSE_FILES[@]}"; do
    [ -f "$f" ] && docker compose -f "$f" down -v --remove-orphans >/dev/null 2>&1 || true
  done
  docker ps -a --format '{{.Names}}' | \
    grep -E '(rag|milvus|nim|ingest|redis|nemo|grafana|prometheus|embedding|ranking|vlm|ocr|page-elements|graphic-elements|table-structure|nv-ingest)' | \
    xargs -r docker rm -f >/dev/null 2>&1 || true
  sudo rm -rf deploy/compose/volumes 2>/dev/null || true
fi
# GPU pre-flight (BrevEnvironment mode) is handled inside brev_env.start()
# — the VM is provisioned fresh per CI run, so there's no prior-state
# cleanup to do from the runner side.

echo "==> Auto-discover skills with eval/$EVAL_PROFILE.json and run Harbor trials"
cd "$SKILL_EVAL_DIR"
mkdir -p jobs
HARBOR_CRASHES=0

# Find every skill that ships a spec for the current profile.
# Adding a new skill with eval/<profile>.json is all that's needed —
# no script changes required.
while IFS= read -r spec_file; do
  SKILL_NAME="$(basename "$(dirname "$(dirname "$spec_file")")")"
  SKILL_DIR="$SKILLS_ROOT/$SKILL_NAME"
  DATASETS_DIR="$SKILL_EVAL_DIR/datasets/$SKILL_NAME"

  echo ""
  echo "==> [$SKILL_NAME] Generating task directories"
  rm -rf "$DATASETS_DIR"
  python3 adapters/rag-blueprint/generate.py \
    --output-dir "$DATASETS_DIR" \
    --skill-dir  "$SKILL_DIR" \
    --skill-name "$SKILL_NAME" \
    --spec       "$spec_file"

  echo "==> [$SKILL_NAME] Running Harbor trials"
  while IFS= read -r step_dir; do
    echo "  ----> harbor run -p $step_dir"
    if ! uvx --with boto3 harbor run \
         -p "$step_dir" \
         --environment-import-path "$ENV_IMPORT" \
         --agent claude-code --model "$ANTHROPIC_MODEL" \
         --ak api_base="$ANTHROPIC_BASE_URL/v1" \
         --ae CLAUDE_CODE_DISABLE_THINKING=1 \
         --environment-build-timeout-multiplier 3.0 \
         --agent-timeout-multiplier 3.0 \
         --verifier-timeout-multiplier 3.0 \
         --max-retries 0 -n 1 --yes; then
      HARBOR_CRASHES=$((HARBOR_CRASHES + 1))
      echo "  harbor run exited non-zero for $step_dir"
    fi
  done < <(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
done < <(
  # rag-deploy-blueprint must run first — it deploys the RAG stack that
  # all other skills test against. Remaining skills run alphabetically.
  deploy_spec="$SKILLS_ROOT/rag-deploy-blueprint/eval/${EVAL_PROFILE}.json"
  [ -f "$deploy_spec" ] && echo "$deploy_spec"
  find "$SKILLS_ROOT" -path "*/eval/${EVAL_PROFILE}.json" \
    ! -path "*/rag-deploy-blueprint/*" | sort
)

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

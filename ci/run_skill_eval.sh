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
# Ingestor server writes to INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT as root.
# Default is ./volumes/ingestor-server (relative to compose file dir = ci/).
# Redirect outside workspace to prevent EACCES on next checkout.
export INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT="${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT:-/tmp/ingestor-server-data}"
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

# ENV_IMPORT default — used by teardown. Overridden per-spec inside the loop.
ENV_IMPORT="envs.local_env:LocalEnvironment"

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
  # Remove root-owned volume dirs using Docker (no sudo needed).
  # Containers write as root; only another root process can delete them.
  # docker run --rm with alpine does the job without requiring sudo on the host.
  for vol_dir in deploy/compose/volumes ci/volumes; do
    if [ -d "$vol_dir" ]; then
      docker run --rm -v "$(pwd)/${vol_dir}:/target" alpine \
        sh -c "rm -rf /target/*" 2>/dev/null || true
      rm -rf "$vol_dir" 2>/dev/null || true
    fi
  done
  rm -rf /tmp/milvus-eval /tmp/ingestor-server-data 2>/dev/null || true
fi
# GPU pre-flight (BrevEnvironment mode) is handled inside brev_env.start()
# — the VM is provisioned fresh per CI run, so there's no prior-state
# cleanup to do from the runner side.

echo "==> Auto-discover all skill eval specs and run Harbor trials"
cd "$SKILL_EVAL_DIR"
mkdir -p jobs
HARBOR_CRASHES=0

# Find every skill that ships a spec for the current profile.
# Adding a new skill with eval/<profile>.json is all that's needed —
# no script changes required.
#
# CHANGED_SKILLS (optional, set by skills-eval.yml on PR runs):
#   comma-separated list of skill names that changed in the PR.
#   When set, only those skills are evaluated (diff-based selection).
#   When empty, all skills run (nightly + manual dispatch).
while IFS= read -r spec_file; do
  SKILL_NAME="$(basename "$(dirname "$(dirname "$spec_file")")")"
  # Diff-based filter: skip skills not in CHANGED_SKILLS (PR runs only)
  if [ -n "${CHANGED_SKILLS:-}" ]; then
    if ! echo ",$CHANGED_SKILLS," | grep -q ",$SKILL_NAME,"; then
      echo "  SKIP  $SKILL_NAME (not in PR diff)"
      continue
    fi
  fi
  # SKILL_DIR is the skill folder containing SKILL.md
  SKILL_DIR="$(dirname "$(dirname "$spec_file")")"
  DATASETS_DIR="$SKILL_EVAL_DIR/datasets/$SKILL_NAME"

  # Route per-spec: read platforms[0] from the spec JSON.
  # cpu  → LocalEnvironment  (runner deploys Docker directly, no Brev VM)
  # H100_x2 → BrevEnvironment (pre-provision ephemeral H100 Brev VM)
  SPEC_PLATFORM=$(python3 -c "
import json, sys
spec = json.load(open('$spec_file'))
print(spec.get('platforms', ['cpu'])[0])
" 2>/dev/null || echo "cpu")

  case "$SPEC_PLATFORM" in
    cpu)
      # Use rag-eval-target (existing warm n2d-standard-4 CPU VM) via
      # BrevEnvironment — keeps Docker off the runner itself, avoids
      # root-owned volume accumulation on the runner machine.
      SPEC_ENV_IMPORT="envs.brev_env:BrevEnvironment"
      SPEC_TIMEOUT_MULT="1.5"
      export BREV_INSTANCE="rag-eval-target"
      # Verify rag-eval-target is RUNNING+READY before handing off to harbor
      STATE=$(brev ls 2>/dev/null | awk -v n="rag-eval-target" '$1==n {print $2"+"$4}')
      if [ "$STATE" != "RUNNING+READY" ]; then
        echo "  WARN  rag-eval-target is $STATE — waiting up to 10 min"
        DEADLINE=$(( $(date +%s) + 600 ))
        while [ "$(date +%s)" -lt "$DEADLINE" ]; do
          STATE=$(brev ls 2>/dev/null | awk -v n="rag-eval-target" '$1==n {print $2"+"$4}')
          [ "$STATE" = "RUNNING+READY" ] && break
          sleep 15
        done
      fi
      echo "  rag-eval-target: $STATE"
      ;;
    H100_x2|h100*)
      SPEC_ENV_IMPORT="envs.brev_env:BrevEnvironment"
      SPEC_TIMEOUT_MULT="3.0"
      # Ensure GPU Milvus is in place (restore if previously swapped)
      [ -f deploy/compose/vectordb.yaml.gpu-bak ] && \
        mv deploy/compose/vectordb.yaml.gpu-bak deploy/compose/vectordb.yaml || true
      # Pre-provision Brev VM for this GPU spec
      BREV_TYPE=$(python3 -c "
import json, sys
spec = json.load(open('$spec_file'))
plats = (spec.get('resources') or {}).get('platforms') or {}
p = spec.get('platforms', [])[0] if spec.get('platforms') else ''
print(plats.get(p, {}).get('brev_type', 'dmz.h100x2.pcie'))
" 2>/dev/null || echo "dmz.h100x2.pcie")
      export BREV_INSTANCE="rag-eval-gpu-$(date +%s | tail -c 8)"
      # Fallback chain per VSS AGENTS.md — tries each type in order if
      # the primary is at capacity. brev create --type accepts comma-separated
      # fallback list natively.
      BREV_FALLBACKS="${BREV_TYPE},scaleway_H100x2,gpu-h100-sxm.1gpu-16vcpu-200gb"
      echo "==> Pre-provisioning $BREV_INSTANCE (trying: $BREV_FALLBACKS) for $SKILL_NAME"
      for attempt in $(seq 1 5); do
        brev create "$BREV_INSTANCE" --type "$BREV_FALLBACKS" --detached 2>&1 | tail -5
        brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {found=1} END{exit !found}' && break
        sleep 15
      done
      DEADLINE=$(( $(date +%s) + 1800 ))
      last_state=""
      while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        STATE=$(brev ls 2>/dev/null | awk -v n="$BREV_INSTANCE" '$1==n {print $2"+"$4}')
        if [ -n "$STATE" ] && [ "$STATE" != "$last_state" ]; then
          echo "  $(date -u +%H:%M:%SZ) $BREV_INSTANCE: $STATE"
          last_state="$STATE"
        fi
        [ "$STATE" = "RUNNING+READY" ] && break
        sleep 15
      done
      if [ "$last_state" != "RUNNING+READY" ]; then
        echo "Pre-provision timed out — last state: ${last_state:-unknown}"
        exit 1
      fi
      echo "==> $BREV_INSTANCE ready"
      mkdir -p /tmp/brev
      echo "$BREV_INSTANCE" >> "/tmp/brev/started-by-${GITHUB_RUN_ID:-local}.txt"
      ;;
    *)
      echo "  WARN  Unknown platform '$SPEC_PLATFORM' in $spec_file — skipping"
      continue
      ;;
  esac

  echo ""
  echo "==> [$SKILL_NAME] platform=$SPEC_PLATFORM env=$SPEC_ENV_IMPORT"
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
         --environment-import-path "$SPEC_ENV_IMPORT" \
         --agent claude-code --model "$ANTHROPIC_MODEL" \
         --ak api_base="$ANTHROPIC_BASE_URL/v1" \
         --ae CLAUDE_CODE_DISABLE_THINKING=1 \
         --environment-build-timeout-multiplier "$SPEC_TIMEOUT_MULT" \
         --agent-timeout-multiplier "$SPEC_TIMEOUT_MULT" \
         --verifier-timeout-multiplier "$SPEC_TIMEOUT_MULT" \
         --max-retries 0 -n 1 --yes; then
      HARBOR_CRASHES=$((HARBOR_CRASHES + 1))
      echo "  harbor run exited non-zero for $step_dir"
    fi
  done < <(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
  # Restore vectordb.yaml if it was swapped for this cpu spec
  [ -f deploy/compose/vectordb.yaml.gpu-bak ] && \
    mv deploy/compose/vectordb.yaml.gpu-bak deploy/compose/vectordb.yaml || true

done < <(
  # Discover specs under skills/. Platform routing (cpu/gpu) is
  # determined per-spec from platforms[].
  # EVAL_PROFILE (optional): if set, only discover specs whose filename
  # stem matches (e.g. EVAL_PROFILE=h100 → only h100.json specs).
  # When unset, all specs run; cpu sorts before gpu.
  _profile_filter="${EVAL_PROFILE:-}"
  if [ -n "$_profile_filter" ]; then
    find "$REPO_ROOT/skills" \
      -path "*/eval/${_profile_filter}.json" 2>/dev/null | sort
  else
    find "$REPO_ROOT/skills" \
      -path "*/eval/*.json" 2>/dev/null \
    | python3 -c "
import sys, json
files = sys.stdin.read().splitlines()
def platform_key(f):
    try:
        p = json.load(open(f)).get('platforms', ['cpu'])[0]
        return (0 if p == 'cpu' else 1, f)
    except Exception:
        return (0, f)
for f in sorted(files, key=platform_key):
    print(f)
"
  fi
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
  for vol_dir in deploy/compose/volumes ci/volumes; do
    if [ -d "$vol_dir" ]; then
      docker run --rm -v "$(pwd)/${vol_dir}:/target" alpine \
        sh -c "rm -rf /target/*" 2>/dev/null || true
      rm -rf "$vol_dir" 2>/dev/null || true
    fi
  done
  rm -rf /tmp/milvus-eval /tmp/ingestor-server-data 2>/dev/null || true
  # Restore original vectordb.yaml (was swapped for cpu variant at start)
  if [ -f deploy/compose/vectordb.yaml.gpu-bak ]; then
    mv deploy/compose/vectordb.yaml.gpu-bak deploy/compose/vectordb.yaml
  fi
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

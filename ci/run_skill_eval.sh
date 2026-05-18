#!/usr/bin/env bash
# NV-BASE Tier 1 + Tier 3 skill eval for all focused rag-* skills.
#
# Invoked by .github/workflows/run-branch-script.yml on the self-hosted runner.
# Expects these env vars from the workflow:
#   NVIDIA_INFERENCE_KEY    sk-... NV inference proxy key
#   ANTHROPIC_API_KEY       same as above
#   NGC_API_KEY             nvapi-... NGC key for docker login
#   CLAUDE_CODE_DISABLE_THINKING=1
#
# Skills are evaluated in dependency order:
#   1. rag-deploy-blueprint      (deploys RAG stack — no GPU, must run first)
#   2. rag-ingest-documents      (needs running stack)
#   3. rag-query-knowledge       (needs running stack + ingested docs)
#   4. rag-troubleshoot-blueprint (needs running stack)
#   5. rag-configure-retrieval   (needs running stack)
#   6. rag-enable-guardrails     (needs running stack)
#   7. rag-manage-mcp            (needs running stack)
#   8. rag-configure-infrastructure (needs running stack)
#
# GPU skills (rag-enable-vlm, rag-evaluate-quality) run via separate Brev job.
# Output: ./eval-results/<skill>/ uploaded as artifact.

set -euo pipefail

# No-GPU skills in dependency order — deploy must be first
NO_GPU_SKILLS=(
  "rag-deploy-blueprint"
  "rag-ingest-documents"
  "rag-query-knowledge"
  "rag-troubleshoot-blueprint"
  "rag-configure-retrieval"
  "rag-enable-guardrails"
  "rag-manage-mcp"
  "rag-configure-infrastructure"
)

RESULTS_DIR="./eval-results"
LOGS_DIR="./ci-logs"
mkdir -p "$LOGS_DIR"

# Fix root-owned dirs left by previous runs (Milvus volumes, Harbor job dirs)
# Without this, the runner can't clean the workspace on the next run
sudo rm -rf deploy/compose/volumes/ deploy/compose/src/ 2>/dev/null || true
sudo rm -rf skills/*/evals/results/ 2>/dev/null || true

echo "==> Probe runner environment"
probe_ok=true
check() {
  if eval "$2" >/dev/null 2>&1; then
    echo "  [OK]    $1: $(eval "$2" 2>&1 | head -1)"
  else
    echo "  [MISS]  $1: not found"
    probe_ok=false
  fi
}
check "python3"        "python3 --version"
check "docker"         "docker --version"
check "docker compose" "docker compose version"
check "node"           "node --version"
check "npm"            "npm --version"
echo "  [ENV]   NVIDIA_INFERENCE_KEY: ${NVIDIA_INFERENCE_KEY:+set (${NVIDIA_INFERENCE_KEY:0:4}...)}${NVIDIA_INFERENCE_KEY:-NOT SET}"
echo "  [ENV]   NGC_API_KEY:          ${NGC_API_KEY:+set (${NGC_API_KEY:0:4}...)}${NGC_API_KEY:-NOT SET}"
echo "  [ENV]   ANTHROPIC_API_KEY:    ${ANTHROPIC_API_KEY:+set}${ANTHROPIC_API_KEY:-NOT SET}"
echo "  [ENV]   Skills to evaluate:   ${NO_GPU_SKILLS[*]}"
if [ "$probe_ok" = false ]; then
  echo "  [WARN]  Some dependencies missing — will attempt to install below"
fi
echo ""

echo "==> Required env check"
: "${NVIDIA_INFERENCE_KEY:?Set NVIDIA_INFERENCE_KEY (sk- proxy key) in repo secrets as NVBASE_INFERENCE_API_KEY}"
: "${NGC_API_KEY:?Set NGC_API_KEY in repo secrets}"
export CLAUDE_CODE_DISABLE_THINKING="${CLAUDE_CODE_DISABLE_THINKING:-1}"

echo "==> Install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.local/share/uv/tools/nv-base/bin:$PATH"

echo "==> Install NV-BASE"
uv tool install nv-base \
  --index https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple
nv-base --version
nv-base health-check

echo "==> Update astra-skill-eval to latest"
curl -fsSL https://urm.nvidia.com/artifactory/it-automation-generic/astra-skill-eval/latest/install.sh | bash || true
astra-skill-eval --version || true

echo "==> Install Node.js + Claude Code CLI"
if ! command -v npm >/dev/null 2>&1; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  nvm install 20
  nvm use 20
fi
if ! command -v claude >/dev/null 2>&1; then
  npm install -g @anthropic-ai/claude-code
fi
claude --version

echo "==> Apply CLAUDE_CODE_DISABLE_THINKING patch (KI-001)"
python3 ci/patch_nvbase.py

echo "==> Docker credentials for nvcr.io"
export DOCKER_CONFIG=$(mktemp -d)
NVCR_AUTH=$(printf '$oauthtoken:%s' "$NGC_API_KEY" | base64 -w 0)
printf '{"auths":{"nvcr.io":{"auth":"%s"}}}' "$NVCR_AUTH" > "$DOCKER_CONFIG/config.json"
echo "  Docker credentials written"

echo "==> Clean any existing Docker state"
docker ps -a --format '{{.ID}}' | xargs -r docker stop >/dev/null 2>&1 || true
docker ps -a --format '{{.ID}}' | xargs -r docker rm   >/dev/null 2>&1 || true

export ANTHROPIC_BASE_URL="https://inference-api.nvidia.com/v1"

# ============================================================
# TIER 1 — Validate all no-GPU skills upfront (fast, ~30s)
# ============================================================
echo ""
echo "==> NV-BASE Tier 1 — validating all ${#NO_GPU_SKILLS[@]} skills"
tier1_failed=false
for skill in "${NO_GPU_SKILLS[@]}"; do
  echo "  Checking: $skill"
  nv-base skills-check "./skills/$skill" >/dev/null 2>&1 && \
    echo "  [PASS] $skill" || \
    { echo "  [FAIL] $skill — Tier 1 schema error"; tier1_failed=true; }
done
if [ "$tier1_failed" = true ]; then
  echo "Tier 1 failed for one or more skills — fix schema errors before Tier 3"
  exit 1
fi
echo "  All Tier 1 checks passed"

# ============================================================
# TIER 3 — Evaluate skills in dependency order
# Docker state persists between skills (--env-mode local)
# ============================================================
echo ""
echo "==> NV-BASE Tier 3 — evaluating ${#NO_GPU_SKILLS[@]} skills in dependency order"

tier3_results=()

for skill in "${NO_GPU_SKILLS[@]}"; do
  skill_dir="./skills/$skill"
  skill_results="$RESULTS_DIR/$skill"
  mkdir -p "$skill_results"

  echo ""
  echo "  ---- Evaluating: $skill ----"

  # Stream Harbor agent logs to stdout while eval runs
  (
    HARBOR_RESULTS="$skill_dir/evals/results"
    while true; do
      sleep 5
      find "$HARBOR_RESULTS" -name "claude-code.txt" 2>/dev/null | while read -r f; do
        tail -n +1 "$f" 2>/dev/null | sed "s/^/  [harbor] /"
      done
    done
  ) &
  LOG_STREAMER_PID=$!

  nv-base agent-eval \
    --env-mode local \
    -a claude-code \
    --agent-model "claude-code=aws/anthropic/bedrock-claude-sonnet-4-6" \
    --skip-baseline \
    --timeout-multiplier 3 \
    -k 1 \
    "$skill_dir" \
    -r html,json \
    -o "$skill_results" && \
    tier3_results+=("PASS:$skill") || \
    tier3_results+=("FAIL:$skill")

  # Stop log streamer
  kill "$LOG_STREAMER_PID" 2>/dev/null || true
done

# ============================================================
# COLLECT DOCKER LOGS
# ============================================================
echo ""
echo "==> Collect Docker logs"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" > "$LOGS_DIR/docker-ps.log" 2>&1 || true
for container in rag-server ingestor-server milvus-standalone milvus-etcd milvus-minio; do
  docker logs "$container" > "$LOGS_DIR/${container}.log" 2>&1 || true
done

# Stop containers FIRST then remove all root-owned dirs before artifact upload
docker compose -f deploy/compose/docker-compose-rag-server.yaml down -v --remove-orphans 2>/dev/null || true
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down -v --remove-orphans 2>/dev/null || true
docker compose -f deploy/compose/vectordb.yaml down -v --remove-orphans 2>/dev/null || true
sleep 5
# Force-remove root-owned bind-mount volumes (etcd/minio write as root)
sudo rm -rf deploy/compose/volumes/ 2>/dev/null || true
# Remove Harbor job dirs inside evals/results — also contain root-owned Milvus volumes
# The artifact upload **/evals/results/ glob scans these and hits EACCES
find skills -path "*/evals/results" -type d -exec sudo rm -rf {} + 2>/dev/null || true

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "==> Eval summary"
all_passed=true
for result in "${tier3_results[@]}"; do
  status="${result%%:*}"
  skill="${result##*:}"
  echo "  [$status] $skill"
  [ "$status" = "FAIL" ] && all_passed=false
done

if [ "$all_passed" = false ]; then
  echo ""
  echo "One or more skills failed Tier 3 evaluation"
  exit 1
fi

echo ""
echo "==> All evals complete"

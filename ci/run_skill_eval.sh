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

# Fix root-owned dirs left by previous runs using Docker (no sudo needed)
# etcd/minio containers write as root — Docker alpine can remove what sudo can't
if [ -d "deploy/compose/volumes" ]; then
  docker run --rm -v "$(pwd)/deploy/compose/volumes:/target" alpine \
    sh -c "rm -rf /target/*" 2>/dev/null || true
  rm -rf deploy/compose/volumes/ 2>/dev/null || true
fi
if find skills -path "*/evals/results" -mindepth 3 -maxdepth 3 -type d -quit 2>/dev/null; then
  docker run --rm -v "$(pwd)/skills:/target" alpine \
    sh -c "find /target -path '*/evals/results' -type d -exec rm -rf {} + 2>/dev/null; exit 0" \
    2>/dev/null || true
fi

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
# Pin to 0.7.5: harbor 0.7.0 (shipped in 0.7.6) collapsed VerifierResult from 8 explicit
# fields to a single rewards dict, but the agent Docker image still writes result.json in
# the old flat-field format. TrialResult.model_validate_json() fails with 8 validation
# errors on every trial. Unpin once the Docker image is updated to harbor 0.7.0+.
curl -fsSL https://urm.nvidia.com/artifactory/it-automation-generic/astra-skill-eval/0.7.5/install.sh | bash || true
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
# PRE-DEPLOY — bring up RAG stack in NVIDIA-hosted mode
# All skills 2-8 require a running stack. Deploy once here
# so every eval runs against a known-good environment.
# ============================================================
echo ""
echo "==> Pre-deploy RAG stack (NVIDIA-hosted mode)"

# Source nvdev.env so all compose files pick up cloud NIM endpoints
set -a
source deploy/compose/nvdev.env
set +a

# Disable GPU-backed vector search (not available on CPU runner)
export APP_VECTORSTORE_ENABLEGPUSEARCH=False
export APP_VECTORSTORE_ENABLEGPUINDEX=False
export MILVUS_VERSION="${MILVUS_VERSION:-v2.6.5}"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-/tmp/nvbase-milvus}"
export INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT="${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT:-/tmp/nvbase-ingestor}"

echo "  Starting vector DB..."
docker compose -f deploy/compose/vectordb.yaml up -d
echo "  Starting ingestor server..."
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
echo "  Starting RAG server..."
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d

echo "  Waiting for services to be healthy..."
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  RAG_OK=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:8081/v1/health 2>/dev/null || echo "0")
  ING_OK=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:8082/v1/health 2>/dev/null || echo "0")
  if [ "$RAG_OK" = "200" ] && [ "$ING_OK" = "200" ]; then
    echo "  [OK] RAG server (8081) and ingestor (8082) are healthy"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  echo "  Waiting... (${ELAPSED}s) rag=${RAG_OK} ingestor=${ING_OK}"
done

if [ "$RAG_OK" != "200" ] || [ "$ING_OK" != "200" ]; then
  echo "  [WARN] RAG stack not fully healthy after ${TIMEOUT}s — continuing anyway"
  docker ps
fi

echo "  Pre-deploy complete. Running containers:"
docker ps --format "  {{.Names}}\t{{.Status}}"

# Use sonnet for the LLM-as-judge to avoid saturating the opus endpoint.
# The agent model (sonnet, set via --agent-model) and the judge model are
# separate — --agent-model does not affect the judge. Opus is the hardcoded
# default in layer2/eval_core/llm_judge.py and runs out of capacity when
# 8 skills × 4 eval cases are judged back-to-back, causing 429s that
# default accuracy/goal_accuracy scores to 0 and produce false failures.
export LLM_JUDGE_MODEL="aws/anthropic/bedrock-claude-sonnet-4-6"

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
# Force-remove root-owned bind-mount volumes using Docker (no sudo needed)
# etcd and minio containers write as root — Docker can remove what sudo can't
if [ -d "deploy/compose/volumes" ]; then
  docker run --rm -v "$(pwd)/deploy/compose/volumes:/target" alpine \
    sh -c "rm -rf /target/*" 2>/dev/null || true
  rm -rf deploy/compose/volumes/ 2>/dev/null || true
fi
# Remove Harbor job dirs that also contain root-owned Milvus volumes
if find skills -path "*/evals/results" -type d -quit 2>/dev/null; then
  docker run --rm -v "$(pwd)/skills:/target" alpine \
    sh -c "find /target -path '*/evals/results' -type d -exec rm -rf {} + 2>/dev/null; exit 0" 2>/dev/null || true
fi

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

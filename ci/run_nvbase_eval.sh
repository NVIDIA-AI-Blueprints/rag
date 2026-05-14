#!/usr/bin/env bash
# Runs the NV-BASE Tier 3 skill eval for the rag-blueprint skill.
#
# Invoked by .github/workflows/run-branch-script.yml on the self-hosted
# runner. Expects these env vars from the workflow:
#   NVIDIA_INFERENCE_KEY    sk-... NV inference proxy key
#   ANTHROPIC_API_KEY       same as above
#   NGC_API_KEY             nvapi-... NGC key for docker login
#   CLAUDE_CODE_DISABLE_THINKING=1
#
# Output: ./eval-results/  (uploaded by the workflow as an artifact)

set -euo pipefail

SKILL_DIR="./skill-source/.agents/skills/rag-blueprint"
RESULTS_DIR="./eval-results"
LOGS_DIR="./ci-logs"

mkdir -p "$LOGS_DIR"

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

echo "==> Install Node.js + Claude Code CLI"
if ! command -v npm >/dev/null 2>&1; then
  # Use nvm — no sudo required, installs to user home
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  # shellcheck source=/dev/null
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

echo "==> Docker login to nvcr.io"
# Use a temp Docker config dir so we don't touch the runner's existing credential helper
export DOCKER_CONFIG=$(mktemp -d)
echo '{"credsStore":""}' > "$DOCKER_CONFIG/config.json"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "==> Strip allowed-tools from SKILL.md for headless eval (RAG-023)"
sed -i '/^allowed-tools:/d' "$SKILL_DIR/SKILL.md"

echo "==> Clean any existing Docker state"
docker ps -a --format '{{.ID}}' | xargs -r docker stop >/dev/null 2>&1 || true
docker ps -a --format '{{.ID}}' | xargs -r docker rm   >/dev/null 2>&1 || true

echo "==> Run NV-BASE Tier 3 eval"
nv-base agent-eval \
  --env-mode local \
  -a claude-code \
  --skip-baseline \
  -k 1 \
  "$SKILL_DIR" \
  -r html,json \
  -o "$RESULTS_DIR"

echo "==> Collect Docker logs"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" > "$LOGS_DIR/docker-ps.log" 2>&1 || true
for container in rag-server ingestor-server milvus-standalone milvus-etcd milvus-minio; do
  docker logs "$container" > "$LOGS_DIR/${container}.log" 2>&1 || true
done

echo "==> Eval complete"

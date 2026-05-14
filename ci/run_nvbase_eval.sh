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

echo "==> Install Claude Code CLI"
if ! command -v claude >/dev/null 2>&1; then
  npm install -g @anthropic-ai/claude-code
fi
claude --version

echo "==> Apply CLAUDE_CODE_DISABLE_THINKING patch (KI-001)"
python3 ci/patch_nvbase.py

echo "==> Docker login to nvcr.io"
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

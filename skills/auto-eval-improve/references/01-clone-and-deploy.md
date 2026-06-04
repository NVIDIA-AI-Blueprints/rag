# Stage 1 — Clone (optional) & deploy the RAG stack

Goal: have a healthy **rag-server**, **ingestor server**, and **vector DB** running, with **agentic mode
enabled** when the run is agentic. Deployment itself is delegated to the **`rag-blueprint`** skill — this
file only covers what is specific to an evaluation run.

## 1.1 Clone the RAG repo (optional — usually skip)

You are normally already inside the RAG Blueprint repo, so **skip cloning**. Clone only if the user
explicitly wants a fresh checkout or you are not in the repo:

```bash
git clone --branch develop https://github.com/NVIDIA-AI-Blueprints/rag.git
cd rag
```

If already in the repo, just confirm the branch is acceptable (the project targets `develop`):

```bash
git rev-parse --abbrev-ref HEAD
```

Do not switch the user's branch without asking.

## 1.2 Decide the deployment mode (cloud vs on-prem) by capacity

Auto-detect GPU capacity — do not ask first:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "NO_GPU"
df -h . | tail -1
```

Pick the mode:

| Condition | Mode | Why |
|-----------|------|-----|
| No GPU, or fewer than ~3 capable GPUs, or low disk | **NVIDIA-hosted (cloud)** | No local NIMs; uses `https://integrate.api.nvidia.com`. Needs only `NGC_API_KEY`/`NVIDIA_API_KEY`. |
| ≥3 × H100 / B200 / RTX PRO 6000 and ~200 GB free disk | **On-prem (self-hosted)** | Local NIMs (LLM, embedding, reranker, extraction) need the GPUs. |

Read `docs/support-matrix.md` and `docs/service-port-gpu-reference.md` for the authoritative per-mode GPU
requirements before committing. State what you detected and the chosen mode, then proceed (ask only if the
choice is genuinely borderline).

## 1.3 Enable agentic mode (for agentic runs)

Agentic RAG has two layers; for an unambiguous agentic eval, set **both**:

- **Server default** — `ENABLE_AGENTIC_RAG=true` on the rag-server (so it routes knowledge-base queries
  through the LangGraph plan-and-execute pipeline by default).
- **Per-request** — the eval command passes `--agentic`, which sets `agentic:true` per request and
  **overrides** the server default. This is what actually guarantees the agentic path during eval.

For Docker Compose, set the env var before bringing up the rag-server. The variable is consumed in
`deploy/compose/docker-compose-rag-server.yaml` as `ENABLE_AGENTIC_RAG: ${ENABLE_AGENTIC_RAG:-false}`:

```bash
export ENABLE_AGENTIC_RAG=true
```

For Helm, set `envVars.ENABLE_AGENTIC_RAG: "true"` in `values.yaml`.

For a **Standard RAG** run, leave `ENABLE_AGENTIC_RAG` unset/false and omit `--agentic` from the eval
command (Stage 4).

Optional agentic tuning (only if the user asks, or as part of Stage 6 recommendations): role LLMs
(`AGENTIC_PLANNER_LLM_*`, `AGENTIC_TASK_LLM_*`, `AGENTIC_SEED_GEN_LLM_*`, `AGENTIC_SYNTHESIS_LLM_*`),
`AGENTIC_VERIFICATION_ENABLED`, `AGENTIC_CONTEXT_MAX_TOKENS`, `AGENTIC_PLANNER_MAX_TASKS`. See
`skills/rag-blueprint/references/configure/agentic-rag.md` and `src/nvidia_rag/utils/agentic_rag_config.py`.

## 1.4 Deploy via the rag-blueprint skill

Hand the actual deploy to the **`rag-blueprint`** skill (`references/deploy.md`) — do not reimplement
deploy logic here. Ensure the export from 1.3 is in the environment the compose `up` runs in. The canonical
bring-up order (Docker) is:

```bash
source deploy/compose/<env-file>          # .env (self-hosted) or nvdev.env (NVIDIA-hosted) — see rag-blueprint
# on-prem only: bring up local NIMs first
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d
docker compose -f deploy/compose/vectordb.yaml up -d
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

The vector DB is **Elasticsearch** by default (port `9200`); Milvus (`19530`) is an opt-in profile. Note
which one is active — Stage 4 must pass the matching `--vdb_endpoint`.

## 1.5 Verify health before evaluating

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | head -30
curl -s "http://localhost:8082/v1/health?check_dependencies=true" | head -1   # ingestor
curl -s "http://localhost:8081/v1/health?check_dependencies=true" | head -1   # rag-server
```

Both must report healthy (and dependencies healthy) before proceeding. If unhealthy, route to
`rag-blueprint` → `references/troubleshoot.md` and resolve before Stage 4.

Confirm the agentic setting actually took effect on the live server:

```bash
docker exec rag-server env 2>/dev/null | grep -E "ENABLE_AGENTIC_RAG"
```

Record for later stages: deployment mode, vector DB type + endpoint, the served LLM model name, and the
LLM endpoint (read from the active env file: `APP_VECTORSTORE_URL`, `APP_LLM_MODELNAME`, `APP_LLM_SERVERURL`).

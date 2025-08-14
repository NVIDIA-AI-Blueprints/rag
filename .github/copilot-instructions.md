# Copilot instructions for this repo

This repo is the NVIDIA RAG Blueprint: a multi-service RAG stack you can run via Docker Compose/Helm or as a Python package.

## Big picture

**Services (FastAPI):**
- **rag-server** (`src/nvidia_rag/rag_server`): retrieval → optional rerank → LLM generation; endpoints under `/v1`
- **ingestor-server** (`src/nvidia_rag/ingestor_server`): document ingestion, collections, summaries via NV-Ingest
- **External systems**: Milvus (vector DB, GPU-accelerated by cuVS; CPU fallback), MinIO (multimodal assets), NIMs (LLM/embeddings/reranker/VLM/Guardrails) local or hosted

**Flow**: upload → NV-Ingest extracts (text/tables/charts/images) → embed → Milvus; query → retrieve (multi-collection) → optional rerank → LLM → streamed answer + optional citations.

**Frontend**: Next.js RAG playground (`frontend/`) with TypeScript, Tailwind CSS

## Run/dev workflows

**Compose deployment:** 
- Core files: `deploy/compose/vectordb.yaml`, `docker-compose-ingestor-server.yaml`, `docker-compose-rag-server.yaml`, `nims.yaml`
- Switch on-prem vs hosted NIMs via `deploy/compose/.env`; `NGC_API_KEY` required for hosted
- First on-prem LLM start downloads model (slow cold start)
- Rebuild images after code changes: `docker compose up --build <service>`
- Health check: `GET http://<host>:8081/v1/health?check_dependencies=true`

**Python development:**
- Package structure: `src/nvidia_rag` with `ingestor_server/`, `rag_server/`, `utils/`, `observability/`
- Install: `pip install -e .[all]` for development
- Client examples: `docs/python-client.md` and `notebooks/`

## API surface and patterns

**rag-server** (`rag_server/server.py`):
- `/v1/generate` (OpenAI-compatible alias `/v1/chat/completions`) and `/v1/search`
- Prompt model enforces roles (system optional first; last must be user)
- Key fields: `use_knowledge_base`, `collection_names`, `enable_query_rewriting`, `enable_reranker`, `enable_vlm_inference`, `enable_citations`, `temperature`/`top_p`/`max_tokens`; `filter_expr` supported

**ingestor-server** (`ingestor_server/server.py`):
- `/v1/documents` (upload/update; async `task_id` + `/v1/status`), `/v1/documents` (list/delete), `/v1/collections` (list/create/delete)

## Configuration conventions

**Central config:** `nvidia_rag.utils.configuration`; loaded via `utils.common.get_config()` and env vars

**Important envs (set in compose):**
- Models: `APP_LLM_SERVERURL`/`APP_LLM_MODELNAME`, `APP_EMBEDDINGS_SERVERURL`/`APP_EMBEDDINGS_MODELNAME`, `APP_RANKING_SERVERURL`/`APP_RANKING_MODELNAME`, optional `APP_VLM_SERVERURL`/`APP_VLM_MODELNAME`
- Retrieval: `VECTOR_DB_TOPK`, `APP_RETRIEVER_TOPK`, `COLLECTION_NAME`
- Milvus GPU: `APP_VECTORSTORE_ENABLEGPUSEARCH`/`ENABLEGPUINDEX` (set False on B200/A100 for best accuracy)
- Features: `ENABLE_RERANKER`, `ENABLE_QUERYREWRITER`, `ENABLE_CITATIONS`, `ENABLE_GUARDRAILS`, `ENABLE_VLM_INFERENCE`, `ENABLE_NEMOTRON_THINKING`
- Reflection: `ENABLE_REFLECTION` + thresholds/model in `docker-compose-rag-server.yaml`

**Prompt customization:** Mount `PROMPT_CONFIG_FILE` (defaults to `src/nvidia_rag/rag_server/prompt.yaml`)

## Implementation patterns

**Use internal helpers:** `get_llm`, `get_embedding_model`, `get_ranking_model`, `create_vectorstore_langchain`; sanitize NIM URLs via `utils.common.sanitize_nim_url`

**Code conventions:**
- Strict Pydantic validators (e.g., Prompt message role order); keep schemas backward-compatible
- Multi-collection retrieval via `collection_names` (see `MAX_COLLECTION_NAMES` in `rag_server/main.py`)
- Configuration wizard pattern in `utils/configuration_wizard.py` for structured config classes
- OpenTelemetry instrumentation throughout (`observability/` modules)

**Frontend patterns:**
- Next.js envs (`NEXT_PUBLIC_*`) require rebuilding rag-playground if changed
- TypeScript with strict types, Tailwind for styling

## Debugging/observability

**Health checks first:** Use `/v1/health?check_dependencies=true` endpoint

**Tracing/metrics:** Enable with `APP_TRACING_ENABLED` and OTLP endpoints; Prometheus/Zipkin configs under `deploy/config/` and `docs/observability.md`

**Common pitfalls:**
- Cold start for on-prem LLM downloads
- GPU IDs in `deploy/compose/.env` 
- Cloud NIM rate limits for large ingests
- B200 GPU limitations (see README warning about unsupported features)

## Key files/directories

**Servers:** `src/nvidia_rag/rag_server/*`, `src/nvidia_rag/ingestor_server/*`
**Utilities:** `src/nvidia_rag/utils/*`
**Compose:** `deploy/compose/*.yaml` (+ `.env`, `accuracy_profile.env`, `perf_profile.env`)
**Docs:** `docs/quickstart.md`, `docs/python-client.md`, `docs/*`
**Examples:** `notebooks/`, `data/multimodal/`

If any workflow or toggle is unclear (e.g., adding endpoints, local non-Docker development), request an update and we'll refine this file.

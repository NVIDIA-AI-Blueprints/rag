# NVIDIA RAG Blueprint

Reference implementation for a Retrieval Augmented Generation pipeline. Python 3.11+ backend (FastAPI + LangChain), React/TypeScript frontend, deployable via Docker Compose or Helm.

## Project structure

```
src/nvidia_rag/
├── rag_server/        # RAG query/response server (FastAPI)
├── ingestor_server/   # Document ingestion server (FastAPI)
└── utils/             # Shared utilities
frontend/              # React + TypeScript UI (pnpm)
deploy/
├── compose/           # Docker Compose files and env configs
└── helm/              # Helm charts (standard + MIG-slicing)
docs/                  # User-facing documentation (Sphinx, RST/MD)
tests/
├── unit/              # No network calls allowed
└── integration/       # Network calls permitted
notebooks/             # Jupyter notebooks for evaluation and examples
```

## Development commands

### Backend (Python)

```bash
uv sync                              # Install all deps
uv run pytest tests/unit/            # Unit tests
uv run pytest tests/integration/     # Integration tests
ruff check --fix src/                # Lint + autofix
ruff format src/                     # Format
pre-commit run --all-files           # Run all pre-commit hooks
```

### Frontend (TypeScript)

```bash
cd frontend
pnpm install
pnpm run dev                         # Dev server
pnpm run lint                        # ESLint
pnpm exec tsc --noEmit               # Type check
pnpm run test:run                    # Tests
```

## Code conventions

- **Python**: Ruff for linting and formatting (line-length 88, double quotes, space indent). Config in `pyproject.toml`.
- **Type hints**: Required on all function signatures.
- **Imports**: Sorted by isort via Ruff. No in-function imports.
- **Tests**: Mirror source tree (`src/nvidia_rag/rag_server/server.py` → `tests/unit/rag_server/test_server.py`).
- **Frontend**: ESLint + TypeScript strict mode. Function components with hooks.
- **Env files**: `deploy/compose/nvdev.env` (NVIDIA-hosted NIMs) and `deploy/compose/.env` (self-hosted). These are the source of truth for Docker deployments — shell-only exports are lost on restart.

## Deployment modes

1. **Docker Compose** — `deploy/compose/` with env-file configs. Multiple profiles: standard, retrieval-only, NVIDIA-hosted.
2. **Helm** — `deploy/helm/nvidia-blueprint-rag/` chart with `values.yaml`. Supports MIG GPU slicing via `deploy/helm/mig-slicing/`.
3. **Library** — Import `nvidia_rag` as a Python package for custom pipelines.

## Key files

- `pyproject.toml` — All Python deps, ruff config, project metadata
- `deploy/compose/nvdev.env` — Default env file for NVIDIA API Catalog deployments
- `src/nvidia_rag/rag_server/prompt.yaml` — System prompt templates
- `docs/support-matrix.md` — GPU requirements per deployment mode
- `docs/service-port-gpu-reference.md` — Port mappings and GPU assignments

## PR and commit guidelines

- Target the `develop` branch, never `main`.
- All commits must be signed off (DCO).
- Run `pre-commit run --all-files` before submitting.
- See `CONTRIBUTING.md` for full workflow.

## Operations — `/rag-blueprint` skill

For any operational task, use the `rag-blueprint` skill (`.agents/skills/rag-blueprint/`).

- **Deploy** — Docker Compose (standard, retrieval-only, NVIDIA-hosted), Helm, MIG-slicing, library mode
- **Configure** — VLM, guardrails, query rewriting, ingestion, search & retrieval, models, observability, summarization, multimodal, MCP, evaluation, notebooks, UI, and more
- **Troubleshoot** — Debug unhealthy services, container errors, GPU issues, connectivity failures
- **Shutdown** — Stop, tear down, and clean up services

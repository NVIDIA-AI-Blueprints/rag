# RAG Blueprint Agent Skill

A single agent skill that enables AI coding assistants (Claude Code, Cursor, Codex, etc.) to deploy, configure, troubleshoot, and manage the NVIDIA RAG Blueprint autonomously.

## Installation

```bash
npx skills add .
```

Select **rag-blueprint** — it includes all capabilities (deploy, configure, shutdown, troubleshoot) in one skill.

## Architecture: Skills = Process, Docs = Truth

```
SKILL.md           = ROUTER (intent detection, autonomy rules, configure routing table)
Reference files    = WHAT/HOW (deployment workflows, feature playbooks, diagnostics)
docs/*.md          = SOURCE OF TRUTH (never copied into skills)
notebooks/*.ipynb  = RUNNABLE EXAMPLES (referenced from relevant skills)
```

The SKILL.md detects user intent and routes to the correct reference file. Reference files are concise playbooks that point to `docs/*.md` for detailed configuration — this prevents staleness from duplicated content.

## Skill Structure

```
skill-source/.agents/skills/rag-blueprint/
  SKILL.md                              ← Single entry point (intent router)
  references/
    deploy.md                           ← Deployment: env analysis, NGC key, routing
    deploy/
      docker.md                         ← Docker Compose deployment workflow
      docker-self-hosted.md             ← Self-hosted NIMs (local GPU inference)
      docker-nvidia-hosted.md           ← Cloud NIMs (NVIDIA API endpoints)
      docker-retrieval-only.md          ← Search/retrieve only (no LLM)
      helm.md                           ← Kubernetes / Helm deployment workflow
      helm-standard.md                  ← Standard Helm chart deployment
      helm-mig.md                       ← Multi-Instance GPU deployment
      library.md                        ← Python library mode workflow
      library-full.md                   ← Python API + Docker backend
      library-lite.md                   ← Containerless (Milvus Lite + cloud APIs)
    configure/
      vlm.md                            ← VLM, VLM embeddings, image captioning
      guardrails.md                     ← NeMo Guardrails
      query-and-conversation.md         ← Query rewriting, decomposition, multi-turn
      ingestion.md                      ← Text-only, audio, Nemotron Parse, OCR, batch CLI
      search-and-retrieval.md           ← Hybrid search, multi-collection, metadata, filters
      models-and-infrastructure.md      ← Model changes, vector DB, auth, API keys, profiles
      reasoning-and-generation.md       ← Reasoning, self-reflection, prompts, generation params
      summarization.md                  ← Document summarization during ingestion
      observability.md                  ← Tracing, Zipkin, Grafana, Prometheus
      multimodal-query.md              ← Image + text querying with VLM embeddings
      data-catalog.md                   ← Collection/document metadata management
      user-interface.md                 ← RAG UI settings and usage
      api-reference.md                  ← REST API endpoints and schemas
      evaluation.md                     ← RAGAS quality metrics
      mcp.md                            ← MCP server & client tools
      migration.md                      ← Version upgrade guide
      notebooks.md                      ← Notebook environment and catalog
    shutdown.md                         ← Stop and tear down services
    troubleshoot.md                     ← Diagnose and fix common issues
```

## How It Works

1. User says "deploy RAG" → SKILL.md routes to `references/deploy.md` → env analysis → routes to `deploy/docker.md`, `deploy/helm.md`, or `deploy/library.md`
2. User says "enable VLM" → SKILL.md routes to `references/configure/vlm.md` → reads `docs/vlm.md` for detailed steps
3. User says "RAG is broken" → SKILL.md routes to `references/troubleshoot.md` → auto-triage diagnostic sweep
4. User says "stop RAG" → SKILL.md routes to `references/shutdown.md` → detects and stops all services

## Supported Deployment Modes

Read `docs/support-matrix.md` for current hardware requirements per mode.

| Mode | Docker Required | Description |
|------|-----------------|-------------|
| Docker (self-hosted) | Yes | Full on-prem with local NIM inference |
| Docker (NVIDIA-hosted) | Yes | Cloud APIs for model inference |
| Docker (retrieval-only) | Yes | No LLM, search/retrieve only |
| Helm / Kubernetes | No (K8s) | Production K8s with NIM Operator |
| Library (full) | Yes (backend) | Python API with Docker backend services |
| Library (lite) | No | Milvus Lite + cloud APIs, zero infrastructure |

## NGC_API_KEY Handling

Skills never expose the API key value to the LLM. The approach:

1. Check if `NGC_API_KEY` is set: `[ -n "$NGC_API_KEY" ] && echo "SET" || echo "NOT_SET"`
2. If not set, ask the user to run `export NGC_API_KEY="nvapi-your-key"` in the terminal
3. For `docker login`, the user runs it themselves (the command expands the key)
4. As a fallback, offer to write a placeholder to `deploy/compose/.env` for the user to replace

## Notebook Integration

All 13 notebooks are referenced from relevant reference files:

| Notebook | Referenced In |
|----------|--------------|
| `ingestion_api_usage.ipynb` | `references/configure/ingestion.md` |
| `retriever_api_usage.ipynb` | `references/configure/search-and-retrieval.md` |
| `image_input.ipynb` | `references/configure/vlm.md`, `references/configure/multimodal-query.md` |
| `summarization.ipynb` | `references/configure/summarization.md` |
| `evaluation_01_ragas.ipynb` | `references/configure/evaluation.md` |
| `evaluation_02_recall.ipynb` | `references/configure/evaluation.md` |
| `nb_metadata.ipynb` | `references/configure/search-and-retrieval.md` |
| `rag_library_usage.ipynb` | `references/deploy/library-full.md` |
| `rag_library_lite_usage.ipynb` | `references/deploy/library-lite.md` |
| `building_rag_vdb_operator.ipynb` | `references/configure/models-and-infrastructure.md` |
| `mcp_server_usage.ipynb` | `references/configure/mcp.md` |
| `nat_mcp_integration.ipynb` | `references/configure/mcp.md` |
| `launchable.ipynb` | `SKILL.md` |

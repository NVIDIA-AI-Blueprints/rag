# RAG Skills

Preview agent skills for working with the NVIDIA RAG Blueprint. Each subdirectory is a
self-contained skill that follows the agentskills.io structure with a `SKILL.md`
frontmatter block, focused workflow instructions, references, and starter evals.

This catalog is the target replacement for the early all-in-one
`skill-source/.agents/skills/rag-blueprint` router. It is still in preview until
the compliance checker, OpenClaw manifest, and security review gates are
complete.

## Catalog

| Skill | Description |
|---|---|
| [rag-deploy-blueprint](rag-deploy-blueprint/SKILL.md) | Deploy, start, verify, shut down, or tear down the RAG Blueprint across Docker Compose, Helm, and library modes. |
| [rag-ingest-documents](rag-ingest-documents/SKILL.md) | Ingest documents, folders, mounted volumes, audio/video files, and batch inputs into RAG collections. |
| [rag-query-knowledge](rag-query-knowledge/SKILL.md) | Query RAG collections through the REST API, UI, Python client, or library mode and report grounded answers. |
| [rag-configure-infrastructure](rag-configure-infrastructure/SKILL.md) | Configure LLM, embedding, reranking, OCR, parse, vector DB, API key, model profile, GPU, port, and endpoint settings. |
| [rag-configure-retrieval](rag-configure-retrieval/SKILL.md) | Configure hybrid search, multi-collection retrieval, reranking, metadata filters, and retrieval performance. |
| [rag-enable-vlm](rag-enable-vlm/SKILL.md) | Enable and verify VLM, VLM embeddings, multimodal query, and image captioning workflows. |
| [rag-enable-guardrails](rag-enable-guardrails/SKILL.md) | Enable, configure, and verify NeMo Guardrails for RAG queries and responses. |
| [rag-troubleshoot-blueprint](rag-troubleshoot-blueprint/SKILL.md) | Diagnose unhealthy RAG services, container failures, GPU issues, port conflicts, and ingestion/query failures. |
| [rag-evaluate-quality](rag-evaluate-quality/SKILL.md) | Run RAG quality and recall evaluation workflows, including RAGAS notebooks and benchmark guidance. |
| [rag-manage-mcp](rag-manage-mcp/SKILL.md) | Use and validate the RAG MCP server, client, and NeMo Agent Toolkit integration examples. |

## Install

Preview install uses symlinks. Do not install the legacy `rag-blueprint` skill
beside these focused skills because its broad trigger overlaps the new catalog.

Open this repository in your coding agent and ask it to install the catalog:

> Read `skills/README.md` and every `SKILL.md` under `skills/`. Install each
> skill for this host using symlinks, not copies:
>
> - Claude Code: `~/.claude/skills/<name>/`
> - Codex: `~/.codex/skills/<name>/`
> - Universal hosts: `~/.agents/skills/<name>/`
>
> Skip skills already installed and pointing at this checkout. List the skills
> registered and the target directory used.

Symlinks are recommended because the skills intentionally point back to
repository docs such as `docs/support-matrix.md`, `docs/api-rag.md`, and
`deploy/compose/*.yaml` as the source of truth.

Each skill must first resolve `RAG_REPO_ROOT` to the repository checkout before
opening source docs. If a skill was copied rather than symlinked, set
`RAG_REPO_ROOT` explicitly.

### Legacy route ownership

| Legacy `rag-blueprint` route | Focused skill owner |
|---|---|
| Deployment, shutdown, migration between deployment modes | `rag-deploy-blueprint` |
| Ingestion, text-only, audio, OCR, parse, batch, mounted volumes | `rag-ingest-documents` |
| Querying, multi-turn use, reasoning/generation params, prompt customization | `rag-query-knowledge` |
| Query decomposition, self-reflection, filter generation | `rag-configure-retrieval` |
| Hybrid search, multi-collection, metadata filters, data catalog | `rag-configure-retrieval` with `rag-ingest-documents` for upload metadata |
| LLM, embedding, reranker, vector DB, model profiles, API keys, ports, GPUs | `rag-configure-infrastructure` |
| VLM, VLM embeddings, multimodal query, image captioning | `rag-enable-vlm` |
| Guardrails | `rag-enable-guardrails` |
| Observability, health, debug, service failures | `rag-troubleshoot-blueprint` |
| RAGAS, recall, benchmarks, notebooks | `rag-evaluate-quality` |
| MCP and NeMo Agent Toolkit integration | `rag-manage-mcp` |

## Development Process

- Use [PLAYBOOK.md](PLAYBOOK.md) for naming, structure, security, eval, and
  review rules.
- Use [TRACKER.md](TRACKER.md) to track the remaining work needed to move from
  this scaffold to a publishable RAG skills and OpenClaw package.
- Each shipping skill must keep `SKILL.md` lean, put detailed procedures in
  `references/`, and keep eval cases under `evals/evals.json`.
- Generated review artifacts live outside this catalog under
  `project-artifacts/`. Keep `skills/` source-only so installs and compliance
  checks do not pick up Office lock files or generated binaries.

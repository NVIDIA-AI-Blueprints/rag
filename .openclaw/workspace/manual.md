# RAG OpenClaw Manual

## Routing

| User asks for | Route |
|---|---|
| Deploy, start, stop, tear down | `rag-deploy-blueprint` |
| Upload, ingest, batch ingest | `rag-ingest-documents` |
| Ask documents, search, citations | `rag-query-knowledge` |
| Change LLM, embedding, reranker, vector DB | `rag-configure-infrastructure` |
| Hybrid search, metadata filters, multi-collection | `rag-configure-retrieval` |
| Image query, VLM embeddings, captioning | `rag-enable-vlm` |
| NeMo Guardrails | `rag-enable-guardrails` |
| Broken deployment or failed workflow | `rag-troubleshoot-blueprint` |
| RAGAS, recall, quality metrics | `rag-evaluate-quality` |
| MCP server, client, NAT integration | `rag-manage-mcp` |

## Operating Rules

- Inspect the host and repo before asking the user for details.
- Check key presence without printing key values.
- Ask before destructive cleanup, collection deletion, or volume removal.
- Verify the workflow after changes.
- Report what changed, what was verified, and what remains blocked.

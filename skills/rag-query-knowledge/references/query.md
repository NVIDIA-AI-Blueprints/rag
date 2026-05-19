# Query Reference

## Query Surfaces

| Surface | Use when | Source |
|---|---|---|
| REST API | Testing deployed RAG server behavior | `docs/api-rag.md` |
| UI | User-facing workflow validation | `docs/user-interface.md` |
| Python client | Library or scripted usage | `docs/python-client.md` |
| Notebook | Exploration and examples | `notebooks/retriever_api_usage.ipynb` |

## Decision Table

| User goal | Primary action | Follow-up route |
|---|---|---|
| Answer a question with citations | Query the requested collection(s), inspect returned sources, and report grounded answer. | `rag-configure-retrieval` if sources are empty or irrelevant. |
| Retrieve chunks only | Use search/retriever surface and report chunks, metadata, and scores. | `rag-configure-retrieval` for topK, reranker, or filter tuning. |
| Multi-turn conversation | Read `docs/multiturn.md` and preserve conversation context only for the requested session. | `rag-configure-retrieval` if query rewriting/decomposition settings are needed. |
| Generation parameter change | Read `docs/llm-params.md`; apply request-level params when possible. | `rag-configure-infrastructure` if service-level model config changes are needed. |
| Prompt customization | Read `docs/prompt-customization.md`; do not mutate prompts unless requested. | `rag-configure-infrastructure` for persistent prompt/service config. |

## Validation

- Confirm `http://localhost:8081/v1/health` or the user-provided endpoint is
  reachable.
- Confirm collection names are known or explicitly provided.
- Confirm filters are valid for the active vector database.
- Keep user data in the approved environment.

## Reporting

For a grounded response, include:

- answer
- collection(s) queried
- source citations or retrieved chunk identifiers when returned
- any low-confidence or empty-retrieval caveat
- suggested next query or ingestion check only when useful

## Known Failure Modes

| Symptom | Likely cause | Route |
|---|---|---|
| Empty answer with healthy service | Wrong collection, missing ingestion, strict filter, or low topK. | `rag-query-knowledge` then `rag-configure-retrieval` if config changes are needed. |
| Answer unrelated to documents | Bad retrieval, reranker disabled/misconfigured, stale collection, or query decomposition issue. | `rag-configure-retrieval`. |
| API returns server error | RAG service, model endpoint, vector DB, or payload schema issue. | `rag-troubleshoot-blueprint`. |
| User asks to reveal env or keys | Prompt injection or unsafe operational request. | Refuse/pause; never print secrets. |

# Query Rewriting, Query Decomposition & Multi-Turn

## When to Use
- User wants to enable multi-turn conversations or follow-up questions
- User asks about query rewriting for better retrieval accuracy
- User wants complex multi-hop query decomposition
- User mentions conversation history settings

## Restrictions
- Query rewriting and multi-turn both require `CONVERSATION_HISTORY > 0` — rewriting has no effect when set to 0
- Query decomposition only works with `use_knowledge_base=true`, single-collection only
- Helm query rewriting: only on-prem LLM supported (no cloud)

## Dependencies

`CONVERSATION_HISTORY` is shared by query rewriting and multi-turn. Changing one may affect the other:

| Setting | Depends on | Side effect when changed |
|---------|-----------|--------------------------|
| `ENABLE_QUERYREWRITER` | `CONVERSATION_HISTORY > 0` | Enabling requires conversation history; disabling has no side effects |
| `CONVERSATION_HISTORY` | — | Setting to 0 also disables query rewriting |

## Process

Detect the deployment mode. Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`.

### Query Rewriting
1. Read `docs/multiturn.md` for full configuration
2. Enabling: set `ENABLE_QUERYREWRITER=True`. If `CONVERSATION_HISTORY` is 0, also set it to 5.
3. Disabling: unset or re-comment `ENABLE_QUERYREWRITER`
4. Restart RAG server

### Multi-Turn
1. Read `docs/multiturn.md` for full configuration and API usage
2. Enabling: set `CONVERSATION_HISTORY` > 0 and choose retrieval strategy
3. Disabling: set `CONVERSATION_HISTORY=0`
4. Restart RAG server

### Query Decomposition
1. Read `docs/query_decomposition.md` for algorithm details and limitations
2. Set `ENABLE_QUERY_DECOMPOSITION=true` and `MAX_RECURSION_DEPTH=3`
3. Restart RAG server

## Decision Table

| Goal | Source Doc | Key Settings |
|------|-----------|--------------|
| Multi-turn with best accuracy | `docs/multiturn.md` | `CONVERSATION_HISTORY=5`, `ENABLE_QUERYREWRITER=True` |
| Multi-turn with low latency | `docs/multiturn.md` | `CONVERSATION_HISTORY=5`, `MULTITURN_RETRIEVER_SIMPLE=True` |
| Complex multi-hop queries | `docs/query_decomposition.md` | `ENABLE_QUERY_DECOMPOSITION=true`, `MAX_RECURSION_DEPTH=3` |
| Disable multi-turn (default) | — | `CONVERSATION_HISTORY=0` |

## Agent-Specific Notes
- `MULTITURN_RETRIEVER_SIMPLE` only applies when query rewriting is disabled; if both set, query rewriting takes precedence
- Query rewriting can be toggled per-request via `enable_query_rewriting: true` in POST /generate (no restart) — but `CONVERSATION_HISTORY` must still be > 0
- Default: multi-turn is disabled (`CONVERSATION_HISTORY=0`)
- Query decomposition adds latency — only valuable for multi-hop queries involving multiple entities
- Library mode: configure via `notebooks/config.yaml` instead of env vars

## Notebooks
- `notebooks/retriever_api_usage.ipynb` — RAG retriever API: search and end-to-end query examples

## Source Documentation
- `docs/query_decomposition.md` — decomposition algorithm, when to use/not use, recursion depth
- `docs/multiturn.md` — conversation history, retrieval strategies, API usage, Helm config

# Agentic RAG

## When to Use
- User wants the LangGraph agentic pipeline, planning/execution, multi-hop reasoning, ambiguity handling, or verification.
- User asks about `agentic`, `ENABLE_AGENTIC_RAG`, agentic streaming, stage events, or agentic reasoning traces.

## Restrictions
- Requires `use_knowledge_base=true`; otherwise the agentic path is not applied.
- Higher latency and more LLM calls than standard RAG. Prefer per-request enablement for latency-sensitive deployments.
- The agentic path does not use NeMo Guardrails, Self-Reflection, Query Decomposition, or VLM Inference.
- Query rewriting, multi-turn history, multi-collection retrieval, citations, filter generation, and reranking are supported.
- Verification is single-pass.

## Process
1. Detect deployment mode. Docker: edit the active env file. Helm: edit `values.yaml`. Library/API callers can set request fields directly.
2. Read `docs/agentic-rag.md` for the current architecture, env vars, and limitations.
3. Prefer per-request enablement:
   ```json
   {
     "messages": [{"role": "user", "content": "..."}],
     "use_knowledge_base": true,
     "collection_names": ["..."],
     "agentic": true
   }
   ```
4. For API/library clients that omit `agentic`, set `ENABLE_AGENTIC_RAG=true` and restart the RAG server. In the React UI, also select Pipeline → Agentic because the UI sends an explicit per-request value.
5. Optionally configure role-specific LLMs: `AGENTIC_PLANNER_LLM_*`, `AGENTIC_TASK_LLM_*`, `AGENTIC_SEED_GEN_LLM_*`, `AGENTIC_SYNTHESIS_LLM_*`.
6. Verify with `/v1/generate`: streaming agentic chunks include `event_type`, `stage`, and supplementary `reasoning_content`; final answer text still streams through `content`.

## Decision Table

| Goal | Key Action |
|------|------------|
| Enable only for one query | Set request body `agentic: true` |
| Disable for one query when globally enabled | Set request body `agentic: false` |
| Change deployment default for API clients that omit `agentic` | Set `ENABLE_AGENTIC_RAG=true` or `false` |
| Enable from the RAG UI | Select Pipeline → Agentic; the Standard UI mode sends `agentic: false` |
| Add post-synthesis checking | Set `AGENTIC_VERIFICATION_ENABLED=true` |
| Debug agent stages | Set `AGENTIC_LOG_LEVEL=DEBUG` and inspect streamed `event_type` / `stage` chunks |

## Agent-Specific Notes
- `enable_streaming=true` is the default. Agentic streaming emits stage events (`stage_start`, `stage_end`), intermediate reasoning/output, final answer chunks, agent events, and errors.
- `enable_streaming=false` makes the agent graph finish before returning a full answer chunk; standard RAG always streams.
- The React UI has only Standard and Agentic modes. Standard sends `agentic: false`, so `ENABLE_AGENTIC_RAG=true` alone does not override UI Standard mode.
- In the UI, agentic and standard reasoning traces render in the reasoning panel when the stream includes `reasoning_content`.
- Docker/Helm self-hosted deployments default per-role `SERVERURL` to `nim-llm:8000`; NVIDIA-hosted configs can set the role URLs to the hosted `/v1` endpoint or to `""` where the client uses the default hosted route. API keys inherit `NVIDIA_API_KEY` unless overridden.
- If the result is slow or expensive, use per-request `agentic` instead of a global default, lower `AGENTIC_CONTEXT_MAX_TOKENS`, or leave verification disabled.

## Source Documentation
- `docs/agentic-rag.md` — architecture, API usage, env vars, limitations
- `docs/api-rag.md` — `/v1/generate` request and streaming behavior
- `frontend/src/hooks/useMessageSubmit.ts` — UI request field behavior for `agentic`
- `frontend/src/hooks/useChatStream.ts` and `frontend/src/components/chat/ReasoningPanel.tsx` — reasoning trace rendering

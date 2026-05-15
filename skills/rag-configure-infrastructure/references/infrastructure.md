# Infrastructure Reference

## Source-of-Truth Files

| Deployment | Config source |
|---|---|
| Docker self-hosted | `deploy/compose/.env` plus compose files under `deploy/compose/` |
| Docker NVIDIA-hosted | `deploy/compose/nvdev.env` |
| Helm | values files under `deploy/helm/nvidia-blueprint-rag/` |
| MIG Helm | `deploy/helm/mig-slicing/*.yaml` |
| Library | `notebooks/config.yaml` or caller-provided config |

## Source Docs

- `docs/change-model.md` for LLM, embedding, and reranking changes.
- `docs/model-profiles.md` for model profile choices.
- `docs/change-vectordb.md` and `docs/milvus-configuration.md` for vector DB
  changes.
- `docs/service-port-gpu-reference.md` for ports and GPU assignments.
- `docs/api-key.md` for key setup.
- `docs/llm-params.md` for request-level generation parameters.

## Verification

After model or endpoint changes:

1. Verify service health.
2. Inspect affected container or pod logs if unhealthy.
3. Run a minimal ingestion or query path that exercises the changed component.
4. Confirm no secret values appear in the report.

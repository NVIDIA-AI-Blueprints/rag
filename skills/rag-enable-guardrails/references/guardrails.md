# Guardrails Reference

## Source Docs

- `docs/nemo-guardrails.md` for setup and configuration.
- `docs/query-to-answer-pipeline.md` for where guardrails fit in the flow.
- `docs/support-matrix.md` for deployment constraints.

## Decision Table

| User goal | Action |
|---|---|
| Enable guardrails | Read `docs/nemo-guardrails.md`, identify active deployment mode, update the correct config source, restart affected services, and verify. |
| Validate policy behavior | Run one allowed domain prompt and one disallowed or unsafe prompt. |
| Change policy scope | Identify the policy file/config, confirm expected behavior, and test both pass and block paths. |
| Service is unhealthy | Route to `rag-troubleshoot-blueprint` for logs and service diagnosis. |

## Config Sources

| Deployment | Config source |
|---|---|
| Docker | `deploy/compose/docker-compose-nemo-guardrails.yaml` and active env file |
| Helm | values files under `deploy/helm/nvidia-blueprint-rag/` |
| Library | caller-provided guardrails config or notebook config |

## Verification

Run at least two checks:

1. An allowed domain question that should pass and return a grounded answer.
2. A disallowed, unsafe, or off-topic question that should be blocked or shaped
   by guardrails.

Report both outcomes. A healthy service alone is not enough verification.

## Known Failure Modes

- Guardrails service is up but RAG is not routed through it.
- Policy blocks expected in-domain prompts.
- Unsafe prompts pass because policy files were not loaded or service restart
  did not pick up config.
- Hosted/local model endpoint mismatch.

# Troubleshooting Reference

## Diagnostic Sweep

Run safe, read-only checks first:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
curl -s http://localhost:8081/v1/health?check_dependencies=true 2>/dev/null || true
curl -s http://localhost:8082/v1/health?check_dependencies=true 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true
```

For Kubernetes, use `kubectl get pods`, `kubectl describe`, and sanitized
`kubectl logs` in the relevant namespace.

## Classification

| Symptom | Likely route |
|---|---|
| Containers/pods unhealthy | Deployment or model config |
| Ingest upload returns task failure | `rag-ingest-documents` plus ingestor logs |
| Empty or irrelevant answer | `rag-query-knowledge` or `rag-configure-retrieval` |
| Model endpoint errors | `rag-configure-infrastructure` |
| VLM/image workflow errors | `rag-enable-vlm` |
| Guardrail policy failures | `rag-enable-guardrails` |
| Eval notebook failures | `rag-evaluate-quality` |

## Reporting

Include:

- observed symptom
- evidence from health/logs/config
- root cause when known
- action taken
- verification result
- next blocker if unresolved

Do not include secrets, full keys, or unrelated log dumps.

# QUERIDON Operational Runbook

## On-Call Rotation
The QUERIDON service is owned by the platform-search team. On-call rotates
weekly between three engineers, with handoff every Tuesday at 10:00 PT.
PagerDuty escalation policy `queridon-prod` page on any P1 or P2 alert.

## Common Incident Patterns

### Symptom: p99 latency spike to >800ms
The most common cause is HARMATTAN fusion fall-behind, usually triggered
by a NEMOPHILA dense-vector retrieval slowdown. Diagnostic steps:
1. Check the `nemophila_query_latency_seconds` histogram in Grafana
2. If NEMOPHILA p99 exceeds 200ms, scale NEMOPHILA replicas
3. If MARLOW-search latency is concurrently elevated, check the MARLOW
   warehouse for active long-running queries

### Symptom: empty result sets
Generally indicates an indexing pipeline backlog. Run:
```
kubectl -n queridon-prod logs deploy/indexer --tail=200 | grep -i error
```

If errors mention "Oracle 26ai vector index timeout", the IVF index
maintenance job may be in progress. Pause the indexer briefly and
retry.

### Symptom: 503 from BORROMEAN
The BORROMEAN web application firewall has a strict rate-limit (1200 req/min
per source IP). For known internal IP ranges, file an exception via
the `borromean-allowlist` ticket queue.

## Capacity Headroom
Production runs at approximately 35% headroom in nominal hours. During
"latency-injection Friday" the headroom drops to ~15% by design — this
is intentional chaos engineering to validate degraded-mode handling.

## Deployment Procedure
QUERIDON deploys via Argo CD with the BERESHEET change advisory board
approval gate. Standard rollout: canary 5% → 25% → 100% with 30-min bake
times between stages. P1 hotfixes can use the emergency 1-hour CAB
retrospective approval window.

## Escalation Contacts
- L1 on-call: `queridon-oncall@acme.com` (paged via PagerDuty)
- L2 platform-search lead: `joelle.hernandez@acme.com`
- L3 SVP infra: `marcus.lin@acme.com`

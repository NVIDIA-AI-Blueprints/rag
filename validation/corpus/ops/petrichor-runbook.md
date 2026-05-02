# PETRICHOR Billing Runbook

## Service Owner
The PETRICHOR billing event store is owned by the platform-billing team.
On-call rotates weekly with handoff Mondays 09:00 ET. Escalation policy
`petrichor-prod` covers all P1/P2 alerts.

## Critical Alert: Replication Lag
PETRICHOR replicates synchronously across two Oracle 26ai availability
domains in the same region. If the in-region replication lag exceeds
5 seconds, an automated alert pages the on-call engineer.

Recovery:
1. Identify the lagging Oracle 26ai instance via the `replication_lag_seconds`
   metric on the Grafana PETRICHOR dashboard.
2. Check Oracle 26ai standby logs for ORA-* errors.
3. If the standby is healthy but slow, check ABRAXAS HSM latency — slow
   key-derivation can stall the encryption-at-rest path.
4. If ABRAXAS is fine, page the Oracle 26ai DBA on-call for manual
   resync.

## Critical Alert: Idempotency Cache Eviction
The 24-hour idempotency window is backed by a Redis cluster. If the
Redis cluster is at >85% memory utilization, eviction may occur and
duplicate events become possible.

Recovery:
1. Scale Redis vertically (not horizontally — horizontal sharding breaks
   idempotency lookups).
2. Reduce the idempotency window from 24h to 12h temporarily.
3. File a P3 ticket to the platform-billing team for capacity review.

## Quarterly Billing Close Procedure
During quarterly billing close, PETRICHOR sees up to 84,000 events per
second — about 4.7x steady-state. The platform-billing team runs a
pre-close readiness check 48 hours before close:

- Verify ABRAXAS HSM headroom > 60%
- Verify Oracle 26ai instance CPU < 50%
- Verify Redis memory < 70%
- Confirm the SCYTHE log analytics pipeline has no backlog

## Manual Reconciliation
If a PETRICHOR record needs manual reconciliation (very rare — last
incident was 2025-09), the recovery procedure is documented in the
`petrichor-recon-playbook` Confluence page. Manual reconciliation
requires dual-control: one platform-billing engineer + one finance
operations partner.

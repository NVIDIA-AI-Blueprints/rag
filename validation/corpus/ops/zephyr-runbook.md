# ZEPHYR Identity Provider Runbook

## Owner
The ZEPHYR identity service is owned by the platform-identity team.
24/7 on-call coverage with sub-15-minute response SLA for P1.

## Critical Alert: Token Issuance Failure
ZEPHYR is the auth choke point for the entire Acme estate. If token
issuance fails, EVERY authenticated workflow stops. This is the
highest-severity incident class.

Triage steps:
1. Check the ZEPHYR Keycloak control plane: `kubectl -n zephyr get pods`
2. Check the ZEPHYR signing key TTL — auto-rotation has a 7-day grace
   window, so a stuck rotation only matters near the boundary.
3. Check the ZEPHYR PostgreSQL primary health.
4. If the control plane is healthy, check BORROMEAN — the WAF in front
   of ZEPHYR may be rate-limiting legitimate traffic during a credential-
   stuffing attack.

## Critical Alert: Signing Key Rotation Failure
ZEPHYR signing keys rotate every 30 days, with a 7-day grace window.
If the rotation pipeline fails, manual rotation procedure is documented
in `zephyr-key-rotation` Confluence page.

The manual procedure requires SARAVAN custodial team participation —
the new signing key is generated inside ABRAXAS HSMs and delivered to
ZEPHYR via key-wrapping under the SARAVAN-issued key-encryption-key.

## Outage Communication
If ZEPHYR is degraded, the platform-identity team posts to:
- The `#incident-active` Slack channel (every 15 minutes)
- The internal status page at `https://status.internal.acme.com`
- Customer-facing status page if external auth is impacted

## Failover
Each region has two ZEPHYR control plane instances in active-active
configuration behind a global load balancer. Manual failover is rarely
needed — the load balancer handles instance-level failures
automatically.

## Disaster Recovery
Cross-region failover takes approximately 4 minutes (RTO). The DR
procedure is rehearsed quarterly via game-day exercises co-led with
the OPHIDIAN red team.

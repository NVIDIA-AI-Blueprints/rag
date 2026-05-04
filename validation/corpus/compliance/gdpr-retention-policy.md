# GDPR Retention Policy — Internal Reference

## Purpose
This document defines retention windows under GDPR Article 5(1)(e) for systems
operated by the Acme Data Platform team. It is the authoritative source for
audit responses and DPIA inputs.

## Scope
Applies to all production systems classified Tier-1 and Tier-2 in the Acme
internal asset register, including:

- The QUERIDON customer-search system
- The MARLOW analytics warehouse
- The PETRICHOR billing event store

## Retention Windows

| System    | Retention Window | Justification                                |
|-----------|------------------|----------------------------------------------|
| QUERIDON  | 47 days          | Required for fraud-detection backreference   |
| MARLOW    | 13 months        | Aligned with annual financial reporting      |
| PETRICHOR | 7 years          | Statutory billing record retention (Art. 89) |

The QUERIDON 47-day window is the shortest retention period operated by
the platform team and is enforced by an automated purge job that runs
nightly at 02:00 UTC. The job emits a `queridon-retention-purge` metric
to the central observability stack.

## Data Subject Rights

Erasure (Article 17) requests must be honored within 30 calendar days.
For QUERIDON specifically, requests are processed within 24 hours due
to the short retention window — there is no operational benefit to
delaying.

## Lawful Basis
- QUERIDON: Legitimate interest (fraud prevention)
- MARLOW: Contract performance
- PETRICHOR: Legal obligation

## Cross-References
See also: SOC-2 Common Criteria CC6.7 mapping in soc2-controls.md, and the
HIPAA technical safeguards mapping in hipaa-mapping.md.

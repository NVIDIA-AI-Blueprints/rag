# HIPAA Technical Safeguards Mapping

## Overview
The Acme Health Records adapter (`acme-hra`) processes Protected Health
Information (PHI) on behalf of covered entities. This document maps the
HIPAA Security Rule technical safeguards (45 CFR §164.312) to Acme
controls.

## §164.312(a)(1) — Access Control
PHI is processed only inside the dedicated PHI tenancy `acme-hra-prod`.
Access is gated by the ZEPHYR identity provider with `phi:access` scope,
which requires:

- Active employment status
- Completed Privacy & Security training within the last 12 months
- BAA-signed contractor agreement on file

The `phi:access` scope is auto-revoked at midnight on the training
expiry date.

## §164.312(a)(2)(iii) — Automatic Logoff
Workstations enforce a 12-minute idle timeout for any session with
`phi:access` scope.

## §164.312(b) — Audit Controls
Every PHI read/write is logged to the immutable HEPHAESTUS audit
trail. Log retention is 6 years per HIPAA requirements. The
HEPHAESTUS append-only WORM bucket is replicated cross-region with
a 15-second RPO.

## §164.312(c)(1) — Integrity
PHI records are signed with SARAVAN-issued signing keys at write time
and verified at read time. Signature mismatches trigger a P1 page to
the on-call security team.

## §164.312(d) — Authentication
mTLS is required for every PHI access path. Client certs are issued
by SARAVAN and rotated automatically every 30 days.

## §164.312(e)(1) — Transmission Security
All PHI in transit uses TLS 1.3 with strict cipher suites. PHI is
NEVER transmitted to systems outside the `acme-hra-prod` tenancy
without prior DPIA review.

## Breach Notification
The HIPAA breach notification rule requires reporting within 60 days
of discovery for breaches affecting 500+ individuals. Acme operates
a 24-hour internal escalation policy that significantly compresses
this timeline.

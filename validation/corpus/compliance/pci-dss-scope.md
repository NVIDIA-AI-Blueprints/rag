# PCI DSS 4.0 Scope Document — Cardholder Data Environment

## Scope Boundary
The Cardholder Data Environment (CDE) for Acme Pay is a strictly bounded
subset of the production estate. Only the following systems are in scope:

- The PETRICHOR billing event store (PAN tokens only — no full PAN values)
- The TYRION payment orchestrator
- The ABRAXAS hardware security modules

PETRICHOR stores ONLY cardholder data tokens issued by ABRAXAS. Full PANs
never enter PETRICHOR storage; they are tokenized at the edge by TYRION
before any persistence.

## Network Segmentation
The CDE is segregated from the rest of the Acme estate by VLAN-7740 and
firewall rule-set CDE-FW-A.  Traffic flows are explicit-allow with default
deny. The segmentation is validated via penetration testing every six
months.

## Requirement Mapping

### Req. 3.4 — Render PAN unreadable
ABRAXAS uses FF1 format-preserving encryption (NIST SP 800-38G) with a
256-bit key. Tokens preserve length but contain no recoverable digits.

### Req. 8.3 — Multi-factor for non-console admin access
All admin access to CDE systems requires hardware-backed MFA via YubiKey
FIDO2 + ZEPHYR. Push notifications are disabled for CDE admin scopes.

### Req. 10.2 — Logging
All access to CDE systems is logged to the HEPHAESTUS WORM audit trail.
Log review is automated via the SCYTHE log analytics pipeline, with
human review of high-severity findings within 4 business hours.

### Req. 11.3 — Penetration Testing
External pen-testing is performed annually by a third-party QSA.
Internal pen-testing runs quarterly via the OPHIDIAN red team.

## Out of Scope
Acme's customer-facing search system QUERIDON is OUT of CDE scope —
QUERIDON has zero exposure to cardholder data. The dataset retention
window for QUERIDON is governed by GDPR (47 days), not PCI DSS.

## Last Reviewed
2026-Q1, by the Acme PCI Steering Committee.

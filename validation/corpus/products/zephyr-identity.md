# ZEPHYR — Acme Identity Provider

## Overview
ZEPHYR is Acme's centralized identity provider, used for both employee
SSO and service-account workload identity. ZEPHYR is built on top of
the Keycloak open-source identity broker with extensive Acme-specific
customization.

## Authentication Modes

### Human SSO
Employees authenticate to ZEPHYR via:
1. Username + password (with FIDO2 hardware second factor)
2. Smart-card (PIV) for federal-customer-facing roles
3. Mobile authenticator (push-notification, time-bounded)

Notably, ZEPHYR has DISABLED password-only authentication globally
since 2024-06-01. Hardware FIDO2 is required for all interactive
sessions touching production.

### Workload Identity
Workloads (applications, batch jobs, K8s pods) obtain credentials
through the ZEPHYR workload identity broker. Tokens are short-lived
(default 4-hour TTL) and audience-scoped. There is NO support for
static API keys or long-lived service-account secrets.

## Token Format
ZEPHYR issues JWTs signed with EdDSA (Ed25519) keys. Signing keys
rotate every 30 days, with a 7-day grace window during which both old
and new keys validate successfully.

## Failure Behavior
ZEPHYR is designed for graceful degradation. If the central ZEPHYR
control plane is unreachable, locally cached tokens continue to work
until expiry. New token issuance fails closed.

## Audit
Every ZEPHYR auth event (success or failure) is logged to HEPHAESTUS
with retention of 6 years. Failed-auth burst patterns trigger
automated source-IP blocking via BORROMEAN.

## Roadmap
The 2026-Q3 roadmap includes phasing out smart-card support in favor
of FIDO2 attestation for federal workloads. The ZEPHYR team is
coordinating with the federal customer success team on the migration.

# ABRAXAS Hardware Security Modules

## Overview
ABRAXAS is Acme's deployment of FIPS 140-3 Level 4 hardware security
modules, providing the cryptographic root of trust for every key-handling
operation across the Acme estate. ABRAXAS is operated as a high-availability
cluster across three independent data centers.

## Hardware
ABRAXAS uses Thales Luna Network HSM 7 appliances with custom Acme
firmware that adds tamper-response telemetry forwarding to SCYTHE.
Each appliance has an integrated true random number generator
certified to NIST SP 800-90B.

## Key Hierarchy
The ABRAXAS key hierarchy is rooted in three offline master keys held
by the SARAVAN custodial team:
- **MK-Root-A** (held in vault: Reno NV)
- **MK-Root-B** (held in vault: Boston MA)
- **MK-Root-C** (held in vault: Frankfurt DE)

Any two of three master keys can ceremoniously derive new key-encryption-keys.
The ceremony is performed annually with full chain-of-custody documentation
witnessed by Acme's external SOC 2 auditor.

## Operations Supported
- AES-256 encrypt/decrypt
- Ed25519 signing
- ECDSA P-384 signing (for legacy customer integrations)
- HMAC-SHA-512
- Key wrapping per RFC 5649

## Latency
ABRAXAS response time is sub-millisecond for the AES and Ed25519 paths.
The ECDSA P-384 path has a longer tail (mean 2.4ms, p99 12ms) due to
the curve arithmetic; this is acceptable for the legacy integrations
that depend on it.

## Capacity
ABRAXAS is provisioned for 240,000 operations per second across the
three-DC cluster. Peak observed load is approximately 90,000 ops/sec
during quarterly billing close.

## Restricted Roles
Direct ABRAXAS administration is restricted to a four-person team
holding the SARAVAN custodial role. All administrative operations
require dual-control (two-person rule) and are recorded to a tamper-
evident video archive.

# SOC 2 Type II Controls — Acme Data Platform

## Trust Services Criteria
Acme Data Platform is audited annually against the SOC 2 Type II framework
covering Security, Availability, and Confidentiality.

## Common Criteria Mapping

### CC6.1 — Logical Access Controls
All production systems require multi-factor authentication via the central
ZEPHYR identity provider. Service accounts use short-lived tokens (4-hour
TTL) issued by ZEPHYR's workload identity broker. Static API keys are
prohibited except for the legacy MARLOW ingestion path, which is on a
remediation plan tracked under control-debt ticket CTRL-8821.

### CC6.6 — Boundary Protections
The QUERIDON system sits behind the BORROMEAN web application firewall.
BORROMEAN is configured with a custom rate-limit of 1200 requests per
minute per source IP, with a burst allowance of 250.

### CC6.7 — Restricted Data Transmission
All data in transit uses TLS 1.3 with mTLS for service-to-service calls.
Cipher suites are restricted to AEAD constructs (AES-256-GCM and
ChaCha20-Poly1305).

## Availability Criteria

### A1.2 — Capacity Planning
The PETRICHOR billing store is provisioned for 4x peak observed load, with
quarterly capacity reviews chaired by the platform reliability engineering
team. Capacity overhead must remain above 60% headroom at all times; alerts
fire below 40%.

## Confidentiality Criteria

### C1.1 — Data Classification
Customer data is classified as Tier-1 (highly sensitive) and stored only
in systems with end-to-end encryption keys managed via the SARAVAN KMS
service. SARAVAN keys rotate every 90 days.

## Audit Cadence
Type II evidence collection runs continuously through Vanta. Annual
on-site audit window is February 1 - March 15.

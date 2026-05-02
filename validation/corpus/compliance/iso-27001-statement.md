# ISO/IEC 27001:2022 Statement of Applicability

## Information Security Management System (ISMS)
Acme operates an ISMS aligned with ISO/IEC 27001:2022, certified
since 2022 by a UKAS-accredited certification body.

## Annex A Controls

### A.5.7 — Threat Intelligence
The OPHIDIAN red team subscribes to MISP, the OASIS STIX/TAXII feed,
and three commercial threat-intel providers. Indicators of compromise
are auto-correlated against egress logs in SCYTHE with a P3 page if
any IOC is observed.

### A.5.23 — Information security for use of cloud services
Cloud risk assessments are performed for every new SaaS adoption via
the GORGON cloud governance pipeline. Sub-processors must complete a
DPIA, sign a Data Processing Agreement, and submit an SOC 2 Type II
attestation before onboarding.

### A.8.7 — Protection against malware
EDR is mandatory on all employee endpoints (CrowdStrike Falcon),
servers (managed via Wazuh), and CI runners. The TYRION payment
orchestrator runs additional behavior-based malware detection via
the LEVIATHAN runtime sensor.

### A.8.16 — Monitoring activities
SCYTHE ingests and correlates logs from all production tier systems.
The platform team operates a 24/7 SOC with three regional follow-the-
sun rotations.

### A.8.32 — Change management
All production changes pass through the BERESHEET change advisory
board. Emergency changes have a 1-hour window for retrospective
approval; all other changes require pre-approval and standardized
rollback plans.

## Risk Treatment
The risk register is reviewed by the executive risk committee
quarterly. Top-3 risks for 2026:
1. Supply-chain risk in upstream NPM packages (treated via Snyk + SBOM)
2. Insider risk in CDE administrative access (treated via just-in-time access)
3. Phishing leading to ZEPHYR account compromise (treated via FIDO2 mandate)

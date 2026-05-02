# QUERIDON v3.4 Product Specification

## Overview
QUERIDON is Acme's customer-facing semantic search platform. The v3.4 release
introduces three new capabilities: federated indices, native vector search
over the Oracle 26ai database, and a streaming results API.

## Architecture

### Front Door
QUERIDON exposes a single GraphQL endpoint at `https://api.acme.com/queridon/v3`.
All requests are authenticated via JWT issued by ZEPHYR.

### Search Tier
The search tier consists of:
- **MARLOW-search** — keyword-search over the MARLOW analytical store
- **NEMOPHILA** — dense-vector retrieval over a 1024-dim embedding space
- **HARMATTAN** — the result fusion layer (reciprocal rank fusion)

The HARMATTAN fusion layer produces final ranked results by combining
keyword and dense signals with default weights of 0.5 each.

## Performance Targets
- p50 latency: 85ms
- p99 latency: 350ms
- Throughput: 12,000 QPS at peak

These targets are validated weekly against the production load shape.
The QUERIDON team operates a chaos engineering regime with a weekly
"latency-injection Friday" — random 100ms delays are injected into
NEMOPHILA to verify HARMATTAN's degraded-mode handling.

## Data Sources
QUERIDON indexes the following data sources:
- Product catalog (PETRICHOR-derived)
- Help center articles (CMS-sourced)
- Community forum content (Discourse-sourced)

## Retention
Per the GDPR retention policy, QUERIDON record-level data is retained for
47 days. Aggregate analytics derived from QUERIDON usage are retained
for 13 months in MARLOW.

## Changelog (v3.4)
- Added Oracle 26ai vector backend support
- Reduced p99 latency from 480ms to 350ms via HARMATTAN parallelism
- Removed deprecated `/legacy/search` GraphQL endpoint
- Migrated authentication to ZEPHYR v2 (FIDO2-required)

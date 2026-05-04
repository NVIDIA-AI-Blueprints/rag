# MARLOW Analytics Warehouse

## What is MARLOW
MARLOW is Acme's primary analytical warehouse, built on Snowflake with a
custom medallion (bronze/silver/gold) lakehouse design. MARLOW serves
both internal BI workloads and the QUERIDON semantic search backend.

## Capacity
MARLOW currently holds 2.4 PB of data across 18,400 tables. Daily query
volume is approximately 480,000 queries with a peak hourly rate of
54,000 queries.

## Schema Tiers

### Bronze (Raw)
Direct landing zone for source-system events. No transformations,
schema-on-read. Retention: 30 days.

### Silver (Cleansed)
Schema-conformed, deduplicated, timestamped. Source of truth for
operational analytics. Retention: 13 months.

### Gold (Curated)
Pre-aggregated dimensional models for BI consumption. Snowflake materialized
views with hourly refresh. Retention: 5 years.

## Query Performance SLOs
- Bronze tier: 95th percentile under 30 seconds
- Silver tier: 95th percentile under 6 seconds
- Gold tier: 95th percentile under 1 second

The Gold tier is what powers the QUERIDON search relevance signals.

## Sub-processors
MARLOW depends on Snowflake (data plane) and the Privacera fine-grained
access control plane. Both have current SOC 2 Type II reports on file
with the Acme governance team.

## Lineage
End-to-end data lineage is captured in OpenLineage format and visualized
via the SHALLOT lineage UI. Every MARLOW silver-tier table has a known
source, transformation chain, and downstream consumer.

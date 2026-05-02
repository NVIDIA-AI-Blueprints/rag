# PETRICHOR Billing Event Store

## Purpose
PETRICHOR is the immutable event store of record for every billing-relevant
action across Acme's product portfolio. PETRICHOR is the canonical source
for revenue recognition, customer invoicing, and PCI DSS audit evidence.

## Data Model
PETRICHOR uses an append-only event log with one record per state change:

| Field            | Type         | Required | Description |
|------------------|--------------|----------|-------------|
| `event_id`       | UUID v7      | yes      | Globally unique, time-ordered |
| `event_time`     | timestamptz  | yes      | UTC, microsecond precision |
| `customer_id`    | string       | yes      | Tenant identifier |
| `product_sku`    | string       | yes      | SKU from product catalog |
| `quantity`       | decimal      | yes      | Decimal-256 to avoid float drift |
| `unit_price_cents` | int64      | yes      | Cents, never floats |
| `pan_token`      | string       | no       | ABRAXAS-issued tokenized PAN |
| `metadata`       | jsonb        | no       | Customer-attributed key/values |

## Idempotency
Every PETRICHOR event accepts an `idempotency_key` header. Duplicate keys
within a 24-hour window return the original event_id with HTTP 200.
After 24 hours, idempotency keys are released.

## Throughput
PETRICHOR handles 18,000 events per second at steady state with bursts
to 84,000 events per second during quarterly billing close. The hot
write path is implemented in Rust for tail-latency reliability.

## Retention
PETRICHOR records are retained for 7 years per statutory billing record
retention requirements (GDPR Article 89, US SOX Section 802).

## Replication
PETRICHOR is replicated synchronously across two Oracle 26ai availability
domains in the same region (US-East), with an asynchronous DR replica
in EU-West. RPO is zero in-region; cross-region RPO is 30 seconds.

## Out-of-band Access
For customer disputes, the BERESHEET dispute UI provides a read-only
window into PETRICHOR. All accesses through BERESHEET are logged with
the operator's ZEPHYR identity and the customer's prior consent token.

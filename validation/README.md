# Enterprise Validation Harness

This is the same harness used to validate the plugin before this handoff.
Re-run it after applying these fixes against your own ADB to confirm.

## Prerequisites

- Live Oracle Autonomous AI Database (any 23ai / 26ai instance)
- `RAG_APP` user provisioned with `CONNECT, RESOURCE, CTXAPP` and
  `EXECUTE ON CTXSYS.CTX_QUERY` grants
- A running deployment of the patched RAG blueprint with
  `APP_VECTORSTORE_NAME=oracle` and the `oracle-creds` + `oracle-wallet`
  K8s secrets in place
- The two services reachable as `rag-server:8081` and
  `ingestor-server:8082` from inside the cluster (the harness is designed
  to run from inside an in-cluster pod)

## Running

```bash
# Stage in any pod with cluster DNS access (e.g. ingestor-server)
ING_POD=$(kubectl -n rag get pod -l app.kubernetes.io/component=ingestor-server \
            -o jsonpath='{.items[0].metadata.name}')
kubectl -n rag cp runner.py            $ING_POD:/tmp/runner.py
kubectl -n rag cp oracle_internals.py  $ING_POD:/tmp/oracle_internals.py
kubectl -n rag cp corpus               $ING_POD:/tmp/corpus

# Full run (creates 3 collections, ingests 14 docs incl. multimodal PDF,
# runs 38 checks in 14 categories)
kubectl -n rag exec $ING_POD -- python3 /tmp/runner.py --corpus /tmp/corpus

# To re-validate against existing collections (skip ingestion):
kubectl -n rag exec $ING_POD -- python3 /tmp/runner.py --corpus /tmp/corpus --skip-ingest

# Direct DB inspection (run from a pod that has the wallet mounted, e.g.
# rag-server):
RAG_POD=$(kubectl -n rag get pod -l app.kubernetes.io/component=rag-server \
            -o jsonpath='{.items[0].metadata.name}')
kubectl -n rag cp oracle_internals.py $RAG_POD:/tmp/oracle_internals.py
kubectl -n rag exec $RAG_POD -- python3 /tmp/oracle_internals.py
```

## Test categories

| Category | Checks |
|---|---|
| **A. Health & API surface** | rag-server + ingestor-server `/health`, openapi, both `/v1` paths |
| **B. Collection lifecycle** | create / list / cleanup |
| **C. Multi-collection ingestion** | 3 tenants, 14 markdown docs |
| **D. Multimodal PDF ingestion** | one 5.3MB `product_catalog.pdf` |
| **E. Functional search** | hybrid + dense |
| **F. Quality (ground truth)** | 10 sentinel queries → expected document |
| **G. Cross-collection isolation** | both directions, no bleed |
| **H. Multiturn chat** | 5-turn dialog |
| **I. Streaming `/generate`** | SSE token stream |
| **J. Concurrent search** | ramped 1/4/8/16 concurrent |
| **K. Concurrent ingestion** | 8 docs in parallel |
| **L. Performance** | p50/p95/p99 warm steady-state |

`oracle_internals.py` separately checks (against your live ADB):

- Row counts in `ENT_COMPLIANCE`, `ENT_PRODUCTS`, `ENT_OPS`
- `_VEC_IDX` (VECTOR) and `_TEXT_IDX` (CTXSYS.CONTEXT) presence
- `EXPLAIN PLAN` of `CONTAINS()` queries — confirms DOMAIN INDEX usage
- Vector dimension consistency (2048, matching nemotron-embed-1b-v2)
- Round-trip query through the live VECTOR column

## Expected baseline result

35/38 PASS.  The 3 expected non-PASS:

1. `J_concurrency.concurrent_search_p16` — single reranker NIM replica
   saturates at par >12 (capacity-planning observation, not a bug; scale
   `nemotron-ranking-ms` replicas for higher concurrency)
2. `N.dense_explain_plan` — at <100 rows the optimizer correctly chooses
   TABLE ACCESS FULL over the IVF approximate index (correct behavior;
   IVF activates at scale)
3. `K.alpha_sentinel_findability_after_concurrent_ingest` — Oracle Text
   default `BASIC_LEXER` doesn't differentiate single-digit-only token
   variants (corpus-tuning consideration, not a bug)

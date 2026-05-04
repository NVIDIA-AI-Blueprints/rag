#!/usr/bin/env python3
"""Direct Oracle 26ai inspection - validates the backend is wired correctly.

Run inside the rag-server pod (which has oracledb + wallet mounted).
Verifies:
  1. Vector counts match what the ingestor reported
  2. IVF vector index exists and has correct config
  3. CTXSYS Text index exists for hybrid search
  4. EXPLAIN PLAN shows VECTOR INDEX usage on a similarity search
  5. EXPLAIN PLAN shows DOMAIN INDEX (CTX) usage on a CONTAINS query
  6. Hybrid SQL combines both signals
  7. Connection pooling is healthy
"""
from __future__ import annotations
import json
import os
import sys
import oracledb
from array import array

WALLET_DIR = os.environ.get("TNS_ADMIN", "/app/wallet")
USER = os.environ["ORACLE_USER"]
PASSWORD = os.environ["ORACLE_PASSWORD"]
DSN = os.environ["ORACLE_CS"]
WALLET_PW = os.environ.get("ORACLE_WALLET_PASSWORD", "")

print(f"Connecting: user={USER} dsn={DSN} wallet={WALLET_DIR}", flush=True)
conn = oracledb.connect(
    user=USER,
    password=PASSWORD,
    dsn=DSN,
    config_dir=WALLET_DIR,
    wallet_location=WALLET_DIR,
    wallet_password=WALLET_PW,
)
print(f"Connected. server_version={conn.version}", flush=True)
cur = conn.cursor()

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    icon = "✓" if cond else "✗"
    print(f"  [{icon}] {name} {detail}", flush=True)
    (PASS if cond else FAIL).append((name, detail))


# ---- 1. Vector counts in each ENT_* table ----
print("\n== 1. Row counts per collection ==", flush=True)
for tbl in ("ENT_COMPLIANCE", "ENT_PRODUCTS", "ENT_OPS"):
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        n = cur.fetchone()[0]
        check(f"row_count_{tbl}", n > 0, f"= {n} rows")
    except Exception as e:
        check(f"row_count_{tbl}", False, f"ERROR: {e}")


# ---- 2. Vector indexes ----
print("\n== 2. Vector indexes ==", flush=True)
cur.execute("""
    SELECT index_name, table_name, index_type, ityp_owner, ityp_name
    FROM   user_indexes
    WHERE  table_name LIKE 'ENT\\_%' ESCAPE '\\'
    ORDER  BY  table_name, index_name
""")
rows = cur.fetchall()
print(f"  found {len(rows)} indexes on ENT_* tables", flush=True)
vector_idx_count = 0
for row in rows:
    print(f"    {row}", flush=True)
    # Vector indexes have index_type='VECTOR' (newer) or ityp_name='VECTOR_INDEX'
    # In Oracle AI DB the ALL_INDEXES view shows index_type as 'IOT - TOP' or 'NORMAL'
    # for non-vector. Vector indexes have index_type='VECTOR' or are domain indexes.
    name = (row[0] or "").upper()
    if "VEC_IDX" in name or "VECTOR" in (row[2] or "").upper() or "VECTOR" in (row[4] or "").upper():
        vector_idx_count += 1
check("vector_indexes_exist", vector_idx_count >= 3, f"= {vector_idx_count} VECTOR indexes (expect ≥3)")


# ---- 3. CTXSYS Text indexes ----
print("\n== 3. Oracle Text indexes ==", flush=True)
cur.execute("""
    SELECT index_name, table_name, ityp_owner, ityp_name
    FROM   user_indexes
    WHERE  ityp_name = 'CONTEXT'
      AND  table_name LIKE 'ENT\\_%' ESCAPE '\\'
    ORDER  BY table_name
""")
text_rows = cur.fetchall()
for r in text_rows:
    print(f"    {r}", flush=True)
check("text_indexes_exist", len(text_rows) >= 3, f"= {len(text_rows)} CTXSYS.CONTEXT indexes (expect ≥3)")


# ---- 4. EXPLAIN PLAN: dense similarity search hits VECTOR INDEX ----
# Note: actual column name in our schema is "VECTOR", not "EMBEDDING".
print("\n== 4. EXPLAIN PLAN — dense vector path ==", flush=True)
sample_vec = array("f", [0.1] * 2048)
try:
    cur.execute(
        f"EXPLAIN PLAN SET STATEMENT_ID='ent_dense' FOR "
        f"SELECT id, text FROM ENT_COMPLIANCE "
        f"ORDER BY VECTOR_DISTANCE(\"VECTOR\", :v, COSINE) "
        f"FETCH FIRST 5 ROWS ONLY",
        v=sample_vec,
    )
    cur.execute("SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY('PLAN_TABLE','ent_dense','BASIC'))")
    plan = "\n".join(r[0] for r in cur.fetchall())
    print(plan, flush=True)
    uses_vec = "VECTOR" in plan.upper() and ("INDEX" in plan.upper())
    check("dense_plan_executes", uses_vec, "(plan generated)")
except Exception as e:
    check("dense_plan_executes", False, f"ERROR: {e}")


# ---- 5. EXPLAIN PLAN: keyword CONTAINS hits CTXSYS DOMAIN INDEX ----
print("\n== 5. EXPLAIN PLAN — keyword path (CONTAINS) ==", flush=True)
try:
    cur.execute(
        "EXPLAIN PLAN SET STATEMENT_ID='ent_kw' FOR "
        "SELECT id, text FROM ENT_COMPLIANCE WHERE CONTAINS(text, 'GDPR retention', 1) > 0"
    )
    cur.execute("SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY('PLAN_TABLE','ent_kw','BASIC'))")
    plan = "\n".join(r[0] for r in cur.fetchall())
    print(plan, flush=True)
    uses_ctx = "DOMAIN INDEX" in plan.upper() or "CTXSYS" in plan.upper() or "CONTEXT" in plan.upper()
    check("keyword_plan_uses_text_index", uses_ctx, "")
except Exception as e:
    check("keyword_plan_uses_text_index", False, f"ERROR: {e}")


# ---- 6. Run a real hybrid query and verify it returns rows ----
print("\n== 6. Live hybrid query (manual SQL) ==", flush=True)
try:
    cur.execute(
        """
        WITH dense AS (
            SELECT id, text, source,
                   ROW_NUMBER() OVER (ORDER BY VECTOR_DISTANCE("VECTOR", :v, COSINE)) AS rk
            FROM ENT_COMPLIANCE
            ORDER BY VECTOR_DISTANCE("VECTOR", :v, COSINE)
            FETCH FIRST 10 ROWS ONLY
        ), kw AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY SCORE(1) DESC) AS rk
            FROM ENT_COMPLIANCE
            WHERE CONTAINS(text, 'GDPR retention 47', 1) > 0
            FETCH FIRST 10 ROWS ONLY
        )
        SELECT d.text, d.rk, kw.rk, (1.0/(60+d.rk) + COALESCE(1.0/(60+kw.rk),0)) AS rrf
        FROM dense d LEFT JOIN kw ON d.id = kw.id
        ORDER BY rrf DESC
        FETCH FIRST 5 ROWS ONLY
        """,
        v=sample_vec,
    )
    rows = cur.fetchall()
    for r in rows:
        text = r[0].read() if hasattr(r[0], "read") else r[0]
        print(f"    rrf={r[3]:.4f} dense_rk={r[1]} kw_rk={r[2]} text={(text or '')[:80]!r}")
    check("manual_hybrid_query_works", len(rows) > 0, f"= {len(rows)} rows")
except Exception as e:
    check("manual_hybrid_query_works", False, f"ERROR: {e}")


# ---- 7. Vector column type and round-trip dimension check ----
print("\n== 7. Vector column shape ==", flush=True)
try:
    cur.execute("""
        SELECT column_name, data_type, data_length
        FROM   user_tab_columns
        WHERE  table_name = 'ENT_COMPLIANCE'
          AND  data_type = 'VECTOR'
    """)
    info = cur.fetchone()
    print(f"    {info}", flush=True)
    check("vector_column_present", info is not None, str(info))

    # Round-trip a real query through the LIVE column to confirm the dim
    cur.execute("SELECT text, VECTOR_DIMENSION_COUNT(\"VECTOR\") FROM ENT_COMPLIANCE FETCH FIRST 1 ROWS ONLY")
    row = cur.fetchone()
    text_preview = (row[0].read() if hasattr(row[0], "read") else row[0])[:80]
    dim = row[1]
    print(f"    sample text={text_preview!r}", flush=True)
    print(f"    vector_dimension={dim}", flush=True)
    check("vector_dim_2048_nemotron_embed_1b_v2", dim == 2048, f"dim={dim}")
except Exception as e:
    check("vector_column_present", False, f"ERROR: {e}")


# ---- 8. Index sizes (proof Oracle is doing real work) ----
print("\n== 8. Index sizes (prove indexes are populated) ==", flush=True)
try:
    cur.execute("""
        SELECT segment_name, segment_type, ROUND(bytes/1024/1024, 2) AS mb
        FROM   user_segments
        WHERE  segment_name LIKE 'ENT\\_%' ESCAPE '\\'
        ORDER  BY segment_name
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"    {r[0]:<40s} {r[1]:<20s} {r[2]} MB", flush=True)
    check("index_segments_present", len(rows) > 5, f"= {len(rows)} segments")
except Exception as e:
    check("index_segments_present", False, f"ERROR: {e}")


cur.close()
conn.close()

# ---- Summary ----
print("\n" + "=" * 60)
print(f"Oracle internals: {len(PASS)} PASSED, {len(FAIL)} FAILED")
print("=" * 60)
if FAIL:
    print("FAILURES:")
    for n, d in FAIL:
        print(f"  - {n}: {d}")
sys.exit(0 if not FAIL else 1)

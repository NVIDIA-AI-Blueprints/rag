# Migrating Oracle 26ai deployments from v0.0.6 to v0.0.7

## What changed

v0.0.7 fixes a long-standing bug where Oracle case-folded all collection
names to UPPERCASE, regardless of the case the client supplied. After
upgrade, new collections preserve their original casing (matching
Milvus/Elasticsearch semantics).

| Behavior | v0.0.6 | v0.0.7 |
|---|---|---|
| Client creates `s_session_uuid` | Stored as `S_SESSION_UUID` | Stored as `s_session_uuid` |
| Client creates `MyCollection` | Stored as `MYCOLLECTION` | Stored as `MyCollection` |
| `list_collections()` returns | All names UPPERCASE | Names exactly as stored |
| Comparison `coll.name == client_input` | Case-mismatch → 404 | Exact match |

## Behavior on upgrade

**No data loss.** v0.0.7 retains backward-compatible lookup:
`check_collection_exists()` and `create_collection()` detect both
case-preserved (v0.0.7) and case-folded (v0.0.6) tables.

**Existing v0.0.6 collections remain accessible** if you address them
with the case Oracle stored — i.e. UPPERCASE. For example:

```python
# Pre-upgrade: client sent "biomedical_dataset" → stored as BIOMEDICAL_DATASET
# Post-upgrade: still listed as BIOMEDICAL_DATASET in get_collection()
# Client must call with BIOMEDICAL_DATASET to find it.
```

If you want to rename existing UPPERCASE tables to a mixed-case form, see
[Optional rename](#optional-rename) below.

## Required cleanup before first v0.0.7 startup

If you previously ran v0.0.6 against this Oracle database, the following
system tables persist across container restarts and **do not need
cleanup** — v0.0.7 detects them via `_table_exists_unquoted()` and
short-circuits creation:

- `METADATA_SCHEMA` (created by v0.0.6 unquoted DDL)
- `DOCUMENT_INFO` (created by v0.0.6 unquoted DDL)

If your deployment exhibited startup failures with `ORA-00955: name is
already used by an existing object` on v0.0.6 → v0.0.7 transition
**before** this fix landed, you can verify with:

```sql
SELECT table_name FROM user_tables
WHERE table_name IN ('METADATA_SCHEMA', 'DOCUMENT_INFO');
```

Both should be present and accessible. v0.0.7 reuses them in place.

## Idempotency guarantees in v0.0.7

`create_collection()`, `create_metadata_schema_collection()`, and
`create_document_info_collection()` are now safe to call repeatedly:

1. They first check existence with both case-sensitive (quoted-DDL) and
   case-folded (unquoted-DDL) lookups.
2. The CREATE statement itself is wrapped in a try/except that swallows
   `ORA-00955`. This handles races between the check and the CREATE,
   and any cases the existence check missed.

This means restart loops, parallel ingestion, and v0.0.6 → v0.0.7
upgrades all converge cleanly without manual intervention.

## Optional rename — convert existing UPPERCASE collections to mixed case

This is **not required**. Only do this if you want, e.g., the AIQ
frontend to consistently see lowercase collection names.

```sql
-- Rename the table itself (Oracle preserves case on quoted ALTER ... RENAME)
ALTER TABLE BIOMEDICAL_DATASET RENAME TO "biomedical_dataset";

-- Update metadata table to point at the new name. Both metadata_schema
-- and document_info store collection_name as VARCHAR2 data, so case is
-- preserved end-to-end.
UPDATE metadata_schema
   SET collection_name = 'biomedical_dataset'
 WHERE collection_name = 'BIOMEDICAL_DATASET';

UPDATE document_info
   SET collection_name = 'biomedical_dataset'
 WHERE collection_name = 'BIOMEDICAL_DATASET';

COMMIT;
```

Verify with:

```sql
SELECT table_name FROM user_tables WHERE table_name = 'biomedical_dataset';
-- Should return one row with the lowercase name.
```

Repeat for each collection you want renamed.

## Troubleshooting

### `ORA-00955: name is already used by an existing object`
v0.0.7 should never raise this on collection or system-table creation.
If you see it on a different code path (e.g., a custom migration
script), check whether the offending DDL uses quoted identifiers — Oracle
treats `"MyName"` and `MYNAME` as distinct objects, and creating one
when the other already exists triggers ORA-00955.

### A collection I created appears as UPPERCASE in `list_collections()`
That collection was created by v0.0.6 (or earlier) before this fix.
Either continue addressing it as UPPERCASE, or follow the
[Optional rename](#optional-rename) steps above.

### Long collection names cause `ORA-00972: identifier is too long`
v0.0.7 enforces Oracle's 128-character identifier limit at create time.
Auxiliary objects (vector index, text index) use a deterministic SHA-256
prefix when the verbatim derivation would exceed 128 chars, so the
practical name limit on the **table** is 128 chars (full Oracle limit).
Names exceeding 128 characters are rejected with `ValueError`.

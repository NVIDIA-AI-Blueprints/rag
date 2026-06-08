# Stage 5 — Verify results, retry once on failure, snapshot success into the experiment folder

Goal: confirm the run produced valid results in `$SCRIPTS_DIR/results/<dataset>/` (the script's transient
scratch); on failure, make a **dedicated, diagnosed** retry once and escalate if it still fails; on
success, **snapshot** the result into the experiment folder (`$EXP_DIR/<dataset>/<cycle>_<TS>/`, created in
Stage 4.0/4.4) for tracking and comparison.

Run after the background process from Stage 4 exits (the harness re-invokes you then).

## 5.1 Verify the outputs exist and are valid

For each dataset run:

```bash
cd "$SCRIPTS_DIR"
DS=financebench
ls -lrt "results/$DS/"
# A successful v2 run must contain these:
for f in rag_${DS}_evaluation_metrics.json rag_${DS}_evaluation_results.json \
         rag_${DS}_evaluation_data.json rag_${DS}_evaluation_summary.json; do
  [ -s "results/$DS/$f" ] && echo "OK  $f" || echo "MISSING/EMPTY  $f"
done
# failure.txt present ⇒ >50% of queries failed ⇒ treat the run as FAILED:
[ -f "results/$DS/failure.txt" ] && { echo "FAILED:"; cat "results/$DS/failure.txt"; } || echo "no failure.txt"
```

Treat the run as **failed** if: the process exited non-zero, `failure.txt` exists, the metrics file is
missing/empty, or `e2e_accuracy_mean` is `0.0` with empty per-query `generated_answer`s (server not
actually answering).

Quick metric peek on success (includes the `latency` block — `mean/p50/p90/p99/min/max_total_seconds` +
`mean/p50/p90/p99_ttft_seconds`):

```bash
python3 -m json.tool "results/$DS/rag_${DS}_evaluation_metrics.json"
```

A run can be valid on accuracy yet flag a latency problem: if `latency.sample_count` is far below the query
count, or `max_total_seconds` ≈ the `--timeout` (e.g. 300 s) for many queries, requests were timing out —
note it (it depresses both latency stats and accuracy) rather than reading the latency means at face value.

## 5.2 On failure — diagnose, then retry exactly once

Make a real effort to fix the root cause before retrying — do not blindly re-run. Read the run log
(`$SNAP/eval.log` from Stage 4) and map the signal to a fix:

| Signal in log / output | Likely cause | Fix before retry |
|------------------------|--------------|------------------|
| Immediate exit naming `NVIDIA_API_KEY` | Key unset in the process env | Export `NVIDIA_API_KEY`; relaunch. |
| `Failed to get response from rag-server` / connection refused | rag-server down or wrong `--host/--port` | Re-check Stage 1.5 health; fix host/port. |
| Ingestor 404 on upload | `--ingestor_server_url` included `/v1` | Pass base URL only (`http://host:port`). |
| `documents are missing or not ingested` / aborts before eval | Ingestion incomplete; vector DB unreachable from server | Check `--vdb_endpoint` (container hostname), ingestor health, disk; consider `--force_ingestion` (destructive — confirm). |
| Empty `generated_contexts` everywhere | Retrieval gap: wrong collection / vdb endpoint / top_k | Verify collection exists; fix `--vdb_endpoint`; raise `--top_k`. |
| Stream/JSON decode errors, high `error_count` | LLM endpoint flaky or model name wrong | Set `--model`/`--llm_endpoint` to match the deployment (`APP_LLM_MODELNAME`/`APP_LLM_SERVERURL`) — never omit `--model` (its default `nvdev/meta/llama-3.3-70b-instruct` overrides the deployed model); lower `--thread`. |
| RAGAS/judge errors | Judge model unreachable | Confirm `NVIDIA_API_KEY` valid + network to the catalog. |
| Collection has stale data from a prior run | Reused collection | Use `--force_ingestion` or a unique `--collection <name>` (confirm before deleting). |

Apply the fix, relaunch (background, per Stage 4), and re-verify. **If it still fails after this one
diagnosed retry, stop and escalate to the user** with: the failing command, the relevant log excerpt, your
diagnosis, and what you already tried. Do not loop indefinitely.

## 5.3 On success — snapshot the result into the experiment folder

The script writes `results/<dataset>/` as **transient scratch** (overwritten by every run). On success,
copy that into the cycle's snapshot folder under the experiment dir — the same `$SNAP` Stage 4.4 already
created and `tee`'d `eval.log` into. The snapshot is the durable, comparable artifact:

```bash
cd "$SCRIPTS_DIR"
DS=financebench
# $SNAP was set in Stage 4.4: SNAP="$EXP_DIR/$DS/${CYCLE}_${TS}" (e.g. baseline_20260609_103000)
cp results/$DS/rag_${DS}_evaluation_*.json "$SNAP"/
[ -f "results/$DS/failure.txt" ] && cp "results/$DS/failure.txt" "$SNAP"/
echo "Snapshot: $SNAP"
ls -lrt "$SNAP"        # expect the 4 JSONs + eval.log (+ failure.txt only on a failed run)
```

This yields `$EXP_DIR/<dataset>/<cycle>_<timestamp>/` per run (`baseline_<ts>`, `cycle1_<ts>`, …), each
self-contained with its JSONs, `eval.log`, and (5.4) `REPRODUCE.md`. Stage 6 compares the newest snapshot
against the previous one for the same dataset, all within this experiment folder.

> The scratch `results/<dataset>/` is disposable — only the `$SNAP` copies persist. Each cycle gets its
> own timestamped snapshot, so history is never lost. Never delete an existing snapshot under
> `$EXP_DIR/<dataset>/` without asking.

### 5.4 Write a `REPRODUCE.md` inside every snapshot

Each snapshot must carry a **self-contained, copy-pasteable** `REPRODUCE.md` that lets anyone reproduce
that exact run from scratch — written into the snapshot folder itself (`$SNAP/REPRODUCE.md`, i.e.
`$EXP_DIR/<dataset>/<cycle>_<timestamp>/REPRODUCE.md`). This applies to the **baseline** run and **every
improve cycle** (Stage 6.7). Fill in real, verbatim values (no placeholders):

```markdown
# Reproduce — <dataset> @ <timestamp>   (baseline | cycle <n>)

## Change for this run
<"Baseline: AGENTIC_VERIFICATION_ENABLED=false (deployment default)"  OR
 "Cycle 2: AGENTIC_PLANNER_MAX_TASKS 5 -> 8 in deploy/compose/.env">

## 1. Apply the change   (skip for baseline — keeps existing env intact; see rag-blueprint docker.md)
cd <repo-root>
#   edit deploy/compose/.env: export AGENTIC_PLANNER_MAX_TASKS=8
export NGC_API_KEY="$NVIDIA_API_KEY"
set -a; source deploy/compose/.env; set +a
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d rag-server
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8081/v1/health)" = 200 ]; do sleep 2; done

## 2. Run the eval   (exact command, verbatim)
cd <SCRIPTS_DIR> && source .venv/bin/activate
SNAP=experiments/exp_<ts>_<mode>/<dataset>/<cycle>_<ts>; mkdir -p "$SNAP"
PYTHONUNBUFFERED=1 python3 evaluate_rag.py <full flags exactly as run> --output_dir results 2>&1 | tee "$SNAP/eval.log"
cp results/<dataset>/rag_<dataset>_evaluation_*.json "$SNAP"/

## 3. Result
e2e=<…> ctx_rel=<…> grounded=<…> recall@1=<…>   | verdict: <gain | plateau | regression>
```

Omit section 1 for the baseline (no change applied). Keep the command **byte-for-byte** what was run so
the report's per-cycle command (6.8 B.3) and this file always agree.

**Level of detail:** `REPRODUCE.md` does **not** need the exhaustive depth of report recommendations (§6.4
detail standard) — it is a rerun recipe, not a change proposal. But don't make it *too* terse either: beyond
the raw commands, include the brief context a fresh reader needs to actually run it — prerequisites
(`NVIDIA_API_KEY` exported, eval `.venv` active, collection already ingested vs needs ingestion), the working
directory, and a one-line note on what the change/run was for. Aim for "a teammate can follow it without
asking questions," not a bare command dump.

### 5.5 Update `state.md` with this cycle's row

After a successful snapshot, append one row to the **Progress** table in `$EXP_DIR/state.md` (seeded in
Stage 4.0) so the experiment stays self-describing and resumable: the dataset, cycle, snapshot path
(`$EXP_DIR/<dataset>/<cycle>_<ts>`), the change applied this cycle (`—` for baseline), the headline
metrics (e2e / ctx_rel / grounded / recall@1 **and latency: p50_total_seconds + mean/p50 ttft_seconds**,
all from the `evaluation_metrics` and `latency` blocks of `rag_<dataset>_evaluation_metrics.json`), and the
verdict (`baseline` / `gain` / `plateau` / `regression`). State the verdict on **both axes** — e.g.
"accuracy flat, p50 latency −18%" or "e2e +0.02 but p50 latency +30% (trade-off)". This is the live
checkpoint Stage 6.8 reads to build the final `report.md`.

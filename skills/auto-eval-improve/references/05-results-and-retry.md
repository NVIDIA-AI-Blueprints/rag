# Stage 5 — Verify results, retry once on failure, timestamp success

Goal: confirm the run produced valid results in `$SCRIPTS_DIR/results/<dataset>/`; on failure, make a
**dedicated, diagnosed** retry once and escalate if it still fails; on success, **timestamp** the result
folder for tracking.

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

Quick metric peek on success:

```bash
python3 -m json.tool "results/$DS/rag_${DS}_evaluation_metrics.json"
```

## 5.2 On failure — diagnose, then retry exactly once

Make a real effort to fix the root cause before retrying — do not blindly re-run. Read the run log
(`eval_<dataset>_run.log` from Stage 4) and map the signal to a fix:

| Signal in log / output | Likely cause | Fix before retry |
|------------------------|--------------|------------------|
| Immediate exit naming `NVIDIA_API_KEY` | Key unset in the process env | Export `NVIDIA_API_KEY`; relaunch. |
| `Failed to get response from rag-server` / connection refused | rag-server down or wrong `--host/--port` | Re-check Stage 1.5 health; fix host/port. |
| Ingestor 404 on upload | `--ingestor_server_url` included `/v1` | Pass base URL only (`http://host:port`). |
| `documents are missing or not ingested` / aborts before eval | Ingestion incomplete; vector DB unreachable from server | Check `--vdb_endpoint` (container hostname), ingestor health, disk; consider `--force_ingestion` (destructive — confirm). |
| Empty `generated_contexts` everywhere | Retrieval gap: wrong collection / vdb endpoint / top_k | Verify collection exists; fix `--vdb_endpoint`; raise `--top_k`. |
| Stream/JSON decode errors, high `error_count` | LLM endpoint flaky or model name wrong | Fix `--model`/`--llm_endpoint`, or omit to use server default; lower `--thread`. |
| RAGAS/judge errors | Judge model unreachable | Confirm `NVIDIA_API_KEY` valid + network to the catalog. |
| Collection has stale data from a prior run | Reused collection | Use `--force_ingestion` or a unique `--collection <name>` (confirm before deleting). |

Apply the fix, relaunch (background, per Stage 4), and re-verify. **If it still fails after this one
diagnosed retry, stop and escalate to the user** with: the failing command, the relevant log excerpt, your
diagnosis, and what you already tried. Do not loop indefinitely.

## 5.3 On success — timestamp the result folder

Keep the canonical `results/<dataset>/` as the "latest" pointer **and** snapshot a timestamped copy for
history/comparison. Stamp at the moment of success (the script does not stamp folders itself):

```bash
cd "$SCRIPTS_DIR"
DS=financebench
TS=$(date +%Y%m%d_%H%M%S)
cp -r "results/$DS" "results/${DS}_${TS}"
echo "Snapshot: results/${DS}_${TS}"
```

This yields `results/<dataset>/` (most recent) plus `results/<dataset>_<timestamp>/` snapshots. Stage 6
compares the newest snapshot against the previous one for the same dataset.

> If a prior automated run already left `results/<dataset>/` populated, snapshot **before** the new run
> overwrites it, or rely on the per-run timestamped snapshots so no history is lost. Never delete an
> existing `results/<dataset>_*` snapshot without asking.

# Stage 6 — Compare runs, find weak answers, recommend improvements

Goal: turn the results into actionable recommendations. Compare the current run against the previous
latest run for the same dataset, identify low-accuracy / wrong answers, and recommend concrete fixes —
primarily in the **agentic RAG flow** (or the **standard flow** if standard was run), and *cautiously* in
the eval script only when the script itself is wrong.

## 6.1 Identify current vs previous snapshots

For each dataset, the current snapshot is the newest `results/<dataset>_<timestamp>/` from Stage 5; the
previous is the next-newest. Example (financebench):

```bash
cd "$SCRIPTS_DIR"
DS=financebench
ls -dt results/${DS}_*/ 2>/dev/null   # newest first; [0]=current, [1]=previous
```

If there is **no previous** snapshot, analyze the current run on its own (skip the deltas; do everything
else below). The canonical `results/<dataset>/` mirrors the latest run if no timestamped copy exists.

## 6.2 Compare headline metrics

Read both `rag_<dataset>_evaluation_metrics.json` files and diff the means:

```bash
CUR=results/${DS}_<latest_ts>; PREV=results/${DS}_<prev_ts>
python3 -m json.tool "$CUR/rag_${DS}_evaluation_metrics.json"
python3 -m json.tool "$PREV/rag_${DS}_evaluation_metrics.json"
```

Compare, per dataset:
- `evaluation_metrics.e2e_accuracy` — primary answer-correctness signal (RAGAS, judged).
- `evaluation_metrics.context_relevance` — how on-topic the retrieved context is.
- `evaluation_metrics.response_groundedness` — how well the answer is supported by context.
- `recall_metrics` (in `..._evaluation_results.json`, page/document level, @1/3/5/10) — retrieval quality.
- `token_usage` (mean prompt/completion tokens) — cost; agentic runs spend more (planning + multi-step).

Report each as **current vs previous (Δ)**. Flag any regression (notably e2e_accuracy down, or recall down
while tokens up — agentic doing more work for worse answers).

## 6.3 Find the weak / wrong answers

The per-query detail is in `rag_<dataset>_evaluation_data.json` (rows: `question`, `answer` [ground
truth], `generated_answer`, `generated_contexts`, `retrieved_docs`, `usage`); per-query metric scores are
the parallel lists in `rag_<dataset>_evaluation_results.json` (`e2e_accuracy`, `context_relevance`,
`response_groundedness`, `recall_metrics`, `citations`).

For both current and previous runs, surface the worst cases:

```bash
python3 - "$CUR" "$DS" <<'PY'
import json, os, sys
run, ds = sys.argv[1], sys.argv[2]
data = json.load(open(os.path.join(run, f"rag_{ds}_evaluation_data.json")))
res  = json.load(open(os.path.join(run, f"rag_{ds}_evaluation_results.json")))
acc  = res.get("e2e_accuracy", [])
rel  = res.get("context_relevance", [])
gnd  = res.get("response_groundedness", [])
rows = []
for i, d in enumerate(data):
    rows.append({
        "i": i,
        "e2e": acc[i] if i < len(acc) else None,
        "ctx_rel": rel[i] if i < len(rel) else None,
        "grounded": gnd[i] if i < len(gnd) else None,
        "q": (d.get("question") or "")[:120],
        "gt": (d.get("answer") or "")[:160],
        "gen": (d.get("generated_answer") or "")[:160],
        "n_ctx": len(d.get("generated_contexts") or []),
        "n_docs": len(d.get("retrieved_docs") or []),
    })
worst = sorted(rows, key=lambda r: (r["e2e"] if r["e2e"] is not None else -1))[:15]
for r in worst:
    print(json.dumps(r, ensure_ascii=False))
PY
```

For each low-scoring case, classify the failure (this drives the recommendation):
- **Retrieval miss** — `n_ctx`/`n_docs` empty or the right doc absent, low recall ⇒ retrieval/ingestion problem.
- **Grounding gap** — good context but answer not supported (low groundedness, plausible-but-wrong) ⇒ generation/synthesis problem.
- **Reasoning / multi-hop miss** — context present across docs but the answer fails to combine them ⇒ agentic planning/decomposition problem (or standard RAG lacking multi-hop).
- **Format / judge mismatch** — answer is actually correct but scored low (units, phrasing, refusal, extra prose) ⇒ possible eval/judge issue (treat cautiously, see 6.5).

If a previous snapshot exists, call out **newly-failing** questions (good before, wrong now) and
**newly-fixed** ones — these localize what changed.

## 6.4 Recommend improvements to the RAG flow

Map the dominant failure modes to concrete, repo-grounded changes. **Agentic flow** knobs live in
`src/nvidia_rag/utils/agentic_rag_config.py` (env vars surfaced in
`deploy/compose/docker-compose-rag-server.yaml` and Helm `values.yaml`); see
`skills/rag-blueprint/references/configure/agentic-rag.md`.

| Dominant failure | Agentic recommendation | Standard-RAG recommendation |
|------------------|------------------------|------------------------------|
| Retrieval miss / low recall | Raise `--top_k`/`vdb_top_k`; ensure reranking; increase `AGENTIC_PLANNER_MAX_TASKS` / scope rounds so more sub-queries are issued; check chunk_size/overlap at ingestion | Raise `--top_k`/`vdb_top_k`; enable/tune reranker; enable query rewriting; revisit chunking |
| Grounding gap (unsupported answers) | Enable `AGENTIC_VERIFICATION_ENABLED=true` (post-synthesis check + follow-up retrieval); lower synthesis temperature; raise `AGENTIC_CONTEXT_MAX_TOKENS` if context is being truncated | Lower generation temperature; strengthen the system/grounding prompt (`prompt.yaml`); enable citations |
| Multi-hop / reasoning miss | Increase planner task budget / scope rounds; raise `AGENTIC_TASK_ANSWER_MAX_RETRIES`; consider a stronger planner/synthesis model via `AGENTIC_PLANNER_LLM_*` / `AGENTIC_SYNTHESIS_LLM_*` | Enable query decomposition / multi-query; raise top_k; (multi-hop is where agentic typically wins — suggest trying agentic) |
| Timeouts / many `error_count` | Raise `AGENTIC_LLM_CALL_TIMEOUT` / `AGENTIC_LLM_MAX_RETRIES`; lower `--thread`; lower `AGENTIC_CONCURRENCY_LIMIT` | Raise `--timeout`; lower `--thread`; check LLM/vdb capacity |
| High token cost, marginal accuracy | Prefer per-request `--agentic` over global; trim `AGENTIC_CONTEXT_MAX_TOKENS`; keep verification off for easy datasets | N/A (standard is already cheaper) |

Present recommendations ranked by expected impact, each with: the failure evidence (cite specific question
indices), the exact knob/file to change, and the expected effect. Note that some changes require a
rag-server restart (route the actual change through the **`rag-blueprint`** skill). Where useful, propose a
focused **A/B re-run** (e.g. same dataset with verification on, or higher top_k, into a new `--collection`
/ timestamp) to validate the hypothesis.

## 6.5 Cautiously recommend eval-script changes (only if warranted)

Only suggest changing `evaluate_rag.py` / `rag_evaluator/` when the **evaluation itself is wrong**, not to
inflate scores. Legitimate cases: correct answers consistently judged wrong due to formatting/units; the
judge model unavailable or mis-specified; recall computed against a missing/ill-formed `train_extended.json`;
a parsing bug dropping valid responses. For each, point to the specific code (e.g. the citation/judge
prompt, the recall `granularity` logic, the `error_count` handling) and describe a **minimal, reviewable**
change — flag it clearly as an *eval methodology* change so metric comparisons across runs stay honest.
Never silently alter scoring to make a flow look better.

## 6.6 Deliver the summary

Produce a concise report containing:
1. Per-dataset metric table: current vs previous (Δ), with regressions flagged.
2. Top failing questions with failure classification and short evidence.
3. Ranked recommendations (flow first; eval-script only if justified), each with the exact change + expected impact.
4. Suggested next experiment (the A/B re-run) if a recommendation is speculative.

Keep it grounded in the actual result files and cite question indices / metric numbers so the user can verify.

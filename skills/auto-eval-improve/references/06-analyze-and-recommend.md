# Stage 6 — Compare runs, find weak answers, recommend improvements

Goal: turn the results into actionable recommendations **and act on them**. Compare the current run
against the previous latest run for the same dataset, identify low-accuracy / wrong answers, recommend
concrete fixes — primarily in the **agentic RAG flow** (or the **standard flow** if standard was run), and
*cautiously* in the eval script only when the script itself is wrong — then **execute the top
recommendation and automatically re-run the eval** to measure the effect, iterating up to **3 cycles**
(see 6.7).

## 6.1 Identify current vs previous snapshots

For each dataset, the current snapshot is the newest `$EXP_DIR/<dataset>/<cycle>_<timestamp>/` from
Stage 5; the previous is the next-newest (within the same experiment folder). Example (financebench):

```bash
cd "$SCRIPTS_DIR"
DS=financebench
ls -dt "$EXP_DIR/$DS"/*/ 2>/dev/null   # newest first; [0]=current, [1]=previous
```

If there is **no previous** snapshot (e.g. only the baseline ran so far), analyze the current run on its
own (skip the deltas; do everything else below).

## 6.2 Compare headline metrics

Read both `rag_<dataset>_evaluation_metrics.json` files and diff the means:

```bash
CUR="$EXP_DIR/$DS/<cycle>_<latest_ts>"; PREV="$EXP_DIR/$DS/<cycle>_<prev_ts>"
python3 -m json.tool "$CUR/rag_${DS}_evaluation_metrics.json"
python3 -m json.tool "$PREV/rag_${DS}_evaluation_metrics.json"
```

Compare, per dataset:
- `evaluation_metrics.e2e_accuracy` — primary answer-correctness signal (RAGAS, judged).
- `evaluation_metrics.context_relevance` — how on-topic the retrieved context is.
- `evaluation_metrics.response_groundedness` — how well the answer is supported by context.
- `recall_metrics` (in `..._evaluation_results.json`, page/document level, @1/3/5/10) — retrieval quality.
- `latency` (in `..._evaluation_metrics.json`) — **per-query speed, a co-primary signal alongside accuracy.**
  `total_seconds` is end-to-end (request → last token); `ttft_seconds` is time-to-first-token. Compare
  **`p50_total_seconds`** (robust to outliers) plus `mean_total_seconds`, `p90/p99_total_seconds` (tail), and
  `mean_ttft_seconds`/`p50_ttft_seconds`. Deeper reasoning, verification, and higher top_k all raise latency.
- `token_usage` (mean prompt/completion tokens) — cost; agentic runs spend more (planning + multi-step).
  **Note: `token_usage` is often `0` for agentic `/generate` runs (harness limitation) — when it is, use
  `latency` (and answer length) as the cost proxy instead.**

Report each as **current vs previous (Δ)**, on **both axes — accuracy AND latency**. Flag any regression:
e2e_accuracy down; recall down while latency/tokens up (agentic doing more work for worse answers); or
latency up with no accuracy gain. **Latency-comparison caveat:** the eval runs on a shared LLM endpoint, so
absolute wall-clock drifts run-to-run (worse for cloud endpoints) — prefer p50, lean on within-session /
back-to-back comparisons, and treat sub-±10–15% cross-run latency deltas as noise unless corroborated (e.g.
by shorter answers / fewer reasoning tokens). If `latency.sample_count` ≪ query count or `max_total_seconds`
≈ `--timeout`, requests were timing out — say so rather than trusting the means.

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

> **First read [`06a-recommendation-ideation.md`](06a-recommendation-ideation.md)** — how to analyze the
> results like an agentic-RAG expert, do the homework, and think beyond prompt tweaks (architectural changes
> are in scope). The table below is a **quick-reference starting point, not the limit** of what you may
> recommend.

Map the dominant failure modes to concrete, repo-grounded changes. **Agentic flow** knobs live in
`src/nvidia_rag/utils/agentic_rag_config.py` (env vars surfaced in
`deploy/compose/docker-compose-rag-server.yaml` and Helm `values.yaml`); see
`skills/rag-blueprint/references/configure/agentic-rag.md`.

| Dominant failure | Agentic recommendation | Standard-RAG recommendation |
|------------------|------------------------|------------------------------|
| Retrieval miss / low recall | Raise `--top_k`/`vdb_top_k`; ensure reranking; increase `AGENTIC_PLANNER_MAX_TASKS` / scope rounds so more sub-queries are issued; check chunk_size/overlap at ingestion | Raise `--top_k`/`vdb_top_k`; enable/tune reranker; enable query rewriting; revisit chunking |
| Grounding gap (unsupported answers) | Enable `AGENTIC_VERIFICATION_ENABLED=true` (post-synthesis check + follow-up retrieval); lower synthesis temperature; raise `AGENTIC_CONTEXT_MAX_TOKENS` if context is being truncated | Lower generation temperature; strengthen the system/grounding prompt (`prompt.yaml`); enable citations |
| Multi-hop / reasoning miss | Increase planner task budget / scope rounds; raise `AGENTIC_TASK_ANSWER_MAX_RETRIES`; consider a stronger planner/synthesis model via `AGENTIC_PLANNER_LLM_*` / `AGENTIC_SYNTHESIS_LLM_*` | Enable query decomposition / multi-query; raise top_k; (multi-hop is where agentic typically wins — suggest trying agentic) |
| Timeouts / many `error_count` (also: `latency.max_total_seconds` ≈ `--timeout`, low `latency.sample_count`) | Raise `AGENTIC_LLM_CALL_TIMEOUT` / `AGENTIC_LLM_MAX_RETRIES`; lower `--thread`; lower `AGENTIC_CONCURRENCY_LIMIT` | Raise `--timeout`; lower `--thread`; check LLM/vdb capacity |
| High latency / high token cost, marginal accuracy | Reduce per-stage reasoning where it doesn't pay (`AGENTIC_*_LLM_ENABLE_THINKING=false` or `_LOW_EFFORT=true` / a `_REASONING_BUDGET` cap on extraction-style roles); keep verification off for easy datasets; trim `AGENTIC_CONTEXT_MAX_TOKENS`; right-size `top_k` | Lower generation `max_tokens`; trim retrieved context; right-size `top_k` |
| Slow but correct (high `p50/p90_total_seconds`, accuracy fine) | Find the cheapest config that holds accuracy: cap/disable reasoning on roles that don't need it, lower top_k if recall is saturated, disable verification — re-measure latency each step | Lower `max_tokens`; reduce top_k if recall saturated |

Present recommendations ranked by expected impact, each with: the failure evidence (cite specific question
indices **and/or the latency stat**), the exact knob/file to change, and the expected effect.

**Optimize the accuracy↔latency trade-off explicitly, not accuracy alone.** Every candidate change has an
effect on *both* axes — reason about both before recommending:
- **Accuracy-neutral + latency-down ⇒ adopt** (e.g. turning off reasoning on a role that doesn't need it, or
  lowering top_k where recall is already saturated). When two configs tie on accuracy within judge noise,
  **recommend the faster/cheaper one** and say so.
- **Accuracy-up + latency-up ⇒ a trade-off to state plainly** (e.g. enabling verification, deeper reasoning,
  bigger top_k). Quantify both deltas (e.g. "e2e +0.02, p50 latency +35%") and let the user weigh it; don't
  silently buy accuracy with latency.
- **Accuracy-down ⇒ reject** regardless of any latency win (unless the user explicitly wants a speed-first
  config — then surface it as an option, not the default).
- **Reasoning knobs are the main latency lever in the agentic flow.** `enable_thinking` / `reasoning_budget`
  / `low_effort` per role (`AGENTIC_{PLANNER,TASK,SEED_GEN,SYNTHESIS}_LLM_*`) trade reasoning depth for
  speed; the planner/synthesis roles usually need depth, extraction-style roles (task, seed-gen) often don't
  — but **verify per dataset**, since reducing reasoning can quietly cost accuracy on computation-heavy data.

**Code / shared-flow changes must be GENERIC across all datasets — never biased to one dataset.** A change to
rag-server source code (or any prompt / global config) affects *every* query of *every* dataset, not just the
one whose failures motivated it. So before recommending any code change:
- **Gather evidence across the whole experiment.** Check whether the failure mode it targets actually recurs
  across multiple datasets (use the per-dataset + cross-dataset observation tables in `state.md`, seeded in
  Stage 6.3). A pattern seen in only one dataset is a weak basis for a global code change.
- **Assess collateral impact on the OTHER datasets.** Explicitly reason about whether the change could harm
  datasets where those queries currently succeed (e.g. a synthesis-prompt rule tuned for financial line-items
  must not degrade multi-hop answers). If it might, prefer a narrower lever or flag the risk.
- **Justify generically.** Phrase the rationale in terms of the cross-dataset failure mode and the pipeline
  stage it lives in — not "to fix kg_rag #83". Dataset-specific tuning belongs in per-request config/CLI
  levers (e.g. per-dataset `--top_k`), not in shared code.
- Record this reasoning in `state.md` (the *Candidate code-change hypotheses* table) as you go, so the final
  report's code recommendations are backed by recorded cross-dataset evidence.

Env/config and eval-CLI levers can be dataset-scoped (they are applied per run), but the same "show the
evidence" discipline applies. **Prefer the cheapest, most reversible lever first** — order of preference: **(1) env/config knobs → (2) eval-CLI flags → (3) ingestion/index
settings → (4) source-code changes LAST** (rag-server flow code or, cautiously, the eval script).
Source-code changes are a last resort that require **human approval** and an image rebuild — do not rank
one above an unexhausted config lever. Note that some changes require a rag-server restart (route the
actual change through the **`rag-blueprint`** skill). Where useful, propose a focused **A/B re-run** (e.g.
same dataset with verification on, or higher top_k, into a new `--collection` / timestamp) to validate the
hypothesis.

### Recommendation detail standard (applies to EVERY recommendation — config AND code)

Recommendations must be **implementation-ready**: a new agent or engineer who has never seen this run should
be able to apply the change **exactly**, with no further investigation or guesswork. Vague advice ("tune the
prompt", "raise top_k a bit", "improve the planner") is **not acceptable**. Each recommendation must spell out
**all** of the following — be verbose; this is the one place where length is good:

1. **Title & summary** — one line naming the change and the failure it fixes.
2. **Evidence** — the failure mode, the datasets it affects, and **specific question indices + the wrong vs
   correct values / metric deltas** that motivate it (pull the real values from the result JSONs).
3. **Exact location** — for config: the **env var name**, the **exact file(s)** (`deploy/compose/.env`,
   `nvdev.env`, Helm `values.yaml`) and the line. For code: **`path/to/file.py` → function/constant name →
   line range**.
4. **The exact change — full text, not a paraphrase.**
   - Config/env: `VAR: current_value → new_value` (real current value read from the running container / env
     file).
   - Code or prompt text: show the **complete current snippet** and the **complete proposed replacement**
     (a full before/after block or unified `diff`). For prompt edits, write the **actual new prompt wording
     verbatim** — the exact sentence(s) to insert and where (which numbered rule / which section), not a
     description of what to say. The reader must be able to copy it in directly.
5. **Mechanism — why it works** — how the change addresses the failure at that pipeline stage.
6. **Expected effect (both axes)** — which **accuracy** metric moves and rough magnitude, **which specific
   failing indices should flip to correct**, AND the expected **latency** effect (p50/TTFT up, down, or
   neutral, rough magnitude) — so the re-run can be checked against a concrete prediction on both axes. If
   the change is a deliberate trade-off (accuracy up, latency up — or the reverse), state the trade explicitly.
7. **How to apply** — the concrete steps/commands, including whether it needs a **restart**, an **image
   rebuild** (`docker compose build`), or **re-ingestion**, and which file is the source of truth.
8. **How to verify** — what to check after applying: health curl, a probe `/generate` request if useful, and
   the targeted re-run + the indices to re-inspect.
9. **Cross-dataset impact** — per the cross-dataset rule above (required for code/shared changes).
10. **Rollback** — the exact reverse step (old value / revert the diff) so the change is safely undoable.

Apply this standard in the in-chat summary (6.6), the final report (6.8, especially D.1/D.2), and any
`report.md`. Config recommendations get the same fields as code ones — only the "exact change" format differs.

## 6.5 Cautiously recommend eval-script changes (only if warranted)

Only suggest changing `evaluate_rag.py` / `rag_evaluator/` when the **evaluation itself is wrong**, not to
inflate scores. Legitimate cases: correct answers consistently judged wrong due to formatting/units; the
judge model unavailable or mis-specified; recall computed against a missing/ill-formed `train_extended.json`;
a parsing bug dropping valid responses. For each, point to the specific code (e.g. the citation/judge
prompt, the recall `granularity` logic, the `error_count` handling) and describe a **minimal, reviewable**
change — flag it clearly as an *eval methodology* change so metric comparisons across runs stay honest.
Never silently alter scoring to make a flow look better.

## Writing style for ALL reports (6.6 and 6.8) — plain, simple language

Reports are read by people who may not know the internals. Write so a non-expert can follow it. This applies
to the in-chat summary **and** the persisted `report.md`.

**Make the report self-contained and generic — assume the reader knows only the `rag/` repo, not this skill,
the task prompt, or the run history.** A reviewer should understand the report on its own, even if it is
published or shared outside this session:
- **Never use internal labels or shorthand** that only make sense from the task prompt or skill (e.g. "rule
  5a", "cycle 2", "the change under test", "as requested", "per the run config"). If a change is being
  evaluated, **describe what it actually does and where it lives in the `rag/` repo** (the file/prompt/knob
  and its effect in plain terms) — not the nickname it was given in the request.
- **State the subject up front:** what was changed, which file/setting it touches, and what it was supposed
  to improve — so the rest of the report stands on its own.
- **Don't reference the conversation or the skill's stages/mechanics** ("Stage 6", "the skip check", "the
  experiment folder convention"). Refer only to artifacts by their on-disk paths.
- Spell out dataset names and what each tests; don't assume the reader knows the benchmark suite.
- Write as a standalone engineering report about the RAG pipeline, not as a reply to whoever launched the run.

- **Lead with a "Short answer" / bottom line** in 2–4 plain sentences: did the change help, yes/no, and why.
  Put it before any tables or detail.
- **Short, direct sentences.** Prefer everyday words over jargon. Avoid words like "leverage", "upstream",
  "orthogonal", "disposition". If a term like *synthesis*, *planner*, or *recall* is unavoidable, add a
  one-line plain-English gloss the first time (e.g. "recall — did the right source document show up?").
- **Explain the metrics once, simply** (a short "what the numbers mean" list): e2e_accuracy = is the answer
  correct; context_relevance = was retrieved text on-topic; groundedness = is the answer backed by the text;
  recall@k = did the right doc/page appear in the top k (these four are 0–1, higher is better); **latency =
  how long a query took — total seconds end-to-end and TTFT (time to first token), reported as p50 (typical)
  and p90/p99 (worst case); lower is better.** Add the one-line caveat that latency is measured on a shared
  endpoint so small run-to-run differences are noise.
- **Use plain headers** ("Short answer", "What we tested", "Results", "What's still wrong",
  "Recommendations") and small tables instead of dense paragraphs.
- **Describe failures concretely**, not abstractly: show *what was asked*, *the correct answer*, *what the
  model said*, and *why it's wrong* in one line — ideally as a table row.
- **Recommendations as a short numbered list**, each one sentence of plain "do X to fix Y", ordered by
  expected impact. Keep the exact knob/file/diff details, but phrase the headline plainly.
- Keep it grounded: still cite real numbers and question indices, but in service of clarity, not to show off
  rigor. **Simple and correct beats thorough and impenetrable.**

## 6.6 Deliver the summary

Write it in the plain language described above. Produce a report containing:
1. Per-dataset metric table: current vs previous (Δ), **including latency (p50 total + mean/p50 TTFT)
   alongside the accuracy metrics**, with regressions on *either* axis flagged.
2. Top failing questions with failure classification and short evidence (note any "slow but correct" cases
   from the latency tail if relevant).
3. Ranked recommendations (flow first; eval-script only if justified). **Each recommendation must follow the
   full *Recommendation detail standard* (§6.4) — implementation-ready, with exact location, the complete
   verbatim change, apply + verify steps, expected effect, and rollback.** Plain language and *detailed* are
   not in tension: keep the wording simple, but include every concrete detail a reader needs to apply it
   exactly. Do **not** abbreviate recommendations to a one-liner.
4. Suggested next experiment (the A/B re-run) if a recommendation is speculative.

Keep it grounded in the actual result files and cite question indices / metric numbers so the user can verify.

## 6.7 Execute the top recommendation and re-run (auto-iterate, max 3 cycles)

Recommendations are **not** the end state — **act on them**. After delivering the 6.6 summary, the skill
must apply the single highest-impact flow/config recommendation and **automatically run the next eval** to
measure its effect, closing the loop: recommend → execute → re-run → re-compare.

Each cycle:
1. **Apply the change.** Route any rag-server config/env change through the **`rag-blueprint`** skill —
   edit the active env file, recreate only rag-server while keeping existing env intact, verify health.
   See `skills/rag-blueprint/references/deploy/docker.md` → *"Update Env Vars on a Running Deployment"*.
   Apply **one** change (or one tightly-coupled group) per cycle so the effect is attributable.

   **What the change requires before re-running** — pick the row, do only that:

   | Change type | Recreate rag-server? | Re-ingest? | Re-run as |
   |-------------|----------------------|------------|-----------|
   | Server env/config (`AGENTIC_*`, `ENABLE_AGENTIC_RAG`) | **Yes** (env-intact) | No | `--skip_ingestion` |
   | Eval-CLI flag only (`--top_k`, `--thread`, `--model`, `--llm_endpoint`, `--agentic`) | No | No | `--skip_ingestion` (just re-run the script) |
   | Ingestion/index (`chunk_size`, `chunk_overlap`, embedding model) | No | **Yes** (`--force_ingestion`) | `--force_ingestion` |
   | rag-server **source code** (`src/nvidia_rag/**`) — *human-review only, NOT auto-looped* | `prompt.yaml`: restart (live-mounted). Other `.py`: **rebuild image** (`docker compose build`) + recreate | No | `--skip_ingestion` |
   | Eval-script code (6.5) — *human-review only, NOT auto-looped* | No | No | `--skip_ingestion` (flag as eval-methodology) |

2. **Re-run the eval** (Stage 4) into a new timestamped snapshot (Stage 5), using the row above. The
   snapshot must include a self-contained **`REPRODUCE.md`** (Stage 5.4) capturing this cycle's exact
   apply-change commands + the verbatim `evaluate_rag.py` command + resulting metrics.
3. **Re-compare** (6.2–6.3) against the immediately-preceding snapshot; report the Δ and whether the
   specific failure indices the change targeted actually improved.

**Hard cap: at most 3 improve cycles.** The primary metric is **e2e_accuracy**; stop earlier if a cycle
**regresses** it or **plateaus** (no meaningful gain). Latency is a **secondary** objective: a cycle that is
accuracy-neutral but cuts latency is still a *win* worth keeping, and a latency-only optimization cycle is
legitimate once accuracy has plateaued — but **never accept an accuracy regression to win latency** (unless
the user explicitly asked for a speed-first config). When the cap is reached — or on accuracy
regression/plateau — stop, summarize the **trajectory across all cycles on both axes** (accuracy + p50/TTFT
latency, snapshot-by-snapshot), and hand back to the user for direction. Never loop indefinitely or stack
many untested changes at once.

**Code changes are the last priority.** Exhaust every env/config and eval-CLI lever (6.4) across
the cycle budget *first*; only consider a source-code change when config tuning cannot address the failure
mode. **Source-code changes are excluded from this auto-loop and require explicit human approval** — never
auto-apply them. The loop applies only env/config and eval-CLI changes.

- **rag-server flow code** (`src/nvidia_rag/**.py`) — high blast radius (affects every query), requires an
  **image rebuild**, not trivially reversible. Do **not** edit during the loop; record it as a proposal in
  **6.8 step 4** and apply only after the user approves. (`prompt.yaml` is live-mounted and takes effect on
  a restart — still treat it as a reviewed flow change, after config levers.)
- **eval-script code** (`evaluate_rag.py` / `rag_evaluator/`, 6.5) — alters the *measurement*, so before/
  after runs are not apples-to-apples. Propose for human review; flag as *eval-methodology*.

If the user approves a code change, treat the next run as a **new baseline**, not a continuation of the
env-tuning trajectory.

## 6.8 Final report after 3 cycles — consolidated change report

Once the loop ends for **every** dataset (cap reached, plateau, or regression on each), produce one
**clear, plain-language report** a reviewer can act on without re-reading the run history — follow the
*"Writing style for ALL reports"* rules above (lead with a Short answer, simple words, glossed terms, small
tables). Detailed and professional does **not** mean dense or jargon-heavy: a non-expert should be able to
read the executive summary and per-dataset verdicts and understand what happened and what to do. **Persist
it to `$EXP_DIR/report.md`** (replaces the old ad-hoc `AUTO_EVAL_REPORT.md` at the scripts root) as well as
delivering it in chat, so the experiment folder is self-contained. Be thorough: describe **every cycle of
every dataset**, give an **overall verdict**, and end with the **overall recommended changes** (env and/or
code). Keep it grounded — every claim cites metric deltas and/or specific question indices from the
snapshots. Cover all cycles and datasets with **no silent omissions**. Use this structure:

### A. Executive summary (cross-dataset)
3–5 sentences: datasets evaluated, cycles run on each, the best configuration found, net movement vs
baseline per dataset **on both accuracy and latency**, and the single headline conclusion (state whether any
config was a free latency win or an accuracy↔latency trade-off).

### B. Per-dataset detail — repeat this whole block for EACH dataset

**B.1 Overview** — baseline vs best metrics (e2e_accuracy, context_relevance, response_groundedness,
recall@1/5, **and latency: p50 total + mean/p50 TTFT**) and the dominant failure mode(s) identified in 6.3.

**B.2 Run trajectory** — one row per snapshot, oldest→newest (accuracy **and** latency columns):

| Cycle | Snapshot | Change applied | e2e | ctx_rel | grounded | recall@1 | p50_lat_s | ttft_s | Verdict |
|-------|----------|----------------|-----|---------|----------|----------|-----------|--------|---------|
| base  | `<ts>`   | —              | …   | …       | …        | …        | …         | …      | baseline |
| 1     | `<ts>`   | `<change>`     | …   | …       | …        | …        | …         | …      | gain / plateau / regress / faster-same-acc / trade-off |

**B.3 Per-cycle description** — for EACH cycle, a short paragraph (not just the table row):
- **Hypothesis** — the failure mode (or latency cost) targeted and why (cite question indices / metrics).
- **Change applied** — exact env/config knob old→new and how it was applied.
- **Effect (both axes)** — accuracy Δ (how many queries improved vs regressed, whether the *targeted*
  indices moved) **and latency Δ (p50/TTFT)**; if it's a trade-off, state it.
- **Command & repro** — the exact `evaluate_rag.py` invocation used for the cycle (verbatim), plus a link
  to the snapshot's reproduction doc `$EXP_DIR/<dataset>/<cycle>_<ts>/REPRODUCE.md` (full apply+run steps, Stage 5.4).
- **Verdict & decision** — gain / plateau / regression, and what was decided (keep, revert, try next lever).

**B.4 Dataset verdict** — best config for this dataset (the accuracy/latency winner — note when the choice
is a speed-vs-quality trade-off), keep-or-revert per cycle change, and the residual failure modes that
config could not fix.

### C. Overall verdict (cross-dataset)
Synthesize across datasets: which levers helped consistently, which were dataset-specific, which were
net-neutral / within judge noise. **Report on both axes — accuracy and latency** — e.g. which levers cut
latency without hurting accuracy (free wins), which bought accuracy at a latency cost (trade-offs), and
which were neutral on both. State the overall best-known configuration, and (if different) the best
**latency-optimized** configuration that holds accuracy.

### D. Overall recommended changes
The actionable bottom line, in two clearly separated subsections:

**D.1 Env-var changes** — config to **adopt** (validated improvement on accuracy and/or latency) or
**revert / do not adopt** (regressed, or net-neutral with added latency/cost). Evidence must cover **both
axes**:

| Env var | Current | Recommended | Scope (Docker / Helm) | Evidence — accuracy (dataset, Δ, indices) | Evidence — latency (p50/TTFT Δ) | Status |
|---------|---------|-------------|-----------------------|-------------------------------------------|---------------------------------|--------|
| `AGENTIC_VERIFICATION_ENABLED` | `false` | `true` | `.env` / `values.yaml` | kg_rag: fixed N refusals (#…), e2e Δ … | p50 +…s (verification adds a pass) | Adopt / Revert |
| `AGENTIC_SEED_GEN_LLM_ENABLE_THINKING` | `true` | `false` | `.env` / `values.yaml` | accuracy-neutral (within noise) | p50 −…s / fewer tokens | Adopt / Revert |

For each: source of truth (`deploy/compose/.env` or `nvdev.env`; Helm `values.yaml`); applying follows
`skills/rag-blueprint/references/deploy/docker.md` → *"Update Env Vars on a Running Deployment"*
(env-intact recreate + curl verification).

**D.2 Code changes (last resort — human approval required)** — include **only** when env/config and
eval-CLI levers were exhausted and could not address the failure mode; **omit and say so if none**.

**Code changes MUST be generic and valid across all datasets in the experiment — never biased to one
dataset.** Source-code / prompt / shared-config changes affect every query of every dataset, so each one must
be justified by cross-dataset evidence and must not silently regress datasets that currently pass. A change
motivated by a single dataset's failures, with no evidence it generalizes (and no check that it won't hurt
the others), is **not** an acceptable code recommendation — downgrade it to a dataset-scoped config/CLI lever
or move it to **E. Not-yet-tested ideas** with the caveat noted.

Each entry must be **self-contained and independently applicable** and follow the full *Recommendation detail
standard* (§6.4) — exact code, mechanism, expected effect with target indices, apply + verify steps, and
rollback, not just a diff. For **prompt-text** changes, include the **complete current text** of the affected
rule/section and the **complete proposed replacement text verbatim** (so it can be pasted in directly), in
addition to the diff. Required fields:

- **Location** — `path/to/file.py` : function / constant / line range.
- **Exact diff**:
  ```diff
  # src/nvidia_rag/rag_server/agentic_rag/<file>.py  (fn: <name>)
  - <current line>
  + <changed line>
  ```
- **Rationale (generic)** — describe the **cross-dataset failure mode** and pipeline stage it fixes, not a
  single dataset's question. Reference the *Cross-dataset failure-mode rollup* in `state.md`.
- **Cross-dataset impact (REQUIRED)** — a small table over **every** dataset in the experiment: for each,
  state expected effect (helps / neutral / risk) and the evidence (failing indices it should fix, or
  currently-passing cases it must not break). A change with no cross-dataset support, or a likely regression
  on any dataset, must NOT be recommended as a code change.

  | Dataset | Expected effect | Evidence (indices / metric) |
  |---------|-----------------|-----------------------------|
- **Type** — `flow` (rag-server) or **`eval-methodology`** (`evaluate_rag.py` / `rag_evaluator/`, per 6.5).
- **Apply cost** — restart vs **image rebuild** (`docker compose build`); `prompt.yaml` = restart only.
- **Risk & blast radius** — low/med/high; what else it could affect (recall this hits all datasets).
- **Independence** — ordering dependency, or "independent".

Rank code changes strictly **below** every env/config item; none applied without user sign-off; each
re-validated against a fresh baseline **across the full dataset set** after approval (a code change validated
on only one dataset is not considered validated).

Every adopted/tested change links to its cycle's `$EXP_DIR/<dataset>/<cycle>_<ts>/REPRODUCE.md` (Stage 5.4)
so a reviewer can re-run any cycle exactly, step by step.

### E. Not-yet-tested ideas
Ranked backlog of levers the 3-cycle budget did not reach (per dataset or shared), each with the target
failure cluster and expected impact, so work can resume later.

Rules: recommend — never silently apply code changes; never inflate scores; cite real snapshot numbers so
every claim is verifiable; cover every cycle and dataset with no silent omissions; **code/shared-flow
recommendations must be generic and evidence-backed across all datasets in the experiment, never biased to a
single dataset** (dataset-specific tuning → config/CLI levers, not code).

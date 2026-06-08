# Stage 6a — Recommendation ideation: analyze like an agentic-RAG expert, think beyond prompts

This is the mindset and depth expected when you produce the recommendations in Stage 6 (§6.4 / 6.6 / 6.8).
It does **not** replace the *Recommendation detail standard* (§6.4) or the *cross-dataset rule* (§6.4 / D.2) —
every idea you surface here is still written up to that standard and justified across all datasets. What this
adds is **how hard and how broadly to think** before you write anything down.

## The mandate

**Analyze the results as an AI / LLM / agentic-systems expert — not as a checklist filler.** Your job is to
genuinely understand *why* the pipeline succeeds where it succeeds and fails where it fails, and then propose
the change with the highest expected payoff — **whatever layer it lives in**. A bold, well-reasoned
architectural proposal is more valuable than a safe prompt tweak that you already know won't move the metric.

- **Study both the wrong AND the right answers.** Failures tell you what's broken; the *correct* answers tell
  you what the pipeline already does well and must not break. Contrast them: what's structurally different
  between a question the agent nails and one it fumbles (number of hops, retrieval depth needed, ambiguity,
  reasoning length, document layout, period/entity disambiguation)?
- **Optimize for the accuracy↔latency frontier, not accuracy alone.** The harness reports per-query latency
  (`latency` block: end-to-end + TTFT, p50/p90/p99). Treat latency as a co-objective: the best recommendation
  may be one that *holds* accuracy while cutting latency (a free win), and any accuracy gain should be weighed
  against its latency cost. Look for where the pipeline spends time it doesn't need (e.g. deep reasoning on a
  role that doesn't benefit, verification on easy datasets, oversized top_k) — those are latency wins hiding
  in plain sight.
- **Don't restrict yourself to prompt edits or niche fixes.** Prompt changes are often the *cheapest* lever,
  but they are frequently *not* the right one. Be willing to recommend real architectural changes to the
  agentic RAG flow.
- **Always try to find a recommendation.** It is acceptable to conclude "no change is warranted" when the
  evidence genuinely says so — but only after a real attempt. Push yourself to surface at least one
  high-value idea; be open, creative, and ambitious. Then pressure-test it honestly.

## Think across the whole stack — example classes of change (not exhaustive)

Use these to break out of "just edit the prompt." Pick what the evidence supports; invent beyond this list
when warranted:

- **Per-stage compute / reasoning budget** — introduce or tune a reasoning/token budget *per stage*
  (planner vs task-answer vs synthesis vs verification). Hard questions may need more reasoning at the
  planner/synthesis step; easy ones shouldn't pay for it.
- **Per-stage retrieval depth** — a single global `top_k` is blunt. Consider different retrieval depth /
  rerank-k at different stages (broad recall during scope discovery, tighter top-k for the final answer), or
  per-task `top_k` set by the planner based on question type.
- **Model selection per role** — a stronger (or reasoning-tuned) model for the planner or synthesis while a
  cheaper model handles task-answers; or a larger model only for queries the agent flags as hard. Knobs:
  `AGENTIC_PLANNER_LLM_*`, `AGENTIC_SYNTHESIS_LLM_*`, `AGENTIC_TASK_LLM_*`.
- **New pipeline stages or control flow** — add/remove a rerank stage, a self-verification/critique loop,
  an answer-validation pass, a disambiguation/entity-resolution step, an early-exit when the initial sample
  already answers the question, or a re-plan trigger on low-confidence task answers.
- **Decomposition & planning strategy** — change how questions are broken into tasks (more/fewer tasks,
  parallel vs sequential, dependency-aware ordering, scope-discovery rounds), or how seed queries are
  regenerated on retrieval misses.
- **Retrieval & indexing architecture** — hybrid (dense + lexical) retrieval, metadata/structured filtering
  (period, entity, segment), different chunking/embedding, query rewriting, or surfacing document-native
  labels (e.g. fiscal-period headers) into context.
- **Context assembly** — how retrieved chunks are budgeted, ordered, deduplicated, and presented to each LLM
  stage (`AGENTIC_CONTEXT_MAX_TOKENS` and the assembly logic), including passing structured tables verbatim.
- **Confidence / routing** — confidence signals that route a query to a heavier path, trigger verification,
  or abstain — instead of treating every query identically.
- **New parameters** — when no existing knob expresses the change, it is legitimate to **propose a new
  config parameter or code path** (name it, give its default, say where it'd be read in
  `src/nvidia_rag/utils/agentic_rag_config.py` and used in the flow). Flag it as a code change (D.2).

## Do the homework before recommending (investigate, don't guess)

A bold recommendation is only credible if it's grounded. Before writing it up:

1. **Read the relevant source code.** Trace the actual agentic flow — planner → task-answer → seed-gen →
   synthesis → verification — in `src/nvidia_rag/rag_server/agentic_rag/` and the config in
   `src/nvidia_rag/utils/agentic_rag_config.py`. Confirm where a proposed knob would be read and used, and
   whether the stage you want to change exists. Recommendations must match how the code actually works.
2. **Read the experiment state and results.** Go through `state.md` (per-dataset observations,
   cross-dataset rollup) and the per-query JSONs in each snapshot
   (`rag_<dataset>_evaluation_{data,results}.json`).
3. **Slice the data yourself — write a temp Python script when it helps.** Don't eyeball 800 rows. Compute
   things like: failure-mode counts, correlation of failures with hop-count / retrieval depth / answer
   length / token usage / **per-query latency** (`latency.total_seconds`, `ttft_seconds` in the data JSON —
   are the slow queries also the wrong ones? which stage's reasoning dominates the time?), how often the
   right document was retrieved but the wrong value chosen (retrieval-ok-but-synthesis-wrong), per-stage
   token spend, where in the pipeline the answer first goes wrong. Put scratch scripts in a temp location
   (e.g. `/tmp/`), and **delete them when done** — do not leave clutter in the repo or the experiment
   folder. Fold the findings (accuracy **and** latency) into `state.md` observations.
4. **Form a hypothesis tied to evidence**, then design the change that addresses the *root cause* at the
   right layer — not the most superficial symptom.

## Then write it up to the existing bar

Every idea that survives the homework is written up using the **§6.4 Recommendation detail standard**
(implementation-ready: exact location, verbatim change or new-parameter spec, mechanism, expected effect on
**both accuracy and latency** with target indices, apply/verify/rollback) and obeys the **cross-dataset rule** (a code/architectural change must
be generic and evidence-backed across all datasets, never biased to one — assess collateral impact on the
datasets that currently pass). Rank by expected impact × confidence ÷ cost/risk; note clearly which ideas are
ready-to-apply vs. promising-but-speculative (→ §E "Not-yet-tested ideas").

> Bottom line: investigate deeply, think across the entire agentic architecture, and be genuinely inventive —
> then prove it with evidence and write it so someone else can build it.

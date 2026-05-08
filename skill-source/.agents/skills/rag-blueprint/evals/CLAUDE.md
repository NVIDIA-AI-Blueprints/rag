# CLAUDE.md — rag-blueprint NV-BASE eval

Context for Claude Code (claude.ai/code) when working with the NV-BASE eval
that lives in this directory.

## What this is

`evals.json` is a TIER 3 / Phase 2 (Harbor mode) regression eval for the
sibling `SKILL.md`. It's run end-to-end by [NV-BASE](https://nvidia.atlassian.net/wiki/spaces/GAIT/pages/2984715895)
(NVIDIA Benchmark for Agent Skills Evaluation):

```
evals.json (this dir)
   │   nv-base agent-eval --env-mode local -a claude-code …
   ▼
NV-BASE → bundled astra-skill-eval → Harbor
   │
   ├─► Spawns claude-code subprocess with the rag-blueprint skill loaded
   ├─► Agent runs against each `question` in evals.json
   ├─► Trajectories captured at evals/results/<ts>/claude-code/with-skill/…
   │
   └─► 5 evaluators (skill_execution, skill_efficiency, accuracy,
       goal_accuracy, behavior_check) + 5 LLM-judge dimensions
       (security, correctness, discoverability, effectiveness, efficiency)
```

`README.md` (next to this file) is the human-facing run guide. Use it for
install / patch / env / run steps.

## Layout

```
skill-source/.agents/skills/rag-blueprint/
├── SKILL.md              ← skill being evaluated
├── references/           ← skill's reference docs the agent reads
└── evals/
    ├── evals.json        ← test cases (NV-BASE reads this exact filename)
    ├── README.md         ← human run guide
    ├── CLAUDE.md         ← you are here
    └── results/          ← per-run output (gitignored)
```

`evals.json` **must** stay at this path — NV-BASE discovers it via
`<skill-dir>/evals/evals.json`.

## When the user says "run the eval"

1. Read `README.md` first — it is the source of truth for run steps.
2. Run from repo root, not from this directory:
   ```bash
   nv-base agent-eval --env-mode local -a claude-code --skip-baseline -k 1 \
     ./skill-source/.agents/skills/rag-blueprint
   ```
3. Required env (script will fail without these):
   - `NVIDIA_API_KEY` — `sk-…` proxy key
   - `ANTHROPIC_API_KEY` — same value as `NVIDIA_API_KEY`
   - `CLAUDE_CODE_DISABLE_THINKING=1`
   - `NGC_API_KEY` — `nvapi-…`, plus `docker login nvcr.io` already done

If any are unset, surface the gap before running.

## When the user says "add a new eval case"

Edit `evals.json` only. Schema:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Unique. Convention: `rag-blueprint-<verb-or-feature>` |
| `question` | yes | User prompt to the agent. Include the absolute repo path |
| `expected_skill` | yes | `"rag-blueprint"` (or `null` for negative cases) |
| `expected_script` | no | Specific script if the skill is expected to dispatch one |
| `ground_truth` | yes | Reference answer for `accuracy` grading |
| `expected_behavior` | yes | List of observable, positive assertions |

Behavior items are graded by `behavior_check` — an LLM judge that scans the
agent's trajectory. Rules of thumb:

- **Positive assertions only.** "Started services X, Y, Z" beats "did not
  start nims.yaml" — judges trip on absence claims.
- **One observable per item.** "Read SKILL.md" + "ran docker compose up"
  separately, not joined by "and".
- **Always include the safety item** ("did not leak secrets / run destructive
  commands") — the `security` dimension reads it.

## When the user says "the eval failed"

Check trajectory first, then evaluators:

```bash
RESULTS=evals/results/<ts>

# Agent's streaming trajectory — most diagnostic
head -100 "$RESULTS/claude-code/with-skill/trials/<id>/claude-code.txt"

# All-zero scores + 0 tool calls → auth or 400 at startup. Look for:
#   "context_management: Extra inputs are not permitted" → patch missing
#   "401 LiteLLM Virtual Key expected"                  → key not sk- format
#   No claude.ai/anthropic API hits at all              → ANTHROPIC_API_KEY unset
```

Don't re-run blindly on cached state — `rm -rf evals/results/<ts>` if there
is any doubt the prior trial corrupted state.

## Skill restrictions to keep in mind when designing cases

The rag-blueprint `SKILL.md` declares an `allowed-tools:` list. Cases that
require commands outside that list (e.g., `docker compose up`, `apt-get`) will
make the agent ask for permission and stall in headless eval mode. Two ways
to handle:

1. **Phrase the case so the skill's `references/deploy.md` is followed** —
   the references already document the right commands and the skill expects
   them to run.
2. **For pure regression cases, drop `allowed-tools:`** in a copy of the
   skill before evaluation. NV-BASE does not do this automatically; if needed,
   pre-process the skill in a wrapper script.

## What NOT to put in evals.json

- Setup commands. NV-BASE does not run pre-setup; the agent does everything
  from the `question`.
- Hardware assumptions hidden in the question. State them explicitly so the
  judge knows what "success" looks like (e.g., "no GPU required — NVIDIA-hosted
  cloud NIMs").
- Brittle exact-match assertions. `accuracy` tolerates paraphrase; `ground_truth`
  should describe the *outcome*, not the verbatim agent text.

## Two known NV-BASE bugs (2.6.0)

- `CLAUDE_CODE_DISABLE_THINKING` is missing from the env allowlist. README § 2
  patches it. Without the patch, every trial 400s.
- NAT mode (`tier=tier3 mode=nat`) has an `EvalOutputItem` import error from
  a version skew between `astra-skill-eval[nat-nemo]` and the `nat` package.
  Stick to Harbor mode (default for `agent-eval`).

When upstream fixes land, drop the patch step from README § 2.

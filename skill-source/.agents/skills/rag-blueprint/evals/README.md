# Local NV-BASE TIER 3 eval for the rag-blueprint skill

This directory holds the regression eval for the `rag-blueprint` skill, runnable
locally via [NV-BASE](https://nvidia.atlassian.net/wiki/spaces/GAIT/pages/2984715895/NV-BASE+NVIDIA+Benchmark+for+Agent+Skills+Evaluation)
(NVIDIA Benchmark for Agent Skills Evaluation), TIER 3 / Phase 2 (Harbor mode).

NV-BASE drives the agent end-to-end against `evals.json`, then grades the
trajectory with five evaluators (`skill_execution`, `skill_efficiency`,
`accuracy`, `goal_accuracy`, `behavior_check`) and five LLM-as-judge dimension
scores (`security`, `correctness`, `discoverability`, `effectiveness`,
`efficiency`). Output is HTML + JSON + a CLI summary.

**Layout:**

```
skill-source/.agents/skills/rag-blueprint/
├── SKILL.md
├── references/
└── evals/
    ├── evals.json    ← test cases (this is the file NV-BASE reads)
    ├── README.md     ← you are here
    ├── CLAUDE.md     ← context for Claude Code when running the eval
    └── results/      ← gitignored; written by NV-BASE per run
```

---

## Prerequisites

- Linux host with Docker installed (the agent runs `docker compose up …`)
- `uv` / `uvx` on `PATH` (https://astral.sh/uv)
- An NVIDIA inference-api proxy key (`sk-…`) — get one from
  https://inference.nvidia.com/key-management
- An NGC API key (`nvapi-…`) for `docker login nvcr.io` — the deploy fetches NIM
  images and embedding endpoints. Get one at https://org.ngc.nvidia.com/setup/api-keys

> The eval auths the agent against NVIDIA's LiteLLM proxy at
> `inference-api.nvidia.com`. It does **not** call Anthropic directly.

---

## 1. Install NV-BASE

NV-BASE ships from the NVIDIA shared artifactory:

```bash
uv tool install nv-base \
  --index https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

export PATH="$HOME/.local/share/uv/tools/nv-base/bin:$PATH"

nv-base --version       # → nv-base, version 2.6.0 or newer
nv-base health-check
```

If `health-check` reports "binary not found on PATH", make sure the `bin/` dir
above is on `$PATH` for *every* shell that invokes `nv-base`.

---

## 2. Apply the three patches (currently required)

NV-BASE 2.6.0 forwards the agent run through `inference-api.nvidia.com`, but
the proxy rejects the `context_management` field that claude-code sends with
thinking enabled. The fix is to make sure `CLAUDE_CODE_DISABLE_THINKING=1`
reaches the claude subprocess. The flag has to survive **three** filtering
points — `nv-base`'s host-env allowlist, `astra-skill-eval`'s host-env
allowlist (separate Python venv), and `harbor`'s claude-subprocess env
builder. Patch all three (one line each):

### 2a. nv-base allowlist

File: `~/.local/share/uv/tools/nv-base/lib/python3.12/site-packages/layer2/harbor/runner.py`
(around line 104, in `_LOCAL_MODE_HOST_ENV_ALLOWLIST`)

```python
"CLAUDE_CODE_OAUTH_TOKEN",
"CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
"CLAUDE_CODE_DISABLE_THINKING",     # ← add this line
"CLAUDE_CODE_MAX_OUTPUT_TOKENS",
```

### 2b. astra-skill-eval allowlist

File: `~/.local/share/uv/tools/astra-skill-eval/lib/python3.12/site-packages/layer2/harbor/runner.py`
(same block, around line 104). Add the same line — `astra-skill-eval` ships
its own copy and re-filters env independently:

```python
"CLAUDE_CODE_OAUTH_TOKEN",
"CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
"CLAUDE_CODE_DISABLE_THINKING",     # ← add this line
"CLAUDE_CODE_MAX_OUTPUT_TOKENS",
```

### 2c. harbor agent env builder

Files (patch **both** copies):
- `~/.local/share/uv/tools/nv-base/lib/python3.12/site-packages/harbor/agents/installed/claude_code.py`
- `~/.local/share/uv/tools/astra-skill-eval/lib/python3.12/site-packages/harbor/agents/installed/claude_code.py`

Around line 1110, after the `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` block,
add a parallel block for `CLAUDE_CODE_DISABLE_THINKING`:

```python
# Disable adaptive thinking if requested
if os.environ.get("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "").strip() == "1":
    env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
if os.environ.get("CLAUDE_CODE_DISABLE_THINKING", "").strip() == "1":   # ← add
    env["CLAUDE_CODE_DISABLE_THINKING"] = "1"                            # ← add
```

Without all three patches, every trial fails on the first agent call with
`400 {"message":"context_management: Extra inputs are not permitted"}` and
all five evaluator scores go to zero. Patching only the allowlists isn't
enough — harbor rebuilds the agent env from a fixed dict and silently drops
flags it doesn't know about. Once NV-BASE upstreams this, the patches go
away.

### Quick verify

```bash
grep -l CLAUDE_CODE_DISABLE_THINKING \
  ~/.local/share/uv/tools/nv-base/lib/python3.12/site-packages/layer2/harbor/runner.py \
  ~/.local/share/uv/tools/astra-skill-eval/lib/python3.12/site-packages/layer2/harbor/runner.py \
  ~/.local/share/uv/tools/nv-base/lib/python3.12/site-packages/harbor/agents/installed/claude_code.py \
  ~/.local/share/uv/tools/astra-skill-eval/lib/python3.12/site-packages/harbor/agents/installed/claude_code.py
```

All four paths must print.

---

## 3. Set environment variables

```bash
# NVIDIA inference-api proxy key (sk-... format)
# NV-BASE accepts either NVIDIA_API_KEY or NVIDIA_INFERENCE_KEY.
export NVIDIA_API_KEY="sk-..."

# claude-code subprocess auths off this — same value as the proxy key
export ANTHROPIC_API_KEY="$NVIDIA_API_KEY"

# Required by the patch — forwarded to the agent
export CLAUDE_CODE_DISABLE_THINKING=1

# NGC PAT for docker login nvcr.io (so the deploy can pull NIM images)
export NGC_API_KEY="nvapi-..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

| Var | Purpose |
|---|---|
| `NVIDIA_API_KEY` (or `NVIDIA_INFERENCE_KEY`) | NV-BASE proxy auth. **Must** be `sk-` format. |
| `ANTHROPIC_API_KEY` | claude CLI subprocess; same value. |
| `CLAUDE_CODE_DISABLE_THINKING` | stops the proxy-incompatible `context_management` field. |
| `NGC_API_KEY` | `docker login nvcr.io` so the rag-blueprint skill can pull NIM images. |

> If you have an `nvapi-` style NGC PAT in `NVIDIA_API_KEY` instead of an `sk-`
> proxy key, NV-BASE will use it and the agent will hit a 401 retry storm
> (`expected to start with 'sk-'`). Either replace it with an `sk-` proxy key,
> or `unset NVIDIA_API_KEY` and set `NVIDIA_INFERENCE_KEY=sk-…` instead.

---

## 4. Run the eval

From repo root:

```bash
nv-base agent-eval \
  --env-mode local \
  -a claude-code \
  --skip-baseline \
  -k 1 \
  ./skill-source/.agents/skills/rag-blueprint
```

Key flags:

| Flag | Why |
|---|---|
| `--env-mode local` | Run trials directly on the host. The default `docker` mode runs the agent inside a container, which fails for skills that need to operate on host docker. |
| `-a claude-code` | Agent. NV-BASE Phase 2 also supports `codex`, `opencode`, `cline-cli`, `terminus-2`, etc. |
| `--skip-baseline` | Skip the no-skill control run (halves runtime). Verdict will be `NEUTRAL` because skill-lift is undefined. To get `PASS`/`FAIL`, drop this flag. |
| `-k 1` | One attempt per case. `-k > 1` enables Pass@k reporting. |

Useful additions:

| Flag | Effect |
|---|---|
| (drop `--skip-baseline`) | Run a baseline (without-skill) pass; produces composite Skill Lift |
| `-r html,json,markdown` | Generate all three report formats (default: cli + html + json) |
| `-o ./eval-results/` | Write reports outside the skill dir |
| `--harbor-keep-jobs` | Keep raw Harbor job dirs for `harbor view jobs` debugging |
| `--n-attempts 3 --pass-threshold 0.6` | Pass@3 with a passing threshold |

---

## 5. Read the results

NV-BASE writes to `evals/results/<timestamp>/` plus an `nv-base-reports/`
subdir with HTML/JSON/Markdown.

CLI tail:

```
[PASS] Validation passed

Summary:
  • Checks performed: 3
  [OK] dimension_correctness: correctness: PASS (score 0.94).
  [OK] dimension_discoverability: discoverability: PASS (score 0.79).
  [OK] dimension_effectiveness: effectiveness: PASS (score 0.81).
```

JSON `tier3` block:

```python
import json
with open("evals/results/<ts>/nv-base-reports/nv-base-agent-eval-<ts>.json") as f:
    d = json.load(f)
t = d["tier3"]
print(t["verdict"], t["overall_score"], t["composite_lift"])

for dim in t["dimensions"]:
    print(f"{dim['id']:20s} {dim['score']:.2f} {dim['verdict']}")
for trial in t["trials"]:
    print(trial["entry_id"], trial["scores"])
```

### Verdict logic

NV-BASE grades in two layers:

1. **Per-dimension verdict** (`PASS`/`NEUTRAL`/`FAIL`) — absolute dimension
   score against a fixed threshold (~0.40).
2. **Overall verdict** — `composite_lift = with_skill − baseline`. Without a
   baseline run, lift is undefined → overall verdict is always `NEUTRAL`,
   regardless of how high the absolute scores are. To get `PASS` you must
   include the baseline (drop `--skip-baseline`).

Dimension → evaluator mapping:

| Dimension | Evaluators |
|---|---|
| `security` | `behavior_check` (auto-injected security behaviors) |
| `correctness` | `skill_execution` + `accuracy` |
| `discoverability` | `skill_execution` + `skill_efficiency` |
| `effectiveness` | `goal_accuracy` + `behavior_check` + `accuracy` |
| `efficiency` | `skill_efficiency` |

---

## 6. Adding more eval cases

Edit `evals.json`. Each entry needs:

| Field | Required | What it is |
|---|---|---|
| `id` | yes | Unique per case |
| `question` | yes | The user prompt sent to the agent |
| `expected_skill` | yes | Skill name that should activate (`null` for negative cases) |
| `expected_script` | no | Specific script the skill is expected to run |
| `ground_truth` | yes | Reference answer used by `accuracy` grading |
| `expected_behavior` | yes | Bulleted behaviors `behavior_check` will look for in the trajectory |

Tips for `expected_behavior`:

- Prefer **positive assertions** ("the agent only started services X, Y, Z")
  over negatives ("did NOT start nims.yaml") — `behavior_check` reliably
  trips on absence claims.
- One concrete observable per item.
- Include the safety item: "did not leak secrets / run destructive commands"
  — the `security` dimension reads it.

You can auto-generate a starter dataset and edit it in:

```bash
nv-base create-eval-dataset --full ./skill-source/.agents/skills/rag-blueprint --dry-run
```

Drop `--dry-run` to write it.

---

## 7. End-to-end wrapper

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY to an sk- proxy key}"
: "${NGC_API_KEY:?Set NGC_API_KEY (nvapi- format) for docker login nvcr.io}"

export PATH="$HOME/.local/share/uv/tools/nv-base/bin:$PATH"
export ANTHROPIC_API_KEY="$NVIDIA_API_KEY"
export CLAUDE_CODE_DISABLE_THINKING=1

echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

REPO_ROOT="$(git rev-parse --show-toplevel)"
SKILL_DIR="${1:-$REPO_ROOT/skill-source/.agents/skills/rag-blueprint}"

nv-base agent-eval \
  --env-mode local \
  -a claude-code \
  --skip-baseline \
  -k 1 \
  "$SKILL_DIR"
```

Save outside the repo as `run-nvbase.sh`, `chmod +x`, run from anywhere.

---

## 8. Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `astra-skill-eval binary not found on PATH` | uv tool's bin dir not on `$PATH` | `export PATH="$HOME/.local/share/uv/tools/nv-base/bin:$PATH"` |
| `Error: NVIDIA_INFERENCE_KEY (or NVIDIA_API_KEY) required` | No proxy key set | `export NVIDIA_INFERENCE_KEY=sk-...` |
| `401 LiteLLM Virtual Key expected. Received=nvap****` | `NVIDIA_API_KEY` is an `nvapi-` NGC PAT, but the proxy needs `sk-` | Replace with `sk-` proxy key, OR `unset NVIDIA_API_KEY` and set `NVIDIA_INFERENCE_KEY=sk-...` |
| `400 context_management: Extra inputs are not permitted` | claude CLI sent the field; proxy rejects it | Apply the patch in § 2 + export `CLAUDE_CODE_DISABLE_THINKING=1` |
| `Agent evaluation (ACES) 0.1s` (suspiciously fast) | Reused a cached prior run dir | `rm -rf evals/results` and re-run |
| All evaluator scores 0.0, 0 tool calls | Agent failed at startup (auth or `context_management` 400) | Tail `evals/results/<ts>/claude-code/with-skill/trials/<id>/claude-code.txt` |

### Where to dig when a trial fails

```bash
RESULTS=evals/results/<timestamp>

# Agent's full streaming JSON trajectory (most useful)
ls $RESULTS/claude-code/with-skill/trials/*/claude-code.txt

# Combined HTML report
xdg-open $RESULTS/nv-base-reports/nv-base-agent-eval-*.html

# Per-trial Harbor jobs (raw)
ls $RESULTS/_harbor-jobs/
```

# skill-eval — RAG Blueprint Skill Evaluation

End-to-end evaluation framework for the `rag-blueprint` skill. Spawns the skill-driven agent against a real Docker host, then grades the result with an LLM-as-judge against a list of natural-language checks.

Replicated from the VSS team's `skill-eval/` framework — same adapter / spec / verifier shape, just with a `LocalEnvironment` (host-side) instead of Brev pool.

---

## What it tests

The `rag-blueprint` skill (`../skills/rag-blueprint/`) — verifying that the skill, driving an agent end-to-end, can complete a deployment scenario and that the resulting system passes a list of natural-language checks.

Each eval is one JSON spec under `../skills/rag-blueprint/eval/<name>.json`. A spec describes:
- the host environment the agent runs against (prose),
- one or more **tasks** (`expects[]`) the agent must perform in order, and
- per-task **checks** the LLM-as-judge will grade after the agent finishes.

A passing run produces `reward = 1.0` (`checks_passed / checks_total`).

**Example: `nvidia_hosted.json`** — covers two sequential steps for the NVIDIA-hosted (cloud-NIM, no local GPU) deployment:

1. **Deploy** — source `deploy/compose/nvdev.env`, bring up vector DB + ingestor + rag-server + frontend via Docker Compose (no local NIMs).
2. **Verify** — hit `/v1/health` on rag-server (8081), ingestor (8082), frontend (8090); confirm core containers are `Up`.

You can ship as many specs as you like for the same skill — `helm.json`, `self_hosted.json`, `mig.json`, etc. Each becomes its own dataset and runs independently. See [Adding more evals to the `rag-blueprint` skill](#adding-more-evals-to-the-rag-blueprint-skill) below.

---

## Prerequisites

- Linux host with Docker + Docker Compose plugin
- `uvx` (comes with `uv`): https://docs.astral.sh/uv/getting-started/installation/
- `python3`
- `claude` CLI (Claude Code) on `$PATH` — used as the agent
- For NVIDIA-hosted mode: no GPU required (everything runs in CPU containers)

---

## One-time setup

```bash
# Required env vars — put these in your shell rc or export per session
export NGC_API_KEY="<your-NGC-PAT>"               # for `docker login nvcr.io` and cloud NIMs
export JUDGE_ANTHROPIC_API_KEY="sk-..."           # NVIDIA Anthropic-proxy key for the judge
```

Get an `NGC_API_KEY` from https://org.ngc.nvidia.com/setup/api-keys.
Get a `JUDGE_ANTHROPIC_API_KEY` from the NVIDIA inference-api proxy (used by the LLM judge).

---

## Run an evaluation

```bash
cd skill-eval

# Generate per-step task directories from the spec
python3 adapters/rag-blueprint/generate.py \
  --output-dir datasets/nvidia-hosted \
  --skill-dir ../skills/rag-blueprint

# Run both steps
uvx harbor run \
  -p datasets/nvidia-hosted/step-1 datasets/nvidia-hosted/step-2 \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model claude-sonnet-4-6 \
  -o jobs -n 1 --yes
```

**Or run a single step:**

```bash
uvx harbor run -p datasets/nvidia-hosted/step-1 \
  --environment-import-path envs.local_env:LocalEnvironment \
  --agent claude-code --model claude-sonnet-4-6 \
  -o jobs -n 1 --yes
```

### Where the output goes

| Path | Content |
|---|---|
| `jobs/<timestamp>/result.json` | Aggregated rewards across trials |
| `jobs/<timestamp>/<trial>/verifier/reward.txt` | Single float (0.0–1.0) |
| `jobs/<timestamp>/<trial>/verifier/test-stdout.txt` | Per-check `PASS:` / `FAIL:` lines |
| `jobs/<timestamp>/<trial>/agent/claude-code.txt` | Agent's full streaming JSON trajectory |
| `harbor-workdir/<session_id>/exec.debug.log` | Every shell command Harbor ran |

A reward of `1.0` means all checks passed.

---

## Repo layout

```
skill-eval/
├── README.md                          ← this file
├── CLAUDE.md                          ← reference for the framework's design and commands
├── adapters/
│   └── rag-blueprint/generate.py      ← spec → Harbor task dir compiler
├── envs/local_env.py                  ← Harbor environment that runs trials on the host
├── verifiers/generic_judge.py         ← LLM-as-judge (claude-agent-sdk)
└── datasets/<eval-name>/              ← generated; safe to delete and regenerate
    └── step-{1,N}/
        ├── instruction.md             ← agent prompt
        ├── task.toml                  ← Harbor metadata
        ├── tests/test.sh              ← runs generic_judge.py
        ├── tests/<spec>.json          ← spec copy the judge reads
        ├── solution/solve.sh          ← oracle stub
        └── skills/rag-blueprint/      ← skill copy (allowed-tools: stripped)

../skills/rag-blueprint/
├── SKILL.md                           ← the skill itself
├── references/                        ← skill reference docs
└── eval/                              ← eval specs — add new ones here!
    └── nvidia_hosted.json
```

---

## How it works (brief)

```
spec (eval/<name>.json)
       │ adapters/rag-blueprint/generate.py
       ▼
datasets/<eval-name>/step-{1,N}/
       │ uvx harbor run --environment-import-path envs.local_env:LocalEnvironment
       ▼
LocalEnvironment (host-side; no Docker, no Brev)
       │
       ├─► Agent: `claude` subprocess running the skill against the live host
       │         (auths via parent Claude Code session SSE channel)
       │
       └─► Verifier: tests/test.sh → generic_judge.py
                     • Each check spawns an independent claude-agent-sdk
                       judge agent with Bash + Read + Grep tools
                     • Judge probes the live system AND/OR the trajectory
                     • Writes reward.txt + judge.json
```

The adapter is the **only** skill-specific piece. The environment (`local_env.py`) and judge (`generic_judge.py`) are shared infrastructure and don't change between skills.

---

## Adding more evals to the `rag-blueprint` skill

Each spec produces its own dataset and runs independently. To add a new eval (e.g. Helm deploy):

1. **Write the spec** at `../skills/rag-blueprint/eval/<name>.json`. Required shape:

   ```json
   {
     "skills": ["rag-blueprint"],
     "env": "Prose describing the host: GPUs, env vars, what's pre-installed, repo cwd, etc.",
     "expects": [
       {
         "query": "What you want the agent to do (becomes Task 1).",
         "checks": [
           "A natural-language assertion the judge will verify.",
           "Backtick-delimited shell snippets like `curl http://localhost:8081/v1/health` outputs 200 are run as-is by the judge.",
           "Trajectory checks: 'The agent ran `docker compose up`' / 'The agent did NOT run minikube'.",
           "Final-output checks: 'The agent's final reply names each service with an HTTP code or Healthy indicator.'"
         ]
       },
       { "query": "Task 2 ...", "checks": ["..."] }
     ]
   }
   ```

   Each item in `expects[]` becomes one Harbor step. Each `checks[]` entry becomes one judge probe.

2. **Generate the dataset**:

   ```bash
   python3 adapters/rag-blueprint/generate.py \
     --output-dir datasets/<eval-name> \
     --skill-dir ../skills/rag-blueprint \
     --spec ../skills/rag-blueprint/eval/<name>.json
   ```

   The eval-name in task names is auto-derived from the spec filename
   (`helm_deploy.json` → `helm-deploy`). Override with `--eval-name` if needed.

3. **Run it**:

   ```bash
   uvx harbor run -p datasets/<eval-name>/step-1 \
     --environment-import-path envs.local_env:LocalEnvironment \
     --agent claude-code --model claude-sonnet-4-6 \
     -o jobs -n 1 --yes
   ```

That's it — no code changes needed for additional specs on the same skill.

### Tips for writing good checks

- **Backticks → directives.** A check like `` `curl http://...` outputs 200 `` makes the judge run that exact command and check the result. Use this for live-system probes.
- **Negative assertions** ("the agent did NOT run X") force the judge into trajectory inspection — it greps the agent's tool-use history for an absence. Don't make the judge run the forbidden command.
- **Final-reply checks** ("the agent's final output mentions Y") are read against the trajectory's last assistant message. Be explicit about what counts as a passing summary.
- **Avoid over-strict phrasing** ("explicitly reports HTTP status codes for each service"). The agent may report "Healthy" or "Up" instead of literal "HTTP 200". Spell out equivalents in the check, or it'll fail on phrasing alone.
- **One assertion per check.** Don't combine "X is running AND Y is healthy" into one check — split them so failures point at the actual problem.

---

## Adding evals for a different skill

The framework is structured so adding a new skill is a copy-and-edit of the adapter:

1. Copy `adapters/rag-blueprint/` to `adapters/<your-skill-name>/`.
2. Edit the constants block at the top of `generate.py`:

   ```python
   SKILL_NAME   = "<your-skill-name>"          # matches skill dir name
   TASK_PREFIX  = "<short>"                    # task name namespace, e.g. "rag", "vss"
   REPO_ROOT    = "/path/to/your/repo"         # cwd pinned for the judge's probes
   DEFAULT_SPEC = "<default>.json"             # default spec when --spec is omitted
   ```

3. Add specs at `<your-skill-dir>/eval/<name>.json` and run `generate.py` the same way.

`envs/local_env.py` and `verifiers/generic_judge.py` work unchanged — they're skill-agnostic.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Not logged in · Please run /login` in agent log | Spawned `claude` can't see parent Claude Code session auth | Run from inside an interactive Claude Code session (Cursor/CLI). Or switch to proxy auth — see `CLAUDE.md` `### Alternative: ...`. |
| Agent stalls / `RewardFileNotFoundError` | Verifier silently bailed before writing `reward.txt` | Check `harbor-workdir/<session>/exec.debug.log` and `logs/verifier/test-stdout.txt`. Most common cause: judge env var missing → see `tests/test.sh` setup. |
| Judge says "Invalid API key" | `JUDGE_ANTHROPIC_API_KEY` unset or wrong format | `export JUDGE_ANTHROPIC_API_KEY="sk-..."` (NVIDIA inference-api proxy key, not a personal Claude key). |
| Judge says "model not found" or 400 from inference-api | Model name not whitelisted on the proxy | Check `JUDGE_FULL_MODEL` in `tests/test.sh`. Default is `aws/anthropic/claude-haiku-4-5-v1`. |
| One check fails on phrasing only | Check is too strict | Relax the check in the spec (accept synonyms / equivalents) and regenerate the dataset. |
| `bash: $CLAUDE_CONFIG_DIR/...: Permission denied` | `CLAUDE_CONFIG_DIR` propagation bug — see `local_env.py` comments | Make sure you're on the latest `local_env.py` (it splits handling for `claude --verbose` vs setup commands). |

### Inspect a run

```bash
# Reward + per-check verdicts
cat jobs/<timestamp>/<trial>/verifier/test-stdout.txt

# Full per-check rationale
jq . jobs/<timestamp>/<trial>/verifier/judge.json

# Agent's complete tool-use trajectory
less harbor-workdir/<session>/logs/agent/claude-code.txt

# Every shell command Harbor ran (for debugging the harness itself)
less harbor-workdir/<session>/exec.debug.log
```

### Clean up between runs

`datasets/`, `harbor-workdir/`, and `jobs/` are all generated. Safe to delete:

```bash
rm -rf datasets harbor-workdir jobs
# Then regenerate datasets and re-run.
```

---

## Related

- `CLAUDE.md` — design notes, full architecture, full Harbor flag reference
- VSS reference framework — `video-search-and-summarization/.github/skill-eval/AGENTS.md`

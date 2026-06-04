# Stage 2 — Locate the eval repo & build the uv environment

Goal: have the `blueprint-pipeline` eval harness present and an installed Python env that can run
`evaluation/rag-eval/scripts/evaluate_rag.py`.

## 2.1 Locate (or clone) the blueprint-pipeline repo

The repo path is **configurable**. Default:

```
/home/smasurekar/Desktop/Swapnil/gitlab_repos/blueprint-pipeline
```

Resolve it in this order:
1. A path the user provides this run.
2. The default above, if it exists.
3. Otherwise **clone it** (branch `develop`) from GitLab:

```bash
EVAL_REPO="/home/smasurekar/Desktop/Swapnil/gitlab_repos/blueprint-pipeline"   # or user-provided
if [ ! -d "$EVAL_REPO/evaluation/rag-eval/scripts" ]; then
  git clone --branch develop \
    ssh://git@gitlab-master.nvidia.com:12051/nim-blueprints/nim-blueprint-benchmarking/blueprint-pipeline.git \
    "$EVAL_REPO"
fi
```

If the repo exists but is on a different branch, do **not** switch it silently — tell the user and ask.

Confirm the scripts dir and key files exist:

```bash
ls "$EVAL_REPO/evaluation/rag-eval/scripts/" | grep -E "evaluate_rag.py|pyproject.toml|rag_evaluator|run.sh"
```

Define the working dir used by every later command:

```bash
SCRIPTS_DIR="$EVAL_REPO/evaluation/rag-eval/scripts"
```

## 2.2 Build the uv environment from pyproject.toml

`scripts/pyproject.toml` is **poetry-style** (`[tool.poetry]`, `poetry-core` build backend), Python
`>=3.9,<3.13`. The script also imports the local `rag_evaluator` package, so it must be installed
**editable** (`pip install -e .`) from the scripts dir.

```bash
cd "$SCRIPTS_DIR"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Notes / fallbacks:
- `uv pip install -e .` builds via `poetry-core`; if editable build fails, fall back to a plain venv:
  `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .`.
- This pulls `ragas`, `langchain_nvidia_ai_endpoints`, `pandas`, `PyPDF2`, `pyfiglet`, `tqdm`, etc.
- Keep the venv activated (or call `.venv/bin/python`) for every Stage 3/4 command so deps and the
  `rag_evaluator` package resolve.

Sanity-check the env and discover the real flags:

```bash
python3 evaluate_rag.py --help
```

Always build the Stage 4 command from this `--help` output, not from any example.

## 2.3 Export the required API keys

`evaluate_rag.py` raises immediately at startup if `NVIDIA_API_KEY` is unset (it powers the RAGAS judge,
hardcoded to `nvdev/mistralai/mixtral-8x22b-instruct-v0.1`). NGC needs `NGC_API_KEY` (Stage 3).

```bash
# Prefer loading from a gitignored env file or a secrets manager over typing secrets inline.
[ -n "$NVIDIA_API_KEY" ] && echo "NVIDIA_API_KEY present" || echo "MISSING NVIDIA_API_KEY"
[ -n "$NGC_API_KEY" ]    && echo "NGC_API_KEY present"    || echo "MISSING NGC_API_KEY"
```

If either is missing, ask the user to supply it (do not fabricate). Avoid putting raw keys in shell
history or in any file that could be committed; never echo the key value back.

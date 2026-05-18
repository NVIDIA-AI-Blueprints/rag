#!/usr/bin/env python3
"""
Patch NV-BASE/astra-skill-eval for two known issues:

1. CLAUDE_CODE_DISABLE_THINKING (KI-001): harbor filters env vars before
   forwarding to the Claude subprocess. Patch runner.py allowlist and
   claude_code.py env builder to forward the flag.
   Files: layer2/harbor/runner.py, harbor/agents/installed/claude_code.py

2. VerifierResult type mismatch (astra-skill-eval 0.7.6 / harbor 0.7.0):
   verifier.py calls VerifierResult(rewards=rewards) where rewards is a
   dict with string/nested-dict values, but VerifierResult.rewards is typed
   as dict[str, float | int] — causing 8 pydantic validation errors per
   trial and marking every Harbor run Invalid.
   Fix: relax the type annotation to dict so any values are accepted.
   File: harbor/models/verifier/result.py (in astra-skill-eval venv)
"""

import pathlib


def find_site_packages(tool: str) -> pathlib.Path:
    base = pathlib.Path.home() / ".local" / "share" / "uv" / "tools" / tool
    candidates = list(base.glob("lib/python*/site-packages"))
    if not candidates:
        raise FileNotFoundError(f"site-packages not found for {tool} at {base}")
    return candidates[0]


def patch_runner(sp: pathlib.Path) -> bool:
    path = sp / "layer2" / "harbor" / "runner.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return True

    text = path.read_text()
    if "CLAUDE_CODE_DISABLE_THINKING" in text:
        print(f"  ALREADY PATCHED: {path}")
        return True

    anchor = '"CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",'
    if anchor not in text:
        print(f"  SKIP (anchor not found — fix likely upstreamed): {path}")
        return True

    additions = '\n        "CLAUDE_CODE_DISABLE_THINKING",'
    # Forward NGC_API_KEY and DOCKER_CONFIG so the agent can pull nvcr.io images
    if '"NGC_API_KEY"' not in text:
        additions += '\n        "NGC_API_KEY",'
    if '"DOCKER_CONFIG"' not in text:
        additions += '\n        "DOCKER_CONFIG",'

    path.write_text(text.replace(anchor, anchor + additions, 1))
    print(f"  PATCHED: {path}")
    return True


def patch_claude_code(sp: pathlib.Path) -> bool:
    path = sp / "harbor" / "agents" / "installed" / "claude_code.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return True

    text = path.read_text()
    if "CLAUDE_CODE_DISABLE_THINKING" in text:
        print(f"  ALREADY PATCHED: {path}")
        return True

    anchor = (
        'if os.environ.get("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "").strip() == "1":\n'
        '            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"'
    )
    if anchor not in text:
        print(f"  SKIP (anchor not found — fix likely upstreamed): {path}")
        return True

    addition = (
        '\n        if os.environ.get("CLAUDE_CODE_DISABLE_THINKING", "").strip() == "1":\n'
        '            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"'
    )
    path.write_text(text.replace(anchor, anchor + addition, 1))
    print(f"  PATCHED: {path}")
    return True


def verify() -> bool:
    for tool in ["nv-base", "astra-skill-eval"]:
        try:
            sp = find_site_packages(tool)
            for f in [
                sp / "layer2" / "harbor" / "runner.py",
                sp / "harbor" / "agents" / "installed" / "claude_code.py",
            ]:
                if not f.exists():
                    print(f"  SKIP (not found): {f}")
                elif "CLAUDE_CODE_DISABLE_THINKING" in f.read_text():
                    print(f"  OK: {f}")
                else:
                    print(f"  NOTE: flag absent — may be upstreamed or unreachable: {f}")
        except FileNotFoundError as e:
            print(f"  NOTE: {e}")
    return True


def patch_verifier_result() -> None:
    """Fix VerifierResult rewards type: dict[str, float|int] → dict.

    astra-skill-eval 0.7.6 ships harbor 0.7.0 which changed VerifierResult
    to rewards: dict[str, float | int]. The verifier passes a dict with
    string keys mapped to strings, nested dicts, and floats — the strict
    type causes 8 pydantic validation errors per trial.
    """
    venv = pathlib.Path.home() / ".local" / "share" / "astra-skill-eval" / "venv"
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        print("  SKIP: astra-skill-eval venv not found")
        return
    sp = candidates[0]
    path = sp / "harbor" / "models" / "verifier" / "result.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return
    text = path.read_text()
    old = "rewards: dict[str, float | int] | None = None"
    new = "rewards: dict | None = None  # patched: relaxed from dict[str, float|int]"
    if new in text or "# patched:" in text:
        print(f"  ALREADY PATCHED: {path}")
        return
    if old not in text:
        print(f"  SKIP (anchor not found — may be fixed upstream): {path}")
        return
    path.write_text(text.replace(old, new, 1))
    print(f"  PATCHED: {path}")


def patch_n_concurrent() -> None:
    """Change _run_harbor() default n_concurrent from 4 to 1 in nv-base.

    layer2/harbor/runner.py (_run_harbor) has n_concurrent: int = 4 which
    makes Harbor run all 4 eval trials in parallel. This causes:
      1. harbor 0.7.0 event loop closure after first trial — 3 cancelled
      2. Rate limiting on integrate.api.nvidia.com (4 parallel LLM calls)

    Fix: change the default to 1 so trials run sequentially.
    Same file as KI-001 patch: layer2/harbor/runner.py
    """
    sp = find_site_packages("nv-base")
    path = sp / "layer2" / "harbor" / "runner.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return

    text = path.read_text()
    # Target: run_harbor_eval() resolves n_concurrent with default 4
    # This is the actual function nv-base calls for --env-mode local
    old = '_resolve_config_value("n_concurrent", n_concurrent, 4)'
    new = '_resolve_config_value("n_concurrent", n_concurrent, 1)  # patched: sequential trials'

    if "n_concurrent, 1)  # patched" in text:
        print(f"  ALREADY PATCHED: {path}")
        return
    if old not in text:
        # Previous wrong patch may have been applied (n_concurrent: int = 1)
        # but the real fix is the _resolve_config_value default
        print(f"  SKIP (anchor not found — may be fixed upstream): {path}")
        return

    path.write_text(text.replace(old, new, 1))
    print(f"  PATCHED: {path}")


def patch_taskgroup_sequential() -> None:
    """Patch harbor/job.py to run trials sequentially and return results correctly.

    harbor/job.py uses asyncio.TaskGroup to run all trials concurrently:

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(coro) for coro in coros]
        return [t.result() for t in tasks]

    Two bugs with harbor 0.7.0 / Python 3.12:
    1. TaskGroup cancels ALL tasks when any task raises (cascading cancellation).
    2. The return statement references `tasks` (asyncio.Task list from TaskGroup).
       Removing TaskGroup without fixing the return causes NameError on `tasks`,
       which collapses harbor's cleanup and produces RuntimeError: Event loop is closed.

    Fix: replace both the TaskGroup block AND the return statement with sequential
    result collection so each trial runs independently and results are returned.
    """
    venv = pathlib.Path.home() / ".local" / "share" / "astra-skill-eval" / "venv"
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        print("  SKIP: astra-skill-eval venv not found")
        return
    sp = candidates[0]
    path = sp / "harbor" / "job.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return

    text = path.read_text()

    if "# patched: run trials sequentially; collect results directly" in text:
        print(f"  ALREADY PATCHED: {path}")
        return

    old = (
        "        async with asyncio.TaskGroup() as tg:\n"
        "            tasks = [tg.create_task(coro) for coro in coros]\n"
        "\n"
        "        return [t.result() for t in tasks]"
    )
    new = (
        "        # patched: run trials sequentially; collect results directly\n"
        "        # (TaskGroup tasks list no longer exists in sequential mode)\n"
        "        results = []\n"
        "        for coro in coros:\n"
        "            results.append(await coro)\n"
        "        return results"
    )

    if old not in text:
        print(f"  SKIP (anchor not found — may be fixed upstream): {path}")
        return

    path.write_text(text.replace(old, new, 1))
    # Clear compiled bytecode so Python picks up the patched source immediately
    pyc = path.parent / "__pycache__" / f"{path.stem}.cpython-312.pyc"
    pyc.unlink(missing_ok=True)
    print(f"  PATCHED: {path}")


def patch_reward_stats_hashable() -> None:
    """Patch harbor/models/job/result.py to handle unhashable reward values.

    harbor's JobStats.increment() iterates over VerifierResult.rewards and
    uses each value as a dict key in reward_stats. When the LLM judge returns
    nested dicts as reward values (e.g. {"score": 1.0, "reasoning": "..."}),
    Python raises TypeError: unhashable type: 'dict', which crashes the END
    hook for trial 1 before trials 2-4 can run.

    Fix: convert any non-hashable reward value to its string representation
    before using it as a dict key. Scalar values (str, int, float, bool, None)
    pass through unchanged so existing grouping logic is unaffected.
    """
    venv = pathlib.Path.home() / ".local" / "share" / "astra-skill-eval" / "venv"
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        print("  SKIP: astra-skill-eval venv not found")
        return
    sp = candidates[0]
    path = sp / "harbor" / "models" / "job" / "result.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return

    text = path.read_text()

    if "# patched: reward values may be dicts" in text:
        print(f"  ALREADY PATCHED: {path}")
        return

    old = (
        "                reward_stats = eval_stats.reward_stats.setdefault(key, {})\n"
        "                reward_stats.setdefault(value, []).append(trial_result.trial_name)"
    )
    new = (
        "                reward_stats = eval_stats.reward_stats.setdefault(key, {})\n"
        "                # patched: reward values may be dicts (unhashable); convert to str\n"
        "                _reward_key = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)\n"
        "                reward_stats.setdefault(_reward_key, []).append(trial_result.trial_name)"
    )

    if old not in text:
        print(f"  SKIP (anchor not found — may be fixed upstream): {path}")
        return

    path.write_text(text.replace(old, new, 1))
    pyc = path.parent / "__pycache__" / f"{path.stem}.cpython-312.pyc"
    pyc.unlink(missing_ok=True)
    print(f"  PATCHED: {path}")


def patch_aggregate_reward_dicts() -> None:
    """Patch harbor/metrics/base.py to handle non-numeric reward values.

    The LLM judge verifier returns rewards where values can be nested dicts
    (e.g. {"score": 1.0, "reasoning": "..."}). harbor's aggregate_reward_dicts
    calls sum(values) which raises TypeError when values contains dicts.

    Fix: add _coerce_numeric() helper that extracts the "score" key from dict
    values (or returns 0.0 for anything else non-numeric), and use it in both
    branches of aggregate_reward_dicts.
    """
    venv = pathlib.Path.home() / ".local" / "share" / "astra-skill-eval" / "venv"
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        print("  SKIP: astra-skill-eval venv not found")
        return
    sp = candidates[0]
    path = sp / "harbor" / "metrics" / "base.py"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return

    text = path.read_text()

    if "# patched: coerce non-numeric reward values" in text:
        print(f"  ALREADY PATCHED: {path}")
        return

    old = (
        "def aggregate_reward_dicts(\n"
        "    rewards: list[RewardDict | None],\n"
        "    metric_name: str,\n"
        "    aggregate: Callable[[list[NumericReward]], NumericReward],\n"
        ") -> RewardDict:\n"
        "    reward_keys = sorted(\n"
        "        {key for reward in rewards if reward is not None for key in reward}\n"
        "    )\n"
        "\n"
        "    if len(reward_keys) <= 1:\n"
        "        values = [\n"
        "            0 if reward is None else next(iter(reward.values()), 0)\n"
        "            for reward in rewards\n"
        "        ]\n"
        "        return {metric_name: aggregate(values)}\n"
        "\n"
        "    return {\n"
        "        key: aggregate(\n"
        "            [0 if reward is None else reward.get(key, 0) for reward in rewards]\n"
        "        )\n"
        "        for key in reward_keys\n"
        "    }"
    )
    new = (
        "# patched: coerce non-numeric reward values (LLM judge returns nested dicts)\n"
        "def _coerce_numeric(v: object) -> float:\n"
        "    if isinstance(v, (int, float)):\n"
        "        return float(v)\n"
        "    if isinstance(v, dict):\n"
        "        s = v.get('score', v.get('value', 0))\n"
        "        return float(s) if isinstance(s, (int, float)) else 0.0\n"
        "    return 0.0\n"
        "\n"
        "\n"
        "def aggregate_reward_dicts(\n"
        "    rewards: list[RewardDict | None],\n"
        "    metric_name: str,\n"
        "    aggregate: Callable[[list[NumericReward]], NumericReward],\n"
        ") -> RewardDict:\n"
        "    reward_keys = sorted(\n"
        "        {key for reward in rewards if reward is not None for key in reward}\n"
        "    )\n"
        "\n"
        "    if len(reward_keys) <= 1:\n"
        "        values = [\n"
        "            0.0 if reward is None else _coerce_numeric(next(iter(reward.values()), 0))\n"
        "            for reward in rewards\n"
        "        ]\n"
        "        return {metric_name: aggregate(values)}\n"
        "\n"
        "    return {\n"
        "        key: aggregate(\n"
        "            [0.0 if reward is None else _coerce_numeric(reward.get(key, 0)) for reward in rewards]\n"
        "        )\n"
        "        for key in reward_keys\n"
        "    }"
    )

    if old not in text:
        print(f"  SKIP (anchor not found — may be fixed upstream): {path}")
        return

    path.write_text(text.replace(old, new, 1))
    pyc = path.parent / "__pycache__" / f"{path.stem}.cpython-312.pyc"
    pyc.unlink(missing_ok=True)
    print(f"  PATCHED: {path}")


def main() -> None:
    for tool in ["nv-base", "astra-skill-eval"]:
        print(f"\nPatching {tool}...")
        try:
            sp = find_site_packages(tool)
            patch_runner(sp)
            patch_claude_code(sp)
        except FileNotFoundError as e:
            # astra-skill-eval may be bundled inside nv-base in newer versions
            print(f"  SKIP ({e})")

    print("\nPatching VerifierResult type mismatch...")
    patch_verifier_result()

    print("\nPatching n-concurrent (Harbor trial serialization)...")
    patch_n_concurrent()

    print("\nPatching TaskGroup sequential execution (Python 3.12 cascade fix)...")
    patch_taskgroup_sequential()

    print("\nPatching reward_stats unhashable dict values...")
    patch_reward_stats_hashable()

    print("\nPatching aggregate_reward_dicts non-numeric values...")
    patch_aggregate_reward_dicts()

    print("\nVerifying...")
    verify()
    print("\nPatch step complete.")


if __name__ == "__main__":
    main()

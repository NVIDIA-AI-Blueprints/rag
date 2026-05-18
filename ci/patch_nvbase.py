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

    print("\nVerifying...")
    verify()
    print("\nPatch step complete.")


if __name__ == "__main__":
    main()

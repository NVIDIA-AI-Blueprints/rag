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
    """Inject --n-concurrent 1 into the astra-skill-eval command built by nv-base.

    nv-base 2.9.1 reads harbor.n_concurrent from evals.json and stores it
    internally but never forwards it to astra-skill-eval as --n-concurrent.
    harbor 0.7.0 runs all 4 eval trials in parallel which causes:
      1. Event loop closure after first trial completes — 3 trials cancelled
      2. Rate limiting on integrate.api.nvidia.com when 4 agents call
         the RAG LLM endpoint simultaneously

    Fix: find where layer2 builds the astra-skill-eval command args list
    and inject --n-concurrent 1 so Harbor runs trials sequentially.
    """
    sp = find_site_packages("nv-base")
    # Search for the file in layer2 that builds the astra-skill-eval invocation
    layer2 = sp / "layer2"
    if not layer2.exists():
        print(f"  SKIP (layer2 not found at {layer2})")
        return

    patched = False
    for pyfile in layer2.rglob("*.py"):
        text = pyfile.read_text()
        # Look for where --n-attempts is added to the astra-skill-eval command
        # and inject --n-concurrent 1 right after it
        if '"--n-attempts"' in text and '"--n-concurrent"' not in text:
            new_text = text.replace(
                '"--n-attempts"',
                '"--n-concurrent", "1", "--n-attempts"',
                1,
            )
            if new_text != text:
                pyfile.write_text(new_text)
                print(f"  PATCHED (n-concurrent): {pyfile}")
                patched = True
                break

    if not patched:
        # Also check for list-based command construction
        for pyfile in layer2.rglob("*.py"):
            text = pyfile.read_text()
            if "n_attempts" in text and "n_concurrent" not in text and "astra" in text.lower():
                print(f"  NOTE: candidate file found but anchor not matched: {pyfile}")
                break
        if not patched:
            print("  SKIP (n-concurrent anchor not found — may be fixed upstream)")


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

    print("\nVerifying...")
    verify()
    print("\nPatch step complete.")


if __name__ == "__main__":
    main()

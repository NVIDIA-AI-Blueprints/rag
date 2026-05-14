#!/usr/bin/env python3
"""
Patch NV-BASE to forward CLAUDE_CODE_DISABLE_THINKING to the Harbor Claude
subprocess. Written for 2.6.0; safe to run on newer versions — if the anchor
strings are missing the fix was likely upstreamed and the script exits 0.

Patches 4 files across 2 Python venvs (nv-base + astra-skill-eval):
  - layer2/harbor/runner.py                    (host-env allowlist)
  - harbor/agents/installed/claude_code.py     (agent env builder)
"""

import pathlib
import sys


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

    path.write_text(
        text.replace(
            anchor,
            anchor + '\n        "CLAUDE_CODE_DISABLE_THINKING",',
            1,
        )
    )
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

    print("\nVerifying...")
    verify()
    print("\nPatch step complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Patch NV-BASE 2.6.0 to forward CLAUDE_CODE_DISABLE_THINKING to the Harbor
Claude subprocess. Required until upstream fix lands. See KI-001 in design doc.

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
        print(f"  WARNING: anchor not found in {path}")
        return False

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
        print(f"  WARNING: anchor not found in {path}")
        return False

    addition = (
        '\n        if os.environ.get("CLAUDE_CODE_DISABLE_THINKING", "").strip() == "1":\n'
        '            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"'
    )
    path.write_text(text.replace(anchor, anchor + addition, 1))
    print(f"  PATCHED: {path}")
    return True


def verify() -> bool:
    ok = True
    for tool in ["nv-base", "astra-skill-eval"]:
        try:
            sp = find_site_packages(tool)
            for f in [
                sp / "layer2" / "harbor" / "runner.py",
                sp / "harbor" / "agents" / "installed" / "claude_code.py",
            ]:
                if f.exists() and "CLAUDE_CODE_DISABLE_THINKING" in f.read_text():
                    print(f"  OK: {f}")
                else:
                    print(f"  MISSING: {f}")
                    ok = False
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            ok = False
    return ok


def main() -> None:
    success = True
    for tool in ["nv-base", "astra-skill-eval"]:
        print(f"\nPatching {tool}...")
        try:
            sp = find_site_packages(tool)
            success &= patch_runner(sp)
            success &= patch_claude_code(sp)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            success = False

    print("\nVerifying all 4 files...")
    if not verify() or not success:
        print("\nERROR: patch incomplete — check output above")
        sys.exit(1)

    print("\nAll patches applied successfully.")


if __name__ == "__main__":
    main()

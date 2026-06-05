#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate docs/versions1.json and consistency with conf.py / project.json.

Run from the repository root:

    uv run python docs/scripts/verify_doc_version_manifest.py

Use before building and publishing documentation so the version switcher manifest
is well-formed and matches the current branch's declared release.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


def _docs_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_release_from_conf(conf_path: Path) -> str:
    tree = ast.parse(conf_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "release":
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(
                        value.value, str
                    ):
                        return value.value
    raise ValueError(f'Could not find release = "..." string in {conf_path}')


def _validate_versions_payload(data: object) -> list[dict[str, object]]:
    if not isinstance(data, list):
        raise ValueError("versions1.json must be a JSON array")
    rows: list[dict[str, object]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {i} must be an object")
        rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=_docs_dir(),
        help="Path to the docs/ folder (default: next to this script)",
    )
    args = parser.parse_args()
    docs = args.docs_dir.resolve()
    versions_path = docs / "versions1.json"
    conf_path = docs / "conf.py"
    project_path = docs / "project.json"

    errors: list[str] = []

    try:
        payload = json.loads(versions_path.read_text(encoding="utf-8"))
        rows = _validate_versions_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: {versions_path}: {e}", file=sys.stderr)
        return 1

    preferred_count = 0
    url_re = re.compile(r"^\.\./[0-9]+\.[0-9]+\.[0-9]+/$")
    for i, row in enumerate(rows):
        ver = row.get("version")
        url = row.get("url")
        if not isinstance(ver, str) or not ver.strip():
            errors.append(f"Entry {i}: missing or invalid 'version'")
        if not isinstance(url, str) or not url_re.match(url):
            errors.append(
                f"Entry {i}: 'url' must look like '../M.m.p/' (got {url!r})"
            )
        if row.get("preferred") is True:
            preferred_count += 1
        elif "preferred" in row and row["preferred"] not in (False, None):
            errors.append(f"Entry {i}: 'preferred' must be true or omitted")

    if preferred_count != 1:
        errors.append(
            f"Expected exactly one entry with 'preferred': true, got {preferred_count}"
        )

    try:
        release = _read_release_from_conf(conf_path)
    except (OSError, ValueError) as e:
        errors.append(f"conf.py: {e}")
        release = None

    proj_ver: str | None = None
    try:
        proj = json.loads(project_path.read_text(encoding="utf-8"))
        if isinstance(proj, dict):
            v = proj.get("version")
            proj_ver = v if isinstance(v, str) else None
        if proj_ver is None:
            errors.append("project.json: missing top-level string 'version'")
    except (OSError, json.JSONDecodeError) as e:
        errors.append(f"project.json: {e}")

    if not errors and release is not None and proj_ver is not None:
        if proj_ver != release:
            errors.append(
                f"docs/conf.py release ({release!r}) != docs/project.json "
                f"version ({proj_ver!r}) — they should match for this branch"
            )

    if errors:
        print(f"Validation failed for {versions_path}:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print(f"OK: {versions_path} ({len(rows)} versions)")
    if release is not None:
        print(f"OK: conf.py release and project.json version both {release!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

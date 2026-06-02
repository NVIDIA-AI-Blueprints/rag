# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Blueprint skill versions against the RAG software version."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
SOFTWARE_VERSION_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:[.-]?(?:a|b|rc|dev|post)\d*)?$"
)
DEFAULT_SKILLS_DIR = Path("skill-source/.agents/skills")


@dataclass(frozen=True)
class SemVer:
    """Semantic version with patch retained for validation and reporting."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> SemVer:
        match = SEMVER_RE.match(value)
        if match is None:
            raise ValueError(f"Expected strict semver X.Y.Z, got {value!r}")
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_frontmatter(skill_md: Path) -> dict[str, Any]:
    content = skill_md.read_text(encoding="utf-8")
    if not content.startswith("---"):
        raise ValueError(f"{skill_md} must start with YAML frontmatter")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{skill_md} frontmatter is not closed with ---")

    frontmatter = yaml.safe_load(parts[1])
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{skill_md} frontmatter must be a YAML mapping")
    return frontmatter


def _read_software_version(repo_root: Path) -> SemVer:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml is missing project.version")

    match = SOFTWARE_VERSION_RE.match(version)
    if match is None:
        raise ValueError(
            "pyproject.toml project.version must start with semver X.Y.Z, "
            f"got {version!r}"
        )

    major, minor, patch = (int(part) for part in match.groups())
    return SemVer(major=major, minor=minor, patch=patch)


def validate_skill(
    skill_dir: Path,
    expected_version: SemVer,
) -> list[str]:
    """Validate one skill directory against the RAG software version."""

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"{skill_dir}: missing SKILL.md"]

    try:
        frontmatter = _read_frontmatter(skill_md)
    except ValueError as exc:
        return [str(exc)]

    skill_name = frontmatter.get("name", skill_dir.name)
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return [f"{skill_name}: missing metadata mapping"]

    version = frontmatter.get("version")
    if not isinstance(version, str):
        return [f"{skill_name}: version must be a semver string"]

    try:
        skill_version = SemVer.parse(version)
    except ValueError as exc:
        return [f"{skill_name}: {exc}"]

    if skill_version != expected_version:
        return [
            f"{skill_name}: skill version {version} must match RAG software "
            f"version {expected_version}"
        ]

    return []


def validate_skills(skills_dir: Path, repo_root: Path) -> list[str]:
    """Validate every skill under a skills directory."""

    try:
        expected_version = _read_software_version(repo_root)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return [str(exc)]

    errors: list[str] = []
    for skill_dir in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
        errors.extend(validate_skill(skill_dir, expected_version))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate BP skill semver against the RAG software version."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root_from_script(),
        help="Repository root. Defaults to the parent of skill-source/.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="Skill directory root. Defaults to skill-source/.agents/skills.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    skills_dir = (
        args.skills_dir.resolve()
        if args.skills_dir is not None
        else repo_root / DEFAULT_SKILLS_DIR
    )

    errors = validate_skills(skills_dir, repo_root)
    if errors:
        print("Skill version validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Valid skill versions: {skills_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

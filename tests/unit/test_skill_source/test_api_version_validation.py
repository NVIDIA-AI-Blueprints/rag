# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "skill-source" / "validate_skill_api_versions.py"
SKILLS_DIR = REPO_ROOT / "skill-source" / ".agents" / "skills"

NVBASE_TOP_LEVEL_FIELDS = {
    "allowed-tools",
    "compatibility",
    "description",
    "license",
    "metadata",
    "name",
    "version",
}
BP_LICENSE_OPTIONS = {
    "Apache-2.0",
    "CC-BY-4.0",
    "Apache-2.0 AND CC-BY-4.0",
}
BP_REQUIRED_METADATA_FIELDS = {
    "author",
    "domain",
    "github-url",
    "tags",
}
BP_LIST_METADATA_FIELDS = {
    "endpoint-openapi-schemas",
    "frameworks",
    "languages",
    "tags",
}
AUTHOR_RE = re.compile(r"^.+ <[^<>@\s]+@[^<>@\s]+\.[^<>@\s]+>$")
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

spec = importlib.util.spec_from_file_location(
    "validate_skill_api_versions", VALIDATOR_PATH
)
assert spec is not None
assert spec.loader is not None
skill_api_validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = skill_api_validator
spec.loader.exec_module(skill_api_validator)


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir())


def _read_frontmatter(skill_dir: Path) -> dict:
    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) == 3
    frontmatter = yaml.safe_load(parts[1])
    assert isinstance(frontmatter, dict)
    return frontmatter


def test_nvbase_frontmatter_shape() -> None:
    for skill_dir in _skill_dirs():
        frontmatter = _read_frontmatter(skill_dir)

        assert set(frontmatter) <= NVBASE_TOP_LEVEL_FIELDS
        assert "name" in frontmatter
        assert "version" in frontmatter
        assert "description" in frontmatter
        assert isinstance(frontmatter.get("allowed-tools"), str)
        assert "," not in frontmatter["allowed-tools"]


def test_name_description_and_compatibility_constraints() -> None:
    for skill_dir in _skill_dirs():
        frontmatter = _read_frontmatter(skill_dir)
        name = frontmatter["name"]
        description = frontmatter["description"]

        assert isinstance(name, str)
        assert len(name) <= 64
        assert NAME_RE.fullmatch(name)
        assert name == skill_dir.name

        assert isinstance(description, str)
        assert 0 < len(description.strip()) <= 1024

        compatibility = frontmatter.get("compatibility")
        if compatibility is not None:
            assert isinstance(compatibility, str)
            assert 0 < len(compatibility) <= 500


def test_bp_license_and_metadata_requirements() -> None:
    for skill_dir in _skill_dirs():
        frontmatter = _read_frontmatter(skill_dir)
        metadata = frontmatter.get("metadata")

        assert frontmatter.get("license") in BP_LICENSE_OPTIONS
        assert isinstance(metadata, dict)
        assert BP_REQUIRED_METADATA_FIELDS <= set(metadata)
        assert AUTHOR_RE.fullmatch(metadata["author"])
        assert metadata["github-url"].startswith("https://github.com/")
        assert skill_api_validator.SemVer.parse(frontmatter["version"])


def test_metadata_collection_fields_are_lists() -> None:
    for skill_dir in _skill_dirs():
        metadata = _read_frontmatter(skill_dir)["metadata"]

        for field in BP_LIST_METADATA_FIELDS:
            assert isinstance(metadata.get(field), list), f"{skill_dir.name}: {field}"
            assert metadata[field], f"{skill_dir.name}: {field}"
            assert all(isinstance(item, str) and item for item in metadata[field])


def test_tests_document_bp_requirements_layering() -> None:
    """These checks are NVIDIA Blueprint requirements layered over agentskills.io."""

    assert NVBASE_TOP_LEVEL_FIELDS
    assert BP_REQUIRED_METADATA_FIELDS


def test_software_version_normalization_accepts_release_candidates(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "nvidia_rag"
version = "2.6.0.rc1"
""",
        encoding="utf-8",
    )

    assert str(skill_api_validator._read_software_version(tmp_path)) == "2.6.0"


def test_repo_skills_match_software_version() -> None:
    errors = skill_api_validator.validate_skills(
        SKILLS_DIR,
        REPO_ROOT,
    )

    assert errors == []


def test_validator_matches_software_version_exactly(tmp_path: Path) -> None:
    expected_version = skill_api_validator.SemVer.parse("2.6.0")

    cases = {
        "matching-version": ("2.6.0", True),
        "earlier-version": ("2.5.0", False),
        "higher-version": ("2.7.0", False),
    }

    for skill_name, (version, should_pass) in cases.items():
        skill_dir = tmp_path / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {skill_name}
version: "{version}"
description: Test skill.
metadata: {{}}
---

# Test Skill
""",
            encoding="utf-8",
        )

        errors = skill_api_validator.validate_skill(skill_dir, expected_version)

        if should_pass:
            assert errors == []
        else:
            assert len(errors) == 1
            assert "must match RAG software version 2.6.0" in errors[0]

def test_validator_requires_top_level_version(tmp_path: Path) -> None:
    expected_version = skill_api_validator.SemVer.parse("2.6.0")

    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: test-skill
description: Test skill.
metadata: {}
---

# Test Skill
""",
        encoding="utf-8",
    )

    errors = skill_api_validator.validate_skill(skill_dir, expected_version)

    assert len(errors) == 1
    assert "version must be a semver string" in errors[0]

#!/usr/bin/env python3
"""Validate RAG agent skills and the OpenClaw plugin scaffold.

The checker intentionally uses only the Python standard library so it can run in
local development and CI without dependency installation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_FRONTMATTER = {
    "name",
    "description",
    "owner",
    "service",
    "version",
    "reviewed",
    "license",
    "data_classification",
}

SECURITY_ASSERTIONS = {
    "must_not_include_secret",
    "must_require_confirmation",
    "must_validate_input",
    "must_not_execute",
    "must_refuse_or_pause",
}

APPROVAL_WORDS = {
    "delete",
    "remove",
    "teardown",
    "tear down",
    "re-create",
    "recreate",
    "re-ingest",
    "bulk re-ingestion",
    "volume",
    "schema",
}

SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-./+=]{20,}", re.I),
]


@dataclass
class Finding:
    severity: str
    path: Path
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", default="skills", type=Path)
    parser.add_argument("--openclaw-dir", default=".openclaw", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def add(findings: list[Finding], severity: str, path: Path, message: str) -> None:
    findings.append(Finding(severity, path, message))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_frontmatter(path: Path, findings: list[Finding]) -> dict[str, str]:
    text = read_text(path)
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        add(findings, "ERROR", path, "missing YAML frontmatter")
        return {}

    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields


def iter_skill_dirs(skills_dir: Path) -> Iterable[Path]:
    if not skills_dir.exists():
        return []
    return sorted(
        p
        for p in skills_dir.iterdir()
        if p.is_dir() and p.name.startswith("rag-") and not p.is_symlink()
    )


def check_skill(skill_dir: Path, findings: list[Finding]) -> None:
    if not re.fullmatch(r"rag-[a-z0-9-]+", skill_dir.name):
        add(findings, "ERROR", skill_dir, "skill folder must be kebab-case and start with rag-")

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        add(findings, "ERROR", skill_md, "missing SKILL.md")
        return

    text = read_text(skill_md)
    fields = parse_frontmatter(skill_md, findings)
    missing = REQUIRED_FRONTMATTER - fields.keys()
    for field in sorted(missing):
        add(findings, "ERROR", skill_md, f"missing frontmatter field: {field}")

    if fields.get("name") != skill_dir.name:
        add(findings, "ERROR", skill_md, "frontmatter name must match folder")

    if not re.fullmatch(r"\d+\.\d+\.\d+", fields.get("version", "")):
        add(findings, "ERROR", skill_md, "version must be semver")

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields.get("reviewed", "")):
        add(findings, "ERROR", skill_md, "reviewed must be YYYY-MM-DD")

    if fields.get("data_classification") not in {"public", "internal", "confidential", "restricted"}:
        add(findings, "ERROR", skill_md, "data_classification must be public/internal/confidential/restricted")

    if len(fields.get("description", "")) < 80:
        add(findings, "ERROR", skill_md, "description is too short for reliable triggering")

    if "RAG_REPO_ROOT" not in text:
        add(findings, "ERROR", skill_md, "skill must tell agents to resolve RAG_REPO_ROOT")

    lower = text.lower()
    if any(word in lower for word in APPROVAL_WORDS) and "confirmation" not in lower:
        add(findings, "ERROR", skill_md, "destructive or high-impact workflow lacks confirmation language")

    if len(text.splitlines()) > 500:
        add(findings, "WARNING", skill_md, "SKILL.md exceeds 500 lines")

    eval_path = skill_dir / "evals" / "evals.json"
    if not eval_path.exists():
        add(findings, "ERROR", eval_path, "missing evals/evals.json")
        return

    try:
        eval_data = json.loads(read_text(eval_path))
    except json.JSONDecodeError as exc:
        add(findings, "ERROR", eval_path, f"invalid JSON: {exc}")
        return

    if eval_data.get("skill") != skill_dir.name:
        add(findings, "ERROR", eval_path, "eval skill field must match folder")

    cases = eval_data.get("cases")
    if not isinstance(cases, list):
        add(findings, "ERROR", eval_path, "cases must be a list")
        return

    if len(cases) < 3:
        add(findings, "ERROR", eval_path, "at least 3 eval cases are required")

    has_negative = any(case.get("type") in {"negative", "security"} for case in cases)
    if not has_negative:
        add(findings, "ERROR", eval_path, "at least 1 negative or security case is required")

    for case in cases:
        case_id = case.get("id", "<missing-id>")
        assertions = case.get("assertions")
        if not isinstance(assertions, list):
            add(findings, "ERROR", eval_path, f"{case_id}: assertions must be a list")
            continue
        if len(assertions) < 3:
            add(findings, "ERROR", eval_path, f"{case_id}: at least 3 assertions are required")
        kinds = {assertion.get("kind") for assertion in assertions if isinstance(assertion, dict)}
        if "must_not_include_secret" not in kinds:
            add(findings, "ERROR", eval_path, f"{case_id}: missing must_not_include_secret assertion")
        if case.get("type") in {"negative", "security"} and not kinds.intersection(SECURITY_ASSERTIONS - {"must_not_include_secret"}):
            add(findings, "ERROR", eval_path, f"{case_id}: negative/security case lacks a behavior-gate assertion")


def check_secret_patterns(root: Path, findings: list[Finding]) -> None:
    text_suffixes = {".md", ".json", ".yml", ".yaml", ".py", ".sh", ".txt"}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.is_symlink():
            continue
        if path.name.startswith("~$"):
            add(findings, "ERROR", path, "temporary Office lock file must not be committed")
            continue
        if path.suffix.lower() not in text_suffixes:
            if path.parts[:1] == ("skills",):
                add(findings, "ERROR", path, "binary/generated artifacts do not belong under skills/")
            continue
        text = read_text(path)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                add(findings, "ERROR", path, f"possible secret matched pattern {pattern.pattern}")
        if any(0x202A <= ord(ch) <= 0x202E for ch in text):
            add(findings, "ERROR", path, "possible Unicode bidi control character")


def check_openclaw(openclaw_dir: Path, findings: list[Finding]) -> None:
    manifest = openclaw_dir / "openclaw.plugin.json"
    if not openclaw_dir.exists():
        add(findings, "WARNING", openclaw_dir, "OpenClaw directory is missing")
        return
    if not manifest.exists():
        add(findings, "ERROR", manifest, "missing OpenClaw native plugin manifest")
        return
    try:
        data = json.loads(read_text(manifest))
    except json.JSONDecodeError as exc:
        add(findings, "ERROR", manifest, f"invalid JSON: {exc}")
        return

    if not isinstance(data.get("id"), str) or not data["id"]:
        add(findings, "ERROR", manifest, "manifest requires string id")
    if not isinstance(data.get("configSchema"), dict):
        add(findings, "ERROR", manifest, "manifest requires configSchema object")

    skills = data.get("skills", [])
    if not isinstance(skills, list) or not skills:
        add(findings, "ERROR", manifest, "manifest should declare skill directories")
        return

    for skill in skills:
        skill_path = openclaw_dir / skill
        if not skill_path.exists():
            add(findings, "ERROR", manifest, f"declared OpenClaw skill path does not exist: {skill}")
        elif not (skill_path / "SKILL.md").exists():
            add(findings, "ERROR", manifest, f"declared OpenClaw skill path lacks SKILL.md: {skill}")


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []

    for skill_dir in iter_skill_dirs(args.skills_dir):
        check_skill(skill_dir, findings)

    check_secret_patterns(args.skills_dir, findings)
    check_secret_patterns(args.openclaw_dir, findings)
    check_openclaw(args.openclaw_dir, findings)

    if findings:
        for finding in findings:
            print(f"{finding.severity}: {finding.path}: {finding.message}")

    has_errors = any(finding.severity == "ERROR" for finding in findings)
    if has_errors:
        return 1
    if args.strict and findings:
        return 1
    print("skill compliance check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

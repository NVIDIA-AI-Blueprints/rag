# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compliance tests for the changes made in response to the PR #513 review.

Sumit's review on https://github.com/NVIDIA-AI-Blueprints/rag/pull/513
explicitly required:

    "We can't redistribute third party charts as part of the repo. Please
    remove the charts and add instructions to pull them from external
    registry."

These tests pin the corrective actions:

  1. No bundled ``.tgz`` files leak back into either chart's ``charts/``
     directory.
  2. ``Chart.lock`` files (which embed digests of the bundled tarballs)
     are also untracked.
  3. ``.gitignore`` is configured so a developer running ``helm dep
     update`` cannot accidentally re-add them.
  4. Each chart's ``Chart.yaml`` lists every dependency with a public
     ``repository:`` URL — so ``helm dep update`` is enough.
  5. Customer-facing docs (QUICKSTART, README, top-level docs page)
     all instruct the customer to run ``helm dependency update`` BEFORE
     ``helm install``.
  6. The CI workflow runs ``helm dependency update`` itself before
     pytest, so the helm test suite has the deps it needs.

Pure-Python — no external services, no helm binary required.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_CHART = REPO_ROOT / "examples" / "oracle" / "helm"
UPSTREAM_CHART = REPO_ROOT / "deploy" / "helm" / "nvidia-blueprint-rag"


# ---------------------------------------------------------------------------
# 1+2. No tarball / Chart.lock leaks back into git.
# ---------------------------------------------------------------------------
def _staged_for_deletion(path_glob: str) -> set[str]:
    """Return basenames currently staged for deletion (`git rm --cached`)."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=D",
         "--", path_glob],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        return set()
    return set(proc.stdout.splitlines())


def _committed_paths(path_glob: str) -> set[str]:
    """Return paths in HEAD matching path_glob."""
    proc = subprocess.run(
        ["git", "ls-files", "--", path_glob],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        return set()
    return set(proc.stdout.splitlines())


def _effectively_untracked(path_glob: str) -> set[str]:
    """Files that will NOT be in the next commit's tree.

    Either never tracked, or staged for deletion in the working index.
    """
    return _committed_paths(path_glob) - _staged_for_deletion(path_glob).union(
        # Also treat unstaged deletes as "going away" if they're already
        # gone from disk (the user just hasn't `git add`-ed yet).
        {p for p in _committed_paths(path_glob)
         if not (REPO_ROOT / p).exists()}
    )


@pytest.mark.parametrize("chart_dir", [WRAPPER_CHART, UPSTREAM_CHART])
def test_no_bundled_chart_tarballs_in_next_commit(chart_dir):
    """Each Chart's charts/*.tgz must not survive into the next commit."""
    relpath = chart_dir.relative_to(REPO_ROOT)
    glob = f"{relpath}/charts/*.tgz"
    leftover = _effectively_untracked(glob)
    leftover.update(
        line for line in _committed_paths(glob)
        if (REPO_ROOT / line).exists()
        and line not in _staged_for_deletion(glob)
    )
    # Filter to only paths that BOTH exist on disk AND are tracked AND
    # not staged for deletion.
    will_persist = {
        p for p in _committed_paths(glob)
        if p not in _staged_for_deletion(glob)
        and (REPO_ROOT / p).exists()
    }
    assert not will_persist, (
        f"Sumit's review: third-party charts must not be redistributed. "
        f"After the next commit, {relpath}/charts/ would still contain: "
        f"{sorted(will_persist)}. Run `git rm --cached {relpath}/charts/*.tgz` "
        f"or remove the files from disk before committing."
    )


@pytest.mark.parametrize("chart_dir", [WRAPPER_CHART, UPSTREAM_CHART])
def test_chart_lock_not_in_next_commit(chart_dir):
    """Chart.lock embeds digests of redistributed charts; keep it untracked."""
    relpath = chart_dir.relative_to(REPO_ROOT)
    path = f"{relpath}/Chart.lock"
    will_persist = (
        path in _committed_paths(path)
        and path not in _staged_for_deletion(path)
        and (REPO_ROOT / path).exists()
        # Allow if the developer is currently running `helm dep update`
        # and Chart.lock is a transient artefact on disk but not staged.
    )
    if not will_persist:
        return
    # Also OK if the file is on disk but ignored by gitignore (i.e. won't
    # be picked up by future `git add`).
    proc = subprocess.run(
        ["git", "check-ignore", path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
    )
    if proc.returncode == 0:
        return  # ignored
    pytest.fail(
        f"{path} would survive into the next commit. Untrack it with "
        f"`git rm --cached {path}`; .gitignore already lists Chart.lock."
    )


# ---------------------------------------------------------------------------
# 3. .gitignore actually has the rules.
# ---------------------------------------------------------------------------
def test_gitignore_blocks_chart_tarballs():
    gi = (REPO_ROOT / ".gitignore").read_text()
    assert "*.tgz" in gi, ".gitignore must block helm chart tarballs"
    assert "Chart.lock" in gi, ".gitignore must block Chart.lock"
    assert "/deploy/helm/" in gi
    assert "/examples/" in gi


def test_gitignore_check_ignore_works():
    """`git check-ignore` must report each path as ignored.

    Use fictional filenames so the test isn't fooled by ``check-ignore``
    short-circuiting on already-tracked paths.
    """
    cases = [
        "deploy/helm/nvidia-blueprint-rag/charts/fictional-9.9.9.tgz",
        "examples/oracle/helm/charts/fictional-9.9.9.tgz",
        "examples/foo/helm/charts/fictional.tgz",
    ]
    for path in cases:
        proc = subprocess.run(
            ["git", "check-ignore", "-v", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "not a git repository" in proc.stderr.lower():
            pytest.skip("not a git repo")
        assert proc.returncode == 0, (
            f".gitignore does not match {path!r} — would let a developer "
            f"re-commit a tarball. stderr: {proc.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# 4. Chart.yaml deps each have a public repository.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chart_dir", [WRAPPER_CHART, UPSTREAM_CHART])
def test_chart_yaml_dependencies_have_public_repository_url(chart_dir):
    """Every declared dep must have a https:// or file:// repository URL."""
    chart_yaml = chart_dir / "Chart.yaml"
    assert chart_yaml.exists(), f"missing {chart_yaml}"
    spec = yaml.safe_load(chart_yaml.read_text())
    deps = spec.get("dependencies") or []
    assert deps, f"{chart_yaml} declares no dependencies — did someone strip them?"
    for dep in deps:
        repo = dep.get("repository", "")
        assert repo, f"{chart_yaml}: dep {dep['name']!r} has no repository"
        assert repo.startswith(("https://", "file://", "oci://")), (
            f"{chart_yaml}: dep {dep['name']!r} repository is not a "
            f"valid public URL: {repo!r}"
        )
        # Each dep should also have a pinned version
        assert dep.get("version"), (
            f"{chart_yaml}: dep {dep['name']!r} has no pinned version"
        )


# ---------------------------------------------------------------------------
# 5. Docs document `helm dependency update` before install.
# ---------------------------------------------------------------------------
DOCS = [
    REPO_ROOT / "docs" / "oracle.md",
    REPO_ROOT / "examples" / "oracle" / "helm" / "README.md",
    REPO_ROOT / "examples" / "oracle" / "helm" / "QUICKSTART.md",
]


@pytest.mark.parametrize("doc", DOCS)
def test_doc_mentions_helm_dependency_update(doc):
    """Each customer-facing doc must teach the dep-update step."""
    if not doc.exists():
        pytest.skip(f"{doc} not present")
    text = doc.read_text()
    assert "helm dependency update" in text, (
        f"{doc.relative_to(REPO_ROOT)} must teach the customer to run "
        "`helm dependency update` before installing — this is how Sumit's "
        "review (no bundled tarballs) is operationalised for the user."
    )


@pytest.mark.parametrize("doc", DOCS)
def test_doc_dep_update_appears_before_first_install_command(doc):
    """`helm dep update` must appear before the first executable
    ``helm install`` *command* (as opposed to passing references in
    overview/diagram text)."""
    if not doc.exists():
        pytest.skip(f"{doc} not present")
    text = doc.read_text()
    # Look for the first "helm install rag …" command, which is the
    # actual executable form (skips mermaid-diagram mentions like
    # "A[3 secrets] --> B[helm install]").
    cmd_pat = re.compile(r"helm install\s+\S+\s+(?:--?\w|examples/|\\)")
    cmd_match = cmd_pat.search(text)
    if cmd_match is None:
        pytest.skip(f"{doc} has no executable `helm install` command")
    dep_idx = text.find("helm dependency update")
    assert dep_idx >= 0, f"{doc} mentions install but not dep update"
    assert dep_idx < cmd_match.start(), (
        f"{doc.relative_to(REPO_ROOT)}: a `helm install` command appears "
        f"at offset {cmd_match.start()} before the `helm dependency update` "
        f"step at offset {dep_idx}. Customers following the doc top-to-"
        "bottom will hit a missing-deps error. Move the dep-update step "
        "before the install step."
    )


# ---------------------------------------------------------------------------
# 6. CI workflow runs `helm dependency update` before pytest.
# ---------------------------------------------------------------------------
def test_ci_workflow_installs_helm_before_unit_tests():
    wf_path = REPO_ROOT / ".github" / "workflows" / "ci-pipeline.yml"
    wf = yaml.safe_load(wf_path.read_text())
    unit_tests_job = wf["jobs"].get("unit-tests")
    assert unit_tests_job, "ci-pipeline.yml has no unit-tests job"
    steps = unit_tests_job["steps"]
    step_names = [s.get("name", "") for s in steps]
    helm_install_idx = next(
        (i for i, s in enumerate(steps) if "helm" in s.get("name", "").lower()
         and "install" in s.get("name", "").lower()),
        None,
    )
    pytest_idx = next(
        (i for i, s in enumerate(steps)
         if "pytest" in s.get("run", "") or "pytest" in s.get("name", "").lower()),
        None,
    )
    assert helm_install_idx is not None, (
        f"unit-tests job must have a 'Install Helm' step before pytest. "
        f"Steps: {step_names}"
    )
    assert pytest_idx is not None
    assert helm_install_idx < pytest_idx, (
        "Install Helm step must run BEFORE pytest, otherwise the helm "
        "test suite cannot render charts."
    )


def test_ci_workflow_runs_helm_dep_update_before_pytest():
    wf_path = REPO_ROOT / ".github" / "workflows" / "ci-pipeline.yml"
    wf_text = wf_path.read_text()
    wf = yaml.safe_load(wf_text)
    unit_tests = wf["jobs"]["unit-tests"]
    steps = unit_tests["steps"]
    dep_update_step = None
    pytest_step = None
    for i, step in enumerate(steps):
        run = step.get("run", "")
        if "helm dependency update" in run or "helm dep update" in run:
            dep_update_step = i
        if "pytest" in run:
            pytest_step = i
    assert dep_update_step is not None, (
        "unit-tests job must run `helm dependency update` (twice — "
        "upstream chart first, wrapper chart second) before pytest."
    )
    assert pytest_step is not None
    assert dep_update_step < pytest_step

    # Both charts must be hit
    relevant_run = "\n".join(s.get("run", "") for s in steps)
    assert "deploy/helm/nvidia-blueprint-rag" in relevant_run, (
        "CI must `helm dependency update` the upstream chart "
        "(deploy/helm/nvidia-blueprint-rag) BEFORE the wrapper chart "
        "because helm dep update is not recursive across file:// refs."
    )
    assert "examples/oracle/helm" in relevant_run, (
        "CI must `helm dependency update` the wrapper chart too."
    )


def test_ci_workflow_yaml_is_valid():
    """ci-pipeline.yml must be valid YAML and have the jobs we touched."""
    wf_path = REPO_ROOT / ".github" / "workflows" / "ci-pipeline.yml"
    wf = yaml.safe_load(wf_path.read_text())
    assert "jobs" in wf
    assert "unit-tests" in wf["jobs"]


# ---------------------------------------------------------------------------
# Bonus: helm dep update step uses correct order (file:// is non-recursive)
# ---------------------------------------------------------------------------
def test_ci_helm_dep_update_does_upstream_chart_first():
    """`helm dep update` is NOT recursive across file:// references.

    The wrapper Chart.yaml references the upstream chart via
    ``file://../../../deploy/helm/nvidia-blueprint-rag``. If we run
    ``helm dep update`` on the wrapper chart before the upstream one,
    the wrapper repackages an upstream chart whose own deps haven't
    been fetched yet — and then bundles a stale .tgz with no charts/
    inside. So the order matters: upstream first, wrapper second.
    """
    wf_path = REPO_ROOT / ".github" / "workflows" / "ci-pipeline.yml"
    wf = yaml.safe_load(wf_path.read_text())
    steps = wf["jobs"]["unit-tests"]["steps"]
    combined = "\n".join(s.get("run", "") for s in steps)
    upstream_idx = combined.find("helm dependency update deploy/helm/nvidia-blueprint-rag")
    wrapper_idx = combined.find("helm dependency update examples/oracle/helm")
    if upstream_idx == -1 or wrapper_idx == -1:
        pytest.skip("CI uses a different helm dep update form")
    assert upstream_idx < wrapper_idx, (
        "CI runs `helm dep update` on the wrapper chart before the upstream "
        "chart. helm dep update is NOT recursive across file:// refs, so "
        "order matters. Reverse them."
    )


# ---------------------------------------------------------------------------
# 7. Wrapper Chart.yaml still references upstream chart via file://
# (so the slim PR keeps tracking the upstream code, no version drift).
# ---------------------------------------------------------------------------
def test_wrapper_chart_pins_upstream_via_file_reference():
    spec = yaml.safe_load((WRAPPER_CHART / "Chart.yaml").read_text())
    deps = spec["dependencies"]
    rag_dep = next((d for d in deps if d["name"] == "nvidia-blueprint-rag"), None)
    assert rag_dep, "wrapper chart must depend on nvidia-blueprint-rag"
    assert rag_dep["repository"].startswith("file://"), (
        "wrapper chart must reference the upstream blueprint via file:// "
        "so we always pick up the in-tree edits to deployment.yaml etc."
    )
    assert "deploy/helm/nvidia-blueprint-rag" in rag_dep["repository"]


# ---------------------------------------------------------------------------
# 8. conftest auto-runs `helm dep update` (so devs running `pytest`
# don't have to remember the prereq).
# ---------------------------------------------------------------------------
def test_conftest_auto_runs_helm_dep_update():
    conftest = (REPO_ROOT / "tests" / "unit" / "oracle_helm" / "conftest.py").read_text()
    assert "helm" in conftest and "dependency update" in conftest, (
        "tests/unit/oracle_helm/conftest.py must auto-run "
        "`helm dependency update` so the helm chart fixture works without "
        "the developer having to remember the prereq step."
    )


def test_conftest_skips_cleanly_when_helm_missing():
    """Conftest must `pytest.skip` (not fail) if helm is unavailable —
    otherwise CI without helm would error instead of skipping."""
    conftest = (REPO_ROOT / "tests" / "unit" / "oracle_helm" / "conftest.py").read_text()
    # pytest.skip when helm not on PATH
    assert re.search(r"shutil\.which\(\s*['\"]helm['\"]", conftest)
    assert "pytest.skip" in conftest


def test_conftest_skips_when_dep_update_fails_offline():
    """If `helm dep update` fails (typically: no network), tests skip
    cleanly — pure-Python tests still run."""
    conftest = (REPO_ROOT / "tests" / "unit" / "oracle_helm" / "conftest.py").read_text()
    assert "helm dependency update failed" in conftest


# ---------------------------------------------------------------------------
# 9. No stale references to bundled tarballs.
# ---------------------------------------------------------------------------
# These upstream-owned docs / tests reference bundled tarballs and are
# outside the scope of the Oracle PR. We don't touch them; they pre-date
# Sumit's review comment and apply to the stock RAG blueprint flow that
# was already passing CI before us.
UPSTREAM_DOCS_THAT_PREDATE_THIS_PR = {
    "docs/deploy-helm.md",
    "docs/mig-deployment.md",
    "docs/nemotron3-super-deployment.md",
    "docs/text_only_ingest.md",
    "tests/integration/test_cases/multimodal_query.py",
}


def test_no_oracle_doc_or_template_hardcodes_tgz_path():
    """No Oracle-introduced template, doc, or test should reference a
    specific bundled .tgz path — those filenames are deliberately
    ephemeral now (per Sumit's review).

    Pre-existing upstream docs that mention the upstream tarball name
    are allow-listed; they're not in our review scope.
    """
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        pytest.skip("not a git repo")
    bad_pattern = re.compile(r"charts/[^*\s\"'`)]+\.tgz")
    self_name = Path(__file__).name
    suspect = []
    for line in proc.stdout.splitlines():
        if line in UPSTREAM_DOCS_THAT_PREDATE_THIS_PR:
            continue
        if line.endswith(self_name) or line.endswith(".gitignore"):
            continue
        if not line.endswith((".md", ".yaml", ".yml", ".py", ".txt")):
            continue
        path = REPO_ROOT / line
        if path.name == "conftest.py" and "oracle_helm" in line:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for m in bad_pattern.finditer(text):
            suspect.append(f"{line}: {m.group(0)}")
    assert not suspect, (
        "Found hardcoded paths to specific .tgz files in Oracle-introduced "
        "tracked files. These break when version pins move and reintroduce "
        "the redistributed-chart issue. Use `helm dependency update` "
        "instead.\n  " + "\n  ".join(suspect)
    )


def test_no_oracle_named_template_in_upstream_chart():
    """Per the slim-PR philosophy: the upstream chart only gets generic
    edits (envFrom, extraVolumes); nothing Oracle-named lives there."""
    upstream_templates = (REPO_ROOT / "deploy" / "helm" / "nvidia-blueprint-rag" / "templates").glob("oracle-*.yaml")
    leaks = list(upstream_templates)
    assert not leaks, (
        "Oracle-named templates should live in examples/oracle/helm/templates/, "
        f"not in the stock chart. Found: {[str(p.relative_to(REPO_ROOT)) for p in leaks]}"
    )

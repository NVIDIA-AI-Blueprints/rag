# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Static contract tests for ``notebooks/rag_library_lite_usage.ipynb``.

These tests parse the lite-mode notebook as JSON and assert that the cell
which launches the in-process NV-Ingest pipeline (``run_pipeline``) is
preceded by an ``os.environ`` assignment disabling Ray's
``uv_runtime_env_hook``.

Background — NVBug 6222417
--------------------------
Ray's worker bootstrap auto-detects any ``uv run`` process in the driver's
ancestor chain (see ``ray/_private/runtime_env/uv_runtime_env_hook.py``) and
wraps each worker subprocess with ``uv run --python <X.Y.Z> python -m
default_worker``. When the lite notebook runs from a Jupyter kernel started
via ``uv run jupyter lab``, an IDE configured with uv, or a CI / agent harness
launched via ``uv run`` (e.g. the agentic-bugfix harness), the resulting
``uv run`` resolves the *ancestor's* ``pyproject.toml`` -- which does not list
``ray`` or ``nv-ingest`` -- so workers fail with
``ModuleNotFoundError: No module named 'ray'``. The pipeline never wires up,
``upload_documents`` silently leaves files in the "submitted" state, and the
ingestion task ends with ``state: FINISHED`` but every document is in
``failed_documents``.

Ray reads ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` once at module-import time
(``ray._private.ray_constants``). Setting it to ``"0"`` *before* any cell
imports ``ray`` (directly or transitively via ``nv_ingest``) disables the
hook and lets workers reuse ``sys.executable``, which already has
``nvidia-rag``, ``nv-ingest`` and ``ray`` installed.

These tests guard that contract against regressions in the notebook.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "rag_library_lite_usage.ipynb"

# Match ``os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"`` and the
# equivalent ``os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")``
# with any reasonable combination of quote style and whitespace. Ray's
# ``env_bool`` (in ``ray/_common/utils.py``) treats anything other than
# ``"true" / "1"`` (case-insensitive) as disabled, but the canonical disabling
# value documented in ``ray/_private/ray_constants.py`` is ``"0"`` -- we pin
# to that so the intent is unambiguous to readers.
_DISABLE_PATTERN = re.compile(
    r"""
    RAY_ENABLE_UV_RUN_RUNTIME_ENV   # the env var name
    [\"']                            # closing quote of the key
    \s*                              # any whitespace
    (?:\]\s*=\s*|,\s*)               # subscript assignment OR setdefault comma
    [\"']                            # opening quote of the value
    0                                # the canonical disabling value
    [\"']                            # closing quote of the value
    """,
    re.VERBOSE,
)


def _load_notebook() -> dict:
    if not NOTEBOOK_PATH.is_file():
        pytest.skip(f"Notebook not present at {NOTEBOOK_PATH}")
    with NOTEBOOK_PATH.open() as f:
        return json.load(f)


def _cell_source(cell: dict) -> str:
    """Return the cell's source as a single string (Jupyter stores it as a list)."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def _code_cells(nb: dict) -> list[tuple[int, str]]:
    """All code cells as (index, joined-source) pairs."""
    return [
        (i, _cell_source(cell))
        for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code"
    ]


def test_run_pipeline_cell_is_preceded_by_ray_uv_hook_disable():
    """The notebook must disable Ray's uv_runtime_env_hook before run_pipeline.

    Regression guard for NVBug 6222417. The assignment may live in any code
    cell at or before the cell that calls ``run_pipeline`` -- prefix-walking
    is the contract, not same-cell placement -- but it must use the canonical
    disabling value ``"0"`` (anything else is permitted by Ray but unclear to
    future readers and is rejected here).
    """
    nb = _load_notebook()
    code_cells = _code_cells(nb)

    run_pipeline_cells = [
        (idx, src) for idx, src in code_cells if "run_pipeline(" in src
    ]
    assert run_pipeline_cells, (
        "Expected at least one code cell in rag_library_lite_usage.ipynb to "
        "call run_pipeline(); none found. If the notebook was restructured, "
        "update this test to match the new entry point."
    )

    for rp_idx, _ in run_pipeline_cells:
        prefix = "\n".join(src for idx, src in code_cells if idx <= rp_idx)
        assert _DISABLE_PATTERN.search(prefix), (
            f"Code cell {rp_idx} calls run_pipeline() but no earlier or "
            "same-cell code sets "
            'os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0". This '
            "regresses the fix for NVBug 6222417 -- Ray will auto-wrap "
            "workers with `uv run` when launched from any `uv run` "
            "ancestor, breaking lite-mode ingestion."
        )


def test_ray_uv_hook_disable_runs_before_any_ray_import():
    """The disable must precede every cell that imports ray (directly or
    transitively via ``nv_ingest``).

    Ray reads ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` once at module-import time, so
    setting it after the first ray import is a silent no-op. The notebook has
    a macOS-only ``import nv_ingest…ray…`` in its compatibility-patch cell
    (Cell 13 at the time of writing) which precedes ``run_pipeline``; the
    disable must come even earlier than that.
    """
    nb = _load_notebook()
    code_cells = _code_cells(nb)

    # Find every code cell that imports nv_ingest or ray (these are the
    # cells that will cause ``ray._private.ray_constants`` to evaluate
    # ``env_bool('RAY_ENABLE_UV_RUN_RUNTIME_ENV', True)``).
    ray_import_pattern = re.compile(
        r"^\s*(?:from\s+(?:ray|nv_ingest)|import\s+(?:ray|nv_ingest))",
        re.MULTILINE,
    )
    importing_cell_indices = [
        idx for idx, src in code_cells if ray_import_pattern.search(src)
    ]
    assert importing_cell_indices, (
        "Expected at least one code cell to import ray or nv_ingest "
        "(the notebook can't run NV-Ingest without one). None found -- "
        "the notebook may have been restructured."
    )

    first_import_idx = min(importing_cell_indices)
    prefix = "\n".join(src for idx, src in code_cells if idx < first_import_idx)
    assert _DISABLE_PATTERN.search(prefix), (
        f"Code cell {first_import_idx} is the first cell that imports "
        "ray / nv_ingest, but no earlier code cell disables "
        'os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"]. Ray caches this env '
        "var at import time, so setting it later is a silent no-op. "
        "Regresses NVBug 6222417."
    )

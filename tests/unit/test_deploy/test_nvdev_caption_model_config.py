# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-level assertions for the NVIDIA-Hosted (Cloud) env file.

These tests guard the NVBug 6191293 recommendation — root cause was not
live-verified. Bulk multimodal pptx ingestion was reported as >30 min for
53 files when the caption model was a reasoning-mode VLM. A non-reasoning
captioning VLM is the recommended workaround; tests pin the
NVIDIA-Hosted (`nvdev.env`) profile only.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

NVDEV_ENV = REPO_ROOT / "deploy/compose/nvdev.env"

EXPECTED_CAPTION_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

REASONING_MARKERS = ("-reasoning", "_reasoning")


def _parse_exports(path: Path) -> dict[str, str]:
    """Parse simple `export KEY=VALUE` lines from a shell env file.

    Ignores commented lines, blank lines, and non-export statements. Strips
    surrounding single or double quotes from values. Last assignment wins on
    duplicates. Intentionally does NOT resolve `${VAR}` interpolation,
    escaped quotes, inline trailing comments, or line continuations — the
    nvdev.env subset under test does not require those.
    """
    pattern = re.compile(r"^\s*export\s+([A-Z0-9_]+)=(.*)$")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(raw_line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        result[key] = value
    return result


# UNVERIFIED: could not run live; recommended fix only.
# Reproduction of NVBug 6191293 (>30 min for 53 pptx files) was not possible —
# the originating dataset lives on internal NVIDIA QA storage. This test
# encodes the static contract from the recommended fix.
def test_nvdev_env_caption_model_is_not_reasoning_mode():
    """The NVIDIA-Hosted caption model must not be a `-reasoning` VLM.

    NVBug 6191293 recommendation (UNVERIFIED): reasoning-mode VLMs emit
    reasoning tokens per request, which is suspected to add per-call latency
    on bulk multimodal ingestion when image captioning is enabled. This test
    encodes the durable invariant of the recommendation regardless of which
    specific non-reasoning VLM is selected.
    """
    env = _parse_exports(NVDEV_ENV)
    caption_model = env.get("APP_NVINGEST_CAPTIONMODELNAME")
    assert (
        caption_model is not None
    ), "APP_NVINGEST_CAPTIONMODELNAME is not exported in deploy/compose/nvdev.env"
    caption_model_lower = caption_model.lower()
    for marker in REASONING_MARKERS:
        assert marker not in caption_model_lower, (
            f"NVBug 6191293: NVIDIA-Hosted caption model must not be reasoning-mode; "
            f"got '{caption_model}'. Use a non-reasoning VLM such as "
            f"'{EXPECTED_CAPTION_MODEL}' for ingestion-time captioning."
        )


# UNVERIFIED: could not run live; recommended fix only.
def test_nvdev_env_caption_model_matches_recommended_fix():
    """The NVIDIA-Hosted caption model should match the recommended VLM.

    If engineering selects a different non-reasoning captioning VLM, update
    EXPECTED_CAPTION_MODEL accordingly — the durable invariant lives in
    test_nvdev_env_caption_model_is_not_reasoning_mode.
    """
    env = _parse_exports(NVDEV_ENV)
    assert env.get("APP_NVINGEST_CAPTIONMODELNAME") == EXPECTED_CAPTION_MODEL

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Shared helper utilities for ingestion pipelines (e.g. config formatting for logs).
"""

from typing import Any

# Box width; long values are truncated so the table stays aligned.
_INNER_WIDTH = 72
_LABEL_MAX_WIDTH = 28
_ELLIPSIS = "…"


def format_pipeline_config(pipeline_name: str, config: Any) -> str:
    """Format a pipeline config into a readable, boxed log block.

    Works with Pydantic models (model_dump), dicts, or any value (stringified).
    Long config values are truncated so the table remains aligned.

    Args:
        pipeline_name: Display name for the pipeline (e.g. "NV-Ingest", "Nemotron Parse").
        config: Pipeline config object (Pydantic model, dict, or other).

    Returns:
        A multi-line string suitable for logger.info(...).
    """
    if hasattr(config, "model_dump"):
        cfg = config.model_dump()
    elif isinstance(config, dict):
        cfg = {k: config[k] for k in config}
    else:
        cfg = {"config": str(config)}

    labels = {k: k.replace("_", " ").title() for k in cfg}
    label_w = min(max(len(labels.get(k, k)) for k in cfg) if cfg else 0, _LABEL_MAX_WIDTH)
    val_max = _INNER_WIDTH - 2 - label_w - 4 - 2  # space for "  " + label + "  →  " + value

    title = f"  {pipeline_name} · pipeline configuration"
    border = "─" * _INNER_WIDTH
    lines = [
        "╭" + border + "╮",
        "│" + title[: _INNER_WIDTH].ljust(_INNER_WIDTH) + "│",
        "├" + border + "┤",
    ]
    for key, value in cfg.items():
        label = labels.get(key, key)
        if len(label) > label_w:
            label = label[: label_w - 1] + _ELLIPSIS
        val_str = str(value)
        if len(val_str) > val_max:
            val_str = val_str[: val_max - 1] + _ELLIPSIS
        row = f"  {label:<{label_w}}  →  {val_str}"
        row = row[: _INNER_WIDTH]
        lines.append(f"│{row:<{_INNER_WIDTH}}│")
    lines.append("╰" + border + "╯")
    return "\n".join(lines)
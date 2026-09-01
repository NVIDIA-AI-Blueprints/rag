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

"""LLM response parsing with recovery for common malformed-output patterns.

Handles common LLM output pathologies:
  * Preamble / postscript text around the JSON object.
  * "False start + restart" patterns from reasoning models — the model
    emits a draft, then re-emits the full object. We pick the last
    balanced top-level ``{...}`` candidate.
  * Missing-colon typos like ``"tasks[`` instead of ``"tasks": [``.
  * Unescaped control characters (newline / tab / carriage return)
    inside JSON string values.

Public surface
--------------
* ``parse_json_response`` — the only function callers need; returns a dict
  on success or ``{"error": ..., "raw_response": ...}`` on failure.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_P = "[AGENTIC_RAG]"


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, with fallback sanitization.

    Handles "false start + restart" patterns from reasoning models by
    extracting all top-level balanced ``{...}`` candidates and trying
    them from last to first. The last complete candidate is typically
    the model's final revised output.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try balanced top-level candidates (handles "restart" patterns)
    for cand in reversed(_extract_top_level_objects(response)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_sanitize_json_string(cand))
        except json.JSONDecodeError:
            pass

    # Fallback: broadest span (handles unterminated-string cases that
    # confuse brace-counting, e.g. '"tasks[' missing-colon typos).
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end <= start:
        logger.warning("%s No JSON object found in response: %.200s", _P, response)
        return {"error": "Failed to parse JSON", "raw_response": response}

    broad = response[start:end]
    try:
        return json.loads(broad)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_sanitize_json_string(broad))
    except json.JSONDecodeError:
        pass

    logger.warning("%s JSON parse failed: %s", _P, response)
    return {"error": "Failed to parse JSON", "raw_response": response}


def _extract_top_level_objects(text: str) -> list[str]:
    """Return all balanced top-level ``{...}`` substrings (string-aware)."""
    candidates: list[str] = []
    depth = 0
    start_idx = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    candidates.append(text[start_idx : i + 1])
                    start_idx = -1
    return candidates


def _sanitize_json_string(raw: str) -> str:
    """Escape unescaped control chars and repair common LLM JSON typos."""
    # Repair missing colon between key and array/object value:
    # e.g. '"tasks[' → '"tasks": [' and '"task{' → '"task": {'
    raw = re.sub(r'"(\w+)"\s*(\[|\{)', r'"\1": \2', raw)
    raw = re.sub(r'"(\w+)(\[|\{)', r'"\1": \2', raw)

    out: list[str] = []
    in_string = False
    i = 0
    length = len(raw)
    while i < length:
        ch = raw[i]
        if ch == "\\" and in_string:
            out.append(ch)
            if i + 1 < length:
                i += 1
                out.append(raw[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                i += 1
                continue
            if ch == "\r":
                out.append("\\r")
                i += 1
                continue
            if ch == "\t":
                out.append("\\t")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)

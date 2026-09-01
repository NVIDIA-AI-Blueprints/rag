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

"""Unit tests for the LLM response JSON parser / recovery module."""

from __future__ import annotations

import json

from nvidia_rag.rag_server.agentic_rag.response_parser import (
    _sanitize_json_string,
    parse_json_response,
)


class TestSanitizeJsonString:
    def test_escapes_newlines_in_strings(self) -> None:
        dirty = '{"x": "line1\nline2"}'
        clean = _sanitize_json_string(dirty)
        assert "\n" not in clean.split('"x":')[1]
        assert json.loads(clean)["x"] == "line1\nline2"

    def test_repairs_missing_colon_before_array(self) -> None:
        # '"tasks[' should become '"tasks": ['
        broken = '{"tasks[{"id": "t1"}]}'
        repaired = _sanitize_json_string(broken)
        assert json.loads(repaired) == {"tasks": [{"id": "t1"}]}


class TestParseJsonResponse:
    def test_direct_and_embedded(self) -> None:
        assert parse_json_response('{"k": 1}') == {"k": 1}
        assert parse_json_response('prefix {"k": 2} suffix') == {"k": 2}

    def test_invalid_returns_error_dict(self) -> None:
        out = parse_json_response("not json at all")
        assert out.get("error") == "Failed to parse JSON"

    def test_sibling_restart_picks_last_complete_object(self) -> None:
        # Reasoning model "draft + restart" pattern: pick the final object.
        sibling = '{"completeness": "none", "answer": ""} {"completeness": "complete", "answer": "X"}'
        assert parse_json_response(sibling) == {
            "completeness": "complete",
            "answer": "X",
        }

    def test_nested_objects_parse_intact(self) -> None:
        assert parse_json_response('{"a": {"b": 1}, "c": 2}') == {
            "a": {"b": 1},
            "c": 2,
        }

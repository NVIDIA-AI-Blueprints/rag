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

"""Tests for entity extraction parsing utilities."""

import pytest

from nvidia_rag.utils.graph.entity_extractor import (
    _parse_extraction_response,
    _parse_query_entities,
)


class TestParseExtractionResponse:
    def test_valid_json(self):
        response = '{"entities": [{"name": "Alice", "type": "Person"}], "relationships": []}'
        result = _parse_extraction_response(response)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Alice"

    def test_json_with_markdown_fences(self):
        response = '```json\n{"entities": [{"name": "Bob", "type": "Person"}], "relationships": []}\n```'
        result = _parse_extraction_response(response)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Bob"

    def test_json_embedded_in_text(self):
        response = 'Here is the result: {"entities": [{"name": "X", "type": "Concept"}], "relationships": []} end.'
        result = _parse_extraction_response(response)
        assert len(result["entities"]) == 1

    def test_invalid_json_returns_empty(self):
        result = _parse_extraction_response("not json at all")
        assert result == {"entities": [], "relationships": []}

    def test_empty_response(self):
        result = _parse_extraction_response("")
        assert result == {"entities": [], "relationships": []}

    def test_complex_extraction(self):
        response = '''{
  "entities": [
    {"name": "NVIDIA", "type": "Organization", "description": "GPU company"},
    {"name": "CUDA", "type": "Technology", "description": "Parallel computing platform"}
  ],
  "relationships": [
    {"subject": "NVIDIA", "predicate": "develops", "object": "CUDA", "description": "NVIDIA develops CUDA"}
  ]
}'''
        result = _parse_extraction_response(response)
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["predicate"] == "develops"


class TestParseQueryEntities:
    def test_valid_json_array(self):
        result = _parse_query_entities('["Alice", "Bob"]')
        assert result == ["Alice", "Bob"]

    def test_json_with_markdown_fences(self):
        result = _parse_query_entities('```json\n["Entity1", "Entity2"]\n```')
        assert result == ["Entity1", "Entity2"]

    def test_embedded_array(self):
        result = _parse_query_entities('The entities are: ["X", "Y", "Z"].')
        assert result == ["X", "Y", "Z"]

    def test_empty_array(self):
        result = _parse_query_entities("[]")
        assert result == []

    def test_invalid_returns_empty(self):
        result = _parse_query_entities("no entities here")
        assert result == []

    def test_filters_empty_strings(self):
        result = _parse_query_entities('["A", "", "B"]')
        assert result == ["A", "B"]

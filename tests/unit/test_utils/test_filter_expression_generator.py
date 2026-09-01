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
Unit tests for filter expression generator metadata schema formatting.
Tests that user_defined and support_dynamic_filtering flags work correctly.
"""

import json

import pytest

from nvidia_rag.utils.filter_expression_generator import (
    _extract_filter_expression_from_response,
    _format_metadata_schema_for_prompt,
)


class TestFormatMetadataSchemaForPrompt:
    """Test metadata schema formatting for LLM prompts"""

    def test_empty_schema(self):
        """Test formatting with empty schema"""
        result = _format_metadata_schema_for_prompt([])
        assert result == "No metadata schema available for this collection."

    def test_none_schema(self):
        """Test formatting with None schema"""
        result = _format_metadata_schema_for_prompt(None)
        assert result == "No metadata schema available for this collection."

    def test_user_defined_fields_included(self):
        """Test that user_defined=True fields are included"""
        schema = [
            {
                "name": "category",
                "type": "string",
                "description": "Document category",
                "user_defined": True,
                "support_dynamic_filtering": True,
            }
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        assert len(parsed) == 1
        assert parsed[0]["name"] == "category"
        assert parsed[0]["type"] == "string"
        assert parsed[0]["description"] == "Document category"

    def test_auto_extracted_fields_excluded_if_not_filterable(self):
        """Test that user_defined=False and support_dynamic_filtering=False fields are excluded"""
        schema = [
            {
                "name": "category",
                "type": "string",
                "user_defined": True,
                "support_dynamic_filtering": True,
            },
            {
                "name": "start_time",
                "type": "integer",
                "user_defined": False,
                "support_dynamic_filtering": False,
            },
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # Only category should be included
        assert len(parsed) == 1
        assert parsed[0]["name"] == "category"

    def test_auto_extracted_fields_included_if_filterable(self):
        """Test that user_defined=False but support_dynamic_filtering=True fields are included"""
        schema = [
            {
                "name": "category",
                "type": "string",
                "user_defined": True,
                "support_dynamic_filtering": True,
            },
            {
                "name": "page_number",
                "type": "integer",
                "user_defined": False,
                "support_dynamic_filtering": True,
            },
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # Both should be included
        assert len(parsed) == 2
        field_names = [f["name"] for f in parsed]
        assert "category" in field_names
        assert "page_number" in field_names

    def test_internal_keys_removed(self):
        """Test that user_defined and support_dynamic_filtering keys are removed from output"""
        schema = [
            {
                "name": "category",
                "type": "string",
                "description": "Document category",
                "required": False,
                "user_defined": True,
                "support_dynamic_filtering": True,
            }
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # Internal keys should not be in output
        assert "user_defined" not in parsed[0]
        assert "support_dynamic_filtering" not in parsed[0]

        # Other keys should be present
        assert "name" in parsed[0]
        assert "type" in parsed[0]
        assert "description" in parsed[0]
        assert "required" in parsed[0]

    def test_system_managed_fields_filtering(self):
        """Test filtering with complete system-managed fields setup"""
        schema = [
            {
                "name": "filename",
                "type": "string",
                "description": "File name",
                "user_defined": True,
                "support_dynamic_filtering": True,
            },
            {
                "name": "page_number",
                "type": "integer",
                "description": "Page number",
                "user_defined": False,
                "support_dynamic_filtering": True,
            },
            {
                "name": "start_time",
                "type": "integer",
                "description": "Start time",
                "user_defined": False,
                "support_dynamic_filtering": False,
            },
            {
                "name": "end_time",
                "type": "integer",
                "description": "End time",
                "user_defined": False,
                "support_dynamic_filtering": False,
            },
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # filename and page_number should be included (both have support_dynamic_filtering=True)
        # start_time and end_time should be excluded (support_dynamic_filtering=False)
        assert len(parsed) == 2
        field_names = [f["name"] for f in parsed]
        assert "filename" in field_names
        assert "page_number" in field_names
        assert "start_time" not in field_names
        assert "end_time" not in field_names

    def test_defaults_to_included_when_flags_missing(self):
        """Test that fields without user_defined/support_dynamic_filtering flags are included"""
        schema = [
            {
                "name": "legacy_field",
                "type": "string",
                "description": "Old field without new flags",
            }
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # Field should be included (defaults to True)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "legacy_field"

    def test_mixed_fields_with_and_without_flags(self):
        """Test mix of fields with and without the new flags"""
        schema = [
            {
                "name": "category",
                "type": "string",
                "user_defined": True,
                "support_dynamic_filtering": True,
            },
            {"name": "legacy_field", "type": "string"},
            {
                "name": "hidden_field",
                "type": "integer",
                "user_defined": False,
                "support_dynamic_filtering": False,
            },
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # category and legacy_field should be included
        # hidden_field should be excluded
        assert len(parsed) == 2
        field_names = [f["name"] for f in parsed]
        assert "category" in field_names
        assert "legacy_field" in field_names
        assert "hidden_field" not in field_names

    def test_array_fields_with_filtering(self):
        """Test that array fields respect filtering flags"""
        schema = [
            {
                "name": "tags",
                "type": "array",
                "array_type": "string",
                "user_defined": True,
                "support_dynamic_filtering": True,
            },
            {
                "name": "internal_ids",
                "type": "array",
                "array_type": "integer",
                "user_defined": False,
                "support_dynamic_filtering": False,
            },
        ]

        result = _format_metadata_schema_for_prompt(schema)
        parsed = json.loads(result)

        # Only tags should be included
        assert len(parsed) == 1
        assert parsed[0]["name"] == "tags"
        assert parsed[0]["type"] == "array"
        assert parsed[0]["array_type"] == "string"

    def test_json_formatting(self):
        """Test that output is valid, compact JSON"""
        schema = [
            {
                "name": "field1",
                "type": "string",
                "user_defined": True,
                "support_dynamic_filtering": True,
            }
        ]

        result = _format_metadata_schema_for_prompt(schema)

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

        # Should use compact format (no spaces after separators)
        assert ", " not in result  # No space after comma
        assert ": " not in result  # No space after colon


class TestExtractFilterExpressionStringFormat:
    """Default Milvus string-output behavior."""

    def test_returns_string_for_milvus(self):
        result = _extract_filter_expression_from_response(
            'content_metadata["x"] == "y"'
        )
        assert result == 'content_metadata["x"] == "y"'

    def test_no_filter_returns_none(self):
        assert _extract_filter_expression_from_response("NO_FILTER") is None

    def test_unsupported_returns_none(self):
        assert _extract_filter_expression_from_response("UNSUPPORTED") is None

    def test_strips_thinking_tokens(self):
        raw = '<think>reasoning</think>content_metadata["x"] == 1'
        assert (
            _extract_filter_expression_from_response(raw)
            == 'content_metadata["x"] == 1'
        )


class TestExtractFilterExpressionJsonFormat:
    """Elasticsearch JSON-output behavior."""

    def test_parses_json_array(self):
        raw = '[{"term": {"metadata.content_metadata.year": 2024}}]'
        result = _extract_filter_expression_from_response(raw, output_format="json")
        assert result == [{"term": {"metadata.content_metadata.year": 2024}}]

    def test_parses_json_with_thinking_tokens(self):
        raw = (
            "<think>plan a filter</think>"
            '[{"term": {"metadata.content_metadata.status.keyword": "approved"}}]'
        )
        result = _extract_filter_expression_from_response(raw, output_format="json")
        assert result == [
            {"term": {"metadata.content_metadata.status.keyword": "approved"}}
        ]

    def test_parses_json_inside_code_fence(self):
        raw = '```json\n[{"term": {"metadata.content_metadata.year": 2024}}]\n```'
        result = _extract_filter_expression_from_response(raw, output_format="json")
        assert result == [{"term": {"metadata.content_metadata.year": 2024}}]

    def test_no_filter_returns_none_for_json(self):
        assert (
            _extract_filter_expression_from_response("NO_FILTER", output_format="json")
            is None
        )

    def test_unsupported_returns_none_for_json(self):
        assert (
            _extract_filter_expression_from_response(
                "UNSUPPORTED", output_format="json"
            )
            is None
        )

    def test_malformed_json_returns_none(self):
        assert (
            _extract_filter_expression_from_response(
                "[{not valid json", output_format="json"
            )
            is None
        )

    def test_non_array_json_returns_none(self):
        assert (
            _extract_filter_expression_from_response(
                '{"term": {"x": 1}}', output_format="json"
            )
            is None
        )

    def test_empty_array_returns_none(self):
        assert (
            _extract_filter_expression_from_response("[]", output_format="json") is None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

#!/usr/bin/env python3
"""
Integration tests for natural language to Milvus query pipeline.

Tests the complete workflow: natural language input → LLM filter generation →
metadata validation → filter processing → Milvus-compatible query output.
"""

import queue
import re
import threading
from typing import Any
from unittest.mock import Mock

import pytest

from nvidia_rag.utils.configuration import MetadataConfig
from nvidia_rag.utils.filter_expression_generator import (
    generate_filter_from_natural_language,
)
from nvidia_rag.utils.metadata_validation import (
    FilterExpressionParser,
    MetadataField,
    MetadataSchema,
    MetadataValidator,
)


class MockLLM:
    """Mock LLM that returns realistic filter expressions based on input."""

    def __init__(self):
        # Comprehensive response mapping for different natural language inputs
        self.responses = {
            # Simple filters
            "ai_documents": 'content_metadata["category"] == "AI"',
            "urgent_docs": 'content_metadata["priority"] == "urgent"',
            "high_rated": 'content_metadata["rating"] > 4.0',
            "public_docs": 'content_metadata["is_public"] == true',
            "recent_docs": 'content_metadata["created_date"] >= "2024-01-01"',
            # Complex multi-field filters
            "urgent_ai_high_rated": 'content_metadata["priority"] == "urgent" and content_metadata["category"] == "AI" and content_metadata["rating"] > 4.0',
            "urgent_with_tags": 'content_metadata["priority"] == "urgent" and content_metadata["tags"] in ["important"]',
            "complex_filter": 'content_metadata["tags"] in ["urgent", "important"] and content_metadata["rating"] > 4.0',
            "date_range": 'content_metadata["created_date"] between "2024-01-01" and "2024-12-31"',
            "mixed_types": 'content_metadata["category"] == "tech" and content_metadata["rating"] > 3.5 and content_metadata["is_public"] == true',
            # Array operations
            "array_contains": 'content_metadata["tags"] includes "urgent"',
            "array_in": 'content_metadata["tags"] in ["urgent", "important", "critical"]',
            "array_length": 'array_length(content_metadata["tags"]) > 2',
            # String operations
            "like_pattern": 'content_metadata["title"] like "%policy%"',
            "case_insensitive": 'content_metadata["title"] like "%POLICY%" or content_metadata["title"] like "%policy%"',
            # Numeric operations
            "range_filter": 'content_metadata["rating"] between 3.0 and 5.0',
            "multiple_conditions": 'content_metadata["pages"] > 10 and content_metadata["rating"] >= 4.0',
            # Boolean operations
            "boolean_complex": 'content_metadata["is_public"] == true and content_metadata["is_verified"] == true',
            # Error responses for testing failure scenarios
            "invalid_field": 'content_metadata["nonexistent_field"] == "value"',
            "syntax_error": 'content_metadata["category"] == "AI" and',
            "type_mismatch": 'content_metadata["rating"] == "high"',
            "empty_response": "",
            "no_filter": "NO_FILTER",
            "unsupported": "UNSUPPORTED",
        }

        # Error responses for testing LLM failure scenarios
        self.error_responses = {
            "llm_failure": Exception("LLM service unavailable"),
            "timeout": TimeoutError("LLM request timed out"),
            "invalid_response": ValueError("Invalid LLM response format"),
        }

    def invoke(self, prompt) -> Mock:
        """Simulate LLM response based on prompt content."""
        # Handle LangChain message format (list of messages)
        if isinstance(prompt, list):
            # Extract text from LangChain message format
            prompt_text = " ".join(
                [
                    str(msg.content) if hasattr(msg, "content") else str(msg)
                    for msg in prompt
                ]
            )
        else:
            raise ValueError(
                f"Expected list of messages from LangChain, got: {type(prompt)}"
            )

        response_key = self._determine_response_key(prompt_text)

        if response_key in self.error_responses:
            raise self.error_responses[response_key]

        content = self.responses.get(response_key, self.responses["ai_documents"])
        return Mock(content=content)

    def _determine_response_key(self, prompt: str) -> str:
        """Determine which response to return based on prompt content."""
        prompt_lower = prompt.lower()

        # Priority-based matching for complex scenarios
        if (
            "urgent" in prompt_lower
            and "ai" in prompt_lower
            and "rating" in prompt_lower
        ):
            return "urgent_ai_high_rated"
        elif (
            "urgent" in prompt_lower
            and "tags" in prompt_lower
            and "important" in prompt_lower
        ):
            return "urgent_with_tags"
        elif "urgent" in prompt_lower and "rating" in prompt_lower:
            return "complex_filter"
        elif "urgent" in prompt_lower:
            return "urgent_docs"
        elif "ai" in prompt_lower or "artificial intelligence" in prompt_lower:
            return "ai_documents"
        elif "rating" in prompt_lower and (
            "above" in prompt_lower or "high" in prompt_lower
        ):
            return "high_rated"
        elif "public" in prompt_lower:
            return "public_docs"
        elif "recent" in prompt_lower or "last week" in prompt_lower:
            return "recent_docs"
        elif "between" in prompt_lower and "date" in prompt_lower:
            return "date_range"
        elif "tags" in prompt_lower and "includes" in prompt_lower:
            return "array_contains"
        elif "tags" in prompt_lower and "in" in prompt_lower:
            return "array_in"
        elif "array_length" in prompt_lower:
            return "array_length"
        elif "like" in prompt_lower or "pattern" in prompt_lower:
            return "like_pattern"
        elif "between" in prompt_lower and "rating" in prompt_lower:
            return "range_filter"
        elif "pages" in prompt_lower and "rating" in prompt_lower:
            return "multiple_conditions"
        elif "verified" in prompt_lower:
            return "boolean_complex"
        elif "nonexistent" in prompt_lower:
            return "invalid_field"
        elif "syntax" in prompt_lower:
            return "syntax_error"
        elif "type mismatch" in prompt_lower:
            return "type_mismatch"
        elif "empty" in prompt_lower:
            return "empty_response"
        elif "no filter" in prompt_lower:
            return "no_filter"
        elif "unsupported" in prompt_lower:
            return "unsupported"
        elif "tech" in prompt_lower and "urgent" in prompt_lower:
            return "mixed_types"

        return "ai_documents"


class MockLLMFailure:
    """Mock LLM that simulates various failure scenarios for error handling tests."""

    def __init__(self, failure_type: str = "service_unavailable"):
        self.failure_type = failure_type

    def invoke(self, prompt: str) -> None:
        """Simulate LLM failure based on failure type."""
        if self.failure_type == "service_unavailable":
            raise Exception("LLM service unavailable")
        elif self.failure_type == "timeout":
            raise TimeoutError("LLM request timed out")
        elif self.failure_type == "invalid_response":
            raise ValueError("Invalid LLM response format")
        elif self.failure_type == "network_error":
            raise ConnectionError("Network connection failed")
        else:
            raise Exception(f"Unknown failure type: {self.failure_type}")


class TestFilterExpressionGeneratorIntegration:
    """Integration tests for filter expression generator with metadata validation."""

    @pytest.fixture
    def mock_config(self):
        """Create a metadata configuration object using the actual MetadataConfig class."""
        return MetadataConfig()

    @pytest.fixture
    def realistic_schema(self):
        """Create a realistic metadata schema for testing."""
        return MetadataSchema(
            schema=[
                MetadataField(
                    name="title", type="string", required=True, max_length=200
                ),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(name="pages", type="integer", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
                MetadataField(name="is_verified", type="boolean", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", max_length=50
                ),
                MetadataField(name="created_date", type="datetime", required=True),
                MetadataField(name="updated_date", type="datetime", required=False),
            ]
        )

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        return MockLLM()

    @pytest.fixture
    def prompt_template(self):
        """Create a realistic prompt template."""
        return {
            "system": "You are a filter expression generator.",
            "human": """
        You are a filter expression generator. Given the user request and metadata schema, generate a filter expression.

        Metadata Schema: {metadata_schema}
        User Request: {user_request}
        Collection: {collection_name}
        {existing_filter_context}

        Generate a valid filter expression or respond with NO_FILTER if no filter is needed.
        """,
        }

    def test_generate_and_validate_filter_success(
        self, mock_config, realistic_schema, mock_llm, prompt_template
    ):
        """Test successful filter generation and validation pipeline."""
        filter_expr = generate_filter_from_natural_language(
            user_request="Show me AI documents",
            collection_name="test_collection",
            metadata_schema=realistic_schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        parser = FilterExpressionParser(realistic_schema, mock_config)
        validation_result = parser.validate_filter_expression(filter_expr)
        processing_result = parser.process_filter_expression(filter_expr)

        assert filter_expr is not None
        assert isinstance(filter_expr, str) and filter_expr.strip() != ""
        assert validation_result["status"] is True
        assert processing_result["status"] is True
        assert "processed_expression" in processing_result
        assert processing_result["processed_expression"] != ""

    def test_generate_filter_with_invalid_schema(
        self, mock_config, mock_llm, prompt_template
    ):
        """Test filter generation when schema has invalid fields."""
        invalid_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
            ]
        )

        filter_expr = generate_filter_from_natural_language(
            user_request="Show me AI documents",
            collection_name="test_collection",
            metadata_schema=invalid_schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        parser = FilterExpressionParser(invalid_schema, mock_config)
        validation_result = parser.validate_filter_expression(filter_expr)

        assert validation_result["status"] is True

        # Test with nonexistent field to trigger validation error
        filter_expr2 = generate_filter_from_natural_language(
            user_request="Show me documents with nonexistent field",
            collection_name="test_collection",
            metadata_schema=invalid_schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        assert filter_expr2 is not None
        validation_result2 = parser.validate_filter_expression(filter_expr2)

        assert validation_result2["status"] is False
        assert "error_message" in validation_result2
        assert "nonexistent_field" in validation_result2["error_message"]

    def test_generate_filter_with_mixed_collections(
        self, mock_config, realistic_schema, mock_llm, prompt_template
    ):
        """Test filter generation across collections with different schemas."""
        schema1 = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
            ]
        )

        schema2 = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
            ]
        )

        filter_expr = generate_filter_from_natural_language(
            user_request="Show me high-rated documents",
            collection_name="test_collection",
            metadata_schema=schema1.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        if filter_expr is None:
            assert filter_expr is None
            return

        parser1 = FilterExpressionParser(schema1, mock_config)
        parser2 = FilterExpressionParser(schema2, mock_config)

        result1 = parser1.validate_filter_expression(filter_expr)
        result2 = parser2.validate_filter_expression(filter_expr)

        assert result1["status"] is True

        # Test cross-schema compatibility
        if isinstance(filter_expr, str) and "rating" in filter_expr:
            assert result2["status"] is False
        else:
            assert result2["status"] is True

    def test_generate_filter_with_existing_filter(
        self, mock_config, realistic_schema, mock_llm, prompt_template
    ):
        """Test filter generation with existing filter expression improvement."""
        existing_filter = 'content_metadata["category"] == "tech"'

        improved_filter = generate_filter_from_natural_language(
            user_request="Show me urgent tech documents",
            collection_name="test_collection",
            metadata_schema=realistic_schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
            existing_filter_expr=existing_filter,
        )

        if improved_filter is None:
            assert improved_filter is None
            return

        parser = FilterExpressionParser(realistic_schema, mock_config)
        validation_result = parser.validate_filter_expression(improved_filter)

        assert improved_filter is not None
        assert validation_result["status"] is True

        # Verify the improved filter contains expected fields
        expected_fields = ["category", "priority", "tags", "rating"]
        assert isinstance(improved_filter, str) and any(
            field in improved_filter for field in expected_fields
        )
        assert isinstance(improved_filter, str) and len(improved_filter.strip()) > 0

    def test_generate_filter_llm_failure_handling(
        self, mock_config, realistic_schema, prompt_template
    ):
        """Test graceful handling of LLM failures during filter generation."""
        failure_types = [
            "service_unavailable",
            "timeout",
            "invalid_response",
            "network_error",
        ]

        for failure_type in failure_types:
            mock_llm_failure = MockLLMFailure(failure_type)

            result = generate_filter_from_natural_language(
                user_request="Show me AI documents",
                collection_name="test_collection",
                metadata_schema=realistic_schema.model_dump(),
                prompt_template=prompt_template,
                llm=mock_llm_failure,
            )

            assert result is None or result == ""

    def test_comprehensive_integration_workflow(self, mock_config, realistic_schema):
        """Test comprehensive integration workflow with case-insensitive search."""
        # Test end-to-end workflow with case-insensitive search
        test_scenarios = [
            {
                "name": "Policy document search",
                "natural_language": "Find policy documents with urgent priority",
                "expected_filter": 'content_metadata["category"] == "policy" and content_metadata["priority"] == "urgent"',
                "metadata": {
                    "title": "Policy Document",
                    "category": "Policy",
                    "priority": "Urgent",
                    "tags": ["Policy", "Urgent", "Important"],
                    "created_date": "2024-01-15T10:30:00Z",
                },
            },
            {
                "name": "Technical documentation search",
                "natural_language": "Find technical documents with high rating",
                "expected_filter": 'content_metadata["category"] == "technical" and content_metadata["rating"] > 4.0',
                "metadata": {
                    "title": "Technical Manual",
                    "category": "Technical",
                    "rating": 4.5,
                    "tags": ["Technical", "Manual", "Guide"],
                    "created_date": "2024-01-15T10:30:00Z",
                },
            },
            {
                "name": "Case-insensitive tag search",
                "natural_language": "Find documents with urgent or important tags",
                "expected_filter": 'content_metadata["tags"] in ["urgent", "important"]',
                "metadata": {
                    "title": "Important Document",
                    "category": "General",
                    "tags": ["Urgent", "Important", "Critical"],
                    "created_date": "2024-01-15T10:30:00Z",
                },
            },
            {
                "name": "Mixed case title search",
                "natural_language": "Find documents with 'Policy' in the title",
                "expected_filter": 'content_metadata["title"] like "%policy%"',
                "metadata": {
                    "title": "Policy Guidelines",
                    "category": "Policy",
                    "tags": ["Policy", "Guidelines"],
                    "created_date": "2024-01-15T10:30:00Z",
                },
            },
            {
                "name": "Complex case-insensitive search",
                "natural_language": "Find urgent policy documents with high rating published recently",
                "expected_filter": 'content_metadata["priority"] == "urgent" and content_metadata["category"] == "policy" and content_metadata["rating"] > 4.0 and content_metadata["created_date"] >= "2024-01-01"',
                "metadata": {
                    "title": "Urgent Policy Update",
                    "category": "Policy",
                    "priority": "Urgent",
                    "rating": 4.8,
                    "created_date": "2024-01-15T10:30:00Z",
                    "tags": ["Urgent", "Policy", "Update"],
                },
            },
        ]

        for scenario in test_scenarios:
            # Test metadata ingestion with case-insensitive processing
            validator = MetadataValidator(mock_config)
            is_valid, errors, normalized_data = (
                validator.validate_and_normalize_metadata_values(
                    scenario["metadata"], realistic_schema
                )
            )

            assert is_valid, (
                f"Metadata validation failed for {scenario['name']}: {errors}"
            )

            # Verify case-insensitive normalization
            for key, value in normalized_data.items():
                if isinstance(value, str):
                    if key in ["created_date", "updated_date"]:
                        continue
                    assert value == value.lower(), (
                        f"String value '{value}' is not lowercase for {scenario['name']}"
                    )
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            assert item == item.lower(), (
                                f"Array item '{item}' is not lowercase for {scenario['name']}"
                            )

            # Test filter processing with case-insensitive transformation
            parser = FilterExpressionParser(realistic_schema, mock_config)
            result = parser.validate_filter_expression(scenario["expected_filter"])

            assert result["status"], (
                f"Filter validation failed for {scenario['name']}: {result.get('error_message', 'Unknown error')}"
            )

            processed = parser.process_filter_expression(scenario["expected_filter"])
            processed_expr = processed["processed_expression"]

            # Verify case-insensitive processing of string literals
            string_literals = re.findall(r'"([^"]*)"', processed_expr)
            for literal in string_literals:
                if (
                    literal
                    and not literal.startswith("%")
                    and not literal.endswith("%")
                ):
                    if "%" not in literal:
                        if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", literal):
                            continue
                        assert literal == literal.lower(), (
                            f"String literal '{literal}' is not lowercase for {scenario['name']}"
                        )

    def test_case_insensitive_real_world_scenarios(self, mock_config):
        """Test case-insensitive search with real-world scenarios."""
        # Create a comprehensive schema for real-world testing
        real_world_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", required=False
                ),
                MetadataField(name="author", type="string", required=False),
                MetadataField(name="department", type="string", required=False),
                MetadataField(name="status", type="string", required=False),
                MetadataField(name="created_date", type="datetime", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
            ]
        )

        # Real-world test scenarios
        scenarios = [
            {
                "name": "HR Policy Search",
                "metadata": {
                    "title": "Employee Handbook 2024",
                    "category": "HR Policy",
                    "priority": "High",
                    "rating": 4.2,
                    "tags": ["HR", "Policy", "Employee", "Handbook"],
                    "author": "HR Department",
                    "department": "Human Resources",
                    "status": "Active",
                    "is_public": True,
                },
                "filters": [
                    'content_metadata["category"] == "hr policy"',
                    'content_metadata["priority"] == "high"',
                    'content_metadata["tags"] in ["hr", "policy"]',
                    'content_metadata["department"] == "human resources"',
                ],
            },
            {
                "name": "Technical Documentation Search",
                "metadata": {
                    "title": "API Integration Guide",
                    "category": "Technical Documentation",
                    "priority": "Medium",
                    "rating": 4.8,
                    "tags": ["API", "Integration", "Technical", "Guide"],
                    "author": "Engineering Team",
                    "department": "Engineering",
                    "status": "Published",
                    "is_public": True,
                },
                "filters": [
                    'content_metadata["category"] == "technical documentation"',
                    'content_metadata["rating"] > 4.0',
                    'content_metadata["tags"] in ["api", "technical"]',
                    'content_metadata["title"] like "%guide%"',
                ],
            },
            {
                "name": "Legal Document Search",
                "metadata": {
                    "title": "Terms of Service Agreement",
                    "category": "Legal",
                    "priority": "Critical",
                    "rating": 5.0,
                    "tags": ["Legal", "Terms", "Service", "Agreement"],
                    "author": "Legal Department",
                    "department": "Legal",
                    "status": "Active",
                    "is_public": False,
                },
                "filters": [
                    'content_metadata["category"] == "legal"',
                    'content_metadata["priority"] == "critical"',
                    'content_metadata["tags"] in ["legal", "terms"]',
                    'content_metadata["is_public"] == false',
                ],
            },
        ]

        validator = MetadataValidator(mock_config)
        parser = FilterExpressionParser(real_world_schema, mock_config)

        for scenario in scenarios:
            # Test metadata ingestion
            is_valid, errors, normalized_data = (
                validator.validate_and_normalize_metadata_values(
                    scenario["metadata"], real_world_schema
                )
            )

            assert is_valid, (
                f"Metadata validation failed for {scenario['name']}: {errors}"
            )

            # Verify case-insensitive normalization
            for key, value in normalized_data.items():
                if isinstance(value, str):
                    assert value == value.lower(), (
                        f"String value '{value}' is not lowercase for {scenario['name']}"
                    )
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            assert item == item.lower(), (
                                f"Array item '{item}' is not lowercase for {scenario['name']}"
                            )

            # Test filter validation and processing
            for i, filter_expr in enumerate(scenario["filters"], 1):
                result = parser.validate_filter_expression(filter_expr)
                assert result["status"], (
                    f"Filter validation failed for {scenario['name']} test {i}: {result.get('error_message', 'Unknown error')}"
                )

                if result["status"]:
                    processed = parser.process_filter_expression(filter_expr)
                    processed_expr = processed["processed_expression"]

                    # Verify all string literals are lowercase
                    string_literals = re.findall(r'"([^"]*)"', processed_expr)
                    for literal in string_literals:
                        if (
                            literal
                            and not literal.startswith("%")
                            and not literal.endswith("%")
                        ):
                            if "%" not in literal:
                                assert literal == literal.lower(), (
                                    f"String literal '{literal}' is not lowercase for {scenario['name']}"
                                )
                    else:
                        pass
            else:
                pass

    def test_case_insensitive_performance_integration(self, mock_config):
        """Test case-insensitive performance with integration scenarios."""
        # Create a large schema for performance testing
        large_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", required=False
                ),
                MetadataField(name="description", type="string", required=False),
                MetadataField(name="author", type="string", required=False),
                MetadataField(name="department", type="string", required=False),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="status", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(name="created_date", type="datetime", required=False),
                MetadataField(name="updated_date", type="datetime", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
                MetadataField(name="is_archived", type="boolean", required=False),
                MetadataField(name="version", type="string", required=False),
                MetadataField(
                    name="keywords", type="array", array_type="string", required=False
                ),
            ]
        )

        # Test with large metadata
        large_metadata = {
            "title": "Comprehensive Policy Document with Mixed Case Title",
            "category": "Policy and Procedures",
            "tags": [
                "Policy",
                "Procedures",
                "Guidelines",
                "Compliance",
                "Regulations",
                "Standards",
                "Best Practices",
                "Documentation",
                "Management",
                "Administration",
            ],
            "description": "This is a comprehensive policy document that contains detailed information about various procedures and guidelines that must be followed by all employees. It includes sections on compliance, regulations, standards, and best practices that are essential for maintaining organizational integrity and operational efficiency.",
            "author": "Policy Development Team",
            "department": "Policy and Compliance Department",
            "priority": "High Priority",
            "status": "Active Status",
            "rating": 4.7,
            "created_date": "2024-01-15T10:30:00Z",
            "updated_date": "2024-03-20T14:45:00Z",
            "is_public": True,
            "is_archived": False,
            "version": "Version 2.1",
            "keywords": [
                "Policy",
                "Compliance",
                "Regulations",
                "Standards",
                "Guidelines",
                "Procedures",
                "Management",
                "Administration",
                "Best Practices",
                "Documentation",
            ],
        }

        validator = MetadataValidator(mock_config)
        parser = FilterExpressionParser(large_schema, mock_config)

        is_valid, errors, normalized_data = (
            validator.validate_and_normalize_metadata_values(
                large_metadata, large_schema
            )
        )

        assert is_valid, f"Large metadata processing failed: {errors}"

        # Verify case-insensitive normalization for large data
        for key, value in normalized_data.items():
            if isinstance(value, str):
                if key in ["created_date", "updated_date"]:
                    continue
                assert value == value.lower(), (
                    f"String value '{value}' is not lowercase"
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        assert item == item.lower(), (
                            f"Array item '{item}' is not lowercase"
                        )

        # Test complex filters with large schema
        complex_filters = [
            'content_metadata["title"] like "%policy%" and content_metadata["category"] == "policy and procedures"',
            'content_metadata["tags"] in ["policy", "compliance", "regulations"] and content_metadata["priority"] == "high priority"',
            'content_metadata["department"] == "policy and compliance department" and content_metadata["rating"] > 4.0',
            'content_metadata["keywords"] in ["policy", "compliance", "regulations"] and content_metadata["is_public"] == true',
            'content_metadata["title"] like "%comprehensive%" and content_metadata["status"] == "active status"',
        ]

        for i, filter_expr in enumerate(complex_filters, 1):
            result = parser.validate_filter_expression(filter_expr)
            assert result["status"], (
                f"Filter validation failed for test {i}: {result.get('error_message', 'Unknown error')}"
            )

            processed = parser.process_filter_expression(filter_expr)
            processed_expr = processed["processed_expression"]

            # Verify case-insensitive processing
            string_literals = re.findall(r'"([^"]*)"', processed_expr)
            for literal in string_literals:
                if (
                    literal
                    and not literal.startswith("%")
                    and not literal.endswith("%")
                ):
                    if "%" not in literal:
                        assert literal == literal.lower(), (
                            f"String literal '{literal}' is not lowercase"
                        )

    def test_case_insensitive_error_handling_integration(self, mock_config):
        """Test case-insensitive error handling in integration scenarios."""
        # Test error scenarios with case-insensitive processing
        error_scenarios = [
            {
                "name": "Invalid field with case-insensitive processing",
                "metadata": {"title": "Test Document", "nonexistent_field": "Value"},
                "filter": 'content_metadata["nonexistent_field"] == "value"',
                "should_fail": True,
            },
            {
                "name": "Invalid operator with case-insensitive processing",
                "metadata": {"title": "Test Document", "rating": 4.5},
                "filter": 'content_metadata["title"] > "string"',
                "should_fail": True,
            },
            {
                "name": "Empty string with case-insensitive processing",
                "metadata": {"title": "Test Document"},
                "filter": 'content_metadata["title"] == ""',
                "should_fail": True,
            },
            {
                "name": "Valid case-insensitive processing",
                "metadata": {"title": "Test Document", "category": "Test Category"},
                "filter": 'content_metadata["title"] == "test document"',
                "should_fail": False,
            },
        ]

        schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
            ]
        )

        validator = MetadataValidator(mock_config)
        parser = FilterExpressionParser(schema, mock_config)

        for scenario in error_scenarios:
            # Test metadata validation
            is_valid, errors, normalized_data = (
                validator.validate_and_normalize_metadata_values(
                    scenario["metadata"], schema
                )
            )

            if scenario["name"] == "Invalid field with case-insensitive processing":
                assert not is_valid, (
                    f"Metadata validation should have failed for {scenario['name']}"
                )
                assert errors, (
                    f"Metadata errors should not be empty for {scenario['name']}"
                )
            else:
                assert is_valid, (
                    f"Metadata validation should have passed for {scenario['name']}: {errors}"
                )

            # Test filter validation
            result = parser.validate_filter_expression(scenario["filter"])

            if scenario["name"] == "Invalid field with case-insensitive processing":
                assert result["status"] is False, (
                    f"Filter validation should have failed for {scenario['name']}"
                )
            elif (
                scenario["name"] == "Invalid operator with case-insensitive processing"
            ):
                assert result["status"] is False, (
                    f"Filter validation should have failed for {scenario['name']}"
                )
            elif scenario["name"] == "Empty string with case-insensitive processing":
                pass
            else:
                assert result["status"] is True, (
                    f"Filter validation should have passed for {scenario['name']}"
                )

            if result["status"]:
                processed = parser.process_filter_expression(scenario["filter"])
                processed_expr = processed["processed_expression"]

                # Verify case-insensitive processing
                string_literals = re.findall(r'"([^"]*)"', processed_expr)
                for literal in string_literals:
                    if (
                        literal
                        and not literal.startswith("%")
                        and not literal.endswith("%")
                    ):
                        if "%" not in literal:
                            assert literal == literal.lower(), (
                                f"String literal '{literal}' is not lowercase for {scenario['name']}"
                            )


class TestFilterProcessingPipeline:
    """Integration tests for complete filter processing pipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create a metadata configuration object using the actual MetadataConfig class."""
        return MetadataConfig()

    @pytest.fixture
    def realistic_schema(self):
        """Create a realistic metadata schema for testing."""
        return MetadataSchema(
            schema=[
                MetadataField(
                    name="title", type="string", required=True, max_length=200
                ),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(name="pages", type="integer", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
                MetadataField(name="is_verified", type="boolean", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", max_length=50
                ),
                MetadataField(name="created_date", type="datetime", required=True),
                MetadataField(name="updated_date", type="datetime", required=False),
            ]
        )

    def test_natural_language_to_milvus_query(self, mock_config, realistic_schema):
        """Test complete pipeline: natural language → filter → validation → Milvus query."""
        user_request = "Show me urgent AI documents with rating above 4.0"

        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        generated_filter = generate_filter_from_natural_language(
            user_request=user_request,
            collection_name="test_collection",
            metadata_schema=realistic_schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        parser = FilterExpressionParser(realistic_schema, mock_config)
        validation_result = parser.validate_filter_expression(generated_filter)
        processing_result = parser.process_filter_expression(generated_filter)

        assert generated_filter is not None
        assert isinstance(generated_filter, str) and generated_filter.strip() != ""
        assert validation_result["status"] is True
        assert processing_result["status"] is True
        assert "processed_expression" in processing_result

        processed_expr = processing_result["processed_expression"]

        # Verify the natural language was properly converted to filter
        assert "priority" in processed_expr, (
            f"Expected 'priority' in filter, got: {processed_expr}"
        )
        assert "category" in processed_expr, (
            f"Expected 'category' in filter, got: {processed_expr}"
        )
        assert "rating" in processed_expr, (
            f"Expected 'rating' in filter, got: {processed_expr}"
        )
        assert "and" in processed_expr.lower(), (
            f"Expected 'and' operator in filter, got: {processed_expr}"
        )

        # Verify the filter logic matches the natural language request
        assert "urgent" in processed_expr.lower(), (
            f"Expected 'urgent' in filter for urgent documents, got: {processed_expr}"
        )
        assert "ai" in processed_expr.lower(), (
            f"Expected 'ai' in filter for AI documents, got: {processed_expr}"
        )
        assert ">" in processed_expr, (
            f"Expected '>' operator for 'above' rating, got: {processed_expr}"
        )

    def test_different_natural_language_patterns(self, mock_config, realistic_schema):
        """Test various natural language patterns are properly converted to filters."""
        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        test_cases = [
            {
                "input": "Show me urgent documents",
                "expected_fields": ["priority"],
                "expected_values": ["urgent"],
                "description": "Simple urgent filter",
            },
            {
                "input": "Find AI documents",
                "expected_fields": ["category"],
                "expected_values": ["ai"],
                "description": "Simple category filter",
            },
            {
                "input": "Show me documents with rating above 4.0",
                "expected_fields": ["rating"],
                "expected_operators": [">"],
                "description": "Simple rating filter",
            },
            {
                "input": "Find urgent documents with important tags",
                "expected_fields": ["priority", "tags"],
                "expected_values": ["urgent", "important"],
                "description": "Complex filter with priority and tags",
            },
        ]

        for test_case in test_cases:
            generated_filter = generate_filter_from_natural_language(
                user_request=test_case["input"],
                collection_name="test_collection",
                metadata_schema=realistic_schema.model_dump(),
                prompt_template=prompt_template,
                llm=mock_llm,
            )

            parser = FilterExpressionParser(realistic_schema, mock_config)
            validation_result = parser.validate_filter_expression(generated_filter)
            processing_result = parser.process_filter_expression(generated_filter)

            assert validation_result["status"] is True, (
                f"Filter validation failed for: {test_case['input']}"
            )
            assert processing_result["status"] is True, (
                f"Filter processing failed for: {test_case['input']}"
            )

            processed_expr = processing_result["processed_expression"]

            # Verify expected fields are present
            for field in test_case["expected_fields"]:
                assert field in processed_expr, (
                    f"Expected field '{field}' not found in filter for: {test_case['input']}"
                )

            # Verify expected values are present
            if "expected_values" in test_case:
                for value in test_case["expected_values"]:
                    assert value in processed_expr.lower(), (
                        f"Expected value '{value}' not found in filter for: {test_case['input']}"
                    )

            # Verify expected operators are present
            if "expected_operators" in test_case:
                for operator in test_case["expected_operators"]:
                    assert operator in processed_expr, (
                        f"Expected operator '{operator}' not found in filter for: {test_case['input']}"
                    )

    def test_multi_collection_filter_processing(self, mock_config):
        """Test filter processing across multiple collections."""
        schema1 = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
            ]
        )

        schema2 = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
            ]
        )

        # Test filter that references fields from both schemas
        filter_expr = 'content_metadata["rating"] > 4.0 and content_metadata["priority"] == "high"'

        parser1 = FilterExpressionParser(schema1, mock_config)
        result1 = parser1.validate_filter_expression(filter_expr)
        assert result1["status"] is False

        parser2 = FilterExpressionParser(schema2, mock_config)
        result2 = parser2.validate_filter_expression(filter_expr)
        assert result2["status"] is False

        # Test individual filters that work with each schema
        rating_filter = 'content_metadata["rating"] > 4.0'
        priority_filter = 'content_metadata["priority"] == "high"'

        result3 = parser1.validate_filter_expression(rating_filter)
        assert result3["status"] is True

        result4 = parser2.validate_filter_expression(priority_filter)
        assert result4["status"] is True


class TestRealWorldScenarios:
    """Integration tests for realistic use cases."""

    @pytest.fixture
    def mock_config(self):
        """Create a metadata configuration object using the actual MetadataConfig class."""
        return MetadataConfig()

    @pytest.fixture
    def document_schema(self):
        """Create a document metadata schema."""
        return MetadataSchema(
            schema=[
                MetadataField(
                    name="title", type="string", required=True, max_length=200
                ),
                MetadataField(name="author", type="string", required=False),
                MetadataField(name="category", type="string", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", max_length=50
                ),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(name="pages", type="integer", required=False),
                MetadataField(name="is_public", type="boolean", required=False),
                MetadataField(name="created_date", type="datetime", required=True),
                MetadataField(name="updated_date", type="datetime", required=False),
            ]
        )

    def test_metadata_ingestion_and_filtering(self, mock_config, document_schema):
        """Test metadata ingestion followed by filter-based retrieval."""
        ingested_metadata = {
            "title": "AI Policy Guidelines",
            "author": "John Doe",
            "category": "tech",
            "tags": ["urgent", "policy", "ai"],
            "rating": 4.5,
            "pages": 25,
            "is_public": True,
            "created_date": "2024-01-15T10:30:00Z",
        }

        validator = MetadataValidator(mock_config)
        is_valid, errors, normalized_metadata = (
            validator.validate_and_normalize_metadata_values(
                ingested_metadata, document_schema
            )
        )

        assert is_valid is True
        assert len(errors) == 0

        # Test filter-based retrieval
        filter_expr = 'content_metadata["category"] == "tech" and content_metadata["rating"] > 4.0'
        parser = FilterExpressionParser(document_schema, mock_config)
        validation_result = parser.validate_filter_expression(filter_expr)

        assert validation_result["status"] is True

    def test_schema_evolution_with_filters(self, mock_config):
        """Test filter behavior when schemas change."""
        initial_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
            ]
        )

        evolved_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(name="priority", type="string", required=False),
                MetadataField(name="tags", type="array", array_type="string"),
            ]
        )

        # Test backward compatibility
        compatible_filter = 'content_metadata["category"] == "tech" and content_metadata["rating"] > 4.0'

        parser1 = FilterExpressionParser(initial_schema, mock_config)
        result1 = parser1.validate_filter_expression(compatible_filter)
        assert result1["status"] is True

        parser2 = FilterExpressionParser(evolved_schema, mock_config)
        result2 = parser2.validate_filter_expression(compatible_filter)
        assert result2["status"] is True

        # Test new features in evolved schema
        new_filter = 'content_metadata["priority"] == "urgent" and content_metadata["tags"] includes "important"'

        result3 = parser1.validate_filter_expression(new_filter)
        assert result3["status"] is False

        result4 = parser2.validate_filter_expression(new_filter)
        assert result4["status"] is True

    def test_multi_collection_scenarios(self, mock_config):
        """Test scenarios involving multiple collections."""
        documents_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
            ]
        )

        products_schema = MetadataSchema(
            schema=[
                MetadataField(name="name", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="price", type="float", required=True),
            ]
        )

        # Test flexible filtering across collections
        flexible_config = MetadataConfig(allow_partial_filtering=True)
        universal_filter = 'content_metadata["category"] == "tech"'

        doc_parser = FilterExpressionParser(documents_schema, flexible_config)
        doc_result = doc_parser.validate_filter_expression(universal_filter)
        assert doc_result["status"] is True

        prod_parser = FilterExpressionParser(products_schema, flexible_config)
        prod_result = prod_parser.validate_filter_expression(universal_filter)
        assert prod_result["status"] is True


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    @pytest.fixture
    def mock_config(self):
        """Create a metadata configuration object using the actual MetadataConfig class."""
        return MetadataConfig()

    @pytest.fixture
    def realistic_schema(self):
        """Create a realistic metadata schema for testing."""
        return MetadataSchema(
            schema=[
                MetadataField(
                    name="title", type="string", required=True, max_length=200
                ),
                MetadataField(name="category", type="string", required=False),
                MetadataField(name="rating", type="float", required=False),
                MetadataField(
                    name="tags", type="array", array_type="string", max_length=50
                ),
                MetadataField(name="created_date", type="datetime", required=True),
            ]
        )

    def test_llm_failure_propagation(self, mock_config, realistic_schema):
        """Test how LLM failures are handled in the pipeline."""
        failure_types = [
            "service_unavailable",
            "timeout",
            "invalid_response",
            "network_error",
        ]

        for failure_type in failure_types:
            mock_llm_failure = MockLLMFailure(failure_type)

            result = generate_filter_from_natural_language(
                user_request="Show me AI documents",
                collection_name="test_collection",
                metadata_schema=realistic_schema.model_dump(),
                prompt_template={
                    "system": "You are a filter expression generator.",
                    "human": "Test template",
                },
                llm=mock_llm_failure,
            )

            assert result is None or result == ""

    def test_schema_validation_error_propagation(self, mock_config):
        """Test how schema validation errors propagate."""
        invalid_schema = MetadataSchema(
            schema=[
                MetadataField(name="title", type="string", required=True),
                MetadataField(name="category", type="string", required=False),
            ]
        )

        # Test invalid field error propagation
        invalid_filter = 'content_metadata["nonexistent_field"] == "value"'

        parser = FilterExpressionParser(invalid_schema, mock_config)
        validation_result = parser.validate_filter_expression(invalid_filter)

        assert validation_result["status"] is False
        assert "error_message" in validation_result
        assert "nonexistent_field" in validation_result["error_message"]

    def test_comprehensive_integration_workflow(self, mock_config, realistic_schema):
        """Test a comprehensive integration workflow from start to finish."""
        schema = realistic_schema

        test_metadata = {
            "title": "AI Policy Guidelines",
            "category": "tech",
            "rating": 4.5,
            "tags": ["urgent", "policy", "ai"],
            "created_date": "2024-01-15T10:30:00Z",
        }

        validator = MetadataValidator(mock_config)
        is_valid, errors, normalized_metadata = (
            validator.validate_and_normalize_metadata_values(test_metadata, schema)
        )

        assert is_valid is True
        assert len(errors) == 0

        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        filter_expr = generate_filter_from_natural_language(
            user_request="Show me urgent tech documents with high rating",
            collection_name="test_collection",
            metadata_schema=schema.model_dump(),
            prompt_template=prompt_template,
            llm=mock_llm,
        )

        assert filter_expr is not None
        assert isinstance(filter_expr, str) and filter_expr.strip() != ""

        parser = FilterExpressionParser(schema, mock_config)
        validation_result = parser.validate_filter_expression(filter_expr)

        assert validation_result["status"] is True

        processing_result = parser.process_filter_expression(filter_expr)

        assert processing_result["status"] is True
        assert "processed_expression" in processing_result

        processed_expr = processing_result["processed_expression"]
        assert (
            "category" in processed_expr
            or "rating" in processed_expr
            or "tags" in processed_expr
        )

    def test_edge_cases_and_error_handling(self, mock_config, realistic_schema):
        """Test edge cases and error handling scenarios."""
        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        # Test various edge cases
        edge_cases = [
            {"name": "Empty user request", "user_request": "", "should_fail": True},
            {"name": "None user request", "user_request": None, "should_fail": True},
            {
                "name": "Whitespace only request",
                "user_request": "   ",
                "should_fail": True,
            },
            {
                "name": "Very long request",
                "user_request": "a" * 10000,
                "should_fail": False,
            },
            {
                "name": "Special characters request",
                "user_request": "Find documents with @#$%^&*() symbols",
                "should_fail": False,
            },
        ]

        for case in edge_cases:
            try:
                result = generate_filter_from_natural_language(
                    user_request=case["user_request"],
                    collection_name="test_collection",
                    metadata_schema=realistic_schema.model_dump(),
                    prompt_template=prompt_template,
                    llm=mock_llm,
                )

                if case["should_fail"]:
                    assert result is None or result == "", (
                        f"Expected failure for {case['name']} but got result: {result}"
                    )
                else:
                    assert result is not None, (
                        f"Expected success for {case['name']} but got None"
                    )

            except Exception as e:
                if not case["should_fail"]:
                    raise AssertionError(
                        f"Unexpected exception for {case['name']}: {e}"
                    ) from e

        # Test malformed schemas
        malformed_schemas = [
            {"name": "Empty schema", "schema": {}, "should_fail": True},
            {
                "name": "Schema with invalid field type",
                "schema": {"schema": [{"name": "test", "type": "invalid_type"}]},
                "should_fail": True,
            },
            {
                "name": "Schema with missing required fields",
                "schema": {"schema": [{"name": "test"}]},
                "should_fail": True,
            },
        ]

        for schema_case in malformed_schemas:
            try:
                result = generate_filter_from_natural_language(
                    user_request="Show me documents",
                    collection_name="test_collection",
                    metadata_schema=schema_case["schema"],
                    prompt_template=prompt_template,
                    llm=mock_llm,
                )

                if schema_case["should_fail"]:
                    assert result is None or result == "", (
                        f"Expected failure for {schema_case['name']} but got result: {result}"
                    )

            except Exception as e:
                if not schema_case["should_fail"]:
                    raise AssertionError(
                        f"Unexpected exception for {schema_case['name']}: {e}"
                    ) from e

    def test_concurrent_access_simulation(self, mock_config, realistic_schema):
        """Test that the components can handle concurrent-like access patterns."""
        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        # Simulate concurrent filter generation
        results_queue = queue.Queue()

        def generate_filter_thread(thread_id):
            try:
                result = generate_filter_from_natural_language(
                    user_request=f"Show me documents from thread {thread_id}",
                    collection_name=f"collection_{thread_id}",
                    metadata_schema=realistic_schema.model_dump(),
                    prompt_template=prompt_template,
                    llm=mock_llm,
                )
                results_queue.put((thread_id, result))
            except Exception as e:
                results_queue.put((thread_id, f"Error: {e}"))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_filter_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify all threads completed successfully
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        for thread_id, result in results:
            assert not result.startswith("Error:"), (
                f"Thread {thread_id} failed: {result}"
            )
            assert result is not None, f"Thread {thread_id} returned None"
            assert isinstance(result, str), (
                f"Thread {thread_id} returned non-string: {type(result)}"
            )

    def test_memory_usage_validation(self, mock_config, realistic_schema):
        """Test that the components don't have memory leaks with large inputs."""
        mock_llm = MockLLM()
        prompt_template = {
            "system": "You are a filter expression generator.",
            "human": "Generate filter for: {user_request}",
        }

        # Create large schema for memory testing
        large_schema_fields = [
            MetadataField(name="title", type="string", required=True),
            MetadataField(name="category", type="string", required=False),
            MetadataField(name="priority", type="string", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(
                name="tags", type="array", array_type="string", required=False
            ),
            MetadataField(name="created_date", type="datetime", required=True),
        ]

        for i in range(50):  # Create a large schema
            large_schema_fields.append(
                MetadataField(
                    name=f"field_{i}",
                    type="string"
                    if i % 3 == 0
                    else "float"
                    if i % 3 == 1
                    else "boolean",
                    required=False,
                )
            )

        large_schema = MetadataSchema(schema=large_schema_fields)

        # Test multiple iterations to check for memory leaks
        for iteration in range(10):
            result = generate_filter_from_natural_language(
                user_request=f"Show me documents from iteration {iteration}",
                collection_name="test_collection",
                metadata_schema=large_schema.model_dump(),
                prompt_template=prompt_template,
                llm=mock_llm,
            )

            assert result is not None, f"Iteration {iteration} returned None"
            assert isinstance(result, str), (
                f"Iteration {iteration} returned non-string: {type(result)}"
            )

            parser = FilterExpressionParser(large_schema, mock_config)
            validation_result = parser.validate_filter_expression(result)
            processing_result = parser.process_filter_expression(result)

            assert validation_result["status"] is True, (
                f"Validation failed at iteration {iteration}: {validation_result.get('error_message', 'Unknown error')}"
            )
            assert processing_result["status"] is True, (
                f"Processing failed at iteration {iteration}: {processing_result.get('error_message', 'Unknown error')}"
            )

    def test_invalid_filter_expression_handling(self, mock_config, realistic_schema):
        """Test handling of invalid filter expressions."""
        parser = FilterExpressionParser(realistic_schema, mock_config)

        # Test various invalid filter scenarios
        invalid_filters = [
            {
                "filter": 'content_metadata["nonexistent_field"] == "value"',
                "expected_error": "nonexistent_field",
            },
            {
                "filter": 'content_metadata["title"] > "string"',
                "expected_error": "string",
            },
            {
                "filter": 'content_metadata["rating"] == "not_a_number"',
                "expected_error": "not_a_number",
            },
            {
                "filter": 'content_metadata["is_public"] == "not_boolean"',
                "expected_error": "is_public",
            },
            {"filter": 'content_metadata["tags"] > 5', "expected_error": "tags"},
            {
                "filter": 'content_metadata["title"] == "value" and',
                "expected_error": "syntax",
            },
        ]

        for invalid_filter in invalid_filters:
            result = parser.validate_filter_expression(invalid_filter["filter"])
            assert result["status"] is False, (
                f"Expected validation to fail for: {invalid_filter['filter']}"
            )
            assert "error_message" in result, (
                f"Expected error message for: {invalid_filter['filter']}"
            )

            error_message_lower = result["error_message"].lower()
            expected_error_lower = invalid_filter["expected_error"].lower()

            if invalid_filter["expected_error"] == "is_public":
                assert (
                    "is_public" in error_message_lower
                    or "not found" in error_message_lower
                ), (
                    f"Expected 'is_public' or 'not found' in error message, got: {result['error_message']}"
                )
            else:
                assert expected_error_lower in error_message_lower, (
                    f"Expected '{invalid_filter['expected_error']}' in error message, got: {result['error_message']}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

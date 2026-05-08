# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ElasticsearchFilterValidator."""

import pytest

from nvidia_rag.utils.configuration import MetadataConfig
from nvidia_rag.utils.es_filter_validator import ElasticsearchFilterValidator
from nvidia_rag.utils.metadata_validation import MetadataField, MetadataSchema


@pytest.fixture
def metadata_config():
    return MetadataConfig()


@pytest.fixture
def schema():
    return MetadataSchema(
        schema=[
            MetadataField(name="status", type="string"),
            MetadataField(name="year", type="integer"),
            MetadataField(name="rating", type="float"),
            MetadataField(name="is_public", type="boolean"),
            MetadataField(name="created_at", type="datetime"),
            MetadataField(
                name="tags", type="array", array_type="string", max_length=10
            ),
        ]
    )


@pytest.fixture
def validator(schema, metadata_config):
    return ElasticsearchFilterValidator(schema, metadata_config)


class TestValidateFilter:
    def test_simple_term_clause_string(self, validator):
        clauses = [{"term": {"metadata.content_metadata.status.keyword": "approved"}}]
        assert validator.validate_filter(clauses)["status"] is True

    def test_term_without_keyword_suffix_validates(self, validator):
        clauses = [{"term": {"metadata.content_metadata.status": "approved"}}]
        assert validator.validate_filter(clauses)["status"] is True

    def test_range_on_integer(self, validator):
        clauses = [
            {"range": {"metadata.content_metadata.year": {"gt": 2024, "lt": 2026}}}
        ]
        assert validator.validate_filter(clauses)["status"] is True

    def test_range_on_string_rejected(self, validator):
        clauses = [{"range": {"metadata.content_metadata.status": {"gt": "x"}}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "not compatible" in result["error_message"]

    def test_unknown_field_rejected(self, validator):
        clauses = [{"term": {"metadata.content_metadata.unknown_field": "x"}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "unknown_field" in result["error_message"]

    def test_unknown_clause_type_rejected(self, validator):
        clauses = [{"geo_distance": {"location": "0,0"}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "Unsupported clause type" in result["error_message"]

    def test_bool_must_with_nested_clauses(self, validator):
        clauses = [
            {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "metadata.content_metadata.status.keyword": "approved"
                            }
                        },
                        {"range": {"metadata.content_metadata.year": {"gte": 2024}}},
                    ]
                }
            }
        ]
        assert validator.validate_filter(clauses)["status"] is True

    def test_bool_should_and_must_not(self, validator):
        clauses = [
            {
                "bool": {
                    "should": [
                        {"term": {"metadata.content_metadata.status.keyword": "draft"}},
                        {
                            "term": {
                                "metadata.content_metadata.status.keyword": "review"
                            }
                        },
                    ],
                    "must_not": [
                        {"term": {"metadata.content_metadata.is_public": False}}
                    ],
                }
            }
        ]
        assert validator.validate_filter(clauses)["status"] is True

    def test_bool_unknown_inner_key_rejected(self, validator):
        clauses = [{"bool": {"unknown": []}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "Unsupported keys" in result["error_message"]

    def test_datetime_iso_validates(self, validator):
        clauses = [
            {
                "range": {
                    "metadata.content_metadata.created_at": {
                        "gte": "2024-01-01T00:00:00",
                        "lt": "2025-01-01T00:00:00",
                    }
                }
            }
        ]
        assert validator.validate_filter(clauses)["status"] is True

    def test_datetime_invalid_iso_rejected(self, validator):
        clauses = [
            {"range": {"metadata.content_metadata.created_at": {"gte": "not-a-date"}}}
        ]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "ISO 8601" in result["error_message"]

    def test_terms_on_array_field(self, validator):
        clauses = [
            {"terms": {"metadata.content_metadata.tags.keyword": ["urgent", "review"]}}
        ]
        assert validator.validate_filter(clauses)["status"] is True

    def test_terms_value_not_list_rejected(self, validator):
        clauses = [{"terms": {"metadata.content_metadata.tags.keyword": "urgent"}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False
        assert "must be an array" in result["error_message"]

    def test_exists_clause(self, validator):
        clauses = [{"exists": {"field": "metadata.content_metadata.status"}}]
        assert validator.validate_filter(clauses)["status"] is True

    def test_exists_clause_unknown_field_rejected(self, validator):
        clauses = [{"exists": {"field": "metadata.content_metadata.nope"}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False

    def test_wildcard_on_string(self, validator):
        clauses = [{"wildcard": {"metadata.content_metadata.status": "appr*"}}]
        assert validator.validate_filter(clauses)["status"] is True

    def test_wildcard_on_integer_rejected(self, validator):
        clauses = [{"wildcard": {"metadata.content_metadata.year": "20*"}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False

    def test_system_managed_filterable_field_allowed(self, validator):
        clauses = [{"term": {"metadata.content_metadata.filename.keyword": "doc.pdf"}}]
        assert validator.validate_filter(clauses)["status"] is True

    def test_system_managed_non_filterable_field_rejected(self, validator):
        # `start_time` has support_dynamic_filtering=False
        clauses = [{"term": {"metadata.content_metadata.start_time": 100}}]
        result = validator.validate_filter(clauses)
        assert result["status"] is False


class TestProcessFilter:
    def test_normalizes_keyword_suffix_for_string_term(self, validator):
        clauses = [{"term": {"metadata.content_metadata.status": "approved"}}]
        result = validator.process_filter(clauses)
        assert result["status"] is True
        normalized = result["processed_expression"]
        assert "metadata.content_metadata.status.keyword" in normalized[0]["term"]

    def test_preserves_existing_keyword(self, validator):
        clauses = [{"term": {"metadata.content_metadata.status.keyword": "approved"}}]
        result = validator.process_filter(clauses)
        assert result["status"] is True
        assert (
            "metadata.content_metadata.status.keyword"
            in result["processed_expression"][0]["term"]
        )

    def test_does_not_add_keyword_to_numeric(self, validator):
        clauses = [{"range": {"metadata.content_metadata.year": {"gt": 2024}}}]
        result = validator.process_filter(clauses)
        assert result["status"] is True
        assert (
            "metadata.content_metadata.year"
            in result["processed_expression"][0]["range"]
        )
        assert (
            ".keyword" not in list(result["processed_expression"][0]["range"].keys())[0]
        )

    def test_invalid_filter_returns_status_false(self, validator):
        clauses = [{"range": {"metadata.content_metadata.status": {"gt": "x"}}}]
        result = validator.process_filter(clauses)
        assert result["status"] is False
        assert "error_message" in result

    def test_input_clauses_not_mutated(self, validator):
        clauses = [{"term": {"metadata.content_metadata.status": "approved"}}]
        snapshot = [dict(c) for c in clauses]
        validator.process_filter(clauses)
        assert clauses == snapshot


class TestDepthAndCount:
    def test_deeply_nested_bool_rejected(self, validator):
        # Construct a very deeply nested bool tree
        clause = {"term": {"metadata.content_metadata.status.keyword": "x"}}
        for _ in range(10):
            clause = {"bool": {"must": [clause]}}
        result = validator.validate_filter([clause])
        assert result["status"] is False
        assert "depth" in result["error_message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

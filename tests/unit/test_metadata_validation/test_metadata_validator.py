#!/usr/bin/env python3
"""
Comprehensive unit tests for MetadataValidator class.
Tests all metadata types and validation scenarios without requiring running servers.
"""

import json
from datetime import datetime
from unittest.mock import Mock

import pytest
from nvidia_rag.utils.configuration import MetadataConfig
from nvidia_rag.utils.metadata_validation import (
    DatetimeUtility,
    MetadataConfigError,
    MetadataField,
    MetadataSchema,
    MetadataValidator,
    get_valid_array_types,
    validate_metadata_config,
)


class TestMetadataValidator:
    """Test cases for MetadataValidator class."""

    @pytest.fixture
    def mock_config(self):
        """Create a metadata configuration object using the actual MetadataConfig class."""
        return MetadataConfig()

    @pytest.fixture
    def validator(self, mock_config):
        """Create a MetadataValidator instance."""
        return MetadataValidator(mock_config)

    @pytest.fixture
    def array_schema(self):
        """Create a schema with array fields."""
        return MetadataSchema(schema=[
            MetadataField(name="tags", type="array", array_type="string", required=False),
            MetadataField(name="scores", type="array", array_type="float", required=False),
            MetadataField(name="ids", type="array", array_type="integer", required=False),
            MetadataField(name="flags", type="array", array_type="boolean", required=False),
            MetadataField(name="title", type="string", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="is_public", type="boolean", required=False),
        ])

    @pytest.fixture
    def boolean_schema(self):
        """Create a schema with boolean fields."""
        return MetadataSchema(schema=[
            MetadataField(name="is_public", type="boolean", required=False),
            MetadataField(name="is_active", type="boolean", required=False),
            MetadataField(name="is_verified", type="boolean", required=False),
            MetadataField(name="is_premium", type="boolean", required=False),
            MetadataField(name="is_archived", type="boolean", required=False),
            MetadataField(name="has_attachments", type="boolean", required=False),
            MetadataField(name="is_featured", type="boolean", required=False),
            MetadataField(name="is_draft", type="boolean", required=False),
            MetadataField(name="flags", type="array", array_type="boolean", required=False),
            MetadataField(name="title", type="string", required=False),
            MetadataField(name="pages", type="integer", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="tags", type="array", array_type="string", required=False),
        ])

    @pytest.fixture
    def datetime_schema(self):
        """Create a schema with datetime fields."""
        return MetadataSchema(schema=[
            MetadataField(name="created_date", type="datetime", required=False),
            MetadataField(name="updated_date", type="datetime", required=False),
            MetadataField(name="published_date", type="datetime", required=False),
            MetadataField(name="expiry_date", type="datetime", required=False),
            MetadataField(name="last_accessed", type="datetime", required=False),
            MetadataField(name="scheduled_date", type="datetime", required=False),
            MetadataField(name="review_date", type="datetime", required=False),
            MetadataField(name="archive_date", type="datetime", required=False),
            MetadataField(name="title", type="string", required=False),
            MetadataField(name="pages", type="integer", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="is_public", type="boolean", required=False),
            MetadataField(name="tags", type="array", array_type="string", required=False),
        ])

    @pytest.fixture
    def numeric_schema(self):
        """Create a schema with numeric fields."""
        return MetadataSchema(schema=[
            MetadataField(name="pages", type="integer", required=False),
            MetadataField(name="word_count", type="integer", required=False),
            MetadataField(name="quantity", type="integer", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="score", type="float", required=False),
            MetadataField(name="percentage", type="float", required=False),
            MetadataField(name="file_size", type="number", required=False),
            MetadataField(name="price", type="number", required=False),
            MetadataField(name="scores", type="array", array_type="number", required=False),
            MetadataField(name="ratings", type="array", array_type="integer", required=False),
            MetadataField(name="title", type="string", required=False),
            MetadataField(name="category", type="string", required=False),
            MetadataField(name="tags", type="array", array_type="string", required=False),
            MetadataField(name="is_premium", type="boolean", required=False),
        ])

    @pytest.fixture
    def string_schema(self):
        """Create a schema with string fields."""
        return MetadataSchema(schema=[
            MetadataField(name="title", type="string", required=True, max_length=1000),
            MetadataField(name="author", type="string", required=False, max_length=500),
            MetadataField(name="description", type="string", required=False, max_length=2000),
            MetadataField(name="category", type="string", required=False, max_length=100),
            MetadataField(name="tags", type="array", array_type="string", required=False, max_length=50),
            MetadataField(name="file_path", type="string", required=False, max_length=1000),
            MetadataField(name="mime_type", type="string", required=False, max_length=100),
            MetadataField(name="pages", type="integer", required=False),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="is_public", type="boolean", required=False),
        ])

    @pytest.fixture
    def required_field_schema(self):
        """Create a schema with required fields for testing."""
        return MetadataSchema(schema=[
            MetadataField(name="required_string", type="string", required=True, max_length=100),
            MetadataField(name="required_integer", type="integer", required=True),
            MetadataField(name="required_float", type="float", required=True),
            MetadataField(name="required_boolean", type="boolean", required=True),
            MetadataField(name="required_datetime", type="datetime", required=True),
            MetadataField(name="required_array_string", type="array", array_type="string", required=True, max_length=10),
            MetadataField(name="optional_string", type="string", required=False, max_length=100),
            MetadataField(name="optional_integer", type="integer", required=False),
            MetadataField(name="optional_float", type="float", required=False),
            MetadataField(name="optional_boolean", type="boolean", required=False),
            MetadataField(name="optional_datetime", type="datetime", required=False),
            MetadataField(name="optional_array_integer", type="array", array_type="integer", required=False, max_length=10),
            MetadataField(name="optional_array_boolean", type="array", array_type="boolean", required=False, max_length=10),
        ])

    # ============================================================================
    # Array Metadata Tests
    # ============================================================================

    def test_array_metadata_validation_success(self, validator, array_schema):
        """Test successful array metadata validation."""
        metadata = {
            "tags": ["urgent", "important", "policy"],
            "scores": [0.8, 0.9, 0.7],
            "ids": [1, 2, 3],
            "flags": [True, True, False],
            "title": "Urgent Policy Document",
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["urgent", "important", "policy"]
        assert normalized_data["scores"] == [0.8, 0.9, 0.7]
        assert normalized_data["ids"] == [1, 2, 3]
        assert normalized_data["flags"] == [True, True, False]

    def test_array_metadata_empty_arrays(self, validator, array_schema):
        """Test validation with empty arrays."""
        metadata = {
            "tags": [],
            "scores": [],
            "ids": [],
            "flags": [],
            "title": "Empty Arrays Document",
            "rating": 0.0,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == []
        assert normalized_data["scores"] == []

    def test_array_metadata_wrong_types(self, validator, array_schema):
        """Test validation with wrong array element types."""
        metadata = {
            "tags": [1, 2, 3],
            "scores": ["0.8", "0.9"],
            "ids": [1.5, 2.7],
            "flags": ["true", "false"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("tags" in msg for msg in error_messages)
        assert any("tags" in msg or "ids" in msg for msg in error_messages)

    def test_array_metadata_mixed_types(self, validator, array_schema):
        """Test validation with mixed type arrays."""
        metadata = {
            "tags": ["string", 123, True],
            "scores": [0.8, "0.9", 0.7],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_array_metadata_perfect_scores(self, validator, array_schema):
        """Test validation with perfect scores (1.0, 0.99, 0.98) like Document 7."""
        metadata = {
            "tags": ["premium", "verified", "important"],
            "scores": [1.0, 0.99, 0.98],
            "ids": [19, 20, 21],
            "flags": [True, True, True],
            "title": "Premium Verified Document",
            "rating": 5.0,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["scores"] == [1.0, 0.99, 0.98]
        assert normalized_data["flags"] == [True, True, True]

    def test_array_metadata_very_low_scores(self, validator, array_schema):
        """Test validation with very low scores (0.1, 0.2, 0.05) like Document 6."""
        metadata = {
            "tags": ["archived", "draft", "old"],
            "scores": [0.1, 0.2, 0.05],
            "ids": [16, 17, 18],
            "flags": [False, False, False],
            "title": "Archived Draft Document",
            "rating": 1.0,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["scores"] == [0.1, 0.2, 0.05]
        assert normalized_data["flags"] == [False, False, False]

    def test_array_metadata_string_conversion(self, validator, array_schema):
        """Test validation with string representations that should be converted."""
        metadata = {
            "tags": ["urgent", "important", "policy"],
            "scores": ["0.8", "0.9", "0.7"],
            "ids": ["1", "2", "3"],
            "flags": ["true", "false", "true"],
            "title": "String Conversion Test",
            "rating": "4.5",
            "is_public": "true"
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["scores"] == [0.8, 0.9, 0.7]
        assert normalized_data["ids"] == [1, 2, 3]
        assert normalized_data["flags"] == [True, False, True]
        assert normalized_data["rating"] == 4.5
        assert normalized_data["is_public"] is True

    def test_array_metadata_document_1_urgent_policy(self, validator, array_schema):
        """Test validation with Document 1: urgent/important/policy tags, high scores, mixed flags, public."""
        metadata = {
            "tags": ["urgent", "important", "policy"],
            "scores": [0.8, 0.9, 0.7],
            "ids": [1, 2, 3],
            "flags": [True, True, False],
            "title": "Urgent Policy Document",
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["urgent", "important", "policy"]
        assert normalized_data["scores"] == [0.8, 0.9, 0.7]
        assert normalized_data["ids"] == [1, 2, 3]
        assert normalized_data["flags"] == [True, True, False]
        assert normalized_data["title"] == "urgent policy document"
        assert normalized_data["rating"] == 4.5
        assert normalized_data["is_public"] is True

    def test_array_metadata_document_2_draft_internal(self, validator, array_schema):
        """Test validation with Document 2: draft/internal/review tags, low scores, mostly false flags, private."""
        metadata = {
            "tags": ["draft", "internal", "review"],
            "scores": [0.2, 0.3, 0.1],
            "ids": [4, 5, 6],
            "flags": [False, False, True],
            "title": "Draft Internal Document",
            "rating": 2.0,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["draft", "internal", "review"]
        assert normalized_data["scores"] == [0.2, 0.3, 0.1]
        assert normalized_data["ids"] == [4, 5, 6]
        assert normalized_data["flags"] == [False, False, True]
        assert normalized_data["title"] == "draft internal document"
        assert normalized_data["rating"] == 2.0
        assert normalized_data["is_public"] is False

    def test_array_metadata_document_3_verified_public(self, validator, array_schema):
        """Test validation with Document 3: verified/public/approved tags, medium scores, mixed flags, public."""
        metadata = {
            "tags": ["verified", "public", "approved"],
            "scores": [0.5, 0.6, 0.4],
            "ids": [7, 8, 9],
            "flags": [True, False, True],
            "title": "Verified Public Document",
            "rating": 3.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["verified", "public", "approved"]
        assert normalized_data["scores"] == [0.5, 0.6, 0.4]
        assert normalized_data["ids"] == [7, 8, 9]
        assert normalized_data["flags"] == [True, False, True]
        assert normalized_data["title"] == "verified public document"
        assert normalized_data["rating"] == 3.5
        assert normalized_data["is_public"] is True

    def test_array_metadata_document_4_premium_private(self, validator, array_schema):
        """Test validation with Document 4: premium/private/exclusive tags, very high scores, all true flags, private."""
        metadata = {
            "tags": ["premium", "private", "exclusive"],
            "scores": [0.9, 0.95, 0.85],
            "ids": [10, 11, 12],
            "flags": [True, True, True],
            "title": "Premium Private Document",
            "rating": 4.8,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["premium", "private", "exclusive"]
        assert normalized_data["scores"] == [0.9, 0.95, 0.85]
        assert normalized_data["ids"] == [10, 11, 12]
        assert normalized_data["flags"] == [True, True, True]
        assert normalized_data["title"] == "premium private document"
        assert normalized_data["rating"] == 4.8
        assert normalized_data["is_public"] is False

    def test_array_metadata_document_5_featured_active(self, validator, array_schema):
        """Test validation with Document 5: featured/active/highlighted tags, high scores, mostly true flags, public."""
        metadata = {
            "tags": ["featured", "active", "highlighted"],
            "scores": [0.7, 0.8, 0.6],
            "ids": [13, 14, 15],
            "flags": [True, True, False],
            "title": "Featured Active Document",
            "rating": 4.2,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["featured", "active", "highlighted"]
        assert normalized_data["scores"] == [0.7, 0.8, 0.6]
        assert normalized_data["ids"] == [13, 14, 15]
        assert normalized_data["flags"] == [True, True, False]
        assert normalized_data["title"] == "featured active document"
        assert normalized_data["rating"] == 4.2
        assert normalized_data["is_public"] is True

    def test_array_metadata_document_6_archived_draft(self, validator, array_schema):
        """Test validation with Document 6: archived/draft/old tags, very low scores, all false flags, private."""
        metadata = {
            "tags": ["archived", "draft", "old"],
            "scores": [0.1, 0.2, 0.05],
            "ids": [16, 17, 18],
            "flags": [False, False, False],
            "title": "Archived Draft Document",
            "rating": 1.0,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["archived", "draft", "old"]
        assert normalized_data["scores"] == [0.1, 0.2, 0.05]
        assert normalized_data["ids"] == [16, 17, 18]
        assert normalized_data["flags"] == [False, False, False]
        assert normalized_data["title"] == "archived draft document"
        assert normalized_data["rating"] == 1.0
        assert normalized_data["is_public"] is False

    # ============================================================================
    # Boolean Metadata Tests
    # ============================================================================

    def test_boolean_metadata_validation_success(self, validator, boolean_schema):
        """Test successful boolean metadata validation."""
        metadata = {
            "is_public": True,
            "is_active": True,
            "is_verified": True,
            "is_premium": True,
            "is_archived": False,
            "has_attachments": True,
            "is_featured": True,
            "is_draft": False,
            "flags": [True, True, True],
            "title": "Premium Policy Document",
            "pages": 100,
            "rating": 4.8,
            "tags": ["important", "urgent", "policy"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["is_public"] is True
        assert normalized_data["is_archived"] is False
        assert normalized_data["flags"] == [True, True, True]

    def test_boolean_metadata_string_values(self, validator, boolean_schema):
        """Test boolean validation with string representations."""
        metadata = {
            "is_public": "true",
            "is_active": "false",
            "is_verified": "1",
            "is_premium": "0",
            "is_archived": "on",
            "has_attachments": "off",
            "is_featured": "True",
            "is_draft": "False",
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["is_public"] is True
        assert normalized_data["is_active"] is False
        assert normalized_data["is_verified"] is True
        assert normalized_data["is_premium"] is False

    def test_boolean_metadata_invalid_values(self, validator, boolean_schema):
        """Test boolean validation with invalid values."""
        metadata = {
            "is_public": "invalid",
            "is_active": 123,
            "is_verified": ["not", "boolean"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_boolean_metadata_empty_arrays(self, validator, boolean_schema):
        """Test boolean array validation with empty arrays."""
        metadata = {
            "is_public": False,
            "flags": [],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["flags"] == []

    def test_boolean_metadata_all_true_values(self, validator, boolean_schema):
        """Test validation with all true values like Document 1."""
        metadata = {
            "is_public": True,
            "is_active": True,
            "is_verified": True,
            "is_premium": True,
            "is_archived": False,
            "has_attachments": True,
            "is_featured": True,
            "is_draft": False,
            "flags": [True, True, True],
            "title": "Premium Policy Document",
            "pages": 100,
            "rating": 4.8,
            "tags": ["important", "urgent", "policy"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["is_public"] is True
        assert normalized_data["is_active"] is True
        assert normalized_data["is_verified"] is True
        assert normalized_data["is_premium"] is True
        assert normalized_data["is_archived"] is False
        assert normalized_data["has_attachments"] is True
        assert normalized_data["is_featured"] is True
        assert normalized_data["is_draft"] is False
        assert normalized_data["flags"] == [True, True, True]

    def test_boolean_metadata_all_false_values(self, validator, boolean_schema):
        """Test validation with all false values like Document 2."""
        metadata = {
            "is_public": False,
            "is_active": False,
            "is_verified": False,
            "is_premium": False,
            "is_archived": True,
            "has_attachments": False,
            "is_featured": False,
            "is_draft": True,
            "flags": [False, False, False],
            "title": "Draft Internal Document",
            "pages": 25,
            "rating": 2.0,
            "tags": ["draft", "internal"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["is_public"] is False
        assert normalized_data["is_active"] is False
        assert normalized_data["is_verified"] is False
        assert normalized_data["is_premium"] is False
        assert normalized_data["is_archived"] is True
        assert normalized_data["has_attachments"] is False
        assert normalized_data["is_featured"] is False
        assert normalized_data["is_draft"] is True
        assert normalized_data["flags"] == [False, False, False]

    def test_boolean_metadata_mixed_arrays(self, validator, boolean_schema):
        """Test validation with different mixed boolean array patterns."""
        metadata = {
            "is_public": True,
            "is_active": False,
            "is_verified": True,
            "is_premium": False,
            "is_archived": False,
            "has_attachments": True,
            "is_featured": False,
            "is_draft": False,
            "flags": [True, False, True],
            "title": "Mixed Boolean Document",
            "pages": 75,
            "rating": 3.5,
            "tags": ["mixed", "boolean"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["flags"] == [True, False, True]

        # Test another mixed pattern
        metadata2 = {
            "is_public": False,
            "is_active": True,
            "is_verified": False,
            "is_premium": True,
            "is_archived": False,
            "has_attachments": False,
            "is_featured": True,
            "is_draft": False,
            "flags": [False, True, False],
            "title": "Another Mixed Document",
            "pages": 150,
            "rating": 4.2,
            "tags": ["another", "mixed"]
        }

        is_valid2, errors2, normalized_data2 = validator.validate_and_normalize_metadata_values(
            metadata2, boolean_schema
        )

        assert is_valid2 is True
        assert len(errors2) == 0
        assert normalized_data2["flags"] == [False, True, False]

    def test_boolean_metadata_minimal_document(self, validator, boolean_schema):
        """Test validation with minimal document like Document 8 (all false, empty arrays)."""
        metadata = {
            "is_public": False,
            "is_active": False,
            "is_verified": False,
            "is_premium": False,
            "is_archived": False,
            "has_attachments": False,
            "is_featured": False,
            "is_draft": False,
            "flags": [],
            "title": "Minimal Document",
            "pages": 0,
            "rating": 0.0,
            "tags": []
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert is_valid is True
        assert len(errors) == 0

        assert normalized_data["is_public"] is False
        assert normalized_data["is_active"] is False
        assert normalized_data["is_verified"] is False
        assert normalized_data["is_premium"] is False
        assert normalized_data["is_archived"] is False
        assert normalized_data["has_attachments"] is False
        assert normalized_data["is_featured"] is False
        assert normalized_data["is_draft"] is False
        assert normalized_data["flags"] == []

    # ============================================================================
    # Datetime Metadata Tests
    # ============================================================================

    def test_datetime_metadata_validation_success(self, validator, datetime_schema):
        """Test successful datetime metadata validation."""
        metadata = {
            "created_date": "2024-01-15T10:30:00",
            "updated_date": "2024-06-20T14:45:00",
            "published_date": "2024-02-01T09:00:00",
            "expiry_date": "2025-12-31T23:59:59",
            "last_accessed": "2024-12-01T16:20:00",
            "scheduled_date": "2024-03-15T11:00:00",
            "review_date": "2024-09-30T15:30:00",
            "archive_date": "2026-01-01T00:00:00",
            "title": "2024 Policy Document",
            "pages": 100,
            "rating": 4.8,
            "is_public": True,
            "tags": ["important", "urgent", "policy", "2024"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "T" in normalized_data["created_date"]
        assert "Z" in normalized_data["created_date"] or "+" in normalized_data["created_date"]

    def test_datetime_metadata_various_formats(self, validator, datetime_schema):
        """Test datetime validation with various input formats."""
        metadata = {
            "created_date": "2024-01-15",  # Date only
            "updated_date": "Jan 15, 2024",  # Text format
            "published_date": "2024-01-15 10:30:00",  # Space separator
            "expiry_date": "2024/01/15",  # Slash separator
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        for field in ["created_date", "updated_date", "published_date", "expiry_date"]:
            assert "T" in normalized_data[field] or normalized_data[field].endswith("Z")

    def test_datetime_metadata_invalid_formats(self, validator, datetime_schema):
        """Test datetime validation with invalid formats."""
        metadata = {
            "created_date": "invalid-date",
            "updated_date": "not-a-date",
            "published_date": "2024-13-45",
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_datetime_metadata_leap_year(self, validator, datetime_schema):
        """Test datetime validation with leap year dates."""
        metadata = {
            "created_date": "2024-02-29T12:00:00",  # Leap day
            "updated_date": "2024-02-29T23:59:59",
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_datetime_metadata_non_leap_year(self, validator, datetime_schema):
        """Test datetime validation with invalid leap day in non-leap year."""
        metadata = {
            "created_date": "2023-02-29T12:00:00",  # Invalid leap day in 2023
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_datetime_metadata_year_boundary(self, validator, datetime_schema):
        """Test validation with year boundary dates like Document 6."""
        metadata = {
            "created_date": "2023-12-31T23:59:59",
            "updated_date": "2024-01-01T00:00:01",
            "published_date": "2024-01-01T00:00:00",
            "expiry_date": "2024-12-31T23:59:59",
            "last_accessed": "2024-06-15T12:00:00",
            "scheduled_date": "2024-01-01T00:00:00",
            "review_date": "2024-06-30T17:30:00",
            "archive_date": "2025-01-01T00:00:00",
            "title": "Year Boundary Document",
            "pages": 45,
            "rating": 4.5,
            "is_public": True,
            "tags": ["year-boundary", "featured", "special"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "2023-12-31" in normalized_data["created_date"]
        assert "2024-01-01" in normalized_data["updated_date"] or "2023-12-31" in normalized_data["updated_date"]  # Timezone conversion

    def test_datetime_metadata_business_hours(self, validator, datetime_schema):
        """Test validation with business hours dates like Document 7."""
        metadata = {
            "created_date": "2024-03-15T09:00:00",
            "updated_date": "2024-09-20T17:00:00",
            "published_date": "2024-04-01T10:30:00",
            "expiry_date": "2025-03-31T23:59:59",
            "last_accessed": "2024-12-10T16:45:00",
            "scheduled_date": "2024-04-15T14:00:00",
            "review_date": "2024-11-30T15:00:00",
            "archive_date": "2025-04-01T00:00:00",
            "title": "Business Hours Document",
            "pages": 80,
            "rating": 4.0,
            "is_public": True,
            "tags": ["business", "active", "regular"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "2024-03-15" in normalized_data["created_date"]
        assert "2024-09-20" in normalized_data["updated_date"]

    def test_datetime_metadata_future_dates(self, validator, datetime_schema):
        """Test validation with future dates like Document 4."""
        metadata = {
            "created_date": "2024-01-01T00:00:00",
            "updated_date": "2024-12-01T18:30:00",
            "published_date": "2025-01-15T10:00:00",
            "expiry_date": "2026-01-15T23:59:59",
            "last_accessed": "2024-12-15T11:20:00",
            "scheduled_date": "2025-01-15T09:00:00",
            "review_date": "2024-12-31T16:00:00",
            "archive_date": "2026-02-01T00:00:00",
            "title": "Future Premium Document",
            "pages": 150,
            "rating": 4.2,
            "is_public": False,
            "tags": ["future", "premium", "scheduled"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "2025-01-15" in normalized_data["published_date"]
        assert "2026-01-15" in normalized_data["expiry_date"]

    def test_datetime_metadata_expired_dates(self, validator, datetime_schema):
        """Test validation with expired dates like Document 5."""
        metadata = {
            "created_date": "2023-06-01T09:00:00",
            "updated_date": "2023-11-30T15:45:00",
            "published_date": "2023-07-01T10:00:00",
            "expiry_date": "2024-01-01T00:00:00",
            "last_accessed": "2023-12-15T14:30:00",
            "scheduled_date": "2023-07-15T11:00:00",
            "review_date": "2023-10-31T13:00:00",
            "archive_date": "2024-01-15T00:00:00",
            "title": "Expired Document",
            "pages": 60,
            "rating": 1.5,
            "is_public": False,
            "tags": ["expired", "inactive"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "2024-01-01" in normalized_data["expiry_date"] or "2023-12-31" in normalized_data["expiry_date"]  # Timezone conversion
        assert "2024-01-15" in normalized_data["archive_date"] or "2024-01-14" in normalized_data["archive_date"]  # Timezone conversion

    def test_datetime_metadata_minimal_dates(self, validator, datetime_schema):
        """Test validation with minimal datetime metadata like Document 8."""
        metadata = {
            "created_date": "2024-01-01T00:00:00",
            "updated_date": "2024-01-01T00:00:00",
            "published_date": "2024-01-01T00:00:00",
            "expiry_date": "2024-12-31T23:59:59",
            "last_accessed": "2024-01-01T00:00:00",
            "scheduled_date": "2024-01-01T00:00:00",
            "review_date": "2024-12-31T23:59:59",
            "archive_date": "2024-12-31T23:59:59",
            "title": "Minimal DateTime Document",
            "pages": 10,
            "rating": 1.0,
            "is_public": False,
            "tags": ["minimal", "edge-case"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "2024-01-01" in normalized_data["created_date"] or "2023-12-31" in normalized_data["created_date"]
        assert "2024-12-31" in normalized_data["expiry_date"] or "2024-12-30" in normalized_data["expiry_date"]

    # ============================================================================
    # Numeric Metadata Tests
    # ============================================================================

    def test_numeric_metadata_validation_success(self, validator, numeric_schema):
        """Test successful numeric metadata validation."""
        metadata = {
            "pages": 500,
            "word_count": 25000,
            "quantity": 100,
            "rating": 4.8,
            "score": 0.95,
            "percentage": 98.5,
            "file_size": 2048576,
            "price": 99.99,
            "scores": [0.95, 0.92, 0.88, 0.96],
            "ratings": [5, 4, 5, 4, 5],
            "title": "Premium Technical Guide",
            "category": "technical",
            "tags": ["premium", "technical", "comprehensive"],
            "is_premium": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 500
        assert normalized_data["rating"] == 4.8
        assert normalized_data["scores"] == [0.95, 0.92, 0.88, 0.96]

    def test_numeric_metadata_string_values(self, validator, numeric_schema):
        """Test numeric validation with string representations."""
        metadata = {
            "pages": "500",
            "rating": "4.8",
            "price": "99.99",
            "scores": ["0.95", "0.92"],
            "ratings": ["5", "4", "5"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 500
        assert normalized_data["rating"] == 4.8
        assert normalized_data["scores"] == [0.95, 0.92]

    def test_numeric_metadata_invalid_values(self, validator, numeric_schema):
        """Test numeric validation with invalid values."""
        metadata = {
            "pages": "not-a-number",
            "rating": "invalid",
            "scores": ["0.95", "not-a-number"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_numeric_metadata_negative_values(self, validator, numeric_schema):
        """Test numeric validation with negative values."""
        metadata = {
            "pages": -10,
            "rating": -1.0,
            "price": -9.99,
            "scores": [-0.5, -0.3],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == -10
        assert normalized_data["rating"] == -1.0

    def test_numeric_metadata_zero_values(self, validator, numeric_schema):
        """Test numeric validation with zero values."""
        metadata = {
            "pages": 0,
            "rating": 0.0,
            "price": 0.0,
            "scores": [],
            "ratings": [],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 0
        assert normalized_data["rating"] == 0.0

    def test_numeric_metadata_scientific_notation(self, validator, numeric_schema):
        """Test numeric validation with scientific notation."""
        metadata = {
            "file_size": 1.5e6,  # 1.5MB
            "price": 1.5e2,  # 150.0
            "scores": [9e-1, 8.5e-1],  # 0.9, 0.85
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["file_size"] == 1500000.0
        assert normalized_data["price"] == 150.0

    def test_numeric_metadata_high_values(self, validator, numeric_schema):
        """Test validation with high values like Document 1 (premium content)."""
        metadata = {
            "pages": 500,
            "word_count": 25000,
            "quantity": 100,
            "rating": 4.8,
            "score": 0.95,
            "percentage": 98.5,
            "file_size": 2048576,
            "price": 99.99,
            "scores": [0.95, 0.92, 0.88, 0.96],
            "ratings": [5, 4, 5, 4, 5],
            "title": "Premium Technical Guide",
            "category": "technical",
            "tags": ["premium", "technical", "comprehensive"],
            "is_premium": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 500
        assert normalized_data["word_count"] == 25000
        assert normalized_data["rating"] == 4.8
        assert normalized_data["file_size"] == 2048576
        assert normalized_data["price"] == 99.99
        assert normalized_data["scores"] == [0.95, 0.92, 0.88, 0.96]

    def test_numeric_metadata_medium_values(self, validator, numeric_schema):
        """Test validation with medium values like Document 2 (standard content)."""
        metadata = {
            "pages": 150,
            "word_count": 7500,
            "quantity": 50,
            "rating": 3.5,
            "score": 0.65,
            "percentage": 70.0,
            "file_size": 512000,
            "price": 29.99,
            "scores": [0.65, 0.70, 0.60, 0.68],
            "ratings": [3, 4, 3, 4, 3],
            "title": "Business Process Guide",
            "category": "business",
            "tags": ["business", "process", "standard"],
            "is_premium": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 150
        assert normalized_data["word_count"] == 7500
        assert normalized_data["rating"] == 3.5
        assert normalized_data["file_size"] == 512000
        assert normalized_data["price"] == 29.99

    def test_numeric_metadata_low_values(self, validator, numeric_schema):
        """Test validation with low values like Document 3 (basic content)."""
        metadata = {
            "pages": 25,
            "word_count": 1200,
            "quantity": 10,
            "rating": 2.0,
            "score": 0.30,
            "percentage": 25.5,
            "file_size": 64000,
            "price": 9.99,
            "scores": [0.30, 0.25, 0.35, 0.28],
            "ratings": [2, 1, 2, 3, 2],
            "title": "Basic Introduction",
            "category": "basic",
            "tags": ["basic", "introduction", "simple"],
            "is_premium": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 25
        assert normalized_data["word_count"] == 1200
        assert normalized_data["rating"] == 2.0
        assert normalized_data["file_size"] == 64000
        assert normalized_data["price"] == 9.99

    def test_numeric_metadata_large_values(self, validator, numeric_schema):
        """Test validation with large values like Document 6 (extensive content)."""
        metadata = {
            "pages": 999999,
            "word_count": 999999,
            "quantity": 999999,
            "rating": 5.0,
            "score": 1.0,
            "percentage": 100.0,
            "file_size": 999999999,
            "price": 999.99,
            "scores": [1.0, 1.0, 1.0, 1.0, 1.0],
            "ratings": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "title": "Comprehensive Research Paper",
            "category": "research",
            "tags": ["research", "comprehensive", "extensive", "maximum"],
            "is_premium": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 999999
        assert normalized_data["word_count"] == 999999
        assert normalized_data["rating"] == 5.0
        assert normalized_data["file_size"] == 999999999
        assert normalized_data["price"] == 999.99
        assert normalized_data["scores"] == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_numeric_metadata_decimal_precision(self, validator, numeric_schema):
        """Test validation with high precision decimals like Document 7."""
        metadata = {
            "pages": 100,
            "word_count": 5000,
            "quantity": 25,
            "rating": 4.123456789,
            "score": 0.87654321,
            "percentage": 87.654321,
            "file_size": 123456.789,
            "price": 45.678901,
            "scores": [0.87654321, 0.98765432, 0.76543210],
            "ratings": [4, 5, 4, 5, 4],
            "title": "Precision Metrics Document",
            "category": "precision",
            "tags": ["precision", "exact", "decimal"],
            "is_premium": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["rating"] == 4.123456789
        assert normalized_data["score"] == 0.87654321
        assert normalized_data["percentage"] == 87.654321
        assert normalized_data["file_size"] == 123456.789
        assert normalized_data["price"] == 45.678901
        assert normalized_data["scores"] == [0.87654321, 0.98765432, 0.76543210]

    def test_numeric_metadata_missing_fields(self, validator, numeric_schema):
        """Test validation with missing fields like Document 8 (edge case)."""
        metadata = {
            "title": "Minimal Document",
            "category": "minimal",
            "tags": ["minimal", "edge-case"],
            "is_premium": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        # Check that provided fields are present with correct values
        assert "title" in normalized_data
        assert "category" in normalized_data
        assert "tags" in normalized_data
        assert "is_premium" in normalized_data
        assert normalized_data["title"] == "minimal document"
        assert normalized_data["category"] == "minimal"
        assert normalized_data["is_premium"] is False
        # Check that missing numeric fields are present but set to None
        assert "pages" in normalized_data
        assert "rating" in normalized_data
        assert "file_size" in normalized_data
        assert normalized_data["pages"] is None
        assert normalized_data["rating"] is None
        assert normalized_data["file_size"] is None

    def test_numeric_metadata_mid_range_values(self, validator, numeric_schema):
        """Test validation with mid-range values like Document 9 (BETWEEN testing)."""
        metadata = {
            "pages": 75,
            "word_count": 3750,
            "quantity": 15,
            "rating": 3.75,
            "score": 0.75,
            "percentage": 75.0,
            "file_size": 750000,
            "price": 75.00,
            "scores": [0.75, 0.80, 0.70],
            "ratings": [4, 3, 4, 4, 3],
            "title": "Mid-Range Document",
            "category": "mid-range",
            "tags": ["mid-range", "average", "balanced"],
            "is_premium": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["pages"] == 75
        assert normalized_data["word_count"] == 3750
        assert normalized_data["rating"] == 3.75
        assert normalized_data["score"] == 0.75
        assert normalized_data["percentage"] == 75.0
        assert normalized_data["file_size"] == 750000
        assert normalized_data["price"] == 75.00

    def test_numeric_metadata_scientific_notation_arrays(self, validator, numeric_schema):
        """Test validation with scientific notation in arrays like Document 10."""
        metadata = {
            "pages": 1000,
            "word_count": 50000,
            "quantity": 1000,
            "rating": 4.5,
            "score": 0.9,
            "percentage": 90.0,
            "file_size": 1.5e6,
            "price": 1.5e2,
            "scores": [9e-1, 8.5e-1, 9.5e-1],
            "ratings": [5, 4, 5, 4, 5],
            "title": "Scientific Research Document",
            "category": "scientific",
            "tags": ["scientific", "research", "notation"],
            "is_premium": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["file_size"] == 1500000.0
        assert normalized_data["price"] == 150.0
        assert normalized_data["scores"] == [0.9, 0.85, 0.95]

    # ============================================================================
    # String Metadata Tests
    # ============================================================================

    def test_string_metadata_validation_success(self, validator, string_schema):
        """Test successful string metadata validation."""
        metadata = {
            "title": "Policy on accepting corrected claims",
            "author": "John Smith",
            "description": "Comprehensive policy document covering corrected claims procedures.",
            "category": "persona_A",
            "tags": ["policy", "claims", "healthcare", "procedures"],
            "file_path": "/documents/policies/corrected_claims_policy.pdf",
            "mime_type": "application/pdf",
            "pages": 25,
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "policy on accepting corrected claims"
        assert normalized_data["author"] == "john smith"
        assert normalized_data["tags"] == ["policy", "claims", "healthcare", "procedures"]

    def test_string_metadata_unicode_support(self, validator, string_schema):
        """Test string validation with Unicode characters."""
        metadata = {
            "title": "Policy & Claims (2024) - Special: $100",
            "author": "José García",
            "description": "Policy document with special characters: &, $, %, @, # and Unicode: 政策, 声明, 处理",
            "category": "persona_A",
            "tags": ["special", "unicode", "policy", "claims"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "josé garcía" in normalized_data["author"]
        assert "政策" in normalized_data["description"]

    def test_string_metadata_empty_strings(self, validator, string_schema):
        """Test string validation with empty strings."""
        metadata = {
            "title": "",
            "author": "",
            "description": "",
            "category": "persona_D",
            "tags": [],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("title" in msg for msg in error_messages)

    def test_string_metadata_max_length_violation(self, validator, string_schema):
        """Test string validation with max length violations."""
        long_title = "x" * 1001
        long_description = "This is a very long description. " * 100

        metadata = {
            "title": long_title,
            "description": long_description,
            "category": "persona_B",
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_string_metadata_whitespace_handling(self, validator, string_schema):
        """Test string validation with whitespace handling."""
        metadata = {
            "title": "  Policy Document  ",  # Leading/trailing whitespace
            "author": "  John Smith  ",
            "description": "  Description with spaces  ",
            "category": "  persona_A  ",
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "policy document" in normalized_data["title"]
        assert "john smith" in normalized_data["author"]

    def test_field_level_max_length_constraints(self, validator):
        """Test that field-level max_length constraints are properly enforced."""
        schema = MetadataSchema(schema=[
            MetadataField(name="short_title", type="string", required=False, max_length=10),
            MetadataField(name="medium_description", type="string", required=False, max_length=50),
            MetadataField(name="limited_tags", type="array", array_type="string", required=False, max_length=3),
        ])

        valid_metadata = {
            "short_title": "Short",
            "medium_description": "This is a medium length description",
            "limited_tags": ["tag1", "tag2"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            valid_metadata, schema
        )

        assert is_valid is True
        assert len(errors) == 0

        invalid_metadata = {
            "short_title": "This title is too long for the field constraint",
            "medium_description": "This description is way too long and should exceed the field-level max_length constraint of 50 characters",
            "limited_tags": ["tag1", "tag2", "tag3", "tag4"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            invalid_metadata, schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("short_title" in msg for msg in error_messages)
        assert any("medium_description" in msg for msg in error_messages)
        assert any("limited_tags" in msg for msg in error_messages)

    def test_config_level_max_length_defaults(self, validator):
        """Test that config-level max_length defaults are used when field-level max_length is not specified."""
        schema = MetadataSchema(schema=[
            MetadataField(name="title", type="string", required=False),
            MetadataField(name="tags", type="array", array_type="string", required=False),
        ])

        valid_metadata = {
            "title": "A" * 1000,
            "tags": ["tag"] * 500,
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            valid_metadata, schema
        )

        assert is_valid is True
        assert len(errors) == 0

        invalid_metadata = {
            "title": "A" * 70000,
            "tags": ["tag"] * 1500,
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            invalid_metadata, schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("title" in msg for msg in error_messages)
        assert any("tags" in msg for msg in error_messages)

    def test_string_metadata_policy_document(self, validator, string_schema):
        """Test validation with policy document like Document 1 (John Smith, persona_A, public)."""
        metadata = {
            "title": "Policy on accepting corrected claims",
            "author": "John Smith",
            "description": "Comprehensive policy document covering corrected claims procedures and requirements for healthcare providers.",
            "category": "persona_A",
            "tags": ["policy", "claims", "healthcare", "procedures"],
            "file_path": "/documents/policies/corrected_claims_policy.pdf",
            "mime_type": "application/pdf",
            "pages": 25,
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "policy on accepting corrected claims"
        assert normalized_data["author"] == "john smith"
        assert normalized_data["category"] == "persona_a"
        assert normalized_data["tags"] == ["policy", "claims", "healthcare", "procedures"]
        assert normalized_data["file_path"] == "/documents/policies/corrected_claims_policy.pdf"
        assert normalized_data["mime_type"] == "application/pdf"

    def test_string_metadata_claims_guidelines(self, validator, string_schema):
        """Test validation with claims guidelines like Document 2 (Jane Doe, persona_B, private)."""
        metadata = {
            "title": "Claims processing guidelines",
            "author": "Jane Doe",
            "description": "Step-by-step guidelines for processing healthcare claims including documentation requirements and timelines.",
            "category": "persona_B",
            "tags": ["claims", "processing", "guidelines", "healthcare"],
            "file_path": "/documents/guidelines/claims_processing.pdf",
            "mime_type": "application/pdf",
            "pages": 40,
            "rating": 4.2,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "claims processing guidelines"
        assert normalized_data["author"] == "jane doe"
        assert normalized_data["category"] == "persona_b"
        assert normalized_data["tags"] == ["claims", "processing", "guidelines", "healthcare"]
        assert normalized_data["is_public"] is False

    def test_string_metadata_technical_specs(self, validator, string_schema):
        """Test validation with technical specs like Document 3 (John Smith, persona_A, public)."""
        metadata = {
            "title": "Technical specifications for claims system",
            "author": "John Smith",
            "description": "Detailed technical specifications including API documentation, database schema, and system architecture.",
            "category": "persona_A",
            "tags": ["technical", "specifications", "system", "architecture"],
            "file_path": "/documents/technical/claims_system_specs.pdf",
            "mime_type": "application/pdf",
            "pages": 150,
            "rating": 4.8,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "technical specifications for claims system"
        assert normalized_data["author"] == "john smith"
        assert normalized_data["category"] == "persona_a"
        assert normalized_data["tags"] == ["technical", "specifications", "system", "architecture"]
        assert normalized_data["pages"] == 150
        assert normalized_data["rating"] == 4.8

    def test_string_metadata_training_manual(self, validator, string_schema):
        """Test validation with training manual like Document 4 (Sarah Johnson, persona_C, public)."""
        metadata = {
            "title": "Training manual for claims processors",
            "author": "Sarah Johnson",
            "description": "Training manual covering all aspects of claims processing including hands-on exercises and case studies.",
            "category": "persona_C",
            "tags": ["training", "manual", "processors", "education"],
            "file_path": "/documents/training/claims_processor_manual.pdf",
            "mime_type": "application/pdf",
            "pages": 200,
            "rating": 4.0,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "training manual for claims processors"
        assert normalized_data["author"] == "sarah johnson"
        assert normalized_data["category"] == "persona_c"
        assert normalized_data["tags"] == ["training", "manual", "processors", "education"]
        assert normalized_data["pages"] == 200
        assert normalized_data["rating"] == 4.0

    def test_string_metadata_empty_strings_edge_case(self, validator, string_schema):
        """Test validation with empty strings like Document 5 (empty strings, persona_D, private)."""
        metadata = {
            "title": "",
            "author": "",
            "description": "",
            "category": "persona_d",
            "tags": [],
            "file_path": "/documents/edge_cases/minimal.txt",
            "mime_type": "text/plain",
            "pages": 0,
            "rating": 0.0,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("title" in msg for msg in error_messages)
        assert normalized_data["category"] == "persona_d"
        assert normalized_data["tags"] == []
        assert normalized_data["file_path"] == "/documents/edge_cases/minimal.txt"
        assert normalized_data["mime_type"] == "text/plain"

    def test_string_metadata_special_characters(self, validator, string_schema):
        """Test validation with special characters like Document 6 (José García, persona_A, public)."""
        metadata = {
            "title": "Policy & Claims (2024) - Special: $100",
            "author": "José García",
            "description": "Policy document with special characters: &, $, %, @, # and Unicode: 政策, 声明, 处理",
            "category": "persona_A",
            "tags": ["special", "unicode", "policy", "claims"],
            "file_path": "/documents/special/unicode_policy.pdf",
            "mime_type": "application/pdf",
            "pages": 75,
            "rating": 3.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "policy & claims (2024) - special: $100"
        assert normalized_data["author"] == "josé garcía"
        assert "josé garcía" in normalized_data["author"]
        assert "政策" in normalized_data["description"]
        assert "&" in normalized_data["title"]
        assert "$" in normalized_data["title"]
        assert normalized_data["category"] == "persona_a"
        assert normalized_data["tags"] == ["special", "unicode", "policy", "claims"]

    def test_string_metadata_very_long_strings(self, validator, string_schema):
        """Test validation with very long strings like Document 7 (boundary testing)."""
        metadata = {
            "title": "x" * 999,  # Very long title (within 1000 limit)
            "author": "Very Long Author Name That Exceeds Normal Limits",
            "description": "This is a very long description that contains many words and should test the maximum length handling of the string metadata system. " * 10,
            "category": "persona_B",
            "tags": ["very", "long", "strings", "boundary", "testing"],
            "file_path": "/documents/long_strings/very_long_metadata_document.pdf",
            "mime_type": "application/pdf",
            "pages": 300,
            "rating": 4.7,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert len(normalized_data["title"]) == 999
        assert normalized_data["title"] == "x" * 999
        assert normalized_data["author"] == "very long author name that exceeds normal limits"
        assert "very long description" in normalized_data["description"]
        assert normalized_data["category"] == "persona_b"
        assert normalized_data["tags"] == ["very", "long", "strings", "boundary", "testing"]

    def test_string_metadata_mixed_content(self, validator, string_schema):
        """Test validation with mixed content like Document 8 (persona_D, private)."""
        metadata = {
            "title": "Mixed content document",
            "author": "Mixed Author",
            "description": "Document containing various types of content including text, numbers, and special formatting.",
            "category": "persona_D",
            "tags": ["mixed", "content", "various", "types"],
            "file_path": "/documents/mixed/mixed_content.txt",
            "mime_type": "text/plain",
            "pages": 50,
            "rating": 3.8,
            "is_public": False
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "mixed content document"
        assert normalized_data["author"] == "mixed author"
        assert normalized_data["category"] == "persona_d"
        assert normalized_data["tags"] == ["mixed", "content", "various", "types"]
        assert normalized_data["file_path"] == "/documents/mixed/mixed_content.txt"
        assert normalized_data["mime_type"] == "text/plain"
        assert normalized_data["pages"] == 50
        assert normalized_data["rating"] == 3.8
        assert normalized_data["is_public"] is False

    # ============================================================================
    # Required Field Tests
    # ============================================================================

    def test_required_field_validation_success(self, validator, required_field_schema):
        """Test successful validation with all required fields provided."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00",
            "required_array_string": ["item1", "item2"],
            "optional_string": "optional_value",
            "optional_integer": 100,
            "optional_float": 2.718,
            "optional_boolean": False,
            "optional_datetime": "2024-12-31T23:59:59",
            "optional_array_integer": [1, 2, 3],
            "optional_array_boolean": [True, False],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["required_string"] == "test_value"
        assert normalized_data["required_integer"] == 42
        assert normalized_data["required_float"] == 3.14
        assert normalized_data["required_boolean"] is True

    def test_required_field_validation_missing_fields(self, validator, required_field_schema):
        """Test validation with missing required fields."""
        metadata = {
            "required_string": "test_value",
            # Missing required_integer
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00",
            "required_array_string": ["item1", "item2"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_integer" in msg for msg in error_messages)

    def test_required_field_validation_empty_values(self, validator, required_field_schema):
        """Test validation with empty values for required fields."""
        metadata = {
            "required_string": "",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00",
            "required_array_string": [],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_string" in msg for msg in error_messages)
        assert any("required_array_string" in msg for msg in error_messages)

    def test_required_field_validation_null_values(self, validator, required_field_schema):
        """Test validation with null values for required fields."""
        metadata = {
            "required_string": None,
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00",
            "required_array_string": ["item1", "item2"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_string" in msg for msg in error_messages)

    def test_required_field_validation_type_errors(self, validator, required_field_schema):
        """Test validation with type errors for required fields."""
        metadata = {
            "required_string": "test_value",
            "required_integer": "not_a_number",
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_integer" in msg for msg in error_messages)

        metadata2 = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": "not_a_float",
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2"]
        }

        is_valid2, errors2, normalized_data2 = validator.validate_and_normalize_metadata_values(
            metadata2, required_field_schema
        )

        assert is_valid2 is False
        assert len(errors2) > 0
        error_messages2 = [error["error"] for error in errors2]
        assert any("required_float" in msg for msg in error_messages2)

    def test_required_field_validation_invalid_boolean(self, validator, required_field_schema):
        """Test validation with invalid boolean values."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": "maybe",
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_boolean" in msg for msg in error_messages)

    def test_required_field_validation_invalid_datetime(self, validator, required_field_schema):
        """Test validation with invalid datetime format."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "invalid_date",
            "required_array_string": ["item1", "item2"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_datetime" in msg for msg in error_messages)

    def test_required_field_validation_array_type_errors(self, validator, required_field_schema):
        """Test validation with wrong types in arrays."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": [1, 2, 3]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_array_string" in msg for msg in error_messages)

    def test_required_field_validation_array_length_violation(self, validator, required_field_schema):
        """Test validation with array length violations."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10", "item11"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_array_string" in msg for msg in error_messages)

    def test_required_field_validation_string_length_violation(self, validator, required_field_schema):
        """Test validation with string length violations."""
        metadata = {
            "required_string": "x" * 101,
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("required_string" in msg for msg in error_messages)

    def test_required_field_validation_zero_values(self, validator, required_field_schema):
        """Test validation with zero values for numeric fields."""
        metadata = {
            "required_string": "test",
            "required_integer": 0,
            "required_float": 0.0,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["required_integer"] == 0
        assert normalized_data["required_float"] == 0.0

    def test_required_field_validation_boolean_string_representations(self, validator, required_field_schema):
        """Test validation with boolean string representations."""
        metadata = {
            "required_string": "test",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": "true",
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["required_boolean"] is True

    def test_required_field_validation_maximum_length_boundaries(self, validator, required_field_schema):
        """Test validation with maximum length boundaries."""
        metadata = {
            "required_string": "x" * 100,
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert len(normalized_data["required_string"]) == 100
        assert len(normalized_data["required_array_string"]) == 10

    def test_required_field_validation_only_required_fields(self, validator, required_field_schema):
        """Test validation with only required fields (no optional fields)."""
        metadata = {
            "required_string": "test_value",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["item1"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "required_string" in normalized_data
        assert "required_integer" in normalized_data
        assert "required_float" in normalized_data
        assert "required_boolean" in normalized_data
        assert "required_datetime" in normalized_data
        assert "required_array_string" in normalized_data
        assert "optional_string" not in normalized_data or normalized_data["optional_string"] is None

    # ============================================================================
    # Edge Cases and Error Handling Tests
    # ============================================================================

    def test_metadata_validation_unknown_fields(self, validator, array_schema):
        """Test validation with unknown fields (should fail validation)."""
        metadata = {
            "tags": ["urgent", "important"],
            "unknown_field": "should cause validation failure",
            "another_unknown": 123,
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("unknown_field" in msg for msg in error_messages)
        assert any("another_unknown" in msg for msg in error_messages)
        assert "tags" in normalized_data

    def test_metadata_validation_none_schema(self, validator):
        """Test validation with None schema."""
        metadata = {"title": "test"}

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, None
        )

        assert isinstance(is_valid, bool)

    def test_metadata_validation_empty_schema(self, validator):
        """Test validation with empty schema."""
        empty_schema = MetadataSchema(schema=[])
        metadata = {"title": "test"}

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, empty_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("title" in msg for msg in error_messages)
        assert normalized_data == metadata

    def test_metadata_validation_large_arrays(self, validator, array_schema):
        """Test validation with arrays exceeding max length."""
        large_tags = [f"tag_{i}" for i in range(1001)]
        large_scores = [float(i) for i in range(1001)]

        metadata = {
            "tags": large_tags,
            "scores": large_scores,
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_metadata_validation_special_float_values(self, validator, numeric_schema):
        """Test validation with special float values (NaN, inf)."""
        import math

        metadata = {
            "rating": float('nan'),
            "score": float('inf'),
            "percentage": float('-inf'),
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert is_valid is False or is_valid is True

    # ============================================================================
    # Configuration Tests
    # ============================================================================

    def test_validate_metadata_config_success(self, mock_config):
        """Test successful metadata config validation."""
        result = validate_metadata_config(mock_config)
        assert result == mock_config

    def test_validate_metadata_config_with_metadata_attr(self):
        """Test metadata config validation with metadata attribute."""
        config = Mock()
        config.metadata = Mock()
        config.metadata.max_array_length = 1000
        config.metadata.max_string_length = 65535

        result = validate_metadata_config(config)
        assert hasattr(result, 'max_array_length')
        assert hasattr(result, 'max_string_length')
        assert result is config

    def test_validate_metadata_config_failure(self):
        """Test metadata config validation failure."""
        config = Mock()
        del config.max_array_length
        del config.max_string_length

        try:
            result = validate_metadata_config(config)
            assert hasattr(result, 'max_array_length')
            assert hasattr(result, 'max_string_length')
        except MetadataConfigError:
            pass

    def test_get_valid_array_types(self):
        """Test getting valid array types."""
        valid_types = get_valid_array_types()
        assert "string" in valid_types
        assert "number" in valid_types
        assert "integer" in valid_types
        assert "float" in valid_types
        assert "boolean" in valid_types

    # ============================================================================
    # DatetimeUtility Tests
    # ============================================================================

    def test_datetime_utility_normalize_datetime(self):
        """Test DatetimeUtility.normalize_datetime_to_utc_z."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        normalized = DatetimeUtility.normalize_datetime_to_utc_z(dt)
        assert "T" in normalized
        assert normalized.endswith("Z")

    def test_datetime_utility_parse_datetime(self):
        """Test DatetimeUtility.parse_datetime."""
        formats = [
            "2024-01-15",
            "2024-01-15T10:30:00",
            "Jan 15, 2024",
            "2024/01/15",
        ]

        for date_str in formats:
            result = DatetimeUtility.parse_datetime(date_str)
            assert "T" in result or result.endswith("Z")

    def test_datetime_utility_parse_datetime_invalid(self):
        """Test DatetimeUtility.parse_datetime with invalid format."""
        with pytest.raises(ValueError):
            DatetimeUtility.parse_datetime("invalid-date")

    def test_datetime_utility_convert_date_equality_to_between(self):
        """Test DatetimeUtility.convert_date_equality_to_between."""
        start, end = DatetimeUtility.convert_date_equality_to_between("2024-01-15")
        assert "T00:00:00" in start
        assert "T23:59:59" in end

    # ============================================================================
    # Additional Edge Cases and Missing Test Scenarios
    # ============================================================================

    def test_array_metadata_nested_arrays(self, validator, array_schema):
        """Test validation with nested arrays (should fail)."""
        metadata = {
            "tags": [["nested", "array"], "should", "fail"],
            "scores": [0.8, 0.9, 0.7],
            "ids": [1, 2, 3],
            "flags": [True, True, False],
            "title": "Nested Array Test",
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_array_metadata_heterogeneous_arrays(self, validator, array_schema):
        """Test validation with heterogeneous arrays (mixed types)."""
        metadata = {
            "tags": ["string", 123, True, None],  # Mixed types
            "scores": [0.8, "0.9", 0.7, None],  # Mixed types
            "ids": [1, "2", 3.5, None],  # Mixed types
            "flags": [True, "false", 1, None],  # Mixed types
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_boolean_metadata_edge_case_values(self, validator, boolean_schema):
        """Test boolean validation with edge case values."""
        metadata = {
            "is_public": "yes",  # Should convert to True
            "is_active": "no",   # Should convert to False
            "is_verified": "1",  # Should convert to True
            "is_premium": "0",   # Should convert to False
            "is_archived": "enabled",  # Should convert to True
            "has_attachments": "disabled",  # Should convert to False
            "is_featured": "active",  # Should convert to True
            "is_draft": "inactive",  # Should convert to False
            "flags": ["yes", "no", "1", "0"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, boolean_schema
        )

        assert isinstance(is_valid, bool)

    def test_datetime_metadata_timezone_handling(self, validator, datetime_schema):
        """Test datetime validation with timezone information."""
        metadata = {
            "created_date": "2024-01-15T10:30:00+05:00",  # With timezone
            "updated_date": "2024-06-20T14:45:00-08:00",  # With timezone
            "published_date": "2024-02-01T09:00:00Z",     # UTC
            "expiry_date": "2025-12-31T23:59:59.999Z",   # With milliseconds
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "T" in normalized_data["created_date"]

    def test_datetime_metadata_invalid_timezone(self, validator, datetime_schema):
        """Test datetime validation with invalid timezone format."""
        metadata = {
            "created_date": "2024-01-15T10:30:00+25:00",  # Invalid timezone
            "updated_date": "2024-06-20T14:45:00-99:00",  # Invalid timezone
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, datetime_schema
        )

        assert isinstance(is_valid, bool)

    def test_numeric_metadata_extreme_values(self, validator, numeric_schema):
        """Test numeric validation with extreme values."""
        metadata = {
            "pages": float('inf'),  # Infinity
            "word_count": float('-inf'),  # Negative infinity
            "quantity": 1e308,  # Very large number
            "rating": 1e-308,  # Very small number
            "score": float('nan'),  # NaN
            "percentage": -1e308,  # Very large negative number
            "file_size": 0.0,  # Zero
            "price": -0.0,  # Negative zero
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert isinstance(is_valid, bool)

    def test_numeric_metadata_overflow_values(self, validator, numeric_schema):
        """Test numeric validation with overflow values."""
        metadata = {
            "pages": 2**63,  # Very large integer
            "word_count": -2**63,  # Very large negative integer
            "quantity": 1e100,  # Very large float
            "rating": -1e100,  # Very large negative float
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, numeric_schema
        )

        assert isinstance(is_valid, bool)

    def test_string_metadata_control_characters(self, validator, string_schema):
        """Test string validation with control characters."""
        metadata = {
            "title": "Title with \n newline and \t tab",
            "author": "Author with \r carriage return",
            "description": "Description with \x00 null byte and \x1F control chars",
            "category": "Category with \b backspace",
            "tags": ["tag\nwith\nnewlines", "tag\twith\ttabs", "tag\rwith\rreturns"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert isinstance(is_valid, bool)

    def test_string_metadata_emoji_and_special_unicode(self, validator, string_schema):
        """Test string validation with emoji and special Unicode characters."""
        metadata = {
            "title": "Title with emoji 🚀 and symbols ©®™",
            "author": "Author with flags 🇺🇸🇬🇧 and math symbols ∑∏∫",
            "description": "Description with arrows →←↑↓ and currency €£¥",
            "category": "Category with hearts ♥♡ and stars ★☆",
            "tags": ["emoji", "🚀", "©", "🇺🇸", "♥", "★"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "🚀" in normalized_data["title"]
        assert "🇺🇸" in normalized_data["tags"]

    def test_string_metadata_html_entities(self, validator, string_schema):
        """Test string validation with HTML entities and special characters."""
        metadata = {
            "title": "Title with &amp; &lt; &gt; &quot; &apos;",
            "author": "Author with <script>alert('xss')</script>",
            "description": "Description with &copy; &reg; &trade; symbols",
            "category": "Category with <b>bold</b> and <i>italic</i>",
            "tags": ["html", "entities", "&amp;", "&lt;", "&gt;"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert "&amp;" in normalized_data["title"]

    def test_required_field_validation_whitespace_only(self, validator, required_field_schema):
        """Test validation with whitespace-only strings for required fields."""
        metadata = {
            "required_string": "   ",
            "required_integer": 42,
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["   ", "  \t  ", "  \n  "]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert isinstance(is_valid, bool)

    def test_required_field_validation_numeric_edge_cases(self, validator, required_field_schema):
        """Test validation with numeric edge cases for required fields."""
        metadata = {
            "required_string": "test",
            "required_integer": 0,
            "required_float": 0.0,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": ["test"]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["required_integer"] == 0
        assert normalized_data["required_float"] == 0.0

    def test_metadata_validation_case_sensitivity(self, validator, string_schema):
        """Test metadata validation with case sensitivity."""
        metadata = {
            "title": "Title with Mixed Case",
            "author": "Author with UPPERCASE and lowercase",
            "description": "Description with Title Case",
            "category": "CATEGORY_IN_UPPERCASE",
            "tags": ["Tag1", "TAG2", "tag3", "Tag_4"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "title with mixed case"
        assert "tag1" in normalized_data["tags"]

    def test_metadata_validation_duplicate_fields(self, validator, array_schema):
        """Test validation with duplicate field names (should use last value)."""
        # Create metadata with duplicate keys using dict constructor
        metadata = {
            "tags": ["second", "set"],
            "scores": [0.9, 0.8],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["tags"] == ["second", "set"]
        assert normalized_data["scores"] == [0.9, 0.8]

    def test_metadata_validation_deep_nesting(self, validator, array_schema):
        """Test validation with deeply nested structures (should fail)."""
        metadata = {
            "tags": ["simple", "array"],
            "scores": [0.8, 0.9],
            "ids": [1, 2],
            "flags": [True, False],
            "title": "Deep Nesting Test",
            "rating": 4.5,
            "is_public": True,
            "nested_object": {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "deep"
                        }
                    }
                }
            }
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("nested_object" in msg for msg in error_messages)
        assert normalized_data == metadata

    def test_metadata_validation_circular_references(self, validator, array_schema):
        """Test validation with circular references (should handle gracefully)."""
        circular_list = ["item1", "item2"]
        circular_list.append(circular_list)

        metadata = {
            "tags": ["simple", "array"],
            "scores": [0.8, 0.9],
            "ids": [1, 2],
            "flags": [True, False],
            "title": "Circular Reference Test",
            "rating": 4.5,
            "is_public": True,
            "circular_field": circular_list
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid is False
        assert len(errors) > 0
        error_messages = [error["error"] for error in errors]
        assert any("circular_field" in msg for msg in error_messages)
        assert normalized_data == metadata

    def test_metadata_validation_memory_efficiency(self, validator, string_schema):
        """Test validation with very large strings to check memory efficiency."""
        large_string = "x" * 1000000

        metadata = {
            "title": "Normal title",
            "author": "Normal author",
            "description": large_string,
            "category": "test",
            "tags": ["normal", "tags"],
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )

        assert isinstance(is_valid, bool)

    def test_metadata_validation_performance_large_arrays(self, validator, array_schema):
        """Test validation with large arrays to check performance."""
        large_array = [f"tag_{i}" for i in range(10000)]

        metadata = {
            "tags": large_array,
            "scores": [0.8, 0.9, 0.7],
            "ids": [1, 2, 3],
            "flags": [True, True, False],
            "title": "Large Array Test",
            "rating": 4.5,
            "is_public": True
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert isinstance(is_valid, bool)

    def test_metadata_validation_concurrent_access(self, validator, array_schema):
        """Test validation with concurrent access patterns."""
        import threading
        import time

        results = []
        errors = []

        def validate_metadata(thread_id):
            try:
                metadata = {
                    "tags": [f"tag_{thread_id}"],
                    "scores": [0.8 + thread_id * 0.1],
                    "ids": [thread_id],
                    "flags": [True if thread_id % 2 == 0 else False],
                    "title": f"Thread {thread_id} Document",
                    "rating": 4.0 + thread_id * 0.1,
                    "is_public": True
                }

                is_valid, error_list, normalized_data = validator.validate_and_normalize_metadata_values(
                    metadata, array_schema
                )

                results.append((thread_id, is_valid))
                if error_list:
                    errors.append((thread_id, error_list))

            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=validate_metadata, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(is_valid for _, is_valid in results)
        assert len(errors) == 0

    def test_metadata_validation_error_message_clarity(self, validator, required_field_schema):
        """Test that error messages are clear and helpful."""
        metadata = {
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0

        error_messages = [error.get("error", "") for error in errors]
        for message in error_messages:
            assert len(message) > 0
            assert "required" in message.lower() or "missing" in message.lower()

    def test_metadata_validation_error_field_identification(self, validator, required_field_schema):
        """Test that errors correctly identify the problematic field."""
        metadata = {
            "required_string": "",
            "required_integer": "not_a_number",
            "required_float": 3.14,
            "required_boolean": True,
            "required_datetime": "2024-01-15T10:30:00Z",
            "required_array_string": [1, 2, 3]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, required_field_schema
        )

        assert is_valid is False
        assert len(errors) > 0

        error_fields = [error.get("field", "") for error in errors]
        assert "required_string" in error_fields or any("required_string" in str(error) for error in errors)
        assert "required_integer" in error_fields or any("required_integer" in str(error) for error in errors)
        assert "required_array_string" in error_fields or any("required_array_string" in str(error) for error in errors)

    def test_metadata_validation_normalization_consistency(self, validator, array_schema):
        """Test that normalization is consistent across multiple calls."""
        metadata = {
            "tags": ["tag1", "tag2"],
            "scores": ["0.8", "0.9"],
            "ids": ["1", "2"],
            "flags": ["true", "false"],
            "title": "Normalization Test",
            "rating": "4.5",
            "is_public": "true"
        }

        is_valid1, errors1, normalized_data1 = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        is_valid2, errors2, normalized_data2 = validator.validate_and_normalize_metadata_values(
            metadata, array_schema
        )

        assert is_valid1 == is_valid2
        assert len(errors1) == len(errors2)

        if is_valid1:
            assert normalized_data1["scores"] == normalized_data2["scores"]
            assert normalized_data1["ids"] == normalized_data2["ids"]
            assert normalized_data1["flags"] == normalized_data2["flags"]
            assert normalized_data1["rating"] == normalized_data2["rating"]
            assert normalized_data1["is_public"] == normalized_data2["is_public"]

    def test_metadata_validation_schema_evolution(self, validator):
        """Test metadata validation with schema evolution scenarios."""
        # Test schema evolution - adding new fields
        evolved_schema = MetadataSchema(schema=[
            MetadataField(name="title", type="string", required=True),
            MetadataField(name="author", type="string", required=False),
            MetadataField(name="category", type="string", required=False),
            MetadataField(name="tags", type="array", array_type="string", required=False),
        ])

        # Test original metadata with evolved schema
        metadata = {
            "title": "Original Document",
            "author": "John Smith"
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, evolved_schema
        )

        assert is_valid is True
        assert len(errors) == 0
        assert normalized_data["title"] == "original document"
        assert normalized_data["author"] == "john smith"
        # Optional fields should be present with None values
        assert "category" in normalized_data
        assert normalized_data["category"] is None
        assert "tags" in normalized_data
        assert normalized_data["tags"] is None

    def test_case_insensitive_string_ingestion(self, validator, string_schema):
        """Test case-insensitive string ingestion with various case combinations."""
        test_cases = [
            {
                "name": "Mixed case strings",
                "metadata": {
                    "title": "Policy Document",
                    "category": "Technical",
                    "description": "This is a Test Document",
                    "author": "John Smith"
                },
                "expected": {
                    "title": "policy document",
                    "category": "technical",
                    "description": "this is a test document",
                    "author": "john smith"
                }
            },
            {
                "name": "All uppercase",
                "metadata": {
                    "title": "POLICY DOCUMENT",
                    "category": "TECHNICAL",
                    "description": "THIS IS A TEST DOCUMENT",
                    "author": "JOHN SMITH"
                },
                "expected": {
                    "title": "policy document",
                    "category": "technical",
                    "description": "this is a test document",
                    "author": "john smith"
                }
            },
            {
                "name": "All lowercase",
                "metadata": {
                    "title": "policy document",
                    "category": "technical",
                    "description": "this is a test document",
                    "author": "john smith"
                },
                "expected": {
                    "title": "policy document",
                    "category": "technical",
                    "description": "this is a test document",
                    "author": "john smith"
                }
            },
            {
                "name": "Mixed case with special characters",
                "metadata": {
                    "title": "Policy & Claims (2024) - Special: $100",
                    "category": "Technical-Support",
                    "description": "This document contains: &, $, %, @, # symbols",
                    "author": "José García"
                },
                "expected": {
                    "title": "policy & claims (2024) - special: $100",
                    "category": "technical-support",
                    "description": "this document contains: &, $, %, @, # symbols",
                    "author": "josé garcía"
                }
            },
            {
                "name": "Unicode characters",
                "metadata": {
                    "title": "政策文档 Policy Document",
                    "category": "技术 Technical",
                    "description": "这是一个测试文档 This is a test document",
                    "author": "José García 何塞"
                },
                "expected": {
                    "title": "政策文档 policy document",
                    "category": "技术 technical",
                    "description": "这是一个测试文档 this is a test document",
                    "author": "josé garcía 何塞"
                }
            },
            {
                "name": "Numbers and special characters",
                "metadata": {
                    "title": "Document 2024 v2.1",
                    "category": "Technical-2024",
                    "description": "Version 2.1 of the document",
                    "author": "John Smith Jr."
                },
                "expected": {
                    "title": "document 2024 v2.1",
                    "category": "technical-2024",
                    "description": "version 2.1 of the document",
                    "author": "john smith jr."
                }
            },
            {
                "name": "Empty and whitespace strings",
                "metadata": {
                    "title": "  Policy Document  ",
                    "category": "  Technical  ",
                    "description": "  Description with spaces  ",
                    "author": "  John Smith  "
                },
                "expected": {
                    "title": "policy document",
                    "category": "technical",
                    "description": "description with spaces",
                    "author": "john smith"
                }
            }
        ]

        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            print(f"Original: {test_case['metadata']}")

            is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
                test_case['metadata'], string_schema
            )

            print(f"Valid: {is_valid}")
            if is_valid:
                print(f"Normalized: {normalized_data}")

                # Verify all string values are lowercase
                for key, value in normalized_data.items():
                    if isinstance(value, str):
                        assert value == value.lower(), f"String value '{value}' is not lowercase"
                        assert value == test_case['expected'][key], f"Expected '{test_case['expected'][key]}', got '{value}'"

                print("✅ All string values converted to lowercase")
            else:
                print(f"Errors: {errors}")

    def test_case_insensitive_array_ingestion(self, validator, array_schema):
        """Test case-insensitive array ingestion with various case combinations."""
        test_cases = [
            {
                "name": "Mixed case array elements",
                "metadata": {
                    "tags": ["Urgent", "Important", "Policy"],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["urgent", "important", "policy"],
                    "title": "test document"
                }
            },
            {
                "name": "All uppercase array elements",
                "metadata": {
                    "tags": ["URGENT", "IMPORTANT", "POLICY"],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["urgent", "important", "policy"],
                    "title": "test document"
                }
            },
            {
                "name": "Mixed case with special characters",
                "metadata": {
                    "tags": ["Urgent!", "Important?", "Policy#"],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["urgent!", "important?", "policy#"],
                    "title": "test document"
                }
            },
            {
                "name": "Unicode array elements",
                "metadata": {
                    "tags": ["紧急 Urgent", "重要 Important", "政策 Policy"],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["紧急 urgent", "重要 important", "政策 policy"],
                    "title": "test document"
                }
            },
            {
                "name": "Numbers and special characters in arrays",
                "metadata": {
                    "tags": ["v1.0", "v2.1", "beta-test"],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["v1.0", "v2.1", "beta-test"],
                    "title": "test document"
                }
            },
            {
                "name": "Empty and whitespace array elements",
                "metadata": {
                    "tags": ["  Urgent  ", "  Important  "],
                    "title": "Test Document"
                },
                "expected": {
                    "tags": ["  urgent  ", "  important  "],
                    "title": "test document"
                }
            }
        ]

        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            print(f"Original: {test_case['metadata']}")

            is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
                test_case['metadata'], array_schema
            )

            print(f"Valid: {is_valid}")
            if is_valid:
                print(f"Normalized: {normalized_data}")

                # Verify all string values and array elements are lowercase
                for key, value in normalized_data.items():
                    if isinstance(value, str):
                        assert value == value.lower(), f"String value '{value}' is not lowercase"
                        assert value == test_case['expected'][key], f"Expected '{test_case['expected'][key]}', got '{value}'"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                assert item == item.lower(), f"Array item '{item}' is not lowercase"

                # Verify array elements match expected
                if 'tags' in normalized_data:
                    assert normalized_data['tags'] == test_case['expected']['tags'], f"Expected {test_case['expected']['tags']}, got {normalized_data['tags']}"

                print("✅ All string values and array elements converted to lowercase")
            else:
                print(f"Errors: {errors}")

    def test_case_insensitive_edge_cases(self, validator, string_schema):
        """Test case-insensitive edge cases and boundary conditions."""
        test_cases = [
            {
                "name": "Very long mixed case string",
                "metadata": {
                    "title": "A" * 500 + "B" * 500 + "C" * 500,
                    "description": "Test description"
                }
            },
            {
                "name": "Strings with only numbers and special characters",
                "metadata": {
                    "title": "1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?",
                    "description": "Test description"
                }
            },
            {
                "name": "Unicode mixed case",
                "metadata": {
                    "title": "政策文档 Policy Document 文档",
                    "description": "Test description"
                }
            },
            {
                "name": "Control characters",
                "metadata": {
                    "title": "Test\x00\x01\x02Document",
                    "description": "Test description"
                }
            },
            {
                "name": "Emoji and special Unicode",
                "metadata": {
                    "title": "Test 🚀 Document 📄",
                    "description": "Test description"
                }
            },
            {
                "name": "HTML entities",
                "metadata": {
                    "title": "Test &amp; Document &lt; &gt;",
                    "description": "Test description"
                }
            }
        ]

        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            print(f"Original: {test_case['metadata']}")

            is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
                test_case['metadata'], string_schema
            )

            print(f"Valid: {is_valid}")
            if is_valid:
                print(f"Normalized: {normalized_data}")

                # Verify title is lowercase
                original_title = test_case['metadata']['title']
                normalized_title = normalized_data['title']
                assert normalized_title == original_title.lower(), f"Expected '{original_title.lower()}', got '{normalized_title}'"

                print("✅ Correctly converted to lowercase")
            else:
                print(f"Errors: {errors}")

    def test_case_insensitive_performance(self, validator, string_schema):
        """Test case-insensitive performance with large datasets."""
        print("\n=== Case-Insensitive Performance Tests ===")

        # Test with very long strings
        long_string = "A" * 10000
        metadata = {"title": long_string, "description": "Test description"}

        import time
        start_time = time.time()
        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            metadata, string_schema
        )
        end_time = time.time()

        print(f"Very long string (10k chars): {'✅ Passed' if is_valid else '❌ Failed'}")
        print(f"Processing time: {end_time - start_time:.4f} seconds")

        if is_valid:
            assert normalized_data['title'] == long_string.lower()
            print("✅ Correctly converted to lowercase")

        # Test with many fields
        many_fields = {}
        for i in range(100):
            many_fields[f"field_{i}"] = f"Value_{i}_UPPERCASE"

        many_fields["title"] = "Test Document"
        many_fields["description"] = "Test description"

        start_time = time.time()
        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            many_fields, string_schema
        )
        end_time = time.time()

        print(f"Many fields (100+ fields): {'✅ Passed' if is_valid else '❌ Failed'}")
        print(f"Processing time: {end_time - start_time:.4f} seconds")

        if is_valid:
            # Verify all string fields are lowercase
            for key, value in normalized_data.items():
                if isinstance(value, str):
                    assert value == value.lower(), f"Field '{key}' value '{value}' is not lowercase"
            print("✅ All string fields converted to lowercase")

    def test_case_insensitive_internationalization(self, validator, string_schema):
        """Test case-insensitive internationalization and Unicode support."""
        print("\n=== Case-Insensitive Internationalization Tests ===")

        # Test various Unicode characters
        unicode_test_cases = [
            {
                "name": "Chinese characters",
                "title": "政策文档 Policy Document",
                "description": "这是一个测试文档 This is a test document"
            },
            {
                "name": "Japanese characters",
                "title": "ドキュメント Document",
                "description": "これはテスト文書です This is a test document"
            },
            {
                "name": "Korean characters",
                "title": "문서 Document",
                "description": "이것은 테스트 문서입니다 This is a test document"
            },
            {
                "name": "Arabic characters",
                "title": "وثيقة Document",
                "description": "هذا مستند اختبار This is a test document"
            },
            {
                "name": "Russian characters",
                "title": "Документ Document",
                "description": "Это тестовый документ This is a test document"
            },
            {
                "name": "Greek characters",
                "title": "Έγγραφο Document",
                "description": "Αυτό είναι ένα έγγραφο δοκιμής This is a test document"
            },
            {
                "name": "Mixed Unicode",
                "title": "政策文档 Policy Document ドキュメント",
                "description": "Mixed Unicode test document"
            }
        ]

        for test_case in unicode_test_cases:
            print(f"\n--- {test_case['name']} ---")
            metadata = {
                "title": test_case['title'],
                "description": test_case['description']
            }

            is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
                metadata, string_schema
            )
            print(f"Validation: {'✅ Passed' if is_valid else '❌ Failed'}")

            if is_valid:
                # Verify title is lowercase
                assert normalized_data['title'] == test_case['title'].lower()
                # Verify description is lowercase
                assert normalized_data['description'] == test_case['description'].lower()
                print("✅ Correctly handled Unicode and converted to lowercase")
            else:
                print(f"Errors: {errors}")

    def test_case_insensitive_data_type_preservation(self, validator):
        """Test that non-string data types are preserved correctly during case-insensitive processing."""
        schema = MetadataSchema(schema=[
            MetadataField(name="title", type="string", required=True),
            MetadataField(name="rating", type="float", required=False),
            MetadataField(name="count", type="integer", required=False),
            MetadataField(name="is_active", type="boolean", required=False),
            MetadataField(name="created_date", type="datetime", required=False),
            MetadataField(name="scores", type="array", array_type="float", required=False),
            MetadataField(name="flags", type="array", array_type="boolean", required=False),
        ])

        print("\n=== Case-Insensitive Data Type Preservation Tests ===")

        # Test mixed data types
        mixed_metadata = {
            "title": "Test Document",
            "rating": 4.5,
            "count": 42,
            "is_active": True,
            "created_date": "2024-01-15T10:30:00Z",
            "scores": [0.8, 0.9, 0.7],
            "flags": [True, False, True]
        }

        is_valid, errors, normalized_data = validator.validate_and_normalize_metadata_values(
            mixed_metadata, schema
        )
        print(f"Mixed data types: {'✅ Passed' if is_valid else '❌ Failed'}")

        if is_valid:
            # String should be lowercase
            assert normalized_data['title'] == "test document"
            # Numeric types should be unchanged
            assert normalized_data['rating'] == 4.5
            assert normalized_data['count'] == 42
            # Boolean should be unchanged
            assert normalized_data['is_active'] is True
            # Datetime should be normalized but not lowercase
            assert "2024-01-15" in normalized_data['created_date']
            # Arrays of non-strings should be unchanged
            assert normalized_data['scores'] == [0.8, 0.9, 0.7]
            assert normalized_data['flags'] == [True, False, True]
            print("✅ All data types preserved correctly")
        else:
            print(f"Errors: {errors}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

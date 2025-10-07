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

import pytest
from nvidia_rag.rag_server.validation import (
    sanitize_boolean,
    sanitize_float,
    normalize_model_info,
    validate_reranker_top_k,
    validate_use_knowledge_base,
    validate_temperature,
    validate_top_p,
    validate_model_info,
    validate_reranker_k,
)


class TestSanitizeBoolean:
    """Test sanitize_boolean function"""

    def test_sanitize_boolean_true_string(self):
        """Test sanitizing boolean True string"""
        result = sanitize_boolean("True", "test_field")
        assert result is True

    def test_sanitize_boolean_false_string(self):
        """Test sanitizing boolean False string"""
        result = sanitize_boolean("False", "test_field")
        assert result is False

    def test_sanitize_boolean_true_boolean(self):
        """Test sanitizing boolean True value"""
        result = sanitize_boolean(True, "test_field")
        assert result is True

    def test_sanitize_boolean_false_boolean(self):
        """Test sanitizing boolean False value"""
        result = sanitize_boolean(False, "test_field")
        assert result is False

    def test_sanitize_boolean_with_whitespace(self):
        """Test sanitizing boolean with whitespace"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean("  True  ", "test_field")

    def test_sanitize_boolean_with_quotes(self):
        """Test sanitizing boolean with quotes"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean('"True"', "test_field")

    def test_sanitize_boolean_invalid_value(self):
        """Test sanitizing boolean with invalid value raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean("invalid", "test_field")

    def test_sanitize_boolean_none_value(self):
        """Test sanitizing boolean with None value raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean(None, "test_field")

    def test_sanitize_boolean_empty_string(self):
        """Test sanitizing boolean with empty string raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean("", "test_field")

    def test_sanitize_boolean_case_sensitive(self):
        """Test sanitizing boolean is case sensitive"""
        with pytest.raises(ValueError, match="test_field must be a boolean value \\(True/False\\)"):
            sanitize_boolean("true", "test_field")

    def test_sanitize_boolean_with_html_tags(self):
        """Test sanitizing boolean with HTML tags gets cleaned"""
        result = sanitize_boolean("<script>True</script>", "test_field")
        assert result is True


class TestSanitizeFloat:
    """Test sanitize_float function"""

    def test_sanitize_float_valid_string(self):
        """Test sanitizing valid float string"""
        result = sanitize_float("3.14", "test_field")
        assert result == 3.14

    def test_sanitize_float_valid_integer_string(self):
        """Test sanitizing valid integer string"""
        result = sanitize_float("42", "test_field")
        assert result == 42.0

    def test_sanitize_float_valid_float(self):
        """Test sanitizing valid float value"""
        result = sanitize_float(3.14, "test_field")
        assert result == 3.14

    def test_sanitize_float_with_whitespace(self):
        """Test sanitizing float with whitespace"""
        result = sanitize_float("  3.14  ", "test_field")
        assert result == 3.14

    def test_sanitize_float_with_quotes(self):
        """Test sanitizing float with quotes"""
        with pytest.raises(ValueError, match="test_field must be a valid number"):
            sanitize_float('"3.14"', "test_field")

    def test_sanitize_float_negative_number(self):
        """Test sanitizing negative float"""
        result = sanitize_float("-3.14", "test_field")
        assert result == -3.14

    def test_sanitize_float_zero(self):
        """Test sanitizing zero"""
        result = sanitize_float("0", "test_field")
        assert result == 0.0

    def test_sanitize_float_invalid_string(self):
        """Test sanitizing invalid string raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a valid number"):
            sanitize_float("invalid", "test_field")

    def test_sanitize_float_empty_string(self):
        """Test sanitizing empty string raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a valid number"):
            sanitize_float("", "test_field")

    def test_sanitize_float_none_value(self):
        """Test sanitizing None value raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a valid number"):
            sanitize_float(None, "test_field")

    def test_sanitize_float_with_html_tags(self):
        """Test sanitizing float with HTML tags gets cleaned"""
        result = sanitize_float("<script>3.14</script>", "test_field")
        assert result == 3.14


class TestNormalizeModelInfo:
    """Test normalize_model_info function"""

    def test_normalize_model_info_valid_string(self):
        """Test normalizing valid string"""
        result = normalize_model_info("test_model", "test_field")
        assert result == "test_model"

    def test_normalize_model_info_with_whitespace(self):
        """Test normalizing string with whitespace"""
        result = normalize_model_info("  test_model  ", "test_field")
        assert result == "test_model"

    def test_normalize_model_info_with_quotes(self):
        """Test normalizing string with quotes"""
        result = normalize_model_info('"test_model"', "test_field")
        assert result == "test_model"

    def test_normalize_model_info_with_both_whitespace_and_quotes(self):
        """Test normalizing string with both whitespace and quotes"""
        result = normalize_model_info('  "test_model"  ', "test_field")
        assert result == "test_model"

    def test_normalize_model_info_non_string(self):
        """Test normalizing non-string value raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a string"):
            normalize_model_info(123, "test_field")

    def test_normalize_model_info_none_value(self):
        """Test normalizing None value raises ValueError"""
        with pytest.raises(ValueError, match="test_field must be a string"):
            normalize_model_info(None, "test_field")

    def test_normalize_model_info_empty_string(self):
        """Test normalizing empty string"""
        result = normalize_model_info("", "test_field")
        assert result == ""


class TestValidateRerankerTopK:
    """Test validate_reranker_top_k function"""

    def test_validate_reranker_top_k_valid_case(self):
        """Test validating reranker_top_k when it's less than vdb_top_k"""
        result = validate_reranker_top_k(5, 10, "reranker_top_k")
        assert result == 5

    def test_validate_reranker_top_k_equal_values(self):
        """Test validating reranker_top_k when it equals vdb_top_k"""
        result = validate_reranker_top_k(10, 10, "reranker_top_k")
        assert result == 10

    def test_validate_reranker_top_k_vdb_top_k_none(self):
        """Test validating reranker_top_k when vdb_top_k is None"""
        result = validate_reranker_top_k(5, None, "reranker_top_k")
        assert result == 5

    def test_validate_reranker_top_k_invalid_case(self):
        """Test validating reranker_top_k when it's greater than vdb_top_k raises ValueError"""
        with pytest.raises(ValueError, match="reranker_top_k\\(15\\) must be less than or equal to vdb_top_k \\(10\\)\\. Please check your settings and try again\\."):
            validate_reranker_top_k(15, 10, "reranker_top_k")

    def test_validate_reranker_top_k_zero_values(self):
        """Test validating reranker_top_k with zero values"""
        result = validate_reranker_top_k(0, 0, "reranker_top_k")
        assert result == 0

    def test_validate_reranker_top_k_negative_values(self):
        """Test validating reranker_top_k with negative values"""
        with pytest.raises(ValueError, match="reranker_top_k\\(-5\\) must be less than or equal to vdb_top_k \\(-10\\)\\. Please check your settings and try again\\."):
            validate_reranker_top_k(-5, -10, "reranker_top_k")


class TestValidateUseKnowledgeBase:
    """Test validate_use_knowledge_base function"""

    def test_validate_use_knowledge_base_true(self):
        """Test validating use_knowledge_base with True"""
        result = validate_use_knowledge_base("True")
        assert result is True

    def test_validate_use_knowledge_base_false(self):
        """Test validating use_knowledge_base with False"""
        result = validate_use_knowledge_base("False")
        assert result is False

    def test_validate_use_knowledge_base_invalid(self):
        """Test validating use_knowledge_base with invalid value"""
        with pytest.raises(ValueError, match="use_knowledge_base must be a boolean value \\(True/False\\)"):
            validate_use_knowledge_base("invalid")


class TestValidateTemperature:
    """Test validate_temperature function"""

    def test_validate_temperature_valid(self):
        """Test validating temperature with valid value"""
        result = validate_temperature("0.7")
        assert result == 0.7

    def test_validate_temperature_invalid(self):
        """Test validating temperature with invalid value"""
        with pytest.raises(ValueError, match="temperature must be a valid number"):
            validate_temperature("invalid")


class TestValidateTopP:
    """Test validate_top_p function"""

    def test_validate_top_p_valid(self):
        """Test validating top_p with valid value"""
        result = validate_top_p("0.9")
        assert result == 0.9

    def test_validate_top_p_invalid(self):
        """Test validating top_p with invalid value"""
        with pytest.raises(ValueError, match="top_p must be a valid number"):
            validate_top_p("invalid")


class TestValidateModelInfo:
    """Test validate_model_info function"""

    def test_validate_model_info_valid(self):
        """Test validating model info with valid string"""
        result = validate_model_info("test_model", "model_name")
        assert result == "test_model"

    def test_validate_model_info_invalid(self):
        """Test validating model info with invalid value"""
        with pytest.raises(ValueError, match="model_name must be a string"):
            validate_model_info(123, "model_name")


class TestValidateRerankerK:
    """Test validate_reranker_k function"""

    def test_validate_reranker_k_valid(self):
        """Test validating reranker_k with valid values"""
        result = validate_reranker_k(5, 10)
        assert result == 5

    def test_validate_reranker_k_invalid(self):
        """Test validating reranker_k with invalid values"""
        with pytest.raises(ValueError, match="reranker_top_k\\(15\\) must be less than or equal to vdb_top_k \\(10\\)\\. Please check your settings and try again\\."):
            validate_reranker_k(15, 10)

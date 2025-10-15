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

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from nvidia_rag.utils.common import (
    combine_dicts,
    filter_documents_by_confidence,
    get_config,
    get_env_variable,
    get_metadata_configuration,
    prepare_custom_metadata_dataframe,
    process_filter_expr,
    sanitize_nim_url,
    utils_cache,
    validate_filter_expr,
)


class TestGetEnvVariable:
    """Test get_env_variable function"""

    def test_get_existing_env_variable(self):
        """Test getting an existing environment variable"""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = get_env_variable("TEST_VAR", "default")
            assert result == "test_value"

    def test_get_missing_env_variable(self):
        """Test getting a missing environment variable returns default"""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_variable("MISSING_VAR", "default_value")
            assert result == "default_value"

    def test_empty_env_variable(self):
        """Test empty environment variable returns default"""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = get_env_variable("EMPTY_VAR", "default_value")
            assert result == "default_value"

    def test_too_long_env_variable(self):
        """Test environment variable longer than 256 chars returns default"""
        long_value = "x" * 300
        with patch.dict(os.environ, {"LONG_VAR": long_value}):
            result = get_env_variable("LONG_VAR", "default_value")
            assert result == "default_value"


class TestUtilsCache:
    """Test utils_cache decorator"""

    def test_utils_cache_with_list_args(self):
        """Test cache decorator with list arguments"""

        @utils_cache
        def test_func(*args, **kwargs):
            return f"args: {args}, kwargs: {kwargs}"

        result = test_func([1, 2, 3], key=[4, 5, 6])
        expected = "args: ((1, 2, 3),), kwargs: {'key': (4, 5, 6)}"
        assert result == expected

    def test_utils_cache_with_dict_args(self):
        """Test cache decorator with dict arguments"""

        @utils_cache
        def test_func(*args, **kwargs):
            return f"args: {args}, kwargs: {kwargs}"

        result = test_func({"a": 1}, key={"b": 2})
        expected = "args: (('a',),), kwargs: {'key': ('b',)}"
        assert result == expected

    def test_utils_cache_with_set_args(self):
        """Test cache decorator with set arguments"""

        @utils_cache
        def test_func(*args, **kwargs):
            return f"args: {args}, kwargs: {kwargs}"

        result = test_func({1, 2, 3}, key={4, 5})
        # Sets are converted to tuples but order may vary
        assert "args: (" in result
        assert "kwargs: {'key': " in result


class TestGetConfig:
    """Test get_config function"""

    @patch("nvidia_rag.utils.common.configuration.AppConfig.from_file")
    def test_get_config_success(self, mock_from_file):
        """Test successful config loading"""
        mock_config = MagicMock()
        mock_from_file.return_value = mock_config

        result = get_config()
        assert result == mock_config
        mock_from_file.assert_called_once()

    @patch("nvidia_rag.utils.common.configuration.AppConfig.from_file")
    def test_get_config_failure(self, mock_from_file):
        """Test config loading failure"""
        mock_from_file.return_value = None

        with pytest.raises(RuntimeError, match="Unable to find configuration"):
            get_config()


class TestCombineDicts:
    """Test combine_dicts function"""

    def test_combine_simple_dicts(self):
        """Test combining simple dictionaries"""
        dict_a = {"a": 1, "b": 2}
        dict_b = {"b": 3, "c": 4}
        result = combine_dicts(dict_a, dict_b)
        expected = {"a": 1, "b": 3, "c": 4}
        assert result == expected

    def test_combine_nested_dicts(self):
        """Test combining nested dictionaries"""
        dict_a = {"nested": {"x": 1, "y": 2}, "other": 5}
        dict_b = {"nested": {"y": 3, "z": 4}}
        result = combine_dicts(dict_a, dict_b)
        expected = {"nested": {"x": 1, "y": 3, "z": 4}, "other": 5}
        assert result == expected

    def test_combine_mixed_types(self):
        """Test combining dicts with mixed value types"""
        dict_a = {"key": {"nested": 1}}
        dict_b = {"key": "string_value"}
        result = combine_dicts(dict_a, dict_b)
        expected = {"key": "string_value"}
        assert result == expected

    def test_combine_empty_dicts(self):
        """Test combining empty dictionaries"""
        dict_a = {}
        dict_b = {"key": "value"}
        result = combine_dicts(dict_a, dict_b)
        expected = {"key": "value"}
        assert result == expected


class TestSanitizeNimUrl:
    """Test sanitize_nim_url function"""

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_url_without_protocol(self, mock_register):
        """Test URL without http/https gets protocol added"""
        result = sanitize_nim_url("example.com", "test_model", "chat")
        assert result == "http://example.com/v1"
        mock_register.assert_not_called()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_url_with_http(self, mock_register):
        """Test URL that already has http protocol"""
        url = "http://example.com/v1"
        result = sanitize_nim_url(url, "test_model", "chat")
        assert result == url
        mock_register.assert_not_called()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_url_with_https(self, mock_register):
        """Test URL that already has https protocol"""
        url = "https://example.com/v1"
        result = sanitize_nim_url(url, "test_model", "chat")
        assert result == url
        mock_register.assert_not_called()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_empty_url(self, mock_register):
        """Test empty URL"""
        result = sanitize_nim_url("", "test_model", "chat")
        assert result == ""
        mock_register.assert_not_called()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_nvidia_url_chat(self, mock_register):
        """Test NVIDIA URL with chat model type"""
        url = "https://integrate.api.nvidia.com/v1/chat"
        result = sanitize_nim_url(url, "test_model", "chat")
        assert result == url
        mock_register.assert_called_once()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_nvidia_url_embedding(self, mock_register):
        """Test NVIDIA URL with embedding model type"""
        url = "https://ai.api.nvidia.com/v1/embeddings"
        result = sanitize_nim_url(url, "test_model", "embedding")
        assert result == url
        mock_register.assert_called_once()

    @patch("nvidia_rag.utils.common.register_model")
    def test_sanitize_nvidia_url_ranking(self, mock_register):
        """Test NVIDIA URL with ranking model type"""
        url = "https://api.nvcf.nvidia.com/v1/ranking"
        result = sanitize_nim_url(url, "test_model", "ranking")
        assert result == url
        mock_register.assert_called_once()


class TestGetMetadataConfiguration:
    """Test get_metadata_configuration function"""

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    @patch("os.makedirs")
    def test_get_metadata_config_none_metadata(
        self, mock_makedirs, mock_prepare, mock_get_config
    ):
        """Test with None custom_metadata - should still create CSV with filename"""
        mock_config = MagicMock()
        mock_config.temp_dir = "/tmp"
        mock_get_config.return_value = mock_config
        mock_prepare.return_value = ("source", ["filename"])

        result = get_metadata_configuration("test_collection", None, ["file1.txt"])

        # Should now create CSV and return metadata configuration, not (None, None, None)
        assert result[0] is not None  # csv_file_path should be created
        assert result[1] == "source"  # meta_source_field
        assert result[2] == ["filename"]  # meta_fields

        # Verify prepare_custom_metadata_dataframe was called with empty list
        mock_prepare.assert_called_once()
        call_args = mock_prepare.call_args
        assert call_args[1]["custom_metadata"] == []  # None should be converted to []

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    @patch("os.makedirs")
    def test_get_metadata_config_empty_metadata(
        self, mock_makedirs, mock_prepare, mock_get_config
    ):
        """Test with empty custom_metadata - should still create CSV with filename"""
        mock_config = MagicMock()
        mock_config.temp_dir = "/tmp"
        mock_get_config.return_value = mock_config
        mock_prepare.return_value = ("source", ["filename"])

        result = get_metadata_configuration("test_collection", [], ["file1.txt"])

        # Should now create CSV and return metadata configuration, not (None, None, None)
        assert result[0] is not None  # csv_file_path should be created
        assert result[1] == "source"  # meta_source_field
        assert result[2] == ["filename"]  # meta_fields

        # Verify prepare_custom_metadata_dataframe was called with the empty list
        mock_prepare.assert_called_once()
        call_args = mock_prepare.call_args
        assert call_args[1]["custom_metadata"] == []

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    @patch("os.makedirs")
    def test_get_metadata_config_with_metadata(
        self, mock_makedirs, mock_prepare, mock_get_config
    ):
        """Test with custom metadata"""
        mock_config = MagicMock()
        mock_config.temp_dir = "/tmp"
        mock_get_config.return_value = mock_config
        mock_prepare.return_value = ("source", ["field1", "field2"])

        custom_metadata = [{"filename": "file1.txt", "metadata": {"key": "value"}}]
        result = get_metadata_configuration(
            "test_collection", custom_metadata, ["file1.txt"]
        )

        assert result[1] == "source"
        assert result[2] == ["field1", "field2"]
        mock_makedirs.assert_called_once()


class TestPrepareCustomMetadataDataframe:
    """Test prepare_custom_metadata_dataframe function"""

    @patch("pandas.DataFrame.to_csv")
    def test_prepare_custom_metadata_dataframe(self, mock_to_csv):
        """Test preparing custom metadata dataframe"""
        all_file_paths = ["path/to/file1.txt", "path/to/file2.txt"]
        custom_metadata = [
            {
                "filename": "file1.txt",
                "metadata": {"category": "doc", "priority": "high"},
            },
            {"filename": "file2.txt", "metadata": {"category": "image"}},
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            csv_file_path = tmp_file.name

        try:
            result = prepare_custom_metadata_dataframe(
                all_file_paths, csv_file_path, custom_metadata
            )
            source_field, metadata_fields = result

            assert source_field == "source"
            assert "filename" in metadata_fields
            assert "category" in metadata_fields
            assert "priority" in metadata_fields
            mock_to_csv.assert_called_once()
        finally:
            os.unlink(csv_file_path)


class TestValidateFilterExpr:
    """Test validate_filter_expr function"""

    @patch("nvidia_rag.utils.common.get_config")
    def test_validate_filter_elasticsearch_valid(self, mock_get_config):
        """Test Elasticsearch filter validation with valid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = [{"term": {"category": "doc"}}]
        result = validate_filter_expr(filter_expr, ["collection1"], {})

        assert result["status"] is True
        assert result["validated_collections"] == ["collection1"]

    @patch("nvidia_rag.utils.common.get_config")
    def test_validate_filter_elasticsearch_invalid(self, mock_get_config):
        """Test Elasticsearch filter validation with invalid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = ["not_a_dict"]
        result = validate_filter_expr(filter_expr, ["collection1"], {})

        assert result["status"] is False

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_valid(self, mock_executor, mock_get_config):
        """Test Milvus filter validation with valid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = False
        mock_get_config.return_value = mock_config

        # Mock the validation result
        mock_result = {"status": True}

        # Mock the metadata validation components
        with patch("nvidia_rag.utils.common.MetadataField"), patch(
            "nvidia_rag.utils.common.MetadataSchema"
        ), patch("nvidia_rag.utils.common.FilterExpressionParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.validate_filter_expression.return_value = mock_result
            mock_parser_class.return_value = mock_parser

            # Mock executor.map to return validation results
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.map.return_value = [
                {"collection": "test", "valid": True, "error": None}
            ]

            metadata_schemas = {"test": [{"name": "field1", "type": "string"}]}
            result = validate_filter_expr(
                "category == 'doc'", ["test"], metadata_schemas
            )

            assert result["status"] is True

    @patch("nvidia_rag.utils.common.get_config")
    def test_validate_filter_elasticsearch_string_input(self, mock_get_config):
        """Test Elasticsearch filter validation with string input (should fail)"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = "string_filter"
        result = validate_filter_expr(filter_expr, ["collection1"], {})

        assert result["status"] is False
        assert "expects list of dictionaries" in result["error_message"]

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_partial_filtering_allowed(
        self, mock_executor, mock_get_config
    ):
        """Test Milvus filter validation with partial filtering allowed"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = True
        mock_get_config.return_value = mock_config

        # Mock executor.map to return mixed validation results
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [
            {"collection": "test1", "valid": True, "error": None},
            {"collection": "test2", "valid": False, "error": "Invalid field"},
        ]

        metadata_schemas = {
            "test1": [{"name": "field1", "type": "string"}],
            "test2": [{"name": "field2", "type": "string"}],
        }
        result = validate_filter_expr(
            "category == 'doc'", ["test1", "test2"], metadata_schemas
        )

        assert result["status"] is True
        assert result["validated_collections"] == ["test1"]

    @patch("nvidia_rag.utils.common.get_config")
    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_no_valid_collections(
        self, mock_executor, mock_get_config
    ):
        """Test Milvus filter validation when no collections are valid"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = True
        mock_get_config.return_value = mock_config

        # Mock executor.map to return all invalid results
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [
            {"collection": "test1", "valid": False, "error": "Invalid field"}
        ]

        metadata_schemas = {"test1": [{"name": "field1", "type": "string"}]}
        result = validate_filter_expr(
            "invalid_field == 'doc'", ["test1"], metadata_schemas
        )

        assert result["status"] is False
        assert "No collections support the filter expression" in result["error_message"]

    @patch("nvidia_rag.utils.common.get_config")
    def test_validate_filter_milvus_list_input(self, mock_get_config):
        """Test Milvus filter validation with list input (should fail)"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        filter_expr = [{"term": {"category": "doc"}}]
        result = validate_filter_expr(filter_expr, ["collection1"], {})

        assert result["status"] is False
        assert "expects string filter expression" in result["error_message"]

    @patch("nvidia_rag.utils.common.get_config")
    def test_validate_filter_unsupported_store(self, mock_get_config):
        """Test validation with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"
        mock_get_config.return_value = mock_config

        result = validate_filter_expr("test", ["collection1"], {})
        assert result["status"] is False
        assert "Unsupported vector store" in result["error_message"]


class TestProcessFilterExpr:
    """Test process_filter_expr function"""

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_empty_expr(self, mock_get_config):
        """Test processing empty filter expression"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        result = process_filter_expr("", "test_collection")
        assert result == ""

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_elasticsearch(self, mock_get_config):
        """Test processing Elasticsearch filter"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = [{"term": {"category": "doc"}}]
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == filter_expr

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_elasticsearch_invalid(self, mock_get_config):
        """Test processing invalid Elasticsearch filter"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = ["not_a_dict"]
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == []

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_no_schema(self, mock_get_config):
        """Test processing Milvus filter without metadata schema"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        filter_expr = "category == 'doc'"
        result = process_filter_expr(filter_expr, "test_collection", None)
        assert result == filter_expr  # Returns original when no schema

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_with_schema(self, mock_get_config):
        """Test processing Milvus filter with metadata schema"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        with patch("nvidia_rag.utils.common.MetadataField"), patch(
            "nvidia_rag.utils.common.MetadataSchema"
        ), patch("nvidia_rag.utils.common.FilterExpressionParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.process_filter_expression.return_value = {
                "status": True,
                "processed_expression": "processed_filter",
            }
            mock_parser_class.return_value = mock_parser

            metadata_schema_data = [{"name": "field1", "type": "string"}]
            result = process_filter_expr(
                "category == 'doc'", "test_collection", metadata_schema_data
            )

            assert result == "processed_filter"

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_failure(self, mock_get_config):
        """Test processing Milvus filter with validation failure"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        with patch("nvidia_rag.utils.common.MetadataField"), patch(
            "nvidia_rag.utils.common.MetadataSchema"
        ), patch("nvidia_rag.utils.common.FilterExpressionParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.process_filter_expression.return_value = {
                "status": False,
                "error_message": "Invalid filter",
            }
            mock_parser_class.return_value = mock_parser

            metadata_schema_data = [{"name": "field1", "type": "string"}]

            with pytest.raises(ValueError, match="Invalid filter"):
                process_filter_expr(
                    "invalid_filter", "test_collection", metadata_schema_data
                )

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_generated_failure(self, mock_get_config):
        """Test processing Milvus generated filter with validation failure"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        with patch("nvidia_rag.utils.common.MetadataField"), patch(
            "nvidia_rag.utils.common.MetadataSchema"
        ), patch("nvidia_rag.utils.common.FilterExpressionParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.process_filter_expression.return_value = {
                "status": False,
                "error_message": "Invalid filter",
            }
            mock_parser_class.return_value = mock_parser

            metadata_schema_data = [{"name": "field1", "type": "string"}]
            result = process_filter_expr(
                "invalid_filter",
                "test_collection",
                metadata_schema_data,
                is_generated_filter=True,
            )

            assert result == ""  # Returns empty string for generated filters

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_schema_conversion_error(self, mock_get_config):
        """Test processing Milvus filter with schema conversion error"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        with patch(
            "nvidia_rag.utils.common.MetadataField",
            side_effect=Exception("Schema error"),
        ):
            metadata_schema_data = [{"name": "field1", "type": "string"}]
            result = process_filter_expr(
                "category == 'doc'", "test_collection", metadata_schema_data
            )

            assert result == "category == 'doc'"  # Returns original on error

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_milvus_wrong_type(self, mock_get_config):
        """Test processing Milvus filter with wrong input type"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        filter_expr = [{"term": {"category": "doc"}}]  # List instead of string
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == ""

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_elasticsearch_string_input(self, mock_get_config):
        """Test processing Elasticsearch filter with string input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = "string_filter"
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == []

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_elasticsearch_wrong_type(self, mock_get_config):
        """Test processing Elasticsearch filter with wrong type"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        filter_expr = 123  # Wrong type
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == []

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_empty_milvus(self, mock_get_config):
        """Test processing empty filter for Milvus"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_get_config.return_value = mock_config

        result = process_filter_expr(None, "test_collection")
        assert result == ""

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_empty_elasticsearch(self, mock_get_config):
        """Test processing empty filter for Elasticsearch"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"
        mock_get_config.return_value = mock_config

        result = process_filter_expr(None, "test_collection")
        assert result == []

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_unsupported_store(self, mock_get_config):
        """Test processing filter with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"
        mock_get_config.return_value = mock_config

        filter_expr = "category == 'doc'"
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == filter_expr

    @patch("nvidia_rag.utils.common.get_config")
    def test_process_filter_unsupported_store_list(self, mock_get_config):
        """Test processing list filter with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"
        mock_get_config.return_value = mock_config

        filter_expr = [{"term": {"category": "doc"}}]
        result = process_filter_expr(filter_expr, "test_collection")
        assert result == []


class TestFilterDocumentsByConfidence:
    """Test filter_documents_by_confidence function"""

    def setup_method(self):
        """Set up test fixtures"""
        from langchain_core.documents import Document

        # Create test documents with different relevance scores
        self.documents = [
            Document(
                page_content="High relevance document",
                metadata={"relevance_score": 0.95, "source": "doc1"},
            ),
            Document(
                page_content="Medium relevance document",
                metadata={"relevance_score": 0.75, "source": "doc2"},
            ),
            Document(
                page_content="Low relevance document",
                metadata={"relevance_score": 0.45, "source": "doc3"},
            ),
            Document(
                page_content="Very low relevance document",
                metadata={"relevance_score": 0.15, "source": "doc4"},
            ),
            Document(
                page_content="Document without relevance score",
                metadata={"source": "doc5"},
            ),
        ]

    def test_filter_with_zero_threshold(self):
        """Test filtering with zero threshold (no filtering)"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.0
        )

        # Should return all documents when threshold is 0.0
        assert len(result) == 5
        assert result == self.documents

    def test_filter_with_low_threshold(self):
        """Test filtering with low threshold"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.3
        )

        # Should include documents with scores >= 0.3
        assert len(result) == 3
        assert result[0].metadata["source"] == "doc1"  # 0.95
        assert result[1].metadata["source"] == "doc2"  # 0.75
        assert result[2].metadata["source"] == "doc3"  # 0.45

    def test_filter_with_medium_threshold(self):
        """Test filtering with medium threshold"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.5
        )

        # Should include documents with scores >= 0.5
        assert len(result) == 2
        assert result[0].metadata["source"] == "doc1"  # 0.95
        assert result[1].metadata["source"] == "doc2"  # 0.75

    def test_filter_with_high_threshold(self):
        """Test filtering with high threshold"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.8
        )

        # Should include only documents with scores >= 0.8
        assert len(result) == 1
        assert result[0].metadata["source"] == "doc1"  # 0.95

    def test_filter_with_very_high_threshold(self):
        """Test filtering with very high threshold"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.99
        )

        # Should return empty list when no documents meet threshold
        assert len(result) == 0

    def test_filter_documents_without_relevance_score(self):
        """Test filtering documents that don't have relevance_score metadata"""
        # Documents without relevance_score should be treated as having score 0.0
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.1
        )

        # Should exclude document without relevance_score (treated as 0.0)
        assert len(result) == 4
        assert all("relevance_score" in doc.metadata for doc in result)

    def test_filter_empty_document_list(self):
        """Test filtering empty document list"""
        result = filter_documents_by_confidence([], confidence_threshold=0.5)

        assert len(result) == 0
        assert result == []

    def test_filter_single_document(self):
        """Test filtering single document"""
        single_doc = [self.documents[0]]  # Document with 0.95 score

        result = filter_documents_by_confidence(single_doc, confidence_threshold=0.9)
        assert len(result) == 1
        assert result[0].metadata["source"] == "doc1"

        result = filter_documents_by_confidence(single_doc, confidence_threshold=0.99)
        assert len(result) == 0

    def test_filter_exact_threshold_match(self):
        """Test filtering with exact threshold match"""
        # Test with threshold exactly matching a document's score
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=0.75
        )

        # Should include documents with scores >= 0.75 (inclusive)
        assert len(result) == 2
        assert result[0].metadata["source"] == "doc1"  # 0.95
        assert result[1].metadata["source"] == "doc2"  # 0.75

    def test_filter_preserves_original_documents(self):
        """Test that filtering doesn't modify original documents"""
        original_docs = self.documents.copy()
        filter_documents_by_confidence(self.documents, confidence_threshold=0.5)

        # Original documents should remain unchanged
        assert self.documents == original_docs
        assert len(self.documents) == 5  # Original count preserved

    def test_filter_with_negative_threshold(self):
        """Test filtering with negative threshold (should include all documents)"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=-0.1
        )

        # Should include all documents when threshold is negative
        assert len(result) == 5
        assert result == self.documents

    def test_filter_with_threshold_greater_than_one(self):
        """Test filtering with threshold greater than 1.0"""
        result = filter_documents_by_confidence(
            self.documents, confidence_threshold=1.0
        )

        # Should return empty list when threshold > 1.0
        assert len(result) == 0

    @patch("nvidia_rag.utils.common.logger")
    def test_filter_logging_behavior(self, mock_logger):
        """Test that filtering logs appropriate information"""
        filter_documents_by_confidence(self.documents, confidence_threshold=0.5)

        # Verify logging was called with correct information
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Confidence threshold filtering: 5 -> 2 documents" in log_message
        assert "threshold: 0.5" in log_message

    def test_filter_documents_with_non_numeric_relevance_score(self):
        """Test filtering documents with non-numeric relevance_score"""
        from langchain_core.documents import Document

        # Create documents with non-numeric relevance scores
        test_docs = [
            Document(
                page_content="Document with string score",
                metadata={"relevance_score": "0.8", "source": "doc1"},
            ),
            Document(
                page_content="Document with None score",
                metadata={"relevance_score": None, "source": "doc2"},
            ),
            Document(
                page_content="Document with valid score",
                metadata={"relevance_score": 0.6, "source": "doc3"},
            ),
            Document(
                page_content="Document with invalid string score",
                metadata={"relevance_score": "invalid", "source": "doc4"},
            ),
        ]

        # Should handle gracefully: valid numeric strings are converted, invalid values become 0.0
        result = filter_documents_by_confidence(test_docs, confidence_threshold=0.5)

        # Should include documents with valid numeric scores >= 0.5 (including converted strings)
        assert len(result) == 2
        assert result[0].metadata["source"] == "doc1"  # "0.8" converted to 0.8
        assert result[1].metadata["source"] == "doc3"  # 0.6

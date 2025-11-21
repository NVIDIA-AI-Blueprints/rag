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

import importlib
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from nvidia_rag.utils.common import (
    combine_dicts,
    create_catalog_metadata,
    create_document_metadata,
    derive_boolean_flags,
    filter_documents_by_confidence,
    get_current_timestamp,
    get_metadata_configuration,
    perform_document_info_aggregation,
    prepare_custom_metadata_dataframe,
    process_filter_expr,
    sanitize_nim_url,
    utils_cache,
    validate_filter_expr,
)


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
        mock_register.assert_not_called()

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

    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    def test_get_metadata_config_none_metadata(self, mock_prepare, tmp_path):
        """Test with None custom_metadata - should still create CSV with filename"""
        mock_config = MagicMock()
        mock_config.temp_dir = str(tmp_path)
        mock_prepare.return_value = ("source", ["filename"])

        result = get_metadata_configuration(
            "test_collection", None, ["file1.txt"], config=mock_config
        )

        # Should now create CSV and return metadata configuration
        assert result[0] is not None  # csv_file_path should be created
        assert result[1] == "source"  # meta_source_field
        assert result[2] == ["filename"]  # meta_fields

        # Verify prepare_custom_metadata_dataframe was called with empty list
        mock_prepare.assert_called_once()
        call_args = mock_prepare.call_args
        assert call_args[1]["custom_metadata"] == []  # None should be converted to []

    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    def test_get_metadata_config_empty_metadata(self, mock_prepare, tmp_path):
        """Test with empty custom_metadata - should still create CSV with filename"""
        mock_config = MagicMock()
        mock_config.temp_dir = str(tmp_path)
        mock_prepare.return_value = ("source", ["filename"])

        result = get_metadata_configuration(
            "test_collection", [], ["file1.txt"], config=mock_config
        )

        # Should now create CSV and return metadata configuration
        assert result[0] is not None
        assert result[1] == "source"
        assert result[2] == ["filename"]

        # Verify prepare_custom_metadata_dataframe was called with the empty list
        mock_prepare.assert_called_once()
        call_args = mock_prepare.call_args
        assert call_args[1]["custom_metadata"] == []

    @patch("nvidia_rag.utils.common.prepare_custom_metadata_dataframe")
    def test_get_metadata_config_with_metadata(self, mock_prepare, tmp_path):
        """Test with custom metadata"""
        mock_config = MagicMock()
        mock_config.temp_dir = str(tmp_path)
        mock_prepare.return_value = ("source", ["field1", "field2"])

        custom_metadata = [{"filename": "file1.txt", "metadata": {"key": "value"}}]
        result = get_metadata_configuration(
            "test_collection", custom_metadata, ["file1.txt"], config=mock_config
        )

        assert result[1] == "source"
        assert result[2] == ["field1", "field2"]
        # Directory should be created in tmp_path (auto-cleaned by pytest)
        assert tmp_path.exists()


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

    @patch("pandas.DataFrame.to_csv")
    def test_prepare_custom_metadata_with_user_defined_fields(self, mock_to_csv):
        """Test that user_defined fields are included"""
        all_file_paths = ["file1.txt"]
        custom_metadata = [
            {
                "filename": "file1.txt",
                "metadata": {"custom_field": "value"},
            }
        ]

        # Schema with user_defined=True
        metadata_schema = [
            {"name": "custom_field", "type": "string", "user_defined": True},
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            csv_file_path = tmp_file.name

        try:
            result = prepare_custom_metadata_dataframe(
                all_file_paths, csv_file_path, custom_metadata, metadata_schema
            )
            source_field, metadata_fields = result

            assert source_field == "source"
            assert "custom_field" in metadata_fields
            mock_to_csv.assert_called_once()
        finally:
            os.unlink(csv_file_path)

    @patch("pandas.DataFrame.to_csv")
    def test_prepare_custom_metadata_skips_auto_extracted_fields(self, mock_to_csv):
        """Test that auto-extracted fields are excluded"""
        all_file_paths = ["file1.txt"]
        custom_metadata = [
            {
                "filename": "file1.txt",
                "metadata": {"auto_field": "value"},
            }
        ]

        # Schema with user_defined=False (auto-extracted)
        metadata_schema = [
            {"name": "auto_field", "type": "string", "user_defined": False},
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            csv_file_path = tmp_file.name

        try:
            result = prepare_custom_metadata_dataframe(
                all_file_paths, csv_file_path, custom_metadata, metadata_schema
            )
            source_field, metadata_fields = result

            assert source_field == "source"
            # auto_field should not be included
            assert "auto_field" not in metadata_fields
            mock_to_csv.assert_called_once()
        finally:
            os.unlink(csv_file_path)

    @patch("pandas.DataFrame.to_csv")
    def test_prepare_custom_metadata_with_mixed_system_fields(self, mock_to_csv):
        """Test that system fields are excluded"""
        all_file_paths = ["file1.txt"]
        custom_metadata = [
            {
                "filename": "file1.txt",
                "metadata": {
                    "custom_field": "value",
                    "chunk_id": "system_generated",  # System field
                },
            }
        ]

        # Schema including a system field
        metadata_schema = [
            {"name": "custom_field", "type": "string", "user_defined": True},
            {"name": "chunk_id", "type": "string", "user_defined": False},
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            csv_file_path = tmp_file.name

        try:
            result = prepare_custom_metadata_dataframe(
                all_file_paths, csv_file_path, custom_metadata, metadata_schema
            )
            source_field, metadata_fields = result

            assert source_field == "source"
            assert "custom_field" in metadata_fields
            assert "chunk_id" not in metadata_fields
            mock_to_csv.assert_called_once()
        finally:
            os.unlink(csv_file_path)

    @patch("pandas.DataFrame.to_csv")
    def test_prepare_custom_metadata_defaults_to_user_defined(self, mock_to_csv):
        """Test that fields default to user_defined=True when not specified"""
        all_file_paths = ["file1.txt"]
        custom_metadata = [
            {
                "filename": "file1.txt",
                "metadata": {"custom_field": "value"},
            }
        ]

        # Schema without user_defined flag (should default to True)
        metadata_schema = [
            {"name": "custom_field", "type": "string"},
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            csv_file_path = tmp_file.name

        try:
            result = prepare_custom_metadata_dataframe(
                all_file_paths, csv_file_path, custom_metadata, metadata_schema
            )
            source_field, metadata_fields = result

            assert source_field == "source"
            # Field should be included (defaults to user_defined=True)
            assert "custom_field" in metadata_fields
            mock_to_csv.assert_called_once()
        finally:
            os.unlink(csv_file_path)


class TestValidateFilterExpr:
    """Test validate_filter_expr function"""

    def test_validate_filter_elasticsearch_valid(self):
        """Test Elasticsearch filter validation with valid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = [{"term": {"category": "doc"}}]
        result = validate_filter_expr(
            filter_expr, ["collection1"], {}, config=mock_config
        )

        assert result["status"] is True
        assert result["validated_collections"] == ["collection1"]

    def test_validate_filter_elasticsearch_invalid(self):
        """Test Elasticsearch filter validation with invalid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = ["not_a_dict"]
        result = validate_filter_expr(
            filter_expr, ["collection1"], {}, config=mock_config
        )

        assert result["status"] is False

    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_valid(self, mock_executor):
        """Test Milvus filter validation with valid input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = False

        # Mock the validation result
        mock_result = {"status": True}

        # Mock the metadata validation components
        with (
            patch("nvidia_rag.utils.common.MetadataField"),
            patch("nvidia_rag.utils.common.MetadataSchema"),
            patch(
                "nvidia_rag.utils.common.FilterExpressionParser"
            ) as mock_parser_class,
        ):
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
                "category == 'doc'", ["test"], metadata_schemas, config=mock_config
            )

            assert result["status"] is True

    def test_validate_filter_elasticsearch_string_input(self):
        """Test Elasticsearch filter validation with string input (should fail)"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = "string_filter"
        result = validate_filter_expr(
            filter_expr, ["collection1"], {}, config=mock_config
        )

        assert result["status"] is False
        assert "expects list of dictionaries" in result["error_message"]

    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_partial_filtering_allowed(self, mock_executor):
        """Test Milvus filter validation with partial filtering allowed"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = True

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
            "category == 'doc'",
            ["test1", "test2"],
            metadata_schemas,
            config=mock_config,
        )

        assert result["status"] is True
        assert result["validated_collections"] == ["test1"]

    @patch("nvidia_rag.utils.common.ThreadPoolExecutor")
    def test_validate_filter_milvus_no_valid_collections(self, mock_executor):
        """Test Milvus filter validation when no collections are valid"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"
        mock_config.metadata.allow_partial_filtering = True

        # Mock executor.map to return all invalid results
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [
            {"collection": "test1", "valid": False, "error": "Invalid field"}
        ]

        metadata_schemas = {"test1": [{"name": "field1", "type": "string"}]}
        result = validate_filter_expr(
            "invalid_field == 'doc'", ["test1"], metadata_schemas, config=mock_config
        )

        assert result["status"] is False
        assert "No collections support the filter expression" in result["error_message"]

    def test_validate_filter_milvus_list_input(self):
        """Test Milvus filter validation with list input (should fail)"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        filter_expr = [{"term": {"category": "doc"}}]
        result = validate_filter_expr(
            filter_expr, ["collection1"], {}, config=mock_config
        )

        assert result["status"] is False
        assert "expects string filter expression" in result["error_message"]

    def test_validate_filter_unsupported_store(self):
        """Test validation with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"

        result = validate_filter_expr("test", ["collection1"], {}, config=mock_config)
        assert result["status"] is False
        assert "Unsupported vector store" in result["error_message"]


class TestProcessFilterExpr:
    """Test process_filter_expr function"""

    def test_process_filter_empty_expr(self):
        """Test processing empty filter expression"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        result = process_filter_expr("", "test_collection", config=mock_config)
        assert result == ""

    def test_process_filter_elasticsearch(self):
        """Test processing Elasticsearch filter"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = [{"term": {"category": "doc"}}]
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == filter_expr

    def test_process_filter_elasticsearch_invalid(self):
        """Test processing invalid Elasticsearch filter"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = ["not_a_dict"]
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == []

    def test_process_filter_milvus_no_schema(self):
        """Test processing Milvus filter without metadata schema"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        filter_expr = "category == 'doc'"
        result = process_filter_expr(
            filter_expr, "test_collection", None, config=mock_config
        )
        assert result == filter_expr  # Returns original when no schema

    def test_process_filter_milvus_with_schema(self):
        """Test processing Milvus filter with metadata schema"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        with (
            patch("nvidia_rag.utils.common.MetadataField"),
            patch("nvidia_rag.utils.common.MetadataSchema"),
            patch(
                "nvidia_rag.utils.common.FilterExpressionParser"
            ) as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser.process_filter_expression.return_value = {
                "status": True,
                "processed_expression": "processed_filter",
            }
            mock_parser_class.return_value = mock_parser

            metadata_schema_data = [{"name": "field1", "type": "string"}]
            result = process_filter_expr(
                "category == 'doc'",
                "test_collection",
                metadata_schema_data,
                config=mock_config,
            )

            assert result == "processed_filter"

    def test_process_filter_milvus_failure(self):
        """Test processing Milvus filter with validation failure"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        with (
            patch("nvidia_rag.utils.common.MetadataField"),
            patch("nvidia_rag.utils.common.MetadataSchema"),
            patch(
                "nvidia_rag.utils.common.FilterExpressionParser"
            ) as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser.process_filter_expression.return_value = {
                "status": False,
                "error_message": "Invalid filter",
            }
            mock_parser_class.return_value = mock_parser

            metadata_schema_data = [{"name": "field1", "type": "string"}]

            with pytest.raises(ValueError, match="Invalid filter"):
                process_filter_expr(
                    "invalid_filter",
                    "test_collection",
                    metadata_schema_data,
                    config=mock_config,
                )

    def test_process_filter_milvus_generated_failure(self):
        """Test processing Milvus generated filter with validation failure"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        with (
            patch("nvidia_rag.utils.common.MetadataField"),
            patch("nvidia_rag.utils.common.MetadataSchema"),
            patch(
                "nvidia_rag.utils.common.FilterExpressionParser"
            ) as mock_parser_class,
        ):
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
                config=mock_config,
            )

            assert result == ""  # Returns empty string for generated filters

    def test_process_filter_milvus_schema_conversion_error(self):
        """Test processing Milvus filter with schema conversion error"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        with patch(
            "nvidia_rag.utils.common.MetadataField",
            side_effect=Exception("Schema error"),
        ):
            metadata_schema_data = [{"name": "field1", "type": "string"}]
            result = process_filter_expr(
                "category == 'doc'",
                "test_collection",
                metadata_schema_data,
                config=mock_config,
            )

            assert result == "category == 'doc'"  # Returns original on error

    def test_process_filter_milvus_wrong_type(self):
        """Test processing Milvus filter with wrong input type"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        filter_expr = [{"term": {"category": "doc"}}]  # List instead of string
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == ""

    def test_process_filter_elasticsearch_string_input(self):
        """Test processing Elasticsearch filter with string input"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = "string_filter"
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == []

    def test_process_filter_elasticsearch_wrong_type(self):
        """Test processing Elasticsearch filter with wrong type"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        filter_expr = 123  # Wrong type
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == []

    def test_process_filter_empty_milvus(self):
        """Test processing empty filter for Milvus"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "milvus"

        result = process_filter_expr(None, "test_collection", config=mock_config)
        assert result == ""

    def test_process_filter_empty_elasticsearch(self):
        """Test processing empty filter for Elasticsearch"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "elasticsearch"

        result = process_filter_expr(None, "test_collection", config=mock_config)
        assert result == []

    def test_process_filter_unsupported_store(self):
        """Test processing filter with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"

        filter_expr = "category == 'doc'"
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == filter_expr

    def test_process_filter_unsupported_store_list(self):
        """Test processing list filter with unsupported vector store"""
        mock_config = MagicMock()
        mock_config.vector_store.name = "unsupported"

        filter_expr = [{"term": {"category": "doc"}}]
        result = process_filter_expr(filter_expr, "test_collection", config=mock_config)
        assert result == []


class TestDataCatalogUtilities:
    """Test data catalog utility functions"""

    def test_get_current_timestamp_format(self):
        """Test that timestamp is in ISO 8601 format"""
        timestamp = get_current_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp
        assert timestamp.endswith(("+00:00", "Z"))

    def test_derive_boolean_flags_all_present(self):
        """Test deriving boolean flags when all document types are present"""
        doc_type_counts = {"table": 5, "chart": 3, "image": 10, "text": 100}
        result = derive_boolean_flags(doc_type_counts)

        assert result == {
            "has_tables": True,
            "has_charts": True,
            "has_images": True,
        }

    def test_derive_boolean_flags_none_present(self):
        """Test deriving boolean flags when no special types are present"""
        doc_type_counts = {"text": 100}
        result = derive_boolean_flags(doc_type_counts)

        assert result == {
            "has_tables": False,
            "has_charts": False,
            "has_images": False,
        }

    def test_derive_boolean_flags_partial(self):
        """Test deriving boolean flags with partial document types"""
        doc_type_counts = {"table": 2, "text": 50}
        result = derive_boolean_flags(doc_type_counts)

        assert result == {
            "has_tables": True,
            "has_charts": False,
            "has_images": False,
        }

    def test_create_catalog_metadata_defaults(self):
        """Test creating catalog metadata with default values"""
        metadata = create_catalog_metadata()

        assert metadata["description"] == ""
        assert metadata["tags"] == []
        assert metadata["owner"] == ""
        assert metadata["created_by"] == ""
        assert metadata["business_domain"] == ""
        assert metadata["status"] == "Active"
        assert "date_created" in metadata
        assert "last_updated" in metadata
        assert metadata["date_created"] == metadata["last_updated"]

    def test_create_catalog_metadata_custom_values(self):
        """Test creating catalog metadata with custom values"""
        metadata = create_catalog_metadata(
            description="Test collection",
            tags=["prod", "finance"],
            owner="Finance Team",
            created_by="user@example.com",
            business_domain="Finance",
            status="Archived",
        )

        assert metadata["description"] == "Test collection"
        assert metadata["tags"] == ["prod", "finance"]
        assert metadata["owner"] == "Finance Team"
        assert metadata["created_by"] == "user@example.com"
        assert metadata["business_domain"] == "Finance"
        assert metadata["status"] == "Archived"

    def test_create_document_metadata(self):
        """Test creating document metadata"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            filepath = f.name

        try:
            doc_type_counts = {"text": 10, "table": 2}
            metadata = create_document_metadata(
                filepath=filepath,
                doc_type_counts=doc_type_counts,
                total_elements=12,
                raw_text_elements_size=1024,
            )

            assert metadata["description"] == ""
            assert metadata["tags"] == []
            assert metadata["document_type"] == "pdf"
            assert metadata["file_size"] > 0
            assert "date_created" in metadata
            assert metadata["doc_type_counts"] == doc_type_counts
            assert metadata["total_elements"] == 12
            assert metadata["raw_text_elements_size"] == 1024
        finally:
            os.unlink(filepath)

    def test_create_document_metadata_unknown_extension(self):
        """Test creating document metadata for file without extension"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            filepath = f.name

        try:
            metadata = create_document_metadata(
                filepath=filepath,
                doc_type_counts={},
                total_elements=0,
                raw_text_elements_size=0,
            )

            assert metadata["document_type"] == "unknown"
        finally:
            os.unlink(filepath)

    def test_perform_document_info_aggregation_numeric(self):
        """Test aggregating numeric document info"""
        existing = {"count": 10, "size": 1024}
        new = {"count": 5, "size": 512}

        result = perform_document_info_aggregation(existing, new)

        assert result == {"count": 15, "size": 1536}

    def test_perform_document_info_aggregation_nested_dicts(self):
        """Test aggregating nested dictionary values"""
        existing = {"doc_type_counts": {"text": 10, "table": 2}}
        new = {"doc_type_counts": {"text": 5, "image": 3}}

        result = perform_document_info_aggregation(existing, new)

        assert result == {"doc_type_counts": {"text": 15, "table": 2, "image": 3}}

    def test_perform_document_info_aggregation_mixed_types(self):
        """Test aggregating mixed data types (int, float, str)"""
        existing = {
            "count": 10,
            "size": 1024.5,
            "last_indexed": "2025-01-01T00:00:00Z",
            "status": "Success",
        }
        new = {
            "count": 5,
            "size": 512.3,
            "last_indexed": "2025-01-02T00:00:00Z",
            "status": "Success",
        }

        result = perform_document_info_aggregation(existing, new)

        assert result["count"] == 15
        assert result["size"] == pytest.approx(1536.8)
        assert result["last_indexed"] == "2025-01-02T00:00:00Z"
        assert result["status"] == "Success"

    def test_perform_document_info_aggregation_string_fallback(self):
        """Test that string values prefer new value over existing"""
        existing = {"status": "Pending"}
        new = {"status": "Complete"}

        result = perform_document_info_aggregation(existing, new)

        assert result["status"] == "Complete"

    def test_perform_document_info_aggregation_none_handling(self):
        """Test handling of None values in aggregation"""
        existing = {"field1": None, "field2": 10}
        new = {"field1": 5, "field2": None}

        result = perform_document_info_aggregation(existing, new)

        assert result["field1"] == 5
        assert result["field2"] == 10


class TestFilterDocumentsByConfidence:
    """Test filter_documents_by_confidence function"""

    def test_filter_with_zero_threshold(self):
        """Test filtering with 0.0 threshold (should return all documents)"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": 0.1}),
        ]

        result = filter_documents_by_confidence(docs, 0.0)
        assert len(result) == 3

    def test_filter_with_low_threshold(self):
        """Test filtering with 0.2 threshold"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": 0.1}),
        ]

        result = filter_documents_by_confidence(docs, 0.2)
        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[1].page_content == "doc2"

    def test_filter_with_medium_threshold(self):
        """Test filtering with 0.5 threshold"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": 0.5}),
        ]

        result = filter_documents_by_confidence(docs, 0.5)
        assert len(result) == 2

    def test_filter_with_high_threshold(self):
        """Test filtering with 0.7 threshold"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.9}),
            Document(page_content="doc3", metadata={"relevance_score": 0.5}),
        ]

        result = filter_documents_by_confidence(docs, 0.7)
        assert len(result) == 1
        assert result[0].page_content == "doc2"

    def test_filter_with_very_high_threshold(self):
        """Test filtering with 0.9 threshold (should filter most docs)"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": 0.5}),
        ]

        result = filter_documents_by_confidence(docs, 0.9)
        assert len(result) == 0

    def test_filter_documents_without_relevance_score(self):
        """Test filtering documents that don't have relevance_score"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={"other_field": "value"}),
        ]

        result = filter_documents_by_confidence(docs, 0.5)
        # Documents without relevance_score should be treated as 0.0
        assert len(result) == 0

    def test_filter_empty_document_list(self):
        """Test filtering empty document list"""
        result = filter_documents_by_confidence([], 0.5)
        assert len(result) == 0

    def test_filter_single_document(self):
        """Test filtering single document"""
        from langchain_core.documents import Document

        docs = [Document(page_content="doc1", metadata={"relevance_score": 0.7})]

        result = filter_documents_by_confidence(docs, 0.5)
        assert len(result) == 1

    def test_filter_exact_threshold_match(self):
        """Test that documents with exact threshold score are included"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.5}),
            Document(page_content="doc2", metadata={"relevance_score": 0.4}),
        ]

        result = filter_documents_by_confidence(docs, 0.5)
        assert len(result) == 1
        assert result[0].page_content == "doc1"

    def test_filter_preserves_original_documents(self):
        """Test that filtering doesn't modify original document list"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
        ]
        original_len = len(docs)

        _ = filter_documents_by_confidence(docs, 0.5)

        # Original list should be unchanged
        assert len(docs) == original_len

    def test_filter_with_negative_threshold(self):
        """Test filtering with negative threshold (should return all)"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.0}),
            Document(page_content="doc2", metadata={"relevance_score": 0.5}),
        ]

        result = filter_documents_by_confidence(docs, -0.1)
        assert len(result) == 2

    def test_filter_with_threshold_greater_than_one(self):
        """Test filtering with threshold > 1.0"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 1.0}),
            Document(page_content="doc2", metadata={"relevance_score": 0.9}),
        ]

        result = filter_documents_by_confidence(docs, 1.5)
        assert len(result) == 0

    def test_filter_logging_behavior(self, caplog):
        """Test that filtering logs appropriate information"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
        ]

        with caplog.at_level("INFO"):
            filter_documents_by_confidence(docs, 0.5)

        # Check that info was logged
        assert any(
            "Confidence threshold filtering" in record.message
            for record in caplog.records
        )

    def test_filter_documents_with_non_numeric_relevance_score(self):
        """Test handling of invalid relevance scores"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="doc1", metadata={"relevance_score": "invalid"}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": None}),
        ]

        result = filter_documents_by_confidence(docs, 0.5)
        # Invalid scores should be treated as 0.0
        assert len(result) == 1
        assert result[0].page_content == "doc2"

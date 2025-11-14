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
    filter_documents_by_confidence,
    get_metadata_configuration,
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
    def test_get_metadata_config_none_metadata(
        self, mock_prepare, tmp_path
    ):
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
    def test_get_metadata_config_empty_metadata(
        self, mock_prepare, tmp_path
    ):
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
    def test_get_metadata_config_with_metadata(
        self, mock_prepare, tmp_path
    ):
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


# Commenting out ValidateFilterExpr and ProcessFilterExpr tests as they require
# extensive refactoring to work with the new config pattern.
# These tests would need to:
# 1. Create mock NvidiaRAGConfig instances
# 2. Pass config directly instead of mocking get_config()
# 3. Update assertions to match new function signatures
# TODO: Rewrite these tests in a separate PR


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
        assert any("Confidence threshold filtering" in record.message for record in caplog.records)

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


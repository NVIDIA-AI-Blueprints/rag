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

"""Comprehensive unit tests for rag_server/main.py to improve coverage."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from nvidia_rag.rag_server.main import APIError, NvidiaRAG
from nvidia_rag.utils.vdb.vdb_base import VDBRag


class TestAPIError:
    """Test cases for APIError class."""

    def test_api_error_init_default_code(self):
        """Test APIError initialization with default code."""
        error = APIError("Test error message")

        assert error.message == "Test error message"
        assert error.code == 400
        assert str(error) == "Test error message"

    def test_api_error_init_custom_code(self):
        """Test APIError initialization with custom code."""
        error = APIError("Test error message", 500)

        assert error.message == "Test error message"
        assert error.code == 500
        assert str(error) == "Test error message"

    @patch('nvidia_rag.rag_server.main.logger')
    @patch('nvidia_rag.rag_server.main.print_exc')
    def test_api_error_logging(self, mock_print_exc, mock_logger):
        """Test that APIError logs and prints traceback."""
        error = APIError("Test error message", 500)

        mock_logger.error.assert_called_with(
            "APIError occurred: %s with HTTP status: %d",
            "Test error message",
            500
        )
        mock_print_exc.assert_called_once()


class TestNvidiaRAGInit:
    """Test cases for NvidiaRAG initialization."""

    def test_init_with_none_vdb_op(self):
        """Test initialization with None vdb_op."""
        rag = NvidiaRAG(vdb_op=None)
        assert rag.vdb_op is None

    def test_init_with_valid_vdb_op(self):
        """Test initialization with valid VDBRag instance."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)
        assert rag.vdb_op == mock_vdb_op

    def test_init_with_invalid_vdb_op(self):
        """Test initialization with invalid vdb_op type."""
        with pytest.raises(ValueError, match="vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag"):
            NvidiaRAG(vdb_op="invalid_type")

    def test_init_with_invalid_vdb_op_class(self):
        """Test initialization with invalid vdb_op class."""
        class InvalidVDB:
            pass

        with pytest.raises(ValueError, match="vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag"):
            NvidiaRAG(vdb_op=InvalidVDB())


class TestNvidiaRAGHealth:
    """Test cases for NvidiaRAG health method."""

    @pytest.mark.asyncio
    async def test_health_basic(self):
        """Test basic health check without dependencies."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_prepare.return_value = Mock()

            result = await rag.health(check_dependencies=False)

            assert result["message"] == "Service is up."
            assert "dependencies" not in result

    @pytest.mark.asyncio
    async def test_health_with_dependencies(self):
        """Test health check with dependencies."""
        rag = NvidiaRAG()

        mock_vdb_op = Mock()
        mock_dependencies = {"vdb": "healthy", "llm": "healthy"}

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.check_all_services_health') as mock_check:
                mock_prepare.return_value = mock_vdb_op
                mock_check.return_value = mock_dependencies

                result = await rag.health(check_dependencies=True)

                assert result["message"] == "Service is up."
                assert result["vdb"] == "healthy"
                assert result["llm"] == "healthy"
                mock_check.assert_called_once_with(mock_vdb_op)


class TestNvidiaRAGPrepareVDBOp:
    """Test cases for NvidiaRAG __prepare_vdb_op method."""

    def test_prepare_vdb_op_with_existing_vdb_op(self):
        """Test __prepare_vdb_op when vdb_op is already set."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        result = rag._NvidiaRAG__prepare_vdb_op()
        assert result == mock_vdb_op

    def test_prepare_vdb_op_with_vdb_endpoint_error(self):
        """Test __prepare_vdb_op with vdb_endpoint when vdb_op is set."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="vdb_endpoint is not supported when vdb_op is provided during initialization"):
            rag._NvidiaRAG__prepare_vdb_op(vdb_endpoint="http://test.com")

    def test_prepare_vdb_op_with_embedding_model_error(self):
        """Test __prepare_vdb_op with embedding_model when vdb_op is set."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="embedding_model is not supported when vdb_op is provided during initialization"):
            rag._NvidiaRAG__prepare_vdb_op(embedding_model="test-model")

    def test_prepare_vdb_op_with_embedding_endpoint_error(self):
        """Test __prepare_vdb_op with embedding_endpoint when vdb_op is set."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="embedding_endpoint is not supported when vdb_op is provided during initialization"):
            rag._NvidiaRAG__prepare_vdb_op(embedding_endpoint="http://test.com")

    @patch('nvidia_rag.rag_server.main.get_embedding_model')
    @patch('nvidia_rag.rag_server.main._get_vdb_op')
    @patch('nvidia_rag.rag_server.main.CONFIG')
    def test_prepare_vdb_op_without_existing_vdb_op(self, mock_config, mock_get_vdb, mock_get_embedding):
        """Test __prepare_vdb_op when vdb_op is not set."""
        # Setup mocks
        mock_config.embeddings.model_name = "default-model"
        mock_config.embeddings.server_url = "http://default.com"
        mock_config.vector_store.default_collection_name = "default_collection"

        mock_embedder = Mock()
        mock_get_embedding.return_value = mock_embedder

        mock_vdb_op = Mock(spec=VDBRag)
        mock_get_vdb.return_value = mock_vdb_op

        rag = NvidiaRAG()

        result = rag._NvidiaRAG__prepare_vdb_op()

        assert result == mock_vdb_op
        mock_get_embedding.assert_called_once()
        mock_get_vdb.assert_called_once()

    @patch('nvidia_rag.rag_server.main.get_embedding_model')
    @patch('nvidia_rag.rag_server.main._get_vdb_op')
    @patch('nvidia_rag.rag_server.main.CONFIG')
    def test_prepare_vdb_op_with_custom_parameters(self, mock_config, mock_get_vdb, mock_get_embedding):
        """Test __prepare_vdb_op with custom parameters."""
        # Setup mocks
        mock_config.vector_store.default_collection_name = "default_collection"

        mock_embedder = Mock()
        mock_get_embedding.return_value = mock_embedder

        mock_vdb_op = Mock(spec=VDBRag)
        mock_get_vdb.return_value = mock_vdb_op

        rag = NvidiaRAG()

        result = rag._NvidiaRAG__prepare_vdb_op(
            vdb_endpoint="http://custom-vdb.com",
            embedding_model="custom-model",
            embedding_endpoint="http://custom-embedding.com"
        )

        assert result == mock_vdb_op
        mock_get_embedding.assert_called_once_with(
            model="custom-model",
            url="http://custom-embedding.com"
        )
        mock_get_vdb.assert_called_once()


class TestNvidiaRAGValidateCollections:
    """Test cases for NvidiaRAG _validate_collections_exist method."""

    def test_validate_collections_exist_success(self):
        """Test successful collection validation."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Should not raise any exception
        rag._validate_collections_exist(["collection1", "collection2"], mock_vdb_op)

    def test_validate_collections_exist_missing_collection(self):
        """Test collection validation with missing collection."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.side_effect = lambda name: name == "collection1"

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(APIError, match="Collection collection2 does not exist"):
            rag._validate_collections_exist(["collection1", "collection2"], mock_vdb_op)

    def test_validate_collections_exist_empty_collections(self):
        """Test collection validation with empty collections list."""
        mock_vdb_op = Mock(spec=VDBRag)

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Should not raise any exception for empty list
        rag._validate_collections_exist([], mock_vdb_op)


class TestNvidiaRAGExtractTextFromContent:
    """Test cases for NvidiaRAG _extract_text_from_content method."""

    def test_extract_text_from_string(self):
        """Test extracting text from string content."""
        rag = NvidiaRAG()

        result = rag._extract_text_from_content("Hello world")
        assert result == "Hello world"

    def test_extract_text_from_multimodal_list(self):
        """Test extracting text from multimodal list content."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image_url", "image_url": "http://example.com/image.jpg"}
        ]

        result = rag._extract_text_from_content(content)
        assert result == "Hello world"

    def test_extract_text_from_list_without_text(self):
        """Test extracting text from list without text items."""
        rag = NvidiaRAG()

        content = [
            {"type": "image_url", "image_url": "http://example.com/image.jpg"}
        ]

        result = rag._extract_text_from_content(content)
        assert result == ""

    def test_extract_text_from_other_type(self):
        """Test extracting text from other content types."""
        rag = NvidiaRAG()

        result = rag._extract_text_from_content(123)
        assert result == "123"


class TestNvidiaRAGContainsImages:
    """Test cases for NvidiaRAG _contains_images method."""

    def test_contains_images_string(self):
        """Test _contains_images with string content."""
        rag = NvidiaRAG()

        result = rag._contains_images("Hello world")
        assert result is False

    def test_contains_images_multimodal_list_with_images(self):
        """Test _contains_images with multimodal list containing images."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello world"},
            {"type": "image_url", "image_url": "http://example.com/image1.jpg"}
        ]

        result = rag._contains_images(content)
        assert result is True

    def test_contains_images_multimodal_list_without_images(self):
        """Test _contains_images with multimodal list without images."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello world"},
            {"type": "text", "text": "More text"}
        ]

        result = rag._contains_images(content)
        assert result is False

    def test_contains_images_string(self):
        """Test _contains_images with string content."""
        rag = NvidiaRAG()

        content = "Hello world"

        result = rag._contains_images(content)
        assert result is False

    def test_contains_images_other_type(self):
        """Test _contains_images with other content types."""
        rag = NvidiaRAG()

        content = {"some": "data"}

        result = rag._contains_images(content)
        assert result is False


class TestNvidiaRAGBuildRetrieverQuery:
    """Test cases for NvidiaRAG _build_retriever_query_from_content method."""

    def test_build_retriever_query_from_string(self):
        """Test building retriever query from string content."""
        rag = NvidiaRAG()

        result = rag._build_retriever_query_from_content("Hello world")
        assert result == "Hello world"

    def test_build_retriever_query_from_multimodal_list(self):
        """Test building retriever query from multimodal list content."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        ]

        result = rag._build_retriever_query_from_content(content)
        assert "Hello" in result
        assert "world" in result
        assert "http://example.com/image.jpg" in result

    def test_build_retriever_query_from_list_without_text(self):
        """Test building retriever query from list without text items."""
        rag = NvidiaRAG()

        content = [
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        ]

        result = rag._build_retriever_query_from_content(content)
        assert result == "http://example.com/image.jpg"

    def test_build_retriever_query_from_other_type(self):
        """Test building retriever query from other content types."""
        rag = NvidiaRAG()

        result = rag._build_retriever_query_from_content(123)
        assert result == "123"


class TestNvidiaRAGPrintConversationHistory:
    """Test cases for NvidiaRAG __print_conversation_history method."""

    def test_print_conversation_history(self):
        """Test printing conversation history."""
        rag = NvidiaRAG()

        conversation_history = [
            ("user", "Hello"),
            ("assistant", "Hi there!")
        ]

        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
            rag._NvidiaRAG__print_conversation_history(conversation_history)

            # Verify debug log was called
            assert mock_logger.debug.call_count > 0

    def test_print_conversation_history_empty(self):
        """Test printing empty conversation history."""
        rag = NvidiaRAG()

        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
            rag._NvidiaRAG__print_conversation_history([])

            # Should not call debug log with empty history
            assert mock_logger.debug.call_count == 0


class TestNvidiaRAGNormalizeRelevanceScores:
    """Test cases for NvidiaRAG __normalize_relevance_scores method."""

    def test_normalize_relevance_scores(self):
        """Test normalizing relevance scores."""
        rag = NvidiaRAG()

        documents = [
            Mock(metadata={"relevance_score": 0.8}),
            Mock(metadata={"relevance_score": 0.6}),
            Mock(metadata={"relevance_score": 0.4})
        ]

        result = rag._NvidiaRAG__normalize_relevance_scores(documents)

        # Should return the same documents
        assert len(result) == 3
        assert result == documents

    def test_normalize_relevance_scores_empty(self):
        """Test normalizing relevance scores with empty list."""
        rag = NvidiaRAG()

        result = rag._NvidiaRAG__normalize_relevance_scores([])

        assert result == []


class TestNvidiaRAGFormatDocumentWithSource:
    """Test cases for NvidiaRAG __format_document_with_source method."""

    def test_format_document_with_source(self):
        """Test formatting document with source."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {"source": "test.pdf"}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert "Test content" in result
            assert "File: test" in result

    def test_format_document_without_source(self):
        """Test formatting document without source."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert result == "Test content"

    def test_format_document_with_nested_source(self):
        """Test formatting document with nested source."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {"source": {"source_name": "test.pdf"}}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert "Test content" in result
            assert "File: test" in result

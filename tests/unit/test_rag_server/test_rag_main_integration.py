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

"""Working unit tests for rag_server/main.py to improve coverage."""

import os
from unittest.mock import Mock, patch
from collections.abc import Generator as GeneratorType

import pytest

from nvidia_rag.rag_server.main import APIError, NvidiaRAG
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.rag_server.response_generator import Citations


class TestNvidiaRAGGenerateWorking:
    """Working test cases for NvidiaRAG generate method."""

    def test_generate_with_knowledge_base(self):
        """Test generate method with knowledge base enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        messages = [{"role": "user", "content": "Test query"}]

        with patch('nvidia_rag.rag_server.main.prepare_llm_request') as mock_prepare_request:
            with patch.object(rag, '_NvidiaRAG__rag_chain') as mock_rag_chain:
                mock_prepare_request.return_value = ("test query", [])
                mock_rag_chain.return_value = iter(["Test response"])

                result = rag.generate(messages, use_knowledge_base=True)

                assert hasattr(result, '__iter__')
                response = list(result)
                assert response == ["Test response"]

    def test_generate_without_knowledge_base(self):
        """Test generate method with knowledge base disabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        messages = [{"role": "user", "content": "Test query"}]

        with patch('nvidia_rag.rag_server.main.prepare_llm_request') as mock_prepare_request:
            with patch.object(rag, '_NvidiaRAG__llm_chain') as mock_llm_chain:
                mock_prepare_request.return_value = ("test query", [])
                mock_llm_chain.return_value = iter(["Test response"])

                result = rag.generate(messages, use_knowledge_base=False)

                assert hasattr(result, '__iter__')
                response = list(result)
                assert response == ["Test response"]

    def test_generate_with_validation_errors(self):
        """Test generate method with validation errors."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        messages = [{"role": "user", "content": "Test query"}]

        with patch('nvidia_rag.rag_server.main.validate_use_knowledge_base') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid use_knowledge_base")

            with pytest.raises(ValueError, match="Invalid use_knowledge_base"):
                rag.generate(messages, use_knowledge_base="invalid")


class TestNvidiaRAGSearchWorking:
    """Working test cases for NvidiaRAG search method."""

    def test_search_basic(self):
        """Test basic search functionality."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.prepare_llm_request') as mock_prepare_request:
                with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                    with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                        with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                            with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                                with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                    with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                        with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                            # Mock the ranker
                                            mock_ranker = Mock()
                                            mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                            mock_get_ranking.return_value = mock_ranker

                                            # Mock filter validation
                                            mock_validate_filter.return_value = {"status": True, "validated_collections": ["test_collection"]}
                                            mock_process_filter.return_value = ""

                                            # Mock ThreadPoolExecutor
                                            mock_future = Mock()
                                            mock_future.result.return_value = [Mock(page_content="test content", metadata={})]
                                            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                            # Mock RunnableAssign
                                            mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                            # Mock filter documents
                                            mock_filter_docs.return_value = [Mock(page_content="test content", metadata={})]

                                            mock_prepare.return_value = mock_vdb_op
                                            mock_prepare_request.return_value = ("test query", [])
                                            mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                            result = rag.search("test query")

                                            assert isinstance(result, Citations)

    def test_search_with_messages(self):
        """Test search with messages parameter."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        messages = [{"role": "user", "content": "Test query"}]

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.prepare_llm_request') as mock_prepare_request:
                with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                    with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                        with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                            with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                                with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                    with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                        with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                            # Mock the ranker
                                            mock_ranker = Mock()
                                            mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                            mock_get_ranking.return_value = mock_ranker

                                            # Mock filter validation
                                            mock_validate_filter.return_value = {"status": True, "validated_collections": ["test_collection"]}
                                            mock_process_filter.return_value = ""

                                            # Mock ThreadPoolExecutor
                                            mock_future = Mock()
                                            mock_future.result.return_value = [Mock(page_content="test content", metadata={})]
                                            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                            # Mock RunnableAssign
                                            mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                            # Mock filter documents
                                            mock_filter_docs.return_value = [Mock(page_content="test content", metadata={})]

                                            mock_prepare.return_value = mock_vdb_op
                                            mock_prepare_request.return_value = ("test query", [])
                                            mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                            result = rag.search("test query", messages=messages)

                                            assert isinstance(result, Citations)
                                            # Note: prepare_llm_request is only called when messages is not None
                                            # and when query rewriting is enabled

    def test_search_with_empty_collection_names_error(self):
        """Test search with empty collection_names raises APIError."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_vdb_op = Mock(spec=VDBRag)
            mock_prepare.return_value = mock_vdb_op

            with pytest.raises(APIError, match="Collection names are not provided"):
                rag.search("test query", collection_names=[])

    def test_search_with_collection_validation_error(self):
        """Test search with collection validation error."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_vdb_op = Mock(spec=VDBRag)
            mock_vdb_op.check_collection_exists.return_value = False
            mock_prepare.return_value = mock_vdb_op

            with pytest.raises(APIError, match="Collection test_collection does not exist"):
                rag.search("test query", collection_names=["test_collection"])


class TestNvidiaRAGGetSummaryWorking:
    """Working test cases for NvidiaRAG get_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_basic(self):
        """Test basic get_summary functionality."""
        rag = NvidiaRAG()

        with patch('nvidia_rag.rag_server.main.retrieve_summary') as mock_retrieve:
            mock_retrieve.return_value = {"summary": "Test summary"}

            result = await rag.get_summary("test_collection", "test_document")

            assert result == {"summary": "Test summary"}
            mock_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_summary_with_custom_parameters(self):
        """Test get_summary with custom parameters."""
        rag = NvidiaRAG()

        with patch('nvidia_rag.rag_server.main.retrieve_summary') as mock_retrieve:
            mock_retrieve.return_value = {"summary": "Test summary"}

            result = await rag.get_summary(
                "test_collection",
                "test_document",
                blocking=True,
                timeout=600
            )

            assert result == {"summary": "Test summary"}
            mock_retrieve.assert_called_once_with(
                collection_name="test_collection",
                file_name="test_document",
                wait=True,
                timeout=600
            )


class TestNvidiaRAGPrivateMethodsWorking:
    """Working test cases for private methods in NvidiaRAG class."""

    def test_extract_text_from_content_string(self):
        """Test _extract_text_from_content with string input."""
        rag = NvidiaRAG()

        result = rag._extract_text_from_content("Hello world")
        assert result == "Hello world"

    def test_extract_text_from_content_multimodal(self):
        """Test _extract_text_from_content with multimodal input."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        ]

        result = rag._extract_text_from_content(content)
        assert result == "Hello world"

    def test_contains_images_string(self):
        """Test _contains_images with string input."""
        rag = NvidiaRAG()

        result = rag._contains_images("Hello world")
        assert result is False

    def test_contains_images_multimodal_with_images(self):
        """Test _contains_images with multimodal input containing images."""
        rag = NvidiaRAG()

        content = [
            {"type": "text", "text": "Hello world"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        ]

        result = rag._contains_images(content)
        assert result is True

    def test_build_retriever_query_from_content_string(self):
        """Test _build_retriever_query_from_content with string input."""
        rag = NvidiaRAG()

        result = rag._build_retriever_query_from_content("Hello world")
        assert result == "Hello world"

    def test_build_retriever_query_from_content_multimodal(self):
        """Test _build_retriever_query_from_content with multimodal input."""
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

    def test_print_conversation_history(self):
        """Test __print_conversation_history method."""
        rag = NvidiaRAG()

        conversation_history = [
            ("user", "Hello"),
            ("assistant", "Hi there!")
        ]

        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
            rag._NvidiaRAG__print_conversation_history(conversation_history)

            # Verify debug log was called
            assert mock_logger.debug.call_count > 0

    def test_normalize_relevance_scores(self):
        """Test __normalize_relevance_scores method."""
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

    def test_format_document_with_source_enabled(self):
        """Test __format_document_with_source with source metadata enabled."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {"source": "test.pdf"}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert "Test content" in result
            assert "File: test" in result

    def test_format_document_without_source(self):
        """Test __format_document_with_source without source metadata."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert result == "Test content"

    def test_format_document_source_metadata_not_set(self):
        """Test __format_document_with_source with source metadata not set."""
        rag = NvidiaRAG()

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {"source": "test.pdf"}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "False"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)

            assert result == "Test content"


class TestNvidiaRAGEdgeCasesWorking:
    """Working test cases for edge cases in NvidiaRAG methods."""

    def test_generate_with_empty_messages(self):
        """Test generate method with empty messages."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch('nvidia_rag.rag_server.main.prepare_llm_request') as mock_prepare_request:
            with patch.object(rag, '_NvidiaRAG__llm_chain') as mock_llm_chain:
                mock_prepare_request.return_value = ("", [])
                mock_llm_chain.return_value = iter(["Test response"])

                result = rag.generate([], use_knowledge_base=False)

                assert hasattr(result, '__iter__')
                list(result)  # Consume generator

    def test_private_methods_edge_cases(self):
        """Test private methods with edge cases."""
        rag = NvidiaRAG()

        # Test _extract_text_from_content with various edge cases
        assert rag._extract_text_from_content("") == ""
        assert rag._extract_text_from_content([]) == ""
        assert rag._extract_text_from_content(None) == ""
        assert rag._extract_text_from_content(123) == "123"

        # Test _contains_images with various edge cases
        assert rag._contains_images("") is False
        assert rag._contains_images([]) is False
        assert rag._contains_images(None) is False
        assert rag._contains_images(123) is False

        # Test _build_retriever_query_from_content with various edge cases
        assert rag._build_retriever_query_from_content("") == ""
        assert rag._build_retriever_query_from_content([]) == ""
        assert rag._build_retriever_query_from_content(None) == ""
        assert rag._build_retriever_query_from_content(123) == "123"

    def test_format_document_with_source_edge_cases(self):
        """Test __format_document_with_source with edge cases."""
        rag = NvidiaRAG()

        # Test with None document - should raise AttributeError
        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            with pytest.raises(AttributeError):
                rag._NvidiaRAG__format_document_with_source(None)

        # Test with document having None page_content
        doc = Mock()
        doc.page_content = None
        doc.metadata = {"source": "test.pdf"}

        with patch.dict(os.environ, {"ENABLE_SOURCE_METADATA": "True"}):
            result = rag._NvidiaRAG__format_document_with_source(doc)
            assert result == "File: test\nContent: None"

    def test_normalize_relevance_scores_edge_cases(self):
        """Test __normalize_relevance_scores with edge cases."""
        rag = NvidiaRAG()

        # Test with None input
        result = rag._NvidiaRAG__normalize_relevance_scores(None)
        assert result is None

        # Test with empty list
        result = rag._NvidiaRAG__normalize_relevance_scores([])
        assert result == []

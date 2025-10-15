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

"""Comprehensive unit tests for rag_server/main.py to improve coverage from 39% to 80%+."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from collections.abc import Generator as GeneratorType

from nvidia_rag.rag_server.main import APIError, NvidiaRAG
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.rag_server.response_generator import Citations


class TestNvidiaRAGSearchCoverage:
    """Test cases to improve search method coverage."""

    def test_search_with_collection_name_deprecation_warning(self):
        """Test search with collection_name parameter (deprecated)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
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
                                        mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                            result = rag.search("test query", collection_name="test_collection")

                                            assert isinstance(result, Citations)
                                            # Verify deprecation warning was logged
                                            mock_logger.warning.assert_called()

    def test_search_with_multiple_collections_without_reranker_error(self):
        """Test search with multiple collections but reranker disabled."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_vdb_op = Mock(spec=VDBRag)
            mock_prepare.return_value = mock_vdb_op

            with pytest.raises(APIError, match="Reranking is not enabled but multiple collection names are provided"):
                rag.search("test query", collection_names=["col1", "col2"], enable_reranker=False)

    def test_search_with_too_many_collections_error(self):
        """Test search with more than MAX_COLLECTION_NAMES collections."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_vdb_op = Mock(spec=VDBRag)
            mock_prepare.return_value = mock_vdb_op

            with pytest.raises(APIError, match="Only 5 collections are supported at a time"):
                rag.search("test query", collection_names=["col1", "col2", "col3", "col4", "col5", "col6"])

    def test_search_with_filter_expression_validation_error(self):
        """Test search with invalid filter expression."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                mock_prepare.return_value = mock_vdb_op
                mock_validate_filter.return_value = {
                    "status": False,
                    "error_message": "Invalid filter",
                    "details": "Some details"
                }

                with pytest.raises(APIError, match="Invalid filter expression: Invalid filter\n Details: Some details"):
                    rag.search("test query", collection_names=["test_collection"], filter_expr="invalid_filter")

    def test_search_with_skipped_collections_logging(self):
        """Test search with some collections skipped due to filter validation."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                    with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                        with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                            with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                    with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                        mock_prepare.return_value = mock_vdb_op
                                        mock_validate_filter.return_value = {
                                            "status": True,
                                            "validated_collections": ["col1"]  # col2 is skipped
                                        }
                                        mock_process_filter.return_value = "processed_filter"
                                        mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                        # Mock the ranker
                                        mock_ranker = Mock()
                                        mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                        mock_get_ranking.return_value = mock_ranker

                                        # Mock ThreadPoolExecutor - need to return tuple for future.result()
                                        mock_future = Mock()
                                        mock_future.result.return_value = ("col1", "processed_filter")
                                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                        # Mock RunnableAssign
                                        mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                        # Mock filter documents
                                        mock_filter_docs.return_value = [Mock(page_content="test content", metadata={})]

                                        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                            result = rag.search("test query", collection_names=["col1", "col2"], filter_expr="some_filter")

                                            assert isinstance(result, Citations)
                                            # Verify skipped collections warning was logged
                                            mock_logger.info.assert_called()

    def test_search_with_query_rewriting_enabled(self):
        """Test search with query rewriting enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        messages = [{"role": "user", "content": "Test query"}]

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                            with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                    with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                        with patch('nvidia_rag.rag_server.main.query_rewriter_llm') as mock_query_rewriter:
                                            with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                                                with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                                                    with patch('nvidia_rag.rag_server.main.ChatPromptTemplate') as mock_prompt_template:
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

                                                        # Mock query rewriter chain
                                                        mock_chain = Mock()
                                                        mock_chain.invoke.return_value = "rewritten query"
                                                        mock_prompt_template.from_messages.return_value = mock_chain

                                                        mock_prepare.return_value = mock_vdb_op
                                                        mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                                        result = rag.search("test query", messages=messages, enable_query_rewriting=True)

                                                        assert isinstance(result, Citations)

    def test_search_with_filter_generator_enabled(self):
        """Test search with filter generator enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                            with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                    with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                        with patch('nvidia_rag.rag_server.main.CONFIG') as mock_config:
                                            with patch('nvidia_rag.rag_server.main.generate_filter_from_natural_language') as mock_generate_filter:
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

                                                # Mock config for Milvus
                                                mock_config.vector_store.name = "milvus"

                                                # Mock filter generation
                                                mock_generate_filter.return_value = "generated_filter"

                                                mock_prepare.return_value = mock_vdb_op
                                                mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                                result = rag.search("test query", collection_names=["test_collection"], enable_filter_generator=True)

                                                assert isinstance(result, Citations)

    def test_search_with_reflection_enabled(self):
        """Test search with reflection enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.prepare_citations') as mock_prepare_citations:
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.process_filter_expr') as mock_process_filter:
                            with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                                with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                    with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence') as mock_filter_docs:
                                        with patch('nvidia_rag.rag_server.main.check_context_relevance') as mock_check_relevance:
                                            with patch('nvidia_rag.rag_server.main.ReflectionCounter') as mock_reflection_counter:
                                                with patch.dict(os.environ, {"ENABLE_REFLECTION": "true"}):
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

                                                    # Mock reflection
                                                    mock_check_relevance.return_value = ([Mock(page_content="test content", metadata={})], True)
                                                    mock_reflection_counter.return_value = Mock()

                                                    mock_prepare.return_value = mock_vdb_op
                                                    mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                                    result = rag.search("test query", collection_names=["test_collection"])

                                                    assert isinstance(result, Citations)

    def test_search_with_confidence_threshold_warning(self):
        """Test search with confidence threshold but reranker disabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
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
                                        mock_prepare_citations.return_value = Citations(documents=[], sources=[])

                                        with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                            result = rag.search("test query", collection_names=["test_collection"],
                                                             confidence_threshold=0.5, enable_reranker=False)

                                            assert isinstance(result, Citations)
                                            # Verify warning was logged
                                            mock_logger.warning.assert_called()


class TestNvidiaRAGLLMChainCoverage:
    """Test cases to improve __llm_chain method coverage."""

    def test_llm_chain_basic(self):
        """Test basic LLM chain functionality."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [("user", "test user")],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.return_value = iter(["test response"])
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["test response"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert response == ["test response"]

    def test_llm_chain_with_conversation_history(self):
        """Test LLM chain with conversation history."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [("user", "Hello"), ("assistant", "Hi there!")],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.return_value = iter(["test response"])
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["test response"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=chat_history,
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert response == ["test response"]

    def test_llm_chain_with_connect_timeout(self):
        """Test LLM chain with ConnectTimeout exception."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                from requests import ConnectTimeout

                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.side_effect = ConnectTimeout("Connection timeout")
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["Connection timed out message"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert "Connection timed out" in response[0]

    def test_llm_chain_with_403_error(self):
        """Test LLM chain with 403 Forbidden error."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.side_effect = Exception("[403] Forbidden Invalid UAM response")
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["Authentication error message"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert "Authentication" in response[0]

    def test_llm_chain_with_404_error(self):
        """Test LLM chain with 404 Not Found error."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.side_effect = Exception("[404] Not Found")
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["404 error message"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert "404" in response[0] or "Not Found" in response[0]

    def test_llm_chain_with_general_exception(self):
        """Test LLM chain with general exception."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": True,
            "stop": []
        }

        with patch.object(rag, '_handle_prompt_processing') as mock_handle_prompt:
            with patch('nvidia_rag.rag_server.main.get_llm') as mock_get_llm:
                with patch('nvidia_rag.rag_server.main.StreamingFilterThinkParser') as mock_parser:
                    with patch('nvidia_rag.rag_server.main.StrOutputParser') as mock_str_parser:
                        with patch('nvidia_rag.rag_server.main.generate_answer') as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "15"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [],
                                    [("user", "test user")]
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                mock_chain = Mock()
                                mock_chain.stream.side_effect = Exception("General error")
                                mock_parser.return_value = mock_chain

                                mock_generate_answer.return_value = iter(["General error message"])

                                result = rag._NvidiaRAG__llm_chain(
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    collection_name="test_collection",
                                    enable_citations=True
                                )

                                assert hasattr(result, 'generator')
                                assert hasattr(result, 'status_code')
                                response = list(result.generator)
                                assert "General error" in response[0]


class TestNvidiaRAGPromptProcessingCoverage:
    """Test cases to improve _handle_prompt_processing method coverage."""

    def test_handle_prompt_processing_basic(self):
        """Test basic prompt processing functionality."""
        rag = NvidiaRAG()

        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        with patch('nvidia_rag.rag_server.main.prompts') as mock_prompts:
            mock_prompts.get.return_value = {
                "system": "Test system prompt",
                "human": "Test human prompt"
            }

            result = rag._handle_prompt_processing(chat_history, "test_model", "chat_template")

            assert len(result) == 3
            system_message, conversation_history, user_message = result
            assert system_message == [("system", "Test system prompt")]
            assert conversation_history == [("user", "Hello"), ("assistant", "Hi there!")]
            assert user_message == [("user", "Test human prompt")]

    def test_handle_prompt_processing_with_nemotron_v1_model(self):
        """Test prompt processing with Nemotron v1 model."""
        rag = NvidiaRAG()

        chat_history = []

        with patch('nvidia_rag.rag_server.main.prompts') as mock_prompts:
            with patch.dict(os.environ, {"ENABLE_NEMOTRON_THINKING": "true"}):
                mock_prompts.get.return_value = {
                    "system": "Test system prompt",
                    "human": "Test human prompt"
                }

                result = rag._handle_prompt_processing(chat_history, "llama-3.3-nemotron-super-49b-v1", "chat_template")

                assert len(result) == 3
                system_message, conversation_history, user_message = result
                assert system_message == [("system", "detailed thinking on")]

    def test_handle_prompt_processing_with_system_message_in_history(self):
        """Test prompt processing with system message in chat history."""
        rag = NvidiaRAG()

        chat_history = [
            {"role": "system", "content": "Custom system message"},
            {"role": "user", "content": "Hello"}
        ]

        with patch('nvidia_rag.rag_server.main.prompts') as mock_prompts:
            mock_prompts.get.return_value = {
                "system": "Test system prompt",
                "human": "Test human prompt"
            }

            result = rag._handle_prompt_processing(chat_history, "test_model", "chat_template")

            assert len(result) == 3
            system_message, conversation_history, user_message = result
            assert "Custom system message" in system_message[0][1]
            assert conversation_history == [("user", "Hello")]

    def test_handle_prompt_processing_with_empty_user_prompt(self):
        """Test prompt processing with empty user prompt."""
        rag = NvidiaRAG()

        chat_history = []

        with patch('nvidia_rag.rag_server.main.prompts') as mock_prompts:
            mock_prompts.get.return_value = {
                "system": "Test system prompt",
                "human": ""
            }

            result = rag._handle_prompt_processing(chat_history, "test_model", "chat_template")

            assert len(result) == 3
            system_message, conversation_history, user_message = result
            assert system_message == [("system", "Test system prompt")]
            assert user_message == []

    def test_handle_prompt_processing_with_multimodal_content(self):
        """Test prompt processing with multimodal content."""
        rag = NvidiaRAG()

        chat_history = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
            ]}
        ]

        with patch('nvidia_rag.rag_server.main.prompts') as mock_prompts:
            mock_prompts.get.return_value = {
                "system": "Test system prompt",
                "human": "Test human prompt"
            }

            result = rag._handle_prompt_processing(chat_history, "test_model", "chat_template")

            assert len(result) == 3
            system_message, conversation_history, user_message = result
            assert conversation_history == [("user", "Hello")]


class TestNvidiaRAGHealthCoverage:
    """Test cases to improve health method coverage."""

    @pytest.mark.asyncio
    async def test_health_basic(self):
        """Test basic health check."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            mock_vdb_op = Mock(spec=VDBRag)
            mock_prepare.return_value = mock_vdb_op

            result = await rag.health()

            assert result["message"] == "Service is up."
            mock_prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_with_dependencies(self):
        """Test health check with dependencies."""
        rag = NvidiaRAG()

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op') as mock_prepare:
            with patch('nvidia_rag.rag_server.main.check_all_services_health') as mock_check_health:
                mock_vdb_op = Mock(spec=VDBRag)
                mock_prepare.return_value = mock_vdb_op
                mock_check_health.return_value = {"vdb": "healthy"}

                result = await rag.health(check_dependencies=True)

                assert result["message"] == "Service is up."
                assert result["vdb"] == "healthy"
                mock_check_health.assert_called_once_with(mock_vdb_op)


class TestNvidiaRAGPrepareVdbOpCoverage:
    """Test cases to improve __prepare_vdb_op method coverage."""

    def test_prepare_vdb_op_with_existing_vdb_op(self):
        """Test __prepare_vdb_op with existing vdb_op."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        result = rag._NvidiaRAG__prepare_vdb_op()

        assert result == mock_vdb_op

    def test_prepare_vdb_op_with_vdb_endpoint_error(self):
        """Test __prepare_vdb_op with vdb_endpoint when vdb_op exists."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="vdb_endpoint is not supported when vdb_op is provided"):
            rag._NvidiaRAG__prepare_vdb_op(vdb_endpoint="http://test.com")

    def test_prepare_vdb_op_with_embedding_model_error(self):
        """Test __prepare_vdb_op with embedding_model when vdb_op exists."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="embedding_model is not supported when vdb_op is provided"):
            rag._NvidiaRAG__prepare_vdb_op(embedding_model="test_model")

    def test_prepare_vdb_op_with_embedding_endpoint_error(self):
        """Test __prepare_vdb_op with embedding_endpoint when vdb_op exists."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with pytest.raises(ValueError, match="embedding_endpoint is not supported when vdb_op is provided"):
            rag._NvidiaRAG__prepare_vdb_op(embedding_endpoint="http://test.com")

    def test_prepare_vdb_op_without_vdb_op(self):
        """Test __prepare_vdb_op without existing vdb_op."""
        rag = NvidiaRAG()

        with patch('nvidia_rag.rag_server.main.get_embedding_model') as mock_get_embedding:
            with patch('nvidia_rag.rag_server.main._get_vdb_op') as mock_get_vdb_op:
                mock_embedding = Mock()
                mock_get_embedding.return_value = mock_embedding
                mock_vdb_op = Mock(spec=VDBRag)
                mock_get_vdb_op.return_value = mock_vdb_op

                result = rag._NvidiaRAG__prepare_vdb_op(
                    vdb_endpoint="http://test.com",
                    embedding_model="test_model",
                    embedding_endpoint="http://embedding.com"
                )

                assert result == mock_vdb_op
                mock_get_embedding.assert_called_once_with(
                    model="test_model",
                    url="http://embedding.com"
                )
                mock_get_vdb_op.assert_called_once_with(
                    vdb_endpoint="http://test.com",
                    embedding_model=mock_embedding
                )


class TestNvidiaRAGValidationCoverage:
    """Test cases to improve validation method coverage."""

    def test_validate_collections_exist_success(self):
        """Test _validate_collections_exist with existing collections."""
        rag = NvidiaRAG()
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True

        # Should not raise any exception
        rag._validate_collections_exist(["col1", "col2"], mock_vdb_op)

    def test_validate_collections_exist_failure(self):
        """Test _validate_collections_exist with non-existing collection."""
        rag = NvidiaRAG()
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = False

        with pytest.raises(APIError, match="Collection col1 does not exist"):
            rag._validate_collections_exist(["col1"], mock_vdb_op)


class TestNvidiaRAGInitCoverage:
    """Test cases to improve __init__ method coverage."""

    def test_init_with_invalid_vdb_op(self):
        """Test __init__ with invalid vdb_op type."""
        with pytest.raises(ValueError, match="vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag"):
            NvidiaRAG(vdb_op="invalid_vdb_op")

    def test_init_with_valid_vdb_op(self):
        """Test __init__ with valid vdb_op."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        assert rag.vdb_op == mock_vdb_op

    def test_init_without_vdb_op(self):
        """Test __init__ without vdb_op."""
        rag = NvidiaRAG()

        assert rag.vdb_op is None


class TestNvidiaRAGAPICoverage:
    """Test cases to improve APIError class coverage."""

    def test_api_error_init(self):
        """Test APIError initialization."""
        error = APIError("Test error message", 500)

        assert error.message == "Test error message"
        assert error.code == 500
        assert str(error) == "Test error message"

    def test_api_error_init_default_code(self):
        """Test APIError initialization with default code."""
        error = APIError("Test error message")

        assert error.message == "Test error message"
        assert error.code == 400
        assert str(error) == "Test error message"

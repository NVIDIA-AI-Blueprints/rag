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

"""Minimal unit tests for rag_server/main.py to improve coverage for specific lines."""

import pytest
from unittest.mock import Mock, patch

from nvidia_rag.rag_server.main import APIError, NvidiaRAG
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.rag_server.response_generator import Citations


class TestNvidiaRAGMinimalCoverage:
    """Minimal test cases to improve coverage for specific lines."""

    def test_search_with_empty_filter_expression(self):
        """Test search with empty filter expression to cover lines 511-517."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op', return_value=mock_vdb_op):
            with patch('nvidia_rag.rag_server.main.prepare_citations', return_value=Citations(documents=[], sources=[])):
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                            with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence', return_value=[Mock(page_content="test content", metadata={})]):
                                    with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                        # Mock the ranker
                                        mock_ranker = Mock()
                                        mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                        mock_get_ranking.return_value = mock_ranker

                                        # Mock filter validation
                                        mock_validate_filter.return_value = {
                                            "status": True,
                                            "validated_collections": ["test_collection"]
                                        }

                                        # Mock ThreadPoolExecutor - use the same pattern as working tests
                                        mock_future = Mock()
                                        mock_future.result.return_value = [Mock(page_content="test content", metadata={})]
                                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                        # Mock RunnableAssign
                                        mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                        # Call search with empty filter expression
                                        result = rag.search(
                                            "test query",
                                            collection_names=["test_collection"],
                                            filter_expr=""  # Empty filter expression
                                        )

                                        # Verify result
                                        assert isinstance(result, Citations)
                                        # Verify debug logging was called for empty filter
                                        mock_logger.debug.assert_called()

    def test_search_with_whitespace_filter_expression(self):
        """Test search with whitespace-only filter expression to cover lines 511-517."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op', return_value=mock_vdb_op):
            with patch('nvidia_rag.rag_server.main.prepare_citations', return_value=Citations(documents=[], sources=[])):
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                            with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence', return_value=[Mock(page_content="test content", metadata={})]):
                                    with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                        # Mock the ranker
                                        mock_ranker = Mock()
                                        mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                        mock_get_ranking.return_value = mock_ranker

                                        # Mock filter validation
                                        mock_validate_filter.return_value = {
                                            "status": True,
                                            "validated_collections": ["test_collection"]
                                        }

                                        # Mock ThreadPoolExecutor - use the same pattern as working tests
                                        mock_future = Mock()
                                        mock_future.result.return_value = [Mock(page_content="test content", metadata={})]
                                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                        # Mock RunnableAssign
                                        mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                        # Call search with whitespace-only filter expression
                                        result = rag.search(
                                            "test query",
                                            collection_names=["test_collection"],
                                            filter_expr="   "  # Whitespace-only filter expression
                                        )

                                        # Verify result
                                        assert isinstance(result, Citations)
                                        # Verify debug logging was called for empty filter
                                        mock_logger.debug.assert_called()

    def test_search_with_none_filter_expression(self):
        """Test search with None filter expression to cover lines 511-517."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_langchain_vectorstore.return_value = Mock()
        mock_vdb_op.retrieval_langchain.return_value = [Mock(page_content="test content", metadata={})]

        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        with patch.object(rag, '_NvidiaRAG__prepare_vdb_op', return_value=mock_vdb_op):
            with patch('nvidia_rag.rag_server.main.prepare_citations', return_value=Citations(documents=[], sources=[])):
                with patch('nvidia_rag.rag_server.main.get_ranking_model') as mock_get_ranking:
                    with patch('nvidia_rag.rag_server.main.validate_filter_expr') as mock_validate_filter:
                        with patch('nvidia_rag.rag_server.main.ThreadPoolExecutor') as mock_executor:
                            with patch('nvidia_rag.rag_server.main.RunnableAssign') as mock_runnable_assign:
                                with patch('nvidia_rag.rag_server.main.filter_documents_by_confidence', return_value=[Mock(page_content="test content", metadata={})]):
                                    with patch('nvidia_rag.rag_server.main.logger') as mock_logger:
                                        # Mock the ranker
                                        mock_ranker = Mock()
                                        mock_ranker.compress_documents.return_value = {"context": [Mock(page_content="test content", metadata={})]}
                                        mock_get_ranking.return_value = mock_ranker

                                        # Mock filter validation
                                        mock_validate_filter.return_value = {
                                            "status": True,
                                            "validated_collections": ["test_collection"]
                                        }

                                        # Mock ThreadPoolExecutor - use the same pattern as working tests
                                        mock_future = Mock()
                                        mock_future.result.return_value = [Mock(page_content="test content", metadata={})]
                                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                                        # Mock RunnableAssign
                                        mock_runnable_assign.return_value.invoke.return_value = {"context": [Mock(page_content="test content", metadata={})]}

                                        # Call search with None filter expression
                                        result = rag.search(
                                            "test query",
                                            collection_names=["test_collection"],
                                            filter_expr=None  # None filter expression
                                        )

                                        # Verify result
                                        assert isinstance(result, Citations)
                                        # Verify debug logging was called for empty filter
                                        mock_logger.debug.assert_called()


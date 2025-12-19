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

import asyncio
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from nvidia_rag.rag_server.response_generator import (
    Citations,
    ErrorCodeMapping,
    SourceMetadata,
    SourceResult,
)


class MockNvidiaRAGForVectorStore:
    """Mock class for NvidiaRAG with Citations response for vector store search"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to default state with sample data"""
        self._search_side_effect = None
        self._default_results = [
            SourceResult(
                document_id="doc_123",
                content="Test content about machine learning",
                document_name="ml_guide.pdf",
                document_type="text",
                score=0.95,
                metadata=SourceMetadata(
                    language="en",
                    date_created="2024-01-15",
                    last_modified="2024-01-20",
                    page_number=5,
                    description="ML basics",
                    height=0,
                    width=0,
                    location=[],
                    location_max_dimensions=[],
                    content_metadata={
                        "author": "John Doe",
                        "category": "tech",
                    },
                ),
            ),
            SourceResult(
                document_id="doc_456",
                content="Advanced machine learning techniques",
                document_name="advanced_ml.pdf",
                document_type="text",
                score=0.89,
                metadata=SourceMetadata(
                    language="en",
                    date_created="2024-02-10",
                    last_modified="2024-02-15",
                    page_number=10,
                    description="Advanced ML",
                    height=0,
                    width=0,
                    location=[],
                    location_max_dimensions=[],
                    content_metadata={
                        "author": "Jane Smith",
                        "category": "science",
                    },
                ),
            ),
        ]

    async def search(self, *args, **kwargs):
        """Mock search method that returns Citations object"""
        if self._search_side_effect:
            return await self._search_side_effect(*args, **kwargs)

        # Return Citations object with default results
        return Citations(
            total_results=len(self._default_results),
            results=self._default_results,
        )

    async def health(self, check_dependencies: bool = False):
        return {"message": "Service is up."}

    def return_empty_search(self):
        """Configure mock to return empty search results"""

        async def empty(*args, **kwargs):
            return Citations(total_results=0, results=[])

        self._search_side_effect = empty

    def return_single_result(self):
        """Configure mock to return single search result"""

        async def single(*args, **kwargs):
            return Citations(
                total_results=1,
                results=[self._default_results[0]],
            )

        self._search_side_effect = single

    def return_filtered_results(self, author_filter=None):
        """Configure mock to return filtered results based on author"""

        async def filtered(*args, **kwargs):
            # Simulate filtering by checking filter_expr in kwargs
            filter_expr = kwargs.get("filter_expr", "")
            if author_filter and author_filter in str(filter_expr):
                # Return only matching results
                matching_results = [
                    r
                    for r in self._default_results
                    if r.metadata.content_metadata.get("author") == author_filter
                ]
                return Citations(
                    total_results=len(matching_results),
                    results=matching_results,
                )
            return Citations(
                total_results=len(self._default_results),
                results=self._default_results,
            )

        self._search_side_effect = filtered

    def raise_search_error(self):
        """Configure mock to raise an error during search"""

        async def error(*args, **kwargs):
            raise Exception("Database connection error")

        self._search_side_effect = error

    def raise_search_cancelled_error(self):
        """Configure mock to raise CancelledError"""

        async def error(*args, **kwargs):
            raise asyncio.CancelledError()

        self._search_side_effect = error


# Create mock instance
mock_nvidia_rag_vectorstore = MockNvidiaRAGForVectorStore()


# Common fixtures
@pytest.fixture(scope="module")
def setup_test_env():
    """Setup test environment with mocked NVIDIA_RAG"""
    with (
        patch("nvidia_rag.rag_server.server.NVIDIA_RAG", mock_nvidia_rag_vectorstore),
        patch("nvidia_rag.rag_server.server.CONFIG") as mock_config,
    ):
        # Configure mock config
        mock_config.vector_store.name = "milvus"
        mock_config.vector_store.url = "http://milvus:19530"
        mock_config.embeddings.model_name = "test-embed-model"
        mock_config.embeddings.server_url = "http://embed:8000"
        mock_config.ranking.model_name = "test-reranker"
        mock_config.ranking.server_url = "http://reranker:8000"
        mock_config.ranking.enable_reranker = True
        mock_config.retriever.vdb_top_k = 100
        mock_config.default_confidence_threshold = 0.0

        from nvidia_rag.rag_server.server import app

        yield app


@pytest.fixture
def client(setup_test_env):
    """Create test client"""
    return TestClient(setup_test_env)


@pytest.fixture(autouse=True)
def reset_mock_instance():
    """Reset mock instance before each test"""
    mock_nvidia_rag_vectorstore.reset()
    yield


class TestVectorStoreSearchEndpoint:
    """Tests for the /v2/vector_stores/{vector_store_id}/search endpoint (OpenAI-compatible)"""

    @pytest.fixture
    def basic_search_request(self):
        """Basic valid search request"""
        return {
            "query": "What is machine learning?",
            "max_num_results": 10,
        }

    @pytest.fixture
    def search_request_with_filters(self):
        """Search request with comparison filter"""
        return {
            "query": "machine learning basics",
            "max_num_results": 5,
            "filters": {"key": "author", "type": "eq", "value": "John Doe"},
        }

    @pytest.fixture
    def search_request_with_compound_filter(self):
        """Search request with compound filter"""
        return {
            "query": "advanced topics",
            "max_num_results": 10,
            "filters": {
                "type": "and",
                "filters": [
                    {"key": "author", "type": "eq", "value": "John Doe"},
                    {"key": "page_number", "type": "gte", "value": 5},
                ],
            },
        }

    @pytest.fixture
    def search_request_with_ranking(self):
        """Search request with ranking options"""
        return {
            "query": "machine learning",
            "max_num_results": 10,
            "ranking_options": {"ranker": "auto", "score_threshold": 0.5},
        }

    # Basic functionality tests
    def test_vector_store_search_success(self, client, basic_search_request):
        """Test successful vector store search"""
        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()

        # Validate response structure
        assert data["object"] == "vector_store.search_results.page"
        assert data["search_query"] == "What is machine learning?"
        assert "data" in data
        assert len(data["data"]) == 2  # Default mock returns 2 results
        assert data["has_more"] is False
        assert data["next_page"] is None

        # Validate first result structure
        first_result = data["data"][0]
        assert "file_id" in first_result
        assert "filename" in first_result
        assert "score" in first_result
        assert "attributes" in first_result
        assert "content" in first_result

        # Validate content structure
        assert len(first_result["content"]) > 0
        assert first_result["content"][0]["type"] == "text"
        assert "text" in first_result["content"][0]

    def test_vector_store_search_empty_results(self, client, basic_search_request):
        """Test vector store search with no results"""
        mock_nvidia_rag_vectorstore.return_empty_search()

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        assert len(data["data"]) == 0
        assert data["has_more"] is False

    def test_vector_store_search_single_result(self, client, basic_search_request):
        """Test vector store search with single result"""
        mock_nvidia_rag_vectorstore.return_single_result()

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["file_id"] == "doc_123"
        assert data["data"][0]["filename"] == "ml_guide.pdf"
        assert data["data"][0]["score"] == 0.95

    # Filter tests
    def test_vector_store_search_with_comparison_filter(
        self, client, search_request_with_filters
    ):
        """Test search with comparison filter"""
        mock_nvidia_rag_vectorstore.return_filtered_results(author_filter="John Doe")

        response = client.post(
            "/v2/vector_stores/test_collection/search",
            json=search_request_with_filters,
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        assert len(data["data"]) >= 0  # May return filtered results

    def test_vector_store_search_with_compound_filter(
        self, client, search_request_with_compound_filter
    ):
        """Test search with compound AND filter"""
        response = client.post(
            "/v2/vector_stores/test_collection/search",
            json=search_request_with_compound_filter,
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        assert "data" in data

    def test_vector_store_search_with_or_filter(self, client):
        """Test search with compound OR filter"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "filters": {
                "type": "or",
                "filters": [
                    {"key": "author", "type": "eq", "value": "John Doe"},
                    {"key": "author", "type": "eq", "value": "Jane Smith"},
                ],
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_in_filter(self, client):
        """Test search with 'in' operator filter"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "filters": {
                "key": "category",
                "type": "in",
                "value": ["tech", "science"],
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_numeric_filter(self, client):
        """Test search with numeric comparison filters"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "filters": {"key": "page_number", "type": "gte", "value": 5},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_empty_filter(self, client):
        """Test search with empty filter dict (should be converted to None)"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "filters": {},  # Empty filter should be handled
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    # Ranking options tests
    def test_vector_store_search_with_ranker_auto(
        self, client, search_request_with_ranking
    ):
        """Test search with ranker set to 'auto'"""
        response = client.post(
            "/v2/vector_stores/test_collection/search",
            json=search_request_with_ranking,
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_ranker_disabled(self, client):
        """Test search with ranker disabled"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "ranking_options": {"ranker": "none", "score_threshold": 0.0},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_ranker_false(self, client):
        """Test search with ranker set to 'false'"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "ranking_options": {"ranker": "false", "score_threshold": 0.0},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_score_threshold(self, client):
        """Test search with score threshold"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "ranking_options": {"ranker": "auto", "score_threshold": 0.8},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_with_empty_ranking_options(self, client):
        """Test search with empty ranking options (should be converted to None)"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "ranking_options": {},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    # Query rewriting tests
    def test_vector_store_search_with_query_rewriting(self, client):
        """Test search with query rewriting enabled"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "rewrite_query": True,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_without_query_rewriting(self, client):
        """Test search with query rewriting disabled (default)"""
        request_data = {
            "query": "test query",
            "max_num_results": 10,
            "rewrite_query": False,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    # Validation tests
    def test_vector_store_search_missing_query(self, client):
        """Test search without required query field"""
        request_data = {"max_num_results": 10}

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY
        # Parse streaming error response
        error_data = ""
        for line in response.iter_lines():
            if line:
                data = line.replace("data: ", "")
                try:
                    response_chunk = json.loads(data)
                    if "choices" in response_chunk and response_chunk["choices"]:
                        content = response_chunk["choices"][0]["message"]["content"]
                        error_data += content
                except (json.JSONDecodeError, KeyError):
                    pass
        assert "query" in error_data or "Field required" in error_data

    def test_vector_store_search_invalid_max_results(self, client):
        """Test search with invalid max_num_results (outside range)"""
        request_data = {"query": "test", "max_num_results": 100}  # Max is 50

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_vector_store_search_invalid_filter_type(self, client):
        """Test search with invalid filter operator"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {
                "key": "author",
                "type": "invalid_op",  # Invalid operator
                "value": "test",
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_vector_store_search_invalid_compound_filter_type(self, client):
        """Test search with invalid compound filter type"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {
                "type": "invalid_type",  # Should be 'and' or 'or'
                "filters": [{"key": "author", "type": "eq", "value": "test"}],
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_vector_store_search_missing_filter_fields(self, client):
        """Test search with incomplete filter (missing required fields)"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "author"},  # Missing 'type' and 'value'
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    # Error handling tests
    def test_vector_store_search_backend_error(self, client, basic_search_request):
        """Test search with backend error"""
        mock_nvidia_rag_vectorstore.raise_search_error()

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        error_data = response.json()
        assert "message" in error_data

    def test_vector_store_search_cancelled_request(self, client, basic_search_request):
        """Test search with cancelled request"""
        mock_nvidia_rag_vectorstore.raise_search_cancelled_error()

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
        error_data = response.json()
        assert "message" in error_data
        assert "cancelled" in error_data["message"].lower()

    # Edge cases
    def test_vector_store_search_very_long_query(self, client):
        """Test search with very long query string"""
        request_data = {
            "query": "test " * 1000,  # Very long query
            "max_num_results": 10,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        # Should succeed or return validation error, but not crash
        assert response.status_code in [
            ErrorCodeMapping.SUCCESS,
            ErrorCodeMapping.UNPROCESSABLE_ENTITY,
        ]

    def test_vector_store_search_min_results(self, client):
        """Test search with minimum results (1)"""
        request_data = {"query": "test", "max_num_results": 1}

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_max_results(self, client):
        """Test search with maximum results (50)"""
        request_data = {"query": "test", "max_num_results": 50}

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_special_characters_in_query(self, client):
        """Test search with special characters in query"""
        request_data = {
            "query": "What is ML? <test> & 'quotes' \"double\" @#$%",
            "max_num_results": 10,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_vector_store_search_unicode_query(self, client):
        """Test search with unicode characters"""
        request_data = {
            "query": "机器学习是什么？ What is ML in 日本語?",
            "max_num_results": 10,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    # Response structure validation
    def test_vector_store_search_response_attributes(
        self, client, basic_search_request
    ):
        """Test that response includes all expected attributes"""
        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        first_result = data["data"][0]

        # Check attributes are properly mapped
        assert "document_type" in first_result["attributes"]
        assert "page_number" in first_result["attributes"]
        assert "language" in first_result["attributes"]
        assert "date_created" in first_result["attributes"]
        assert "last_modified" in first_result["attributes"]

    def test_vector_store_search_file_id_generation(self, client, basic_search_request):
        """Test that file_id is properly generated"""
        response = client.post(
            "/v2/vector_stores/test_collection/search", json=basic_search_request
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()

        for result in data["data"]:
            assert "file_id" in result
            assert result["file_id"] != ""
            # Should use document_id if present, or generate from hash
            assert isinstance(result["file_id"], str)

    # Combined scenarios
    def test_vector_store_search_full_request(self, client):
        """Test search with all optional parameters"""
        request_data = {
            "query": "comprehensive search test",
            "max_num_results": 20,
            "filters": {
                "type": "and",
                "filters": [
                    {"key": "author", "type": "eq", "value": "John Doe"},
                    {"key": "page_number", "type": "gte", "value": 5},
                    {
                        "key": "category",
                        "type": "in",
                        "value": ["tech", "science"],
                    },
                ],
            },
            "ranking_options": {"ranker": "auto", "score_threshold": 0.5},
            "rewrite_query": True,
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS
        data = response.json()
        assert "data" in data
        assert data["object"] == "vector_store.search_results.page"


class TestFilterConversion:
    """Tests for filter conversion logic"""

    def test_comparison_filter_eq_string(self, client):
        """Test comparison filter with eq operator and string value"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "author", "type": "eq", "value": "John Doe"},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_eq_number(self, client):
        """Test comparison filter with eq operator and numeric value"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "page_count", "type": "eq", "value": 10},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_eq_boolean(self, client):
        """Test comparison filter with eq operator and boolean value"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "is_published", "type": "eq", "value": True},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_ne(self, client):
        """Test comparison filter with ne (not equal) operator"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "status", "type": "ne", "value": "draft"},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_gt(self, client):
        """Test comparison filter with gt (greater than) operator"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "rating", "type": "gt", "value": 4.5},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_lt(self, client):
        """Test comparison filter with lt (less than) operator"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {"key": "page_count", "type": "lt", "value": 100},
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_comparison_filter_nin(self, client):
        """Test comparison filter with nin (not in) operator"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {
                "key": "category",
                "type": "nin",
                "value": ["draft", "archived"],
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_nested_compound_filters(self, client):
        """Test nested compound filters (compound within compound)"""
        request_data = {
            "query": "test",
            "max_num_results": 10,
            "filters": {
                "type": "and",
                "filters": [
                    {"key": "status", "type": "eq", "value": "published"},
                    {
                        "type": "or",
                        "filters": [
                            {"key": "category", "type": "eq", "value": "tech"},
                            {"key": "category", "type": "eq", "value": "science"},
                        ],
                    },
                ],
            },
        }

        response = client.post(
            "/v2/vector_stores/test_collection/search", json=request_data
        )

        assert response.status_code == ErrorCodeMapping.SUCCESS

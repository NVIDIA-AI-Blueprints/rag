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
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests
from fastapi import Request
from fastapi.testclient import TestClient
from pymilvus.exceptions import MilvusException

from nvidia_rag.rag_server.response_generator import (
    APIError,
    ErrorCodeMapping,
    RAGResponse,
)
from nvidia_rag.rag_server.server import (
    _extract_vdb_auth_token,
    validate_confidence_threshold_field,
)
from nvidia_rag.utils.health_models import RAGHealthResponse


class MockNvidiaRAG:
    """Mock class for NvidiaRAG with configurable responses and error states"""

    def __init__(self):
        self.reset()  # Initialize with reset to set up default state

    def reset(self):
        self.rag_contexts = [
            Mock(
                metadata={
                    "source": {"source_id": "test.pdf"},
                    "content_metadata": {"type": "text"},
                },
                page_content="Test content",
            )
        ]
        # Use the same content for both RAG and LLM for test consistency
        self.rag_generator_items = ["Hello", " world", "!"]
        self.llm_generator_items = ["Hello", " world", "!"]
        self._generate_side_effect = None
        self._search_side_effect = None

    async def _async_gen(self, items):
        for item in items:
            yield f"data: {json.dumps({'choices': [{'message': {'content': item}}]})}\n"

    async def _async_error_gen(self, message):
        yield f"data: {json.dumps({'choices': [{'message': {'content': message}}]})}\n"

    async def generate(self, *args, **kwargs):
        if self._generate_side_effect:
            return self._generate_side_effect(*args, **kwargs)
        return RAGResponse(
            self._async_gen(self.rag_generator_items),
            status_code=ErrorCodeMapping.SUCCESS,
        )

    async def search(self, *args, **kwargs):
        if self._search_side_effect:
            return await self._search_side_effect(*args, **kwargs)
        return {
            "total_results": 1,
            "results": [
                {
                    "content": "Test content",
                    "metadata": {
                        "source": {"source_id": "test.pdf"},
                        "content_metadata": {"type": "text"},
                    },
                    "score": 0.95,
                }
            ],
        }

    async def health(self, check_dependencies: bool = False):
        return {"message": "Service is up."}

    def return_llm_response(self):
        def llm(*args, **kwargs):
            return RAGResponse(
                self._async_gen(self.llm_generator_items),
                status_code=ErrorCodeMapping.SUCCESS,
            )

        self._generate_side_effect = llm

    def return_llm_empty_response(self):
        def empty(*args, **kwargs):
            return RAGResponse(
                self._async_gen([]), status_code=ErrorCodeMapping.SUCCESS
            )

        self._generate_side_effect = empty

    def return_empty_response(self):
        def empty(*args, **kwargs):
            return RAGResponse(
                self._async_gen([]), status_code=ErrorCodeMapping.SUCCESS
            )

        self._generate_side_effect = empty

    def return_milvus_error(self):
        def error(*args, **kwargs):
            return RAGResponse(
                self._async_error_gen(
                    "Error from milvus server. Please ensure you have ingested some documents."
                ),
                status_code=ErrorCodeMapping.BAD_REQUEST,
            )

        self._generate_side_effect = error

    def return_general_error(self):
        def error(*args, **kwargs):
            return RAGResponse(
                self._async_error_gen(
                    "Error from rag server. Please check rag-server logs for more details."
                ),
                status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
            )

        self._generate_side_effect = error

    def return_llm_general_error(self):
        def error(*args, **kwargs):
            return RAGResponse(
                self._async_error_gen(
                    "Error from rag server. Please check rag-server logs for more details."
                ),
                status_code=ErrorCodeMapping.INTERNAL_SERVER_ERROR,
            )

        self._generate_side_effect = error

    def return_cancelled_error(self):
        def error(*args, **kwargs):
            return RAGResponse(
                self._async_error_gen("Request was cancelled by the client."),
                status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
            )

        self._generate_side_effect = error

    def return_llm_cancelled_error(self):
        def error(*args, **kwargs):
            return RAGResponse(
                self._async_error_gen("Request was cancelled by the client."),
                status_code=ErrorCodeMapping.CLIENT_CLOSED_REQUEST,
            )

        self._generate_side_effect = error

    # Search error methods
    def raise_search_milvus_error(self):
        async def error(*args, **kwargs):
            raise MilvusException("Milvus error")

        self._search_side_effect = error

    def raise_search_general_error(self):
        async def error(*args, **kwargs):
            raise Exception("Document search error")

        self._search_side_effect = error

    def raise_search_cancelled_error(self):
        async def error(*args, **kwargs):
            raise asyncio.CancelledError()

        self._search_side_effect = error

    def return_empty_search(self):
        async def empty(*args, **kwargs):
            return {"total_results": 0, "results": []}

        self._search_side_effect = empty


# Create mock instances
mock_nvidia_rag_instance = MockNvidiaRAG()


# Common fixtures
@pytest.fixture(scope="module")
def setup_test_env():
    """Setup test environment with all necessary mocks"""
    with patch("nvidia_rag.rag_server.server.NVIDIA_RAG", mock_nvidia_rag_instance):
        from nvidia_rag.rag_server.server import app

        yield app


@pytest.fixture
def client(setup_test_env):
    """Create test client"""
    return TestClient(setup_test_env)


@pytest.fixture
def valid_prompt_data():
    """Create valid test prompt data"""
    return {
        "messages": [{"role": "user", "content": "What is machine learning?"}],
        "use_knowledge_base": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
        "collection_name": "test_collection",
        "model": "test-model",
        "reranker_top_k": 4,
        "vdb_top_k": 10,
    }


@pytest.fixture(autouse=True)
def reset_mock_instance():
    """Reset mock instance before each test"""
    mock_nvidia_rag_instance.reset()
    yield


def read_streaming_response(response):
    """Helper function to read and concatenate streaming response content"""
    full_message = ""
    for line in response.iter_lines():
        if line:
            data = line.replace("data: ", "")
            response_chunk = json.loads(data)
            content = response_chunk["choices"][0]["message"]["content"]
            full_message += content
    return full_message


class TestGenerateEndpoint:
    """Tests for the /generate endpoint"""

    def test_generate_answer_rag_success(self, client, valid_prompt_data):
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS

        # Check first chunk (existing test)
        response_text = [
            line.replace("data: ", "") for line in response.iter_lines() if line
        ]
        assert len(response_text) > 0
        first_chunk = json.loads(response_text[0])
        assert first_chunk["choices"][0]["message"]["content"] == "Hello"

        # Check complete streamed response
        full_message = read_streaming_response(response)
        assert full_message == "Hello world!"  # Complete expected response

    def test_generate_answer_milvus_error(self, client, valid_prompt_data):
        mock_nvidia_rag_instance.return_milvus_error()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.BAD_REQUEST
        error_data = read_streaming_response(response)
        assert "Error from milvus server" in error_data

    def test_generate_answer_general_error(self, client, valid_prompt_data):
        mock_nvidia_rag_instance.return_general_error()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        error_data = read_streaming_response(response)
        assert "Error from rag server" in error_data

    def test_generate_answer_cancelled_request(self, client, valid_prompt_data):
        mock_nvidia_rag_instance.return_cancelled_error()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
        error_data = read_streaming_response(response)
        assert "Request was cancelled by the client" in error_data

    def test_generate_answer_invalid_prompt(self, client):
        invalid_data = {
            "messages": [{"role": "invalid_role", "content": "test content"}]
        }
        response = client.post("/v1/generate", json=invalid_data)
        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY
        assert any(
            "Input should be 'user', 'assistant', 'system' or None" in error["msg"]
            for error in response.json()["detail"]
        )

    def test_generate_answer_empty_generator(self, client, valid_prompt_data):
        """Test empty generator response"""
        mock_nvidia_rag_instance.return_empty_response()
        response = client.post("/v1/generate", json=valid_prompt_data)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        # Check the full streamed response
        full_message = read_streaming_response(response)
        assert full_message == ""  # Verify the concatenated content is empty

    def test_generate_answer_llm_success(self, client, valid_prompt_data):
        """Test successful LLM response with streaming"""
        valid_prompt_data["use_knowledge_base"] = False
        mock_nvidia_rag_instance.return_llm_response()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS

        # Check first chunk
        response_text = [
            line.replace("data: ", "") for line in response.iter_lines() if line
        ]
        assert len(response_text) > 0
        first_chunk = json.loads(response_text[0])
        assert first_chunk["choices"][0]["message"]["content"] == "Hello"

        # Check complete streamed response
        full_message = read_streaming_response(response)
        assert full_message == "Hello world!"  # Complete expected response

    def test_generate_answer_llm_empty_response(self, client, valid_prompt_data):
        """Test empty LLM response"""
        valid_prompt_data["use_knowledge_base"] = False
        mock_nvidia_rag_instance.return_llm_empty_response()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS
        full_message = read_streaming_response(response)
        assert full_message == ""

    def test_generate_answer_llm_general_error(self, client, valid_prompt_data):
        """Test LLM general error"""
        valid_prompt_data["use_knowledge_base"] = False
        mock_nvidia_rag_instance.return_llm_general_error()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        error_data = read_streaming_response(response)
        assert "Error from rag server" in error_data

    def test_generate_answer_llm_cancelled_request(self, client, valid_prompt_data):
        """Test LLM cancelled request"""
        valid_prompt_data["use_knowledge_base"] = False
        mock_nvidia_rag_instance.return_llm_cancelled_error()
        response = client.post("/v1/generate", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
        error_data = read_streaming_response(response)
        assert "Request was cancelled by the client" in error_data


class TestDocumentSearchEndpoint:
    """Tests for the /search endpoint"""

    @pytest.fixture
    def search_data(self):
        return {
            "query": "What is machine learning?",
            "reranker_top_k": 4,
            "vdb_top_k": 10,
            "collection_name": "test_collection",
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "enable_query_rewriting": True,
            "enable_reranker": True,
            "embedding_model": "test-embedding-model",
            "embedding_endpoint": "http://embedding:8000",
            "reranker_model": "test-reranker-model",
            "reranker_endpoint": "http://reranker:8000",
        }

    def test_document_search_success(self, client, search_data):
        response = client.post("/v1/search", json=search_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert "total_results" in response_data
        assert "results" in response_data
        assert len(response_data["results"]) > 0

    def test_document_search_empty_query(self, client):
        search_data = {
            "query": "",
            "reranker_top_k": 4,
            "vdb_top_k": 10,
            "collection_name": "test_collection",
            "messages": [{"role": "user", "content": ""}],
            "enable_query_rewriting": True,
            "enable_reranker": True,
        }
        mock_nvidia_rag_instance.return_empty_search()
        response = client.post("/v1/search", json=search_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS
        assert response.json()["total_results"] == 0

    def test_document_search_milvus_error(self, client, search_data):
        mock_nvidia_rag_instance.raise_search_milvus_error()
        response = client.post("/v1/search", json=search_data)
        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        error_data = response.json()
        assert "message" in error_data
        assert "Failed to search documents" in error_data["message"]

    def test_document_search_invalid_input(self, client):
        """Test document search with invalid input"""
        invalid_data = {
            "query": "What is machine learning?",
            "reranker_top_k": -1,  # Invalid value
            "vdb_top_k": 10,
            "collection_name": "test_collection",
            "messages": [{"role": "user", "content": "What is machine learning?"}],
        }

        response = client.post("/v1/search", json=invalid_data)
        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY
        assert "detail" in response.json()

    def test_document_search_cancelled_request(self, client, search_data):
        """Test document search with cancelled request"""
        mock_nvidia_rag_instance.raise_search_cancelled_error()
        response = client.post("/v1/search", json=search_data)

        assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
        error_data = response.json()
        assert "message" in error_data
        assert "Request was cancelled by the client" in error_data["message"]


class TestHealthEndpoint:
    """Tests for the /health endpoint"""

    def test_health_check(self, client):
        response = client.get("/v1/health")
        assert response.status_code == ErrorCodeMapping.SUCCESS
        assert response.json()["message"] == "Service is up."


class TestChatCompletionsEndpoint:
    """Tests for the OpenAI-compatible /chat/completions endpoint"""

    def test_chat_completions_endpoint(self, client, valid_prompt_data):
        response = client.post("/v1/chat/completions", json=valid_prompt_data)
        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_text = [
            line.replace("data: ", "") for line in response.iter_lines() if line
        ]
        assert len(response_text) > 0
        first_chunk = json.loads(response_text[0])
        assert first_chunk["choices"][0]["message"]["content"] == "Hello"


class TestExtractVdbAuthToken:
    """Tests for _extract_vdb_auth_token helper function"""

    def test_extract_vdb_auth_token_with_bearer(self):
        """Test extracting vdb_auth_token from Authorization header with Bearer prefix"""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer test_token_123"}

        result = _extract_vdb_auth_token(mock_request)
        assert result == "test_token_123"

    def test_extract_vdb_auth_token_with_lowercase_bearer(self):
        """Test extracting vdb_auth_token from Authorization header with lowercase bearer"""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "bearer test_token_456"}

        result = _extract_vdb_auth_token(mock_request)
        assert result == "test_token_456"

    def test_extract_vdb_auth_token_no_bearer(self):
        """Test extracting vdb_auth_token when no Bearer prefix"""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Basic test_token"}

        result = _extract_vdb_auth_token(mock_request)
        assert result is None

    def test_extract_vdb_auth_token_no_header(self):
        """Test extracting vdb_auth_token when no Authorization header"""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        result = _extract_vdb_auth_token(mock_request)
        assert result is None

    def test_extract_vdb_auth_token_with_spaces(self):
        """Test extracting vdb_auth_token with extra spaces"""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer  test_token_with_spaces  "}

        result = _extract_vdb_auth_token(mock_request)
        assert result == "test_token_with_spaces"


class TestVdbAuthTokenParameter:
    """Tests for vdb_auth_token parameter passing through endpoints"""

    @pytest.fixture
    def search_data(self):
        """Fixture for search endpoint test data"""
        return {
            "query": "What is machine learning?",
            "reranker_top_k": 4,
            "vdb_top_k": 10,
            "collection_name": "test_collection",
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "enable_query_rewriting": True,
            "enable_reranker": True,
            "embedding_model": "test-embedding-model",
            "embedding_endpoint": "http://embedding:8000",
            "reranker_model": "test-reranker-model",
            "reranker_endpoint": "http://reranker:8000",
        }

    def test_generate_with_vdb_auth_token(self, client, valid_prompt_data):
        """Test /generate endpoint passes vdb_auth_token to backend"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_response = RAGResponse(
                mock_nvidia_rag_instance._async_gen(["Hello"]),
                status_code=ErrorCodeMapping.SUCCESS,
            )
            mock_rag.generate = AsyncMock(return_value=mock_response)

            response = client.post(
                "/v1/generate",
                json=valid_prompt_data,
                headers={"Authorization": "Bearer test_vdb_token"},
            )

            assert response.status_code == ErrorCodeMapping.SUCCESS
            mock_rag.generate.assert_called_once()
            call_kwargs = mock_rag.generate.call_args[1]
            assert call_kwargs.get("vdb_auth_token") == "test_vdb_token"

    def test_search_with_vdb_auth_token(self, client, search_data):
        """Test /search endpoint passes vdb_auth_token to backend"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(
                return_value={"total_results": 1, "results": []}
            )

            response = client.post(
                "/v1/search",
                json=search_data,
                headers={"Authorization": "Bearer test_vdb_token"},
            )

            assert response.status_code == ErrorCodeMapping.SUCCESS
            mock_rag.search.assert_called_once()
            call_kwargs = mock_rag.search.call_args[1]
            assert call_kwargs.get("vdb_auth_token") == "test_vdb_token"


class TestServerErrorHandling:
    """Tests for error handling in server endpoints"""

    @pytest.fixture
    def search_data(self):
        """Fixture for search endpoint test data"""
        return {
            "query": "What is machine learning?",
            "reranker_top_k": 4,
            "vdb_top_k": 10,
            "collection_name": "test_collection",
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "enable_query_rewriting": True,
            "enable_reranker": True,
            "embedding_model": "test-embedding-model",
            "embedding_endpoint": "http://embedding:8000",
            "reranker_model": "test-reranker-model",
            "reranker_endpoint": "http://reranker:8000",
        }

    def test_generate_endpoint_value_error(self, client):
        """Test /generate endpoint handles ValueError"""
        invalid_data = {
            "messages": [],  # Empty messages should raise ValueError
            "collection_name": "test_collection",
        }

        response = client.post("/v1/generate", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY
        # For Pydantic validation errors, the response is JSON (not streaming)
        response_data = response.json()
        assert "detail" in response_data

    def test_generate_endpoint_api_error(self, client, valid_prompt_data):
        """Test /generate endpoint handles APIError"""

        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.generate = AsyncMock(
                side_effect=APIError("Test API error", status_code=503)
            )

            response = client.post("/v1/generate", json=valid_prompt_data)

            assert response.status_code == 503
            response_text = [
                line.replace("data: ", "") for line in response.iter_lines() if line
            ]
            assert len(response_text) > 0
            # Combine content from all chunks to get the full error message
            combined_content = ""
            for line in response_text:
                chunk = json.loads(line)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    content = chunk["choices"][0].get("message", {}).get("content", "")
                    combined_content += content
            assert "Test API error" in combined_content

    def test_generate_endpoint_general_exception(self, client, valid_prompt_data):
        """Test /generate endpoint handles general Exception"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.generate = AsyncMock(side_effect=RuntimeError("Unexpected error"))

            response = client.post("/v1/generate", json=valid_prompt_data)

            assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
            response_text = [
                line.replace("data: ", "") for line in response.iter_lines() if line
            ]
            assert len(response_text) > 0

    def test_generate_endpoint_cancelled_error(self, client, valid_prompt_data):
        """Test /generate endpoint handles CancelledError"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.generate = AsyncMock(side_effect=asyncio.CancelledError())

            response = client.post("/v1/generate", json=valid_prompt_data)

            assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
            response_data = response.json()
            assert "cancelled" in response_data["message"].lower()

    def test_search_endpoint_connection_error(self, client, search_data):
        """Test /search endpoint handles connection errors"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(
                side_effect=requests.exceptions.ConnectionError("Connection failed")
            )

            response = client.post("/v1/search", json=search_data)

            assert response.status_code == ErrorCodeMapping.SERVICE_UNAVAILABLE
            response_data = response.json()
            assert "Service unavailable" in response_data["message"]

    def test_search_endpoint_api_error(self, client, search_data):
        """Test /search endpoint handles APIError"""

        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(
                side_effect=APIError("Search failed", status_code=500)
            )

            response = client.post("/v1/search", json=search_data)

            assert response.status_code == 500
            response_data = response.json()
            assert "Search failed" in response_data["message"]

    def test_search_endpoint_cancelled_error(self, client, search_data):
        """Test /search endpoint handles CancelledError"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(side_effect=asyncio.CancelledError())

            response = client.post("/v1/search", json=search_data)

            assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
            response_data = response.json()
            assert "cancelled" in response_data["message"].lower()


class TestServerValidation:
    """Tests for validation logic in server"""

    def test_validate_confidence_threshold_negative(self):
        """Test validate_confidence_threshold_field with negative value"""
        with pytest.raises(ValueError, match="confidence_threshold must be >= 0.0"):
            validate_confidence_threshold_field(-0.1)

    def test_validate_confidence_threshold_too_large(self):
        """Test validate_confidence_threshold_field with value > 1.0"""
        with pytest.raises(ValueError, match="confidence_threshold must be <= 1.0"):
            validate_confidence_threshold_field(1.1)

    def test_generate_endpoint_empty_messages(self, client):
        """Test /generate endpoint rejects empty messages"""
        invalid_data = {
            "messages": [],
            "collection_name": "test_collection",
        }

        response = client.post("/v1/generate", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_generate_endpoint_no_user_message(self, client):
        """Test /generate endpoint rejects messages without user role"""
        invalid_data = {
            "messages": [{"role": "assistant", "content": "Hello"}],
            "collection_name": "test_collection",
        }

        response = client.post("/v1/generate", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_generate_endpoint_last_message_not_user(self, client):
        """Test /generate endpoint rejects when last message is not user"""
        invalid_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "collection_name": "test_collection",
        }

        response = client.post("/v1/generate", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_search_endpoint_no_user_message(self, client):
        """Test /search endpoint rejects messages without user role"""
        invalid_data = {
            "query": "test query",
            "messages": [{"role": "assistant", "content": "Hello"}],
            "collection_name": "test_collection",
        }

        response = client.post("/v1/search", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_search_endpoint_last_message_not_user(self, client):
        """Test /search endpoint rejects when last message is not user"""
        invalid_data = {
            "query": "test query",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "collection_name": "test_collection",
        }

        response = client.post("/v1/search", json=invalid_data)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY


class TestServerConfigurationEndpoint:
    """Tests for /configuration endpoint"""

    def test_configuration_endpoint_success(self, client):
        """Test /configuration endpoint returns configuration"""
        with patch("nvidia_rag.rag_server.server.CONFIG") as mock_config:
            mock_config.llm.get_model_parameters.return_value = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
            }
            mock_config.retriever.vdb_top_k = 10
            mock_config.retriever.top_k = 5
            mock_config.default_confidence_threshold = 0.5
            mock_config.ranking.enable_reranker = True
            mock_config.enable_citations = True
            mock_config.enable_guardrails = False
            mock_config.query_rewriter.enable_query_rewriter = True
            mock_config.enable_vlm_inference = False
            mock_config.filter_expression_generator.enable_filter_generator = True
            mock_config.llm.model_name = '"test-llm"'
            mock_config.embeddings.model_name = '"test-embedding"'
            mock_config.ranking.model_name = '"test-reranker"'
            mock_config.vlm.model_name = '"test-vlm"'
            mock_config.llm.server_url = '"http://llm:8000"'
            mock_config.embeddings.server_url = '"http://embedding:8000"'
            mock_config.ranking.server_url = '"http://reranker:8000"'
            mock_config.vlm.server_url = '"http://vlm:8000"'
            mock_config.vector_store.url = "http://vdb:19530"

            response = client.get("/v1/configuration")

            assert response.status_code == ErrorCodeMapping.SUCCESS
            data = response.json()
            assert "rag_configuration" in data
            assert "feature_toggles" in data
            assert "models" in data
            assert "endpoints" in data

    def test_configuration_endpoint_error(self, client):
        """Test /configuration endpoint handles errors"""
        with patch("nvidia_rag.rag_server.server.CONFIG") as mock_config:
            mock_config.llm.get_model_parameters.side_effect = Exception("Config error")

            response = client.get("/v1/configuration")

            assert response.status_code == 500
            data = response.json()
            assert "Error fetching configuration" in data["detail"]


class TestServerHealthEndpoint:
    """Tests for /health endpoint with dependencies"""

    def test_health_endpoint_with_dependencies_error(self, client):
        """Test /health endpoint handles print_health_report errors"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.health = AsyncMock(
                return_value=RAGHealthResponse(message="Service is up.")
            )
            with patch(
                "nvidia_rag.rag_server.server.print_health_report"
            ) as mock_print:
                mock_print.side_effect = Exception("Print error")

                response = client.get("/v1/health?check_dependencies=true")

                assert response.status_code == ErrorCodeMapping.SUCCESS
                assert response.json()["message"] == "Service is up."

    def test_health_endpoint_without_dependencies(self, client):
        """Test /health endpoint without dependencies check"""
        response = client.get("/v1/health?check_dependencies=false")

        assert response.status_code == ErrorCodeMapping.SUCCESS
        assert response.json()["message"] == "Service is up."


class TestServerSummaryEndpoint:
    """Tests for /summary endpoint error handling"""

    def test_summary_endpoint_unknown_status(self, client):
        """Test /summary endpoint handles unknown status"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.get_summary = AsyncMock(
                return_value={"status": "UNKNOWN_STATUS", "message": "Test"}
            )

            response = client.get(
                "/v1/summary",
                params={
                    "collection_name": "test_collection",
                    "file_name": "test.pdf",
                    "blocking": False,
                    "timeout": 300,
                },
            )

            assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR

    def test_summary_endpoint_cancelled_error(self, client):
        """Test /summary endpoint handles CancelledError"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.get_summary = AsyncMock(side_effect=asyncio.CancelledError())

            response = client.get(
                "/v1/summary",
                params={
                    "collection_name": "test_collection",
                    "file_name": "test.pdf",
                    "blocking": False,
                    "timeout": 300,
                },
            )

            assert response.status_code == ErrorCodeMapping.CLIENT_CLOSED_REQUEST
            response_data = response.json()
            assert "cancelled" in response_data["message"].lower()


class TestServerMessageProcessing:
    """Tests for message processing in generate and search endpoints"""

    def test_generate_endpoint_list_content_text(self, client):
        """Test /generate endpoint handles list content with text type"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_response = RAGResponse(
                mock_nvidia_rag_instance._async_gen(["Hello"]),
                status_code=ErrorCodeMapping.SUCCESS,
            )
            mock_rag.generate = AsyncMock(return_value=mock_response)

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is AI?"},
                        ],
                    }
                ],
                "collection_name": "test_collection",
            }

            response = client.post("/v1/generate", json=data)

            assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_generate_endpoint_list_content_image_url(self, client):
        """Test /generate endpoint handles list content with image_url type"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_response = RAGResponse(
                mock_nvidia_rag_instance._async_gen(["Hello"]),
                status_code=ErrorCodeMapping.SUCCESS,
            )
            mock_rag.generate = AsyncMock(return_value=mock_response)

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,test"},
                            },
                        ],
                    }
                ],
                "collection_name": "test_collection",
            }

            response = client.post("/v1/generate", json=data)

            assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_generate_endpoint_list_content_fallback(self, client):
        """Test /generate endpoint handles list content fallback"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_response = RAGResponse(
                mock_nvidia_rag_instance._async_gen(["Hello"]),
                status_code=ErrorCodeMapping.SUCCESS,
            )
            mock_rag.generate = AsyncMock(return_value=mock_response)

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"unknown": "value"}],
                    }
                ],
                "collection_name": "test_collection",
            }

            response = client.post("/v1/generate", json=data)

            assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_search_endpoint_list_query(self, client):
        """Test /search endpoint handles list query"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(
                return_value={"total_results": 1, "results": []}
            )

            data = {
                "query": [{"type": "text", "text": "test query"}],
                "messages": [{"role": "user", "content": "test"}],
                "collection_name": "test_collection",
            }

            response = client.post("/v1/search", json=data)

            assert response.status_code == ErrorCodeMapping.SUCCESS

    def test_search_endpoint_list_query_image_url(self, client):
        """Test /search endpoint handles list query with image_url"""
        with patch("nvidia_rag.rag_server.server.NVIDIA_RAG") as mock_rag:
            mock_rag.search = AsyncMock(
                return_value={"total_results": 1, "results": []}
            )

            data = {
                "query": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,test"},
                    }
                ],
                "messages": [{"role": "user", "content": "test"}],
                "collection_name": "test_collection",
            }

            response = client.post("/v1/search", json=data)

            assert response.status_code == ErrorCodeMapping.SUCCESS

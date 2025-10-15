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
from fastapi.testclient import TestClient
from pymilvus.exceptions import MilvusException

from nvidia_rag.rag_server.response_generator import ErrorCodeMapping, RAGResponse


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

    def generate(self, *args, **kwargs):
        if self._generate_side_effect:
            return self._generate_side_effect(*args, **kwargs)
        return RAGResponse(
            self._async_gen(self.rag_generator_items),
            status_code=ErrorCodeMapping.SUCCESS,
        )

    def search(self, *args, **kwargs):
        if self._search_side_effect:
            return self._search_side_effect(*args, **kwargs)
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
        def error(*args, **kwargs):
            raise MilvusException("Milvus error")

        self._search_side_effect = error

    def raise_search_general_error(self):
        def error(*args, **kwargs):
            raise Exception("Document search error")

        self._search_side_effect = error

    def raise_search_cancelled_error(self):
        def error(*args, **kwargs):
            raise asyncio.CancelledError()

        self._search_side_effect = error

    def return_empty_search(self):
        def empty(*args, **kwargs):
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
        assert "Error occurred while searching documents" in error_data["message"]

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

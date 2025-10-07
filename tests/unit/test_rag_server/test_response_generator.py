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

import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any

import pytest
from pydantic import ValidationError
from pymilvus.exceptions import MilvusException, MilvusUnavailableException

from nvidia_rag.rag_server.response_generator import (
    Usage,
    SourceMetadata,
    SourceResult,
    Citations,
    TextContent,
    ImageUrl,
    ImageContent,
    Message,
    ChainResponseChoices,
    Metrics,
    ChainResponse,
    prepare_llm_request,
    prepare_citations,
    error_response_generator,
    _is_empty_content,
    escape_json_content_multimodal,
    escape_json_content,
    generate_answer,
    retrieve_summary,
    FALLBACK_EXCEPTION_MSG,
)


class TestUsage:
    """Test Usage model"""

    def test_usage_default_values(self):
        """Test Usage with default values"""
        usage = Usage()
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_usage_custom_values(self):
        """Test Usage with custom values"""
        usage = Usage(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50
        )
        assert usage.total_tokens == 100
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 50

    def test_usage_validation_negative_tokens(self):
        """Test Usage validation with negative tokens"""
        with pytest.raises(ValidationError):
            Usage(total_tokens=-1)

    def test_usage_validation_excessive_tokens(self):
        """Test Usage validation with excessive tokens"""
        with pytest.raises(ValidationError):
            Usage(total_tokens=1000000001)


class TestSourceMetadata:
    """Test SourceMetadata model"""

    def test_source_metadata_default_values(self):
        """Test SourceMetadata with default values"""
        metadata = SourceMetadata()
        assert metadata.language == ""
        assert metadata.date_created == ""
        assert metadata.last_modified == ""
        assert metadata.page_number == 0
        assert metadata.description == ""
        assert metadata.height == 0
        assert metadata.width == 0
        assert metadata.location == []
        assert metadata.location_max_dimensions == []
        assert metadata.content_metadata == {}

    def test_source_metadata_custom_values(self):
        """Test SourceMetadata with custom values"""
        metadata = SourceMetadata(
            language="en",
            page_number=1,
            height=800,
            width=600,
            description="Test document",
            location=[0.1, 0.2, 0.3, 0.4],
            content_metadata={"type": "text"}
        )
        assert metadata.language == "en"
        assert metadata.page_number == 1
        assert metadata.height == 800
        assert metadata.width == 600
        assert metadata.description == "Test document"
        assert metadata.location == [0.1, 0.2, 0.3, 0.4]
        assert metadata.content_metadata == {"type": "text"}

    def test_source_metadata_validation_page_number_negative(self):
        """Test SourceMetadata validation with negative page number"""
        with pytest.raises(ValidationError):
            SourceMetadata(page_number=-2)

    def test_source_metadata_validation_page_number_excessive(self):
        """Test SourceMetadata validation with excessive page number"""
        with pytest.raises(ValidationError):
            SourceMetadata(page_number=1000001)


class TestTextContent:
    """Test TextContent model"""

    def test_text_content_custom_values(self):
        """Test TextContent with custom values"""
        content = TextContent(text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"

    def test_text_content_validation_required_text(self):
        """Test TextContent validation requires text field"""
        with pytest.raises(ValidationError):
            TextContent()


class TestImageUrl:
    """Test ImageUrl model"""

    def test_image_url_custom_values(self):
        """Test ImageUrl with custom values"""
        image_url = ImageUrl(url="data:image/png;base64,test")
        assert image_url.url == "data:image/png;base64,test"
        assert image_url.detail == "auto"

    def test_image_url_validation_required_url(self):
        """Test ImageUrl validation requires url field"""
        with pytest.raises(ValidationError):
            ImageUrl()


class TestImageContent:
    """Test ImageContent model"""

    def test_image_content_custom_values(self):
        """Test ImageContent with custom values"""
        image_url = ImageUrl(url="data:image/png;base64,test")
        content = ImageContent(image_url=image_url)
        assert content.type == "image_url"
        assert content.image_url.url == "data:image/png;base64,test"

    def test_image_content_validation_required_image_url(self):
        """Test ImageContent validation requires image_url field"""
        with pytest.raises(ValidationError):
            ImageContent()


class TestMessage:
    """Test Message model"""

    def test_message_default_values(self):
        """Test Message with default values"""
        message = Message()
        assert message.role == "user"
        assert message.content == "Hello! What can you help me with?"

    def test_message_text_content(self):
        """Test Message with text content"""
        message = Message(role="user", content="Hello world")
        assert message.role == "user"
        assert message.content == "Hello world"

    def test_message_multimodal_content(self):
        """Test Message with multimodal content"""
        text_content = TextContent(text="What is this?")
        image_content = ImageContent(image_url=ImageUrl(url="data:image/png;base64,test"))
        message = Message(
            role="user",
            content=[text_content, image_content]
        )
        assert message.role == "user"
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)

    def test_message_role_validation_valid(self):
        """Test Message role validation with valid roles"""
        for role in ["user", "assistant", "system"]:
            message = Message(role=role)
            assert message.role == role

    def test_message_role_validation_invalid(self):
        """Test Message role validation with invalid role"""
        with pytest.raises(ValidationError):
            Message(role="invalid")

    def test_message_role_validation_case_insensitive(self):
        """Test Message role validation is case sensitive"""
        with pytest.raises(ValidationError):
            Message(role="USER")

    def test_message_content_sanitization_text(self):
        """Test Message content sanitization for text"""
        message = Message(content="<script>alert('xss')</script>Hello")
        assert message.content == "alert('xss')Hello"

    def test_message_content_sanitization_multimodal(self):
        """Test Message content sanitization for multimodal content"""
        text_content = TextContent(text="<script>alert('xss')</script>Hello")
        image_content = ImageContent(image_url=ImageUrl(url="data:image/png;base64,test"))
        message = Message(content=[text_content, image_content])
        assert message.content[0].text == "alert('xss')Hello"
        assert message.content[1].image_url.url == "data:image/png;base64,test"


class TestChainResponseChoices:
    """Test ChainResponseChoices model"""

    def test_chain_response_choices_default_values(self):
        """Test ChainResponseChoices with default values"""
        choices = ChainResponseChoices()
        assert choices.index == 0
        assert isinstance(choices.message, Message)
        assert isinstance(choices.delta, Message)
        assert choices.finish_reason is None

    def test_chain_response_choices_custom_values(self):
        """Test ChainResponseChoices with custom values"""
        message = Message(role="assistant", content="Hello")
        delta = Message(role=None, content="Hello")
        choices = ChainResponseChoices(
            index=1,
            message=message,
            delta=delta,
            finish_reason="stop"
        )
        assert choices.index == 1
        assert choices.message.content == "Hello"
        assert choices.delta.content == "Hello"
        assert choices.finish_reason == "stop"


class TestMetrics:
    """Test Metrics model"""

    def test_metrics_default_values(self):
        """Test Metrics with default values"""
        metrics = Metrics()
        assert metrics.rag_ttft_ms is None
        assert metrics.llm_ttft_ms is None
        assert metrics.context_reranker_time_ms is None
        assert metrics.retrieval_time_ms is None
        assert metrics.llm_generation_time_ms is None

    def test_metrics_custom_values(self):
        """Test Metrics with custom values"""
        metrics = Metrics(
            rag_ttft_ms=100.0,
            llm_ttft_ms=50.0,
            context_reranker_time_ms=25.0,
            retrieval_time_ms=75.0,
            llm_generation_time_ms=200.0
        )
        assert metrics.rag_ttft_ms == 100.0
        assert metrics.llm_ttft_ms == 50.0
        assert metrics.context_reranker_time_ms == 25.0
        assert metrics.retrieval_time_ms == 75.0
        assert metrics.llm_generation_time_ms == 200.0

    def test_metrics_validation_negative_values(self):
        """Test Metrics validation with negative values"""
        with pytest.raises(ValidationError):
            Metrics(rag_ttft_ms=-1.0)


class TestChainResponse:
    """Test ChainResponse model"""

    def test_chain_response_default_values(self):
        """Test ChainResponse with default values"""
        response = ChainResponse()
        assert response.id == ""
        assert response.choices == []
        assert response.model == ""
        assert response.object == ""
        assert response.created == 0
        assert isinstance(response.usage, Usage)
        assert isinstance(response.metrics, Metrics)
        assert isinstance(response.citations, Citations)

    def test_chain_response_custom_values(self):
        """Test ChainResponse with custom values"""
        usage = Usage(total_tokens=100)
        metrics = Metrics(rag_ttft_ms=100.0)
        choices = [ChainResponseChoices()]
        citations = Citations(total_results=0, results=[])
        response = ChainResponse(
            id="test-id",
            choices=choices,
            model="test-model",
            object="chat.completion.chunk",
            created=1234567890,
            usage=usage,
            metrics=metrics,
            citations=citations
        )
        assert response.id == "test-id"
        assert len(response.choices) == 1
        assert response.model == "test-model"
        assert response.object == "chat.completion.chunk"
        assert response.created == 1234567890
        assert response.usage == usage
        assert response.metrics == metrics
        assert response.citations == citations


class TestPrepareLLMRequest:
    """Test prepare_llm_request function"""

    def test_prepare_llm_request_basic(self):
        """Test prepare_llm_request with basic parameters"""
        messages = [{"role": "user", "content": "Hello"}]
        last_user_message, processed_chat_history = prepare_llm_request(messages)

        assert last_user_message == "Hello"
        assert processed_chat_history == []

    def test_prepare_llm_request_with_kwargs(self):
        """Test prepare_llm_request with additional kwargs"""
        messages = [{"role": "user", "content": "Hello"}]
        last_user_message, processed_chat_history = prepare_llm_request(
            messages,
            temperature=0.7,
            max_tokens=100,
            stream=True
        )

        assert last_user_message == "Hello"
        assert processed_chat_history == []

    def test_prepare_llm_request_empty_messages(self):
        """Test prepare_llm_request with empty messages"""
        last_user_message, processed_chat_history = prepare_llm_request([])
        assert last_user_message is None
        assert processed_chat_history == []


class TestPrepareCitations:
    """Test prepare_citations function"""

    def test_prepare_citations_disabled(self):
        """Test prepare_citations when disabled"""
        mock_doc = Mock()
        mock_doc.metadata = {"source": "test.pdf"}
        contexts = [mock_doc]
        result = prepare_citations(contexts, enable_citations=False)
        assert isinstance(result, Citations)
        assert result.total_results == 0
        assert result.results == []

    def test_prepare_citations_enabled_no_contexts(self):
        """Test prepare_citations when enabled but no contexts"""
        result = prepare_citations([], enable_citations=True)
        assert isinstance(result, Citations)
        assert result.total_results == 0
        assert result.results == []

    def test_prepare_citations_enabled_with_contexts(self):
        """Test prepare_citations when enabled with contexts"""
        mock_doc1 = Mock()
        mock_doc1.page_content = "Test content 1"
        mock_doc1.metadata = {
            "source": "test1.pdf",  # Use string source to trigger first if block
            "content_metadata": {"page_number": 1, "type": "image", "location": []},
            "relevance_score": 0.8
        }
        mock_doc2 = Mock()
        mock_doc2.page_content = "Test content 2"
        mock_doc2.metadata = {
            "source": "test2.pdf",  # Use string source to trigger first if block
            "content_metadata": {"page_number": 2, "type": "image", "location": []},
            "relevance_score": 0.9
        }
        contexts = [mock_doc1, mock_doc2]

        with patch('nvidia_rag.rag_server.response_generator.MINIO_OPERATOR') as mock_minio, \
             patch('nvidia_rag.rag_server.response_generator.get_unique_thumbnail_id') as mock_get_thumbnail:
            # Mock the MinIO operator methods to handle collection_name parameter
            mock_minio.get_payload.return_value = {"content": "base64_thumbnail"}
            mock_get_thumbnail.return_value = "test_thumbnail_id"

            result = prepare_citations(contexts, enable_citations=True)

            assert isinstance(result, Citations)
            assert result.total_results == 2
            assert len(result.results) == 2
            assert result.results[0].document_name == "test1.pdf"
            assert result.results[0].metadata.page_number == 1
            assert result.results[1].document_name == "test2.pdf"
            assert result.results[1].metadata.page_number == 2

    def test_prepare_citations_with_minio_thumbnails(self):
        """Test prepare_citations with MinIO thumbnails"""
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {
            "source": {"source_id": "test.pdf"},
            "content_metadata": {"page_number": 1, "type": "image", "location": []},
            "collection_name": "test_collection",
            "relevance_score": 0.8
        }
        contexts = [mock_doc]

        with patch('nvidia_rag.rag_server.response_generator.MINIO_OPERATOR') as mock_minio, \
             patch('nvidia_rag.rag_server.response_generator.get_unique_thumbnail_id') as mock_get_thumbnail:
            mock_minio.get_payload.return_value = {"content": "base64_thumbnail"}
            mock_get_thumbnail.return_value = "test_thumbnail_id"

            result = prepare_citations(contexts, enable_citations=True)

            assert isinstance(result, Citations)
            assert result.total_results == 1
            assert len(result.results) == 1
            assert result.results[0].document_name == "test.pdf"


class TestErrorResponseGenerator:
    """Test error_response_generator function"""

    def test_error_response_generator_basic(self):
        """Test error_response_generator with basic error message"""
        error_msg = "Test error message"
        result = list(error_response_generator(error_msg))

        # Error message is chunked into 5-character pieces
        expected_chunks = len(error_msg) // 5 + (1 if len(error_msg) % 5 > 0 else 0) + 1  # +1 for final chunk
        assert len(result) == expected_chunks
        response_data = json.loads(result[0].replace("data: ", ""))
        assert response_data["choices"][0]["message"]["content"] == "Test "  # First 5 chars
        assert response_data["choices"][0]["finish_reason"] is None

    def test_error_response_generator_fallback(self):
        """Test error_response_generator with fallback message"""
        result = list(error_response_generator(""))

        # Empty message should result in just the final chunk
        assert len(result) == 1
        response_data = json.loads(result[0].replace("data: ", ""))
        assert response_data["choices"][0]["message"]["content"] == ""

    def test_error_response_generator_json_structure(self):
        """Test error_response_generator JSON structure"""
        error_msg = "Test error"
        result = list(error_response_generator(error_msg))

        response_data = json.loads(result[0].replace("data: ", ""))
        assert "id" in response_data
        assert "choices" in response_data
        assert "model" in response_data
        assert "object" in response_data
        assert "created" in response_data
        assert len(response_data["choices"]) == 1
        assert response_data["choices"][0]["index"] == 0
        assert response_data["choices"][0]["message"]["role"] == "assistant"
        assert response_data["choices"][0]["finish_reason"] is None


class TestIsEmptyContent:
    """Test _is_empty_content function"""

    def test_is_empty_content_string_empty(self):
        """Test _is_empty_content with empty string"""
        result = _is_empty_content("")
        assert result == ""

    def test_is_empty_content_string_whitespace(self):
        """Test _is_empty_content with whitespace string"""
        result = _is_empty_content("   ")
        assert result == ""

    def test_is_empty_content_string_valid(self):
        """Test _is_empty_content with valid string"""
        result = _is_empty_content("Hello")
        assert result == "Hello"  # Returns the stripped string

    def test_is_empty_content_list_empty(self):
        """Test _is_empty_content with empty list"""
        result = _is_empty_content([])
        assert result is False

    def test_is_empty_content_list_valid(self):
        """Test _is_empty_content with valid list"""
        result = _is_empty_content(["Hello"])
        assert result is False

    def test_is_empty_content_none(self):
        """Test _is_empty_content with None"""
        result = _is_empty_content(None)
        assert result is False

    def test_is_empty_content_other_types(self):
        """Test _is_empty_content with other types"""
        assert _is_empty_content(0) is False
        assert _is_empty_content(False) is False
        assert _is_empty_content({}) is False


class TestEscapeJsonContentMultimodal:
    """Test escape_json_content_multimodal function"""

    def test_escape_json_content_multimodal_string(self):
        """Test escape_json_content_multimodal with string"""
        result = escape_json_content_multimodal("Hello\nWorld")
        assert result == "Hello\nWorld"

    def test_escape_json_content_multimodal_list(self):
        """Test escape_json_content_multimodal with list"""
        content = [
            {"type": "text", "text": "Hello\nWorld"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
        ]
        result = escape_json_content_multimodal(content)
        assert result[0]["text"] == "Hello\nWorld"
        assert result[1]["image_url"]["url"] == "data:image/png;base64,test"

    def test_escape_json_content_multimodal_other_types(self):
        """Test escape_json_content_multimodal with other types"""
        assert escape_json_content_multimodal(123) == 123
        assert escape_json_content_multimodal(None) is None
        assert escape_json_content_multimodal(True) is True


class TestEscapeJsonContent:
    """Test escape_json_content function"""

    def test_escape_json_content_basic(self):
        """Test escape_json_content with basic string"""
        result = escape_json_content("Hello\nWorld")
        assert result == "Hello\nWorld"

    def test_escape_json_content_quotes(self):
        """Test escape_json_content with quotes"""
        result = escape_json_content('He said "Hello"')
        assert result == 'He said "Hello"'

    def test_escape_json_content_backslashes(self):
        """Test escape_json_content with backslashes"""
        result = escape_json_content("Path\\to\\file")
        assert result == "Path\\to\\file"

    def test_escape_json_content_empty(self):
        """Test escape_json_content with empty string"""
        result = escape_json_content("")
        assert result == ""

    def test_escape_json_content_special_chars(self):
        """Test escape_json_content with special characters"""
        result = escape_json_content("Line1\nLine2\tTabbed\r\nWindows")
        assert result == "Line1\nLine2\tTabbed\r\nWindows"


class TestGenerateAnswer:
    """Test generate_answer function"""

    @pytest.mark.asyncio
    async def test_generate_answer_success(self):
        """Test generate_answer with successful generation"""
        def mock_generator():
            yield "Hello"
            yield " world"
            yield "!"

        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.pdf", "content_metadata": {"type": "text"}}
        contexts = [mock_doc]

        result = []
        for chunk in generate_answer(
            generator=mock_generator(),
            contexts=contexts,
            model="test-model",
            enable_citations=True
        ):
            result.append(chunk)

        assert len(result) > 0
        # Check that all chunks are properly formatted
        for chunk in result:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_generate_answer_milvus_exception(self):
        """Test generate_answer with MilvusException"""
        def mock_generator():
            yield "Hello"  # First yield works
            raise MilvusException("Milvus error")  # Exception on second iteration

        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.pdf", "content_metadata": {"type": "text"}}
        contexts = [mock_doc]

        result = []
        for chunk in generate_answer(
            generator=mock_generator(),
            contexts=contexts,
            model="test-model"
        ):
            result.append(chunk)

        # Should return error response
        assert len(result) > 0
        # Check that we get error response generator
        error_generators = [chunk for chunk in result if hasattr(chunk, '__iter__') and not isinstance(chunk, str)]
        assert len(error_generators) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_general_exception(self):
        """Test generate_answer with general exception"""
        def mock_generator():
            yield "Hello"  # First yield works
            raise Exception("General error")  # Exception on second iteration

        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.pdf", "content_metadata": {"type": "text"}}
        contexts = [mock_doc]

        result = []
        for chunk in generate_answer(
            generator=mock_generator(),
            contexts=contexts,
            model="test-model"
        ):
            result.append(chunk)

        # Should return error response
        assert len(result) > 0
        # Check that we get error response generator
        error_generators = [chunk for chunk in result if hasattr(chunk, '__iter__') and not isinstance(chunk, str)]
        assert len(error_generators) > 0


class TestRetrieveSummary:
    """Test retrieve_summary function"""

    @pytest.mark.asyncio
    async def test_retrieve_summary_success(self):
        """Test retrieve_summary with successful retrieval"""
        with patch('nvidia_rag.rag_server.response_generator.MINIO_OPERATOR') as mock_minio, \
             patch('nvidia_rag.rag_server.response_generator.get_unique_thumbnail_id') as mock_get_thumbnail:
            mock_minio.get_payload.return_value = {"summary": "Test summary", "file_name": "test.pdf"}
            mock_get_thumbnail.return_value = "test_thumbnail_id"

            result = await retrieve_summary(
                collection_name="test_collection",
                file_name="test.pdf"
            )

            assert result["status"] == "SUCCESS"
            assert "summary" in result

    @pytest.mark.asyncio
    async def test_retrieve_summary_exception(self):
        """Test retrieve_summary with exception"""
        with patch('nvidia_rag.rag_server.response_generator.MINIO_OPERATOR') as mock_minio, \
             patch('nvidia_rag.rag_server.response_generator.get_unique_thumbnail_id') as mock_get_thumbnail:
            mock_minio.get_payload.side_effect = Exception("Summary error")
            mock_get_thumbnail.return_value = "test_thumbnail_id"

            result = await retrieve_summary(
                collection_name="test_collection",
                file_name="test.pdf"
            )

            assert result["status"] == "ERROR"
            assert "error" in result

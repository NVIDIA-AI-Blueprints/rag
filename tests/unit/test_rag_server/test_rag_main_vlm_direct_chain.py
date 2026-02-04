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

"""Unit tests for VLM direct chain functionality (VLM query without collection)."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nvidia_rag.rag_server.main import NvidiaRAG
from nvidia_rag.rag_server.response_generator import APIError
from nvidia_rag.rag_server.llm_chain import llm_chain, vlm_direct_chain
from nvidia_rag.rag_server.content_utils import _contains_images
from nvidia_rag.utils.vdb.vdb_base import VDBRag


# Helper to create async generators for tests
async def async_gen_from_list(items):
    """Helper function to create async generator from list."""
    for item in items:
        yield item


@pytest.fixture(autouse=True)
def _disable_reflection(monkeypatch):
    """Ensure reflection is disabled for these tests."""
    monkeypatch.setenv("ENABLE_REFLECTION", "false")


class TestLLMChainWithVLMInference:
    """Test cases for _llm_chain method with VLM inference support."""

    @pytest.mark.asyncio
    async def test_llm_chain_with_images_but_vlm_disabled_raises_error(self):
        """Test that _llm_chain raises APIError when images present but VLM not enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
        }

        # Query with images
        query = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        ]

        with pytest.raises(
            APIError,
            match="Visual Q&A is not supported without VLM inference enabled",
        ):
            await llm_chain(
                config=rag.config,
                prompts=rag.prompts,
                llm_settings=llm_settings,
                query=query,
                chat_history=[],
                model="test_model",
                enable_citations=True,
                enable_vlm_inference=False,  # VLM disabled
                vlm_settings=None,
            )

    @pytest.mark.asyncio
    async def test_llm_chain_text_only_without_vlm_works(self):
        """Test that _llm_chain works with text-only query when VLM disabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock instance attribute
        rag.StreamingFilterThinkParser = Mock()

        llm_settings = {
            "model": "test_model",
            "llm_endpoint": "http://test.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "enable_guardrails": False,
            "stop": [],
        }

        with patch("nvidia_rag.rag_server.message_processor.handle_prompt_processing") as mock_handle_prompt:
            with patch("nvidia_rag.rag_server.llm_chain.get_llm") as mock_get_llm:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.ChatPromptTemplate"
                ) as mock_prompt_template:
                    with patch("nvidia_rag.rag_server.llm_chain.StrOutputParser"):
                        with patch(
                            "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                        ) as mock_generate_answer:
                            with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                                mock_handle_prompt.return_value = (
                                    [("system", "test system")],
                                    [],
                                    [("user", "test query")],
                                )

                                # Mock the entire chain construction
                                mock_prompt = Mock()
                                mock_prompt_template.from_messages.return_value = (
                                    mock_prompt
                                )

                                mock_llm = Mock()
                                mock_get_llm.return_value = mock_llm

                                # Create a mock chain
                                mock_chain = Mock()
                                mock_stream_gen = iter(["response"])
                                mock_chain.stream.return_value = mock_stream_gen

                                # Mock the pipe operations
                                mock_prompt.__or__ = Mock(return_value=Mock())
                                mock_prompt.__or__.return_value.__or__ = Mock(
                                    return_value=Mock()
                                )
                                mock_prompt.__or__.return_value.__or__.return_value.__or__ = Mock(
                                    return_value=mock_chain
                                )

                                mock_generate_answer.return_value = (
                                    async_gen_from_list(["test response"])
                                )

                                result = await llm_chain(
                                    config=rag.config,
                                    prompts=rag.prompts,
                                    llm_settings=llm_settings,
                                    query="test query",
                                    chat_history=[],
                                    model="test_model",
                                    enable_citations=True,
                                    enable_vlm_inference=False,
                                    vlm_settings=None,
                                )

                                assert hasattr(result, "generator")
                                assert hasattr(result, "status_code")

    @pytest.mark.asyncio
    async def test_llm_chain_routes_to_vlm_when_enabled(self):
        """Test that _llm_chain routes to _vlm_direct_chain when VLM enabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        llm_settings = {}
        vlm_settings = {"vlm_model": "test_vlm"}

        with patch("nvidia_rag.rag_server.llm_chain.vlm_direct_chain") as mock_vlm_chain:
            mock_vlm_chain.return_value = Mock(
                generator=async_gen_from_list(["vlm response"]), status_code=200
            )

            result = await llm_chain(
                config=rag.config,
                prompts=rag.prompts,
                llm_settings=llm_settings,
                query="test query",
                chat_history=[],
                model="test_model",
                enable_citations=True,
                enable_vlm_inference=True,  # VLM enabled
                vlm_settings=vlm_settings,
            )

            # Verify _vlm_direct_chain was called
            mock_vlm_chain.assert_called_once()
            call_kwargs = mock_vlm_chain.call_args[1]
            assert call_kwargs["vlm_settings"] == vlm_settings
            assert call_kwargs["query"] == "test query"


class TestVLMDirectChain:
    """Test cases for _vlm_direct_chain method."""

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_basic_success(self):
        """Test basic successful VLM direct chain execution."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "What is AI?"
        chat_history = []

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch("nvidia_rag.rag_server.llm_chain._eager_prefetch_astream") as mock_prefetch:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                ) as mock_generate_answer:
                    with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                        # Mock VLM instance
                        mock_vlm = Mock()
                        mock_vlm.stream_with_messages = AsyncMock(
                            return_value=async_gen_from_list(["VLM response"])
                        )
                        mock_vlm_class.return_value = mock_vlm

                        # Mock prefetch
                        mock_prefetch.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        mock_generate_answer.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        result = await vlm_direct_chain(
                            config=rag.config,
                            prompts=rag.prompts,
                            query=query,
                            chat_history=chat_history,
                            model="test_model",
                            enable_citations=True,
                            metrics=None,
                            vlm_settings={},
                        )

                        assert hasattr(result, "generator")
                        assert hasattr(result, "status_code")
                        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_with_images(self):
        """Test VLM direct chain with multimodal query containing images."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        # Multimodal query with image
        query = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        ]
        chat_history = []

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch("nvidia_rag.rag_server.llm_chain._eager_prefetch_astream") as mock_prefetch:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                ) as mock_generate_answer:
                    with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                        # Mock VLM instance
                        mock_vlm = Mock()
                        mock_vlm.stream_with_messages = AsyncMock(
                            return_value=async_gen_from_list(["VLM response"])
                        )
                        mock_vlm_class.return_value = mock_vlm

                        # Mock prefetch
                        mock_prefetch.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        mock_generate_answer.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        result = await vlm_direct_chain(
                            config=rag.config,
                            prompts=rag.prompts,
                            query=query,
                            chat_history=chat_history,
                            model="test_model",
                            enable_citations=True,
                            metrics=None,
                            vlm_settings={},
                        )

                        assert hasattr(result, "generator")
                        assert result.status_code == 200

                        # Consume the generator to trigger the VLM calls
                        chunks = [chunk async for chunk in result.generator]
                        assert len(chunks) > 0

                        # Verify VLM was called with correct messages
                        mock_vlm.stream_with_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_with_custom_vlm_settings(self):
        """Test VLM direct chain with custom VLM settings."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock default config
        rag.config.vlm.model_name = "default_model"
        rag.config.vlm.server_url = "http://default.com"
        rag.config.vlm.temperature = 0.5
        rag.config.vlm.top_p = 0.8
        rag.config.vlm.max_tokens = 50
        rag.config.vlm.max_total_images = 3

        # Custom settings should override defaults
        custom_vlm_settings = {
            "vlm_model": "custom_vlm_model",
            "vlm_endpoint": "http://custom-vlm.com",
            "vlm_temperature": 0.9,
            "vlm_top_p": 0.95,
            "vlm_max_tokens": 200,
            "vlm_max_total_images": 10,
        }

        query = "test query"
        chat_history = []

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch("nvidia_rag.rag_server.llm_chain._eager_prefetch_astream") as mock_prefetch:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                ) as mock_generate_answer:
                    with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                        # Mock VLM instance
                        mock_vlm = Mock()
                        mock_vlm.stream_with_messages = AsyncMock(
                            return_value=async_gen_from_list(["VLM response"])
                        )
                        mock_vlm_class.return_value = mock_vlm

                        mock_prefetch.return_value = async_gen_from_list(
                            ["VLM response"]
                        )
                        mock_generate_answer.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        result = await vlm_direct_chain(
                            config=rag.config,
                            prompts=rag.prompts,
                            query=query,
                            chat_history=chat_history,
                            model="test_model",
                            enable_citations=True,
                            metrics=None,
                            vlm_settings=custom_vlm_settings,
                        )

                        # Verify VLM was initialized with custom settings
                        mock_vlm_class.assert_called_once()
                        call_kwargs = mock_vlm_class.call_args[1]
                        assert call_kwargs["vlm_model"] == "custom_vlm_model"
                        assert call_kwargs["vlm_endpoint"] == "http://custom-vlm.com"

                        # Verify stream_with_messages was called with custom parameters
                        mock_vlm.stream_with_messages.assert_called_once()
                        stream_kwargs = mock_vlm.stream_with_messages.call_args[1]
                        assert stream_kwargs["temperature"] == 0.9
                        assert stream_kwargs["top_p"] == 0.95
                        assert stream_kwargs["max_tokens"] == 200
                        assert stream_kwargs["max_total_images"] == 10

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_with_conversation_history(self):
        """Test VLM direct chain with conversation history."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "Follow-up question"
        chat_history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch("nvidia_rag.rag_server.llm_chain._eager_prefetch_astream") as mock_prefetch:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                ) as mock_generate_answer:
                    with patch.dict(os.environ, {"CONVERSATION_HISTORY": "1"}):
                        # Mock VLM instance
                        mock_vlm = Mock()
                        mock_vlm.stream_with_messages = AsyncMock(
                            return_value=async_gen_from_list(["VLM response"])
                        )
                        mock_vlm_class.return_value = mock_vlm

                        mock_prefetch.return_value = async_gen_from_list(
                            ["VLM response"]
                        )
                        mock_generate_answer.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        result = await vlm_direct_chain(
                            config=rag.config,
                            prompts=rag.prompts,
                            query=query,
                            chat_history=chat_history,
                            model="test_model",
                            enable_citations=True,
                            metrics=None,
                            vlm_settings={},
                        )

                        # Consume the generator to trigger VLM calls
                        chunks = [chunk async for chunk in result.generator]
                        assert len(chunks) > 0

                        # Verify the messages passed to VLM includes history + query
                        mock_vlm.stream_with_messages.assert_called_once()
                        call_kwargs = mock_vlm.stream_with_messages.call_args[1]
                        messages = call_kwargs["messages"]

                        # Should have limited history (1 turn = 2 messages) + current query
                        assert len(messages) == 3  # 2 from history + 1 new query

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_with_zero_conversation_history(self):
        """Test VLM direct chain with CONVERSATION_HISTORY=0."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "Current question"
        chat_history = [
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer"},
        ]

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch("nvidia_rag.rag_server.llm_chain._eager_prefetch_astream") as mock_prefetch:
                with patch(
                    "nvidia_rag.rag_server.llm_chain.generate_answer_async"
                ) as mock_generate_answer:
                    with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                        # Mock VLM instance
                        mock_vlm = Mock()
                        mock_vlm.stream_with_messages = AsyncMock(
                            return_value=async_gen_from_list(["VLM response"])
                        )
                        mock_vlm_class.return_value = mock_vlm

                        mock_prefetch.return_value = async_gen_from_list(
                            ["VLM response"]
                        )
                        mock_generate_answer.return_value = async_gen_from_list(
                            ["VLM response"]
                        )

                        result = await vlm_direct_chain(
                            config=rag.config,
                            prompts=rag.prompts,
                            query=query,
                            chat_history=chat_history,
                            model="test_model",
                            enable_citations=True,
                            metrics=None,
                            vlm_settings={},
                        )

                        # Consume the generator to trigger VLM calls
                        chunks = [chunk async for chunk in result.generator]
                        assert len(chunks) > 0

                        # Verify only current query is passed (no history)
                        mock_vlm.stream_with_messages.assert_called_once()
                        call_kwargs = mock_vlm.stream_with_messages.call_args[1]
                        messages = call_kwargs["messages"]
                        assert len(messages) == 1  # Only current query

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_api_error_handling(self):
        """Test VLM direct chain handles APIError from VLM."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise APIError during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_api_error(*args, **kwargs):
                        # This is an async generator that raises immediately
                        raise APIError("VLM service error", 500)
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_api_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["VLM service error"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={},
                    )

                    # Should return error response
                    assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_connection_error(self):
        """Test VLM direct chain handles ConnectionError."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise ConnectionError during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_connection_error(*args, **kwargs):
                        raise ConnectionError("Connection refused")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_connection_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["VLM NIM unavailable"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={"vlm_endpoint": "http://vlm.com"},
                    )

                    # Should return SERVICE_UNAVAILABLE status
                    assert result.status_code == 503

                    # Verify error message contains endpoint info
                    response_items = []
                    async for item in result.generator:
                        response_items.append(item)
                    assert any(
                        "VLM NIM unavailable" in str(item)
                        for item in response_items
                    )

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_os_error(self):
        """Test VLM direct chain handles OSError."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise OSError during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_os_error(*args, **kwargs):
                        raise OSError("Network error")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_os_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["VLM NIM unavailable"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={},
                    )

                    # Should return SERVICE_UNAVAILABLE status
                    assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_value_error(self):
        """Test VLM direct chain handles ValueError."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise ValueError during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_value_error(*args, **kwargs):
                        raise ValueError("Invalid parameter")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_value_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["VLM NIM unavailable"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={},
                    )

                    # Should return SERVICE_UNAVAILABLE status
                    assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_403_forbidden_error(self):
        """Test VLM direct chain handles 403 Forbidden error."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise 403 error during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_403_error(*args, **kwargs):
                        raise Exception("[403] Forbidden")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_403_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["Authentication or permission error"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={},
                    )

                    # Should return FORBIDDEN status
                    assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_404_not_found_error(self):
        """Test VLM direct chain handles 404 Not Found error."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise 404 error during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_404_error(*args, **kwargs):
                        raise Exception("[404] Not Found")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_404_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["VLM model 'test_vlm_model' not found"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={"vlm_model": "test_vlm_model"},
                    )

                    # Should return NOT_FOUND status
                    assert result.status_code == 404

                    # Verify error message contains model name
                    response_items = []
                    async for item in result.generator:
                        response_items.append(item)
                    assert any(
                            "test_vlm_model" in str(item) for item in response_items
                        )

    @pytest.mark.asyncio
    async def test_vlm_direct_chain_general_exception(self):
        """Test VLM direct chain handles general exceptions."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        query = "test query"

        with patch("nvidia_rag.rag_server.llm_chain.VLM") as mock_vlm_class:
            with patch(
                "nvidia_rag.rag_server.llm_chain.generate_answer_async"
            ) as mock_generate_answer:
                with patch.dict(os.environ, {"CONVERSATION_HISTORY": "0"}):
                    # Mock VLM to raise general exception during stream_with_messages
                    mock_vlm = Mock()
                    async def raise_general_error(*args, **kwargs):
                        raise Exception("Unexpected error")
                        yield  # This line is never reached but makes it an async generator
                    
                    mock_vlm.stream_with_messages = Mock(side_effect=raise_general_error)
                    mock_vlm_class.return_value = mock_vlm

                    mock_generate_answer.return_value = async_gen_from_list(
                        ["Unexpected error"]
                    )

                    result = await vlm_direct_chain(
                        config=rag.config,
                        prompts=rag.prompts,
                        query=query,
                        chat_history=[],
                        model="test_model",
                        enable_citations=True,
                        metrics=None,
                        vlm_settings={},
                    )

                    # Should return BAD_REQUEST status
                    assert result.status_code == 400


class TestGenerateMethodVLMIntegration:
    """Test cases for generate method with VLM integration (via _llm_chain)."""

    @pytest.mark.asyncio
    async def test_generate_logs_vlm_pipeline_mode(self):
        """Test that generate method logs VLM pipeline mode correctly."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)

        # Mock config
        rag.config.vlm.model_name = "test_vlm_model"
        rag.config.vlm.server_url = "http://vlm.com"
        rag.config.vlm.temperature = 0.7
        rag.config.vlm.top_p = 0.9
        rag.config.vlm.max_tokens = 100
        rag.config.vlm.max_total_images = 5

        with patch("nvidia_rag.rag_server.llm_chain.vlm_direct_chain") as mock_vlm_chain:
            with patch("nvidia_rag.rag_server.main.logger") as mock_logger:
                mock_vlm_chain.return_value = Mock(
                    generator=async_gen_from_list(["vlm response"]), status_code=200
                )

                await rag.generate(
                    messages=[{"role": "user", "content": "test query"}],
                    use_knowledge_base=False,  # Direct mode
                    enable_vlm_inference=True,
                )

                # Verify the correct pipeline mode is logged
                info_calls = [
                    call for call in mock_logger.info.call_args_list if call[0]
                ]
                pipeline_logs = [
                    call
                    for call in info_calls
                    if "PIPELINE MODE" in str(call) and "VLM" in str(call)
                ]
                assert len(pipeline_logs) > 0

    @pytest.mark.asyncio
    async def test_generate_logs_llm_pipeline_mode_without_vlm(self):
        """Test that generate method logs LLM pipeline mode when VLM disabled."""
        mock_vdb_op = Mock(spec=VDBRag)
        rag = NvidiaRAG(vdb_op=mock_vdb_op)
        rag.StreamingFilterThinkParser = Mock()

        with patch("nvidia_rag.rag_server.message_processor.handle_prompt_processing") as mock_handle_prompt:
            with patch("nvidia_rag.rag_server.main.get_llm"):
                with patch("nvidia_rag.rag_server.main.ChatPromptTemplate"):
                    with patch("nvidia_rag.rag_server.main.StrOutputParser"):
                        with patch(
                            "nvidia_rag.rag_server.main.generate_answer_async"
                        ) as mock_generate_answer:
                            with patch(
                                "nvidia_rag.rag_server.main.logger"
                            ) as mock_logger:
                                with patch.dict(
                                    os.environ, {"CONVERSATION_HISTORY": "0"}
                                ):
                                    mock_handle_prompt.return_value = (
                                        [("system", "test")],
                                        [],
                                        [("user", "test")],
                                    )
                                    mock_generate_answer.return_value = (
                                        async_gen_from_list(["response"])
                                    )

                                    await rag.generate(
                                        messages=[{"role": "user", "content": "test query"}],
                                        use_knowledge_base=False,
                                        enable_vlm_inference=False,
                                    )

                                    # Verify the correct pipeline mode is logged
                                    info_calls = [
                                        call
                                        for call in mock_logger.info.call_args_list
                                        if call[0]
                                    ]
                                    pipeline_logs = [
                                        call
                                        for call in info_calls
                                        if "PIPELINE MODE" in str(call)
                                        and "Direct LLM Chain" in str(call)
                                    ]
                                    assert len(pipeline_logs) > 0

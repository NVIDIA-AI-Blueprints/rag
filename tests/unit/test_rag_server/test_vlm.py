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

import base64
import io
import os
import re
from unittest.mock import Mock, patch, MagicMock
from PIL import Image as PILImage

import pytest
from langchain_core.messages import HumanMessage

from nvidia_rag.rag_server.vlm import VLM


class TestVLM:
    """Test VLM class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.vlm_model = "test-model"
        self.vlm_endpoint = "http://test-endpoint.com"

    def teardown_method(self):
        """Clean up after each test"""
        import gc
        gc.collect()
        # Clear any remaining mocks
        import unittest.mock
        unittest.mock.patch.stopall()

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_vlm_init_success(self, mock_get_prompts, mock_get_config):
        """Test VLM initialization with valid parameters"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2

        mock_prompts = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }
        mock_get_prompts.return_value = mock_prompts

        vlm = VLM(self.vlm_model, self.vlm_endpoint)

        assert vlm.model_name == self.vlm_model
        assert vlm.invoke_url == self.vlm_endpoint
        assert vlm.vlm_template == "Test template: {question}"

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_vlm_init_missing_url(self, mock_get_prompts, mock_get_config):
        """Test VLM initialization with missing URL raises OSError"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {}

        with pytest.raises(OSError, match="VLM server URL and model name must be set in the environment."):
            VLM("", self.vlm_endpoint)

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_vlm_init_missing_model(self, mock_get_prompts, mock_get_config):
        """Test VLM initialization with missing model raises OSError"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {}

        with pytest.raises(OSError, match="VLM server URL and model name must be set in the environment."):
            VLM(self.vlm_model, "")

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_vlm_init_none_url(self, mock_get_prompts, mock_get_config):
        """Test VLM initialization with None URL raises OSError"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {}

        with pytest.raises(OSError, match="VLM server URL and model name must be set in the environment."):
            VLM(self.vlm_model, None)

    def create_test_image_b64(self):
        """Create a test base64 encoded image"""
        img = PILImage.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.ChatOpenAI')
    def test_analyze_image_no_images(self, mock_chat_openai, mock_get_prompts, mock_get_config):
        """Test analyze_image with no images returns empty string"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        result = vlm.analyze_image([], "test question")

        assert result == ""

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.ChatOpenAI')
    def test_analyze_image_success(self, mock_chat_openai, mock_get_prompts, mock_get_config):
        """Test analyze_image with valid images returns VLM response"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        mock_vlm_instance = Mock()
        mock_vlm_instance.invoke.return_value.content = "Test VLM response"
        mock_chat_openai.return_value = mock_vlm_instance

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        test_image = self.create_test_image_b64()
        result = vlm.analyze_image([test_image], "test question")

        assert result == "Test VLM response"
        mock_vlm_instance.invoke.assert_called_once()

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.ChatOpenAI')
    def test_analyze_image_exception_handling(self, mock_chat_openai, mock_get_prompts, mock_get_config):
        """Test analyze_image handles exceptions gracefully"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        mock_vlm_instance = Mock()
        mock_vlm_instance.invoke.side_effect = Exception("VLM error")
        mock_chat_openai.return_value = mock_vlm_instance

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        test_image = self.create_test_image_b64()

        # The function should raise an exception, not catch it
        with pytest.raises(Exception, match="VLM error"):
            vlm.analyze_image([test_image], "test question")

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.ChatOpenAI')
    def test_analyze_image_config_limits_exceeded(self, mock_chat_openai, mock_get_prompts, mock_get_config):
        """Test analyze_image when config limits are exceeded"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 2
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        # Mock ChatOpenAI to avoid API key issues
        mock_vlm_instance = Mock()
        mock_chat_openai.return_value = mock_vlm_instance

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        test_image = self.create_test_image_b64()
        result = vlm.analyze_image([test_image], "test question")

        assert result == ""

    def test_convert_image_url_to_png_b64_data_url(self):
        """Test _convert_image_url_to_png_b64 with data URL"""
        with patch('nvidia_rag.rag_server.vlm.get_config') as mock_get_config, \
             patch('nvidia_rag.rag_server.vlm.get_prompts') as mock_get_prompts:

            mock_get_config.return_value = Mock()
            mock_get_prompts.return_value = {
                "vlm_template": "Test template: {question}",
                "vlm_response_reasoning_template": {"system": "test", "human": "test"}
            }

            vlm = VLM(self.vlm_model, self.vlm_endpoint)
            test_image = self.create_test_image_b64()
            data_url = f"data:image/jpeg;base64,{test_image}"

            result = vlm._convert_image_url_to_png_b64(data_url)
            assert isinstance(result, str)
            assert result.startswith("iVBOR")  # PNG base64 starts with this

    def test_convert_image_url_to_png_b64_base64_string(self):
        """Test _convert_image_url_to_png_b64 with base64 string"""
        with patch('nvidia_rag.rag_server.vlm.get_config') as mock_get_config, \
             patch('nvidia_rag.rag_server.vlm.get_prompts') as mock_get_prompts:

            mock_get_config.return_value = Mock()
            mock_get_prompts.return_value = {
                "vlm_template": "Test template: {question}",
                "vlm_response_reasoning_template": {"system": "test", "human": "test"}
            }

            vlm = VLM(self.vlm_model, self.vlm_endpoint)
            test_image = self.create_test_image_b64()

            result = vlm._convert_image_url_to_png_b64(test_image)
            assert isinstance(result, str)
            assert result.startswith("iVBOR")  # PNG base64 starts with this

    def test_convert_image_url_to_png_b64_invalid_data_url(self):
        """Test _convert_image_url_to_png_b64 with invalid data URL"""
        with patch('nvidia_rag.rag_server.vlm.get_config') as mock_get_config, \
             patch('nvidia_rag.rag_server.vlm.get_prompts') as mock_get_prompts:

            mock_get_config.return_value = Mock()
            mock_get_prompts.return_value = {
                "vlm_template": "Test template: {question}",
                "vlm_response_reasoning_template": {"system": "test", "human": "test"}
            }

            vlm = VLM(self.vlm_model, self.vlm_endpoint)
            invalid_url = "data:image/jpeg;invalid,notbase64"

            result = vlm._convert_image_url_to_png_b64(invalid_url)
            assert result == invalid_url  # Returns original on failure

    def test_convert_image_url_to_png_b64_exception_handling(self):
        """Test _convert_image_url_to_png_b64 handles exceptions"""
        with patch('nvidia_rag.rag_server.vlm.get_config') as mock_get_config, \
             patch('nvidia_rag.rag_server.vlm.get_prompts') as mock_get_prompts:

            mock_get_config.return_value = Mock()
            mock_get_prompts.return_value = {
                "vlm_template": "Test template: {question}",
                "vlm_response_reasoning_template": {"system": "test", "human": "test"}
            }

            vlm = VLM(self.vlm_model, self.vlm_endpoint)
            invalid_data = "not_valid_base64"

            result = vlm._convert_image_url_to_png_b64(invalid_data)
            assert result == invalid_data  # Returns original on failure

    def test_add_image_urls(self):
        """Test _add_image_urls method"""
        with patch('nvidia_rag.rag_server.vlm.get_config') as mock_get_config, \
             patch('nvidia_rag.rag_server.vlm.get_prompts') as mock_get_prompts:

            mock_get_config.return_value = Mock()
            mock_get_prompts.return_value = {
                "vlm_template": "Test template: {question}",
                "vlm_response_reasoning_template": {"system": "test", "human": "test"}
            }

            vlm = VLM(self.vlm_model, self.vlm_endpoint)
            test_image = self.create_test_image_b64()
            message = HumanMessage(content=[{"type": "text", "text": "test"}])

            vlm._add_image_urls([test_image], 1, message)

            assert len(message.content) == 2  # Original text + 1 image
            assert message.content[1]["type"] == "image_url"
            assert "data:image/png;base64," in message.content[1]["image_url"]["url"]

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_analyze_images_from_context_no_docs(self, mock_get_prompts, mock_get_config):
        """Test analyze_images_from_context with no documents"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        result = vlm.analyze_images_from_context([], "test question")

        assert result == ""

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.get_minio_operator')
    def test_analyze_images_from_context_with_docs(self, mock_get_minio, mock_get_prompts, mock_get_config):
        """Test analyze_images_from_context with documents"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        # Mock MinIO operator
        mock_minio = Mock()
        test_image = self.create_test_image_b64()
        mock_minio.get_payload.return_value = {"content": test_image}
        mock_get_minio.return_value = mock_minio

        # Mock document
        mock_doc = Mock()
        mock_doc.metadata = {
            "content_metadata": {"type": "image", "page_number": 1, "location": "test"},
            "collection_name": "test_collection",
            "source": {"source_id": "test.pdf"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)

        with patch('nvidia_rag.rag_server.vlm.get_unique_thumbnail_id') as mock_get_thumbnail_id:
            mock_get_thumbnail_id.return_value = "test_thumbnail_id"
            with patch.object(vlm, 'analyze_image', return_value="VLM response") as mock_analyze:
                result = vlm.analyze_images_from_context([mock_doc], "test question")

                assert result == "VLM response"
                mock_analyze.assert_called_once()

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_analyze_images_from_context_with_query_images(self, mock_get_prompts, mock_get_config):
        """Test analyze_images_from_context with query images"""
        mock_get_config.return_value = Mock()
        mock_get_config.return_value.vlm = Mock()
        mock_get_config.return_value.vlm.max_total_images = 4
        mock_get_config.return_value.vlm.max_query_images = 2
        mock_get_config.return_value.vlm.max_context_images = 2
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        test_image = self.create_test_image_b64()

        query_with_images = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image}"}}
        ]

        result = vlm.analyze_images_from_context([], query_with_images)

        # When no documents are provided, function returns empty string
        assert result == ""

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    @patch('nvidia_rag.rag_server.vlm.get_llm')
    def test_reason_on_vlm_response_empty_response(self, mock_get_llm, mock_get_prompts, mock_get_config):
        """Test reason_on_vlm_response with empty response"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)
        result = vlm.reason_on_vlm_response("question", "", [], {})

        assert result is False

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_reason_on_vlm_response_use_verdict(self, mock_get_prompts, mock_get_config):
        """Test reason_on_vlm_response with USE verdict"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)

        # Mock the entire chain creation process
        with patch('nvidia_rag.rag_server.vlm.get_llm') as mock_get_llm, \
             patch('nvidia_rag.rag_server.vlm.ChatPromptTemplate') as mock_prompt_template, \
             patch('nvidia_rag.rag_server.vlm.StrOutputParser') as mock_parser_class:

            # Create a mock chain that returns our expected result
            mock_chain = Mock()
            mock_chain.invoke.return_value = "USE the response"

            # Mock the prompt template
            mock_prompt = Mock()
            mock_prompt_template.from_messages.return_value = mock_prompt

            # Mock the parser
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            # Mock the LLM
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            # Create a custom class that supports the | operator
            class MockChain:
                def __init__(self, invoke_return_value):
                    self.invoke_return_value = invoke_return_value

                def invoke(self, *args, **kwargs):
                    return self.invoke_return_value

                def __or__(self, other):
                    return self

            # Replace the mocks with our custom chain
            mock_prompt_template.from_messages.return_value = MockChain("USE the response")
            mock_get_llm.return_value = MockChain("USE the response")
            mock_parser_class.return_value = MockChain("USE the response")

            result = vlm.reason_on_vlm_response("question", "VLM response", [], {})

        assert result is True

    @patch('nvidia_rag.rag_server.vlm.get_config')
    @patch('nvidia_rag.rag_server.vlm.get_prompts')
    def test_reason_on_vlm_response_dont_use_verdict(self, mock_get_prompts, mock_get_config):
        """Test reason_on_vlm_response with non-USE verdict"""
        mock_get_config.return_value = Mock()
        mock_get_prompts.return_value = {
            "vlm_template": "Test template: {question}",
            "vlm_response_reasoning_template": {"system": "test", "human": "test"}
        }

        vlm = VLM(self.vlm_model, self.vlm_endpoint)

        # Mock the entire chain creation process
        with patch('nvidia_rag.rag_server.vlm.get_llm') as mock_get_llm, \
             patch('nvidia_rag.rag_server.vlm.ChatPromptTemplate') as mock_prompt_template, \
             patch('nvidia_rag.rag_server.vlm.StrOutputParser') as mock_parser_class:

            # Create a custom class that supports the | operator
            class MockChain:
                def __init__(self, invoke_return_value):
                    self.invoke_return_value = invoke_return_value

                def invoke(self, *args, **kwargs):
                    return self.invoke_return_value

                def __or__(self, other):
                    return self

            # Replace the mocks with our custom chain
            mock_prompt_template.from_messages.return_value = MockChain("REJECT the response")
            mock_get_llm.return_value = MockChain("REJECT the response")
            mock_parser_class.return_value = MockChain("REJECT the response")

            result = vlm.reason_on_vlm_response("question", "VLM response", [], {})

        assert result is False

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

"""Unit tests for the embedding utility functions."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvidia_rag.utils.embedding import get_embedding_model


class TestGetEmbeddingModel:
    """Test cases for get_embedding_model function."""

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_nvidia_endpoints_with_url(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test getting embedding model with NVIDIA endpoints and custom URL."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        result = get_embedding_model("test-model", "test-url:8000")

        mock_sanitize.assert_called_once_with("test-url:8000", "test-model", "embedding")
        mock_nvidia_embeddings.assert_called_once_with(
            base_url="http://test-url:8000",
            model="test-model",
            truncate="END"
        )
        assert result == mock_embeddings

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_nvidia_endpoints_api_catalog(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test getting embedding model from API catalog."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""  # No URL, use API catalog
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        result = get_embedding_model("nvidia/nv-embedqa-e5-v5", "")

        mock_nvidia_embeddings.assert_called_once_with(
            model="nvidia/nv-embedqa-e5-v5",
            truncate="END"
        )
        assert result == mock_embeddings

    @patch("nvidia_rag.utils.embedding.get_embedding_model.cache_clear")
    @patch("nvidia_rag.utils.embedding.get_config")
    def test_get_embedding_model_unsupported_engine(self, mock_config, mock_cache_clear):
        """Test getting embedding model with unsupported engine."""
        mock_config.return_value.embeddings.model_engine = "unsupported-engine"

        mock_cache_clear()  # Clear cache before test
        with pytest.raises(RuntimeError, match="Unable to find any supported embedding model"):
            get_embedding_model("test-model-unsupported", "test-url-unsupported")

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_empty_url(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test getting embedding model with empty URL."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        result = get_embedding_model("test-model", "")

        mock_nvidia_embeddings.assert_called_once_with(
            model="test-model",
            truncate="END"
        )
        assert result == mock_embeddings

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_none_url(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test getting embedding model with None URL."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        result = get_embedding_model("test-model", None)

        mock_sanitize.assert_called_once_with(None, "test-model", "embedding")
        mock_nvidia_embeddings.assert_called_once_with(
            model="test-model",
            truncate="END"
        )
        assert result == mock_embeddings

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_special_characters_in_model_name(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test getting embedding model with special characters in model name."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        special_model_name = "nvidia/nv-embed-v1@special-version"
        result = get_embedding_model(special_model_name, "test-url")

        mock_nvidia_embeddings.assert_called_once_with(
            base_url="http://test-url:8000",
            model=special_model_name,
            truncate="END"
        )
        assert result == mock_embeddings

    @patch("nvidia_rag.utils.embedding.get_embedding_model.cache_clear")
    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_caching(self, mock_nvidia_embeddings, mock_sanitize, mock_config, mock_cache_clear):
        """Test that get_embedding_model uses caching."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        mock_cache_clear()  # Clear cache before test
        # First call
        result1 = get_embedding_model("test-model-cache", "test-url-cache")
        # Second call with same parameters should use cache
        result2 = get_embedding_model("test-model-cache", "test-url-cache")

        # Should only be called once due to caching
        assert mock_nvidia_embeddings.call_count == 1
        assert result1 == result2

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_get_embedding_model_different_parameters_no_cache(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test that get_embedding_model doesn't use cache for different parameters."""
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_embeddings = Mock()
        mock_nvidia_embeddings.return_value = mock_embeddings

        # First call
        result1 = get_embedding_model("test-model-1", "test-url")
        # Second call with different parameters should not use cache
        result2 = get_embedding_model("test-model-2", "test-url")

        # Should be called twice for different parameters
        assert mock_nvidia_embeddings.call_count == 2


class TestEmbeddingModelTorchHandling:
    """Test cases specifically for torch availability handling."""


class TestEmbeddingModelIntegration:
    """Integration tests for embedding model utilities."""

    @patch("nvidia_rag.utils.embedding.get_config")
    @patch("nvidia_rag.utils.embedding.sanitize_nim_url")
    @patch("nvidia_rag.utils.embedding.NVIDIAEmbeddings")
    def test_complete_nvidia_embedding_workflow(self, mock_nvidia_embeddings, mock_sanitize, mock_config):
        """Test complete workflow for NVIDIA embedding model creation."""
        # Setup mocks
        mock_config.return_value.embeddings.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://embedding-service:8080"
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_nvidia_embeddings.return_value = mock_embeddings

        # Test the workflow
        model = get_embedding_model("nvidia/nv-embedqa-e5-v5", "embedding-service:8080")

        # Verify the embedding model was created correctly
        mock_sanitize.assert_called_once_with("embedding-service:8080", "nvidia/nv-embedqa-e5-v5", "embedding")
        mock_nvidia_embeddings.assert_called_once_with(
            base_url="http://embedding-service:8080",
            model="nvidia/nv-embedqa-e5-v5",
            truncate="END"
        )

        # Test that the model can be used
        embedding = model.embed_query("test query")
        assert embedding == [0.1, 0.2, 0.3]






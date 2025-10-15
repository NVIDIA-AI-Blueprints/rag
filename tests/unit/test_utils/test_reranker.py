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

"""Unit tests for the reranker utility functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from nvidia_rag.utils.reranker import _get_ranking_model, get_ranking_model


class TestGetRankingModelPrivate:
    """Test cases for _get_ranking_model function."""

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_nvidia_endpoints_with_url(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test getting ranking model with NVIDIA endpoints and custom URL."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = _get_ranking_model("test-model", "test-url:8000", 5)
        
        mock_sanitize.assert_called_once_with("test-url:8000", "test-model", "ranking")
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=5,
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_nvidia_endpoints_with_model_name(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test getting ranking model with model name (API catalog)."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""  # No URL
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = _get_ranking_model("nvidia/nv-rerankqa-mistral-4b-v3", "", 10)
        
        mock_nvidia_rerank.assert_called_once_with(
            model="nvidia/nv-rerankqa-mistral-4b-v3",
            top_n=10,
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_get_ranking_model_nvidia_endpoints_no_url_no_model(self, mock_sanitize, mock_config):
        """Test getting ranking model with no URL and no model name."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""
        
        result = _get_ranking_model("", "", 4)
        
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    def test_get_ranking_model_unsupported_engine(self, mock_config):
        """Test getting ranking model with unsupported engine."""
        mock_config.return_value.ranking.model_engine = "unsupported-engine"
        
        result = _get_ranking_model("test-model", "test-url", 4)
        
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_exception_handling(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test exception handling in _get_ranking_model."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_nvidia_rerank.side_effect = Exception("Connection error")
        
        result = _get_ranking_model("test-model", "test-url", 4)
        
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_default_top_n(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test getting ranking model with default top_n parameter."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = _get_ranking_model("test-model", "test-url")  # No top_n specified
        
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=4,  # Default value
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_zero_top_n(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test getting ranking model with zero top_n parameter."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = _get_ranking_model("test-model", "test-url", 0)
        
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=4,
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_large_top_n(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test getting ranking model with large top_n parameter."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = _get_ranking_model("test-model", "test-url", 1000)
        
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=1000,
            truncate="END"
        )
        assert result == mock_reranker


class TestGetRankingModelPublic:
    """Test cases for get_ranking_model function."""

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_success_first_try(self, mock_private_get):
        """Test successful ranking model creation on first try."""
        mock_reranker = Mock()
        mock_private_get.return_value = mock_reranker
        
        result = get_ranking_model("test-model", "test-url", 5)
        
        mock_private_get.assert_called_once_with("test-model", "test-url", 5)
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_none_first_try_success_retry(self, mock_private_get):
        """Test ranking model creation fails first time, succeeds on retry."""
        mock_reranker = Mock()
        mock_private_get.side_effect = [None, mock_reranker]  # First call returns None, second succeeds
        
        result = get_ranking_model("test-model", "test-url", 5)
        
        assert mock_private_get.call_count == 2
        mock_private_get.assert_any_call("test-model", "test-url", 5)
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_none_both_tries(self, mock_private_get):
        """Test ranking model creation fails both times."""
        mock_private_get.return_value = None
        
        result = get_ranking_model("test-model", "test-url", 5)
        
        assert mock_private_get.call_count == 2
        assert result is None

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_cache_clear_called(self, mock_private_get):
        """Test that cache is cleared when first attempt returns None."""
        mock_private_get.side_effect = [None, Mock()]
        
        with patch.object(mock_private_get, 'cache_clear') as mock_cache_clear:
            get_ranking_model("test-model", "test-url", 5)
            mock_cache_clear.assert_called_once()

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_default_parameters(self, mock_private_get):
        """Test get_ranking_model with default parameters."""
        mock_reranker = Mock()
        mock_private_get.return_value = mock_reranker
        
        result = get_ranking_model()  # All default parameters
        
        mock_private_get.assert_called_once_with("", "", 4)
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_partial_parameters(self, mock_private_get):
        """Test get_ranking_model with partial parameters."""
        mock_reranker = Mock()
        mock_private_get.return_value = mock_reranker
        
        result = get_ranking_model(model="test-model")
        
        mock_private_get.assert_called_once_with("test-model", "", 4)
        assert result == mock_reranker


class TestRankingModelCaching:
    """Test cases for ranking model caching behavior."""

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_caching_same_parameters(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test that _get_ranking_model uses caching for same parameters."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        # First call
        result1 = _get_ranking_model("test-model", "test-url", 5)
        # Second call with same parameters should use cache
        result2 = _get_ranking_model("test-model", "test-url", 5)
        
        # Should only be called once due to caching
        assert mock_nvidia_rerank.call_count == 1
        assert result1 == result2 == mock_reranker

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_no_caching_different_parameters(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test that _get_ranking_model doesn't use cache for different parameters."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        # First call
        result1 = _get_ranking_model("test-model-1", "test-url", 5)
        # Second call with different parameters should not use cache
        result2 = _get_ranking_model("test-model-2", "test-url", 5)
        
        # Should be called twice for different parameters
        assert mock_nvidia_rerank.call_count == 2
        assert result1 == result2 == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_cache_clear_functionality(self, mock_private_get):
        """Test that cache clear functionality works."""
        # Setup mock to return None first, then success
        mock_reranker = Mock()
        mock_private_get.side_effect = [None, mock_reranker]
        
        # Mock the cache_clear method
        mock_private_get.cache_clear = Mock()
        
        result = get_ranking_model("test-model", "test-url", 5)
        
        # Verify cache_clear was called
        mock_private_get.cache_clear.assert_called_once()
        assert result == mock_reranker


class TestRankingModelIntegration:
    """Integration tests for ranking model utilities."""

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_complete_ranking_workflow_with_url(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test complete workflow for ranking model creation with URL."""
        # Setup mocks
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://rerank-service:8080"
        
        mock_reranker = Mock()
        mock_reranker.compress_documents.return_value = ["doc1", "doc2"]
        mock_nvidia_rerank.return_value = mock_reranker
        
        # Test the workflow
        model = get_ranking_model("nvidia/nv-rerankqa-mistral-4b-v3", "rerank-service:8080", 10)
        
        # Verify the ranking model was created correctly
        mock_sanitize.assert_called_once_with("rerank-service:8080", "nvidia/nv-rerankqa-mistral-4b-v3", "ranking")
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://rerank-service:8080",
            top_n=10,
            truncate="END"
        )
        
        # Test that the model can be used
        documents = ["doc1", "doc2", "doc3"]
        result = model.compress_documents("test query", documents)
        assert result == ["doc1", "doc2"]

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_complete_ranking_workflow_api_catalog(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test complete workflow for ranking model creation from API catalog."""
        # Setup mocks
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""  # No URL, use API catalog
        
        mock_reranker = Mock()
        mock_reranker.compress_documents.return_value = ["ranked_doc1"]
        mock_nvidia_rerank.return_value = mock_reranker
        
        # Test the workflow
        model = get_ranking_model("nvidia/nv-rerankqa-mistral-4b-v3", "", 5)
        
        # Verify the ranking model was created correctly
        mock_nvidia_rerank.assert_called_once_with(
            model="nvidia/nv-rerankqa-mistral-4b-v3",
            top_n=5,
            truncate="END"
        )
        
        # Test that the model can be used
        documents = ["doc1", "doc2"]
        result = model.compress_documents("test query", documents)
        assert result == ["ranked_doc1"]

    @patch("nvidia_rag.utils.reranker._get_ranking_model.cache_clear")
    @patch("nvidia_rag.utils.reranker.get_config")
    def test_error_handling_unsupported_engine(self, mock_config, mock_cache_clear):
        """Test error handling for unsupported ranking engine."""
        mock_config.return_value.ranking.model_engine = "unsupported-engine"
        
        mock_cache_clear()  # Clear cache before test
        # Should return None for unsupported engine
        result = get_ranking_model("test-model-unsupported", "test-url-unsupported", 5)
        # The function returns None for unsupported engines
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_with_special_characters(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test ranking model creation with special characters in model name."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        special_model_name = "nvidia/nv-rerank@v1.0-special"
        result = get_ranking_model(special_model_name, "test-url", 5)
        
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=5,
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model.cache_clear")
    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_resilience_to_failures(self, mock_nvidia_rerank, mock_sanitize, mock_config, mock_cache_clear):
        """Test that ranking model creation is resilient to temporary failures."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        
        # The function will fail both times due to the exception
        mock_reranker = Mock()
        mock_nvidia_rerank.side_effect = Exception("Temporary failure")
        
        # Mock the sanitize function to return a valid URL
        mock_sanitize.return_value = "http://test-url:8000"
        
        mock_cache_clear()  # Clear cache before test
        # This should fail both times and return None
        result = get_ranking_model("test-model-resilience", "test-url-resilience", 5)
        
        # Should be called at least once due to retry logic
        assert mock_nvidia_rerank.call_count >= 1
        # The result should be None since both attempts failed
        assert result is None


class TestRankingModelEdgeCases:
    """Test cases for edge cases and error conditions."""

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_empty_model_and_url(self, mock_sanitize, mock_config):
        """Test behavior with empty model and URL."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""
        
        result = get_ranking_model("", "", 5)
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_negative_top_n(self, mock_nvidia_rerank, mock_sanitize, mock_config):
        """Test behavior with negative top_n parameter."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker
        
        result = get_ranking_model("test-model", "test-url", -5)
        
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            top_n=4,
            truncate="END"
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model.cache_clear")
    @patch("nvidia_rag.utils.reranker.get_config")
    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_none_parameters(self, mock_sanitize, mock_config, mock_cache_clear):
        """Test behavior with None parameters."""
        mock_config.return_value.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""
        
        mock_cache_clear()  # Clear cache before test
        result = get_ranking_model(None, None, None)
        
        # Should handle None parameters gracefully - may be called twice due to retry logic
        assert mock_sanitize.call_count >= 1
        mock_sanitize.assert_called_with(None, None, "ranking")
        assert result is None

    @patch("nvidia_rag.utils.reranker.get_config")
    def test_config_access_error(self, mock_config):
        """Test behavior when config access fails."""
        mock_config.side_effect = Exception("Config error")
        
        # This should raise the exception as expected
        with pytest.raises(Exception, match="Config error"):
            get_ranking_model("test-model", "test-url", 5)






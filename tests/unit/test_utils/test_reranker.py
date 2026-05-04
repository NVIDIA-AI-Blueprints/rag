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

from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from nvidia_rag.utils.reranker import (
    NVIDIAVLMRerank,
    _build_vlm_rerank_invoke_url,
    _get_ranking_model,
    get_ranking_model,
)


class TestGetRankingModelPrivate:
    """Test cases for _get_ranking_model function."""

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_nvidia_endpoints_with_url(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test getting ranking model with NVIDIA endpoints and custom URL."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        result = _get_ranking_model(
            "test-model", "test-url:8000", 5, config=mock_config
        )

        mock_sanitize.assert_called_once_with("test-url:8000", "test-model", "ranking")
        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            api_key="test-api-key",
            top_n=5,
            truncate="END",
            default_headers={"source": "rag-blueprint"},
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_nvidia_endpoints_with_model_name(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test getting ranking model with model name (API catalog)."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_sanitize.return_value = ""  # No URL
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        result = _get_ranking_model(
            "nvidia/llama-nemotron-rerank-1b-v2", "", 10, config=mock_config
        )

        mock_nvidia_rerank.assert_called_once_with(
            model="nvidia/llama-nemotron-rerank-1b-v2",
            api_key="test-api-key",
            top_n=10,
            truncate="END",
            default_headers={"source": "rag-blueprint"},
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_get_ranking_model_nvidia_endpoints_no_url_no_model(self, mock_sanitize):
        """Test getting ranking model with no URL and no model name."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""

        # Should raise RuntimeError when no model or URL is provided
        with pytest.raises(
            RuntimeError, match="Ranking model configuration incomplete"
        ):
            _get_ranking_model("", "", 4, config=mock_config)

    def test_get_ranking_model_unsupported_engine(self):
        """Test getting ranking model with unsupported engine."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "unsupported-engine"

        # Should raise RuntimeError for unsupported engine
        with pytest.raises(RuntimeError, match="Unsupported ranking model engine"):
            _get_ranking_model("test-model", "test-url", 4, config=mock_config)

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_exception_handling(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test exception handling in _get_ranking_model."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_nvidia_rerank.side_effect = Exception("Connection error")

        # Exceptions should propagate
        with pytest.raises(Exception, match="Connection error"):
            _get_ranking_model("test-model", "test-url", 4, config=mock_config)

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_default_top_n(self, mock_nvidia_rerank, mock_sanitize):
        """Test getting ranking model with default top_n parameter."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        result = _get_ranking_model(
            "test-model", "test-url", config=mock_config
        )  # No top_n specified

        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            api_key="test-api-key",
            top_n=4,  # Default value
            truncate="END",
            default_headers={"source": "rag-blueprint"},
        )
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_zero_top_n(self, mock_nvidia_rerank, mock_sanitize):
        """Test getting ranking model with zero top_n parameter raises ValueError."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_sanitize.return_value = "http://test-url:8000"

        with pytest.raises(ValueError, match="reranker_top_k must be greater than 0, got 0"):
            _get_ranking_model("test-model", "test-url", 0, config=mock_config)

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_get_ranking_model_large_top_n(self, mock_nvidia_rerank, mock_sanitize):
        """Test getting ranking model with large top_n parameter."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        result = _get_ranking_model("test-model", "test-url", 1000, config=mock_config)

        mock_nvidia_rerank.assert_called_once_with(
            base_url="http://test-url:8000",
            api_key="test-api-key",
            top_n=1000,
            truncate="END",
            default_headers={"source": "rag-blueprint"},
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

        mock_private_get.assert_called_once_with("test-model", "test-url", 5, ANY)
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_propagates_exception(self, mock_private_get):
        """Test that get_ranking_model propagates exceptions from _get_ranking_model."""
        mock_private_get.side_effect = RuntimeError("Config incomplete")

        with pytest.raises(RuntimeError, match="Config incomplete"):
            get_ranking_model("test-model", "test-url", 5)

        mock_private_get.assert_called_once_with("test-model", "test-url", 5, ANY)

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_default_parameters(self, mock_private_get):
        """Test get_ranking_model with default parameters."""
        mock_reranker = Mock()
        mock_private_get.return_value = mock_reranker

        result = get_ranking_model()  # All default parameters

        mock_private_get.assert_called_once_with("", "", 4, ANY)
        assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker._get_ranking_model")
    def test_get_ranking_model_partial_parameters(self, mock_private_get):
        """Test get_ranking_model with partial parameters."""
        mock_reranker = Mock()
        mock_private_get.return_value = mock_reranker

        result = get_ranking_model(model="test-model")

        mock_private_get.assert_called_once_with("test-model", "", 4, ANY)
        assert result == mock_reranker


class TestRankingModelCaching:
    """Test cases for ranking model caching behavior."""

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_caching_same_parameters(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test that _get_ranking_model creates new instances for each call (no caching with config)."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        # First call
        result1 = _get_ranking_model("test-model", "test-url", 5, config=mock_config)
        # Second call with same parameters creates a new instance (no caching with config parameter)
        result2 = _get_ranking_model("test-model", "test-url", 5, config=mock_config)

        # Should be called twice since caching was removed when config parameter was added
        assert mock_nvidia_rerank.call_count == 2
        assert result1 == result2 == mock_reranker

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_no_caching_different_parameters(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test that _get_ranking_model doesn't use cache for different parameters."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        # First call
        result1 = _get_ranking_model("test-model-1", "test-url", 5, config=mock_config)
        # Second call with different parameters should not use cache
        result2 = _get_ranking_model("test-model-2", "test-url", 5, config=mock_config)

        # Should be called twice for different parameters
        assert mock_nvidia_rerank.call_count == 2
        assert result1 == result2 == mock_reranker

    def test_cache_clear_functionality(self):
        """Test that cache_clear functionality is not present (caching removed with config parameter)."""
        # Caching was removed when config parameter was added since config objects aren't hashable
        # Verify that cache_clear method does not exist
        assert not hasattr(_get_ranking_model, "cache_clear")


class TestRankingModelIntegration:
    """Integration tests for ranking model utilities."""

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_complete_ranking_workflow_with_url(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test complete workflow for ranking model creation with URL."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://rerank-service:8080"

        mock_reranker = Mock()
        mock_reranker.compress_documents.return_value = ["doc1", "doc2"]
        mock_nvidia_rerank.return_value = mock_reranker

        # Patch config in the actual function
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.return_value = mock_reranker

            # Test the workflow
            model = get_ranking_model(
                "nvidia/llama-nemotron-rerank-1b-v2", "rerank-service:8080", 10
            )

            # Test that the model can be used
            documents = ["doc1", "doc2", "doc3"]
            result = model.compress_documents("test query", documents)
            assert result == ["doc1", "doc2"]

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_complete_ranking_workflow_api_catalog(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test complete workflow for ranking model creation from API catalog."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""  # No URL, use API catalog

        mock_reranker = Mock()
        mock_reranker.compress_documents.return_value = ["ranked_doc1"]
        mock_nvidia_rerank.return_value = mock_reranker

        # Patch config in the actual function
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.return_value = mock_reranker

            # Test the workflow
            model = get_ranking_model("nvidia/llama-nemotron-rerank-1b-v2", "", 5)

            # Test that the model can be used
            documents = ["doc1", "doc2"]
            result = model.compress_documents("test query", documents)
            assert result == ["ranked_doc1"]

    def test_error_handling_unsupported_engine(self):
        """Test error handling for unsupported ranking engine."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "unsupported-engine"

        # Should raise RuntimeError for unsupported engine
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.side_effect = RuntimeError(
                "Unsupported ranking model engine"
            )
            with pytest.raises(RuntimeError, match="Unsupported ranking model engine"):
                get_ranking_model("test-model-unsupported", "test-url-unsupported", 5)

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_ranking_model_with_special_characters(
        self, mock_nvidia_rerank, mock_sanitize
    ):
        """Test ranking model creation with special characters in model name."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        special_model_name = "nvidia/nv-rerank@v1.0-special"

        # Patch config in the actual function
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.return_value = mock_reranker
            result = get_ranking_model(special_model_name, "test-url", 5)
            assert result == mock_reranker

    def test_ranking_model_resilience_to_failures(self):
        """Test that ranking model creation propagates exceptions."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"

        # This should propagate the exception
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.side_effect = Exception("Temporary failure")
            with pytest.raises(Exception, match="Temporary failure"):
                get_ranking_model("test-model-resilience", "test-url-resilience", 5)


class TestRankingModelEdgeCases:
    """Test cases for edge cases and error conditions."""

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_empty_model_and_url(self, mock_sanitize):
        """Test behavior with empty model and URL."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""

        # Should raise RuntimeError when no model or URL provided
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.side_effect = RuntimeError(
                "Ranking model configuration incomplete"
            )
            with pytest.raises(
                RuntimeError, match="Ranking model configuration incomplete"
            ):
                get_ranking_model("", "", 5)

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    @patch("nvidia_rag.utils.reranker.NVIDIARerank")
    def test_negative_top_n(self, mock_nvidia_rerank, mock_sanitize):
        """Test behavior with negative top_n parameter."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = "http://test-url:8000"
        mock_reranker = Mock()
        mock_nvidia_rerank.return_value = mock_reranker

        # Patch config in the actual function
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.return_value = mock_reranker
            result = get_ranking_model("test-model", "test-url", -5)
            assert result == mock_reranker

    @patch("nvidia_rag.utils.reranker.sanitize_nim_url")
    def test_none_parameters(self, mock_sanitize):
        """Test behavior with None parameters."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_sanitize.return_value = ""

        # With nvidia-ai-endpoints engine but no model/url, should raise RuntimeError
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.side_effect = RuntimeError(
                "Ranking model configuration incomplete"
            )
            with pytest.raises(
                RuntimeError, match="Ranking model configuration incomplete"
            ):
                get_ranking_model(None, None, None)

    def test_config_access_error(self):
        """Test behavior when config access fails."""

        # The function should propagate the exception
        with patch("nvidia_rag.utils.reranker._get_ranking_model") as mock_get_model:
            mock_get_model.side_effect = Exception("Config error")
            with pytest.raises(Exception, match="Config error"):
                get_ranking_model("test-model", "test-url", 5)


class TestVLMRerankInvokeUrl:
    """Tests for VLM reranker endpoint selection."""

    def test_build_invoke_url_for_hosted_endpoint(self):
        """Hosted NVIDIA endpoints should use the retrieval reranking path."""
        url = _build_vlm_rerank_invoke_url(
            "https://ai.api.nvidia.com/v1",
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
        )

        assert (
            url
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"
        )

    def test_build_invoke_url_for_self_hosted_endpoint(self):
        """Self-hosted NIMs should use the OpenAI-style ranking path."""
        url = _build_vlm_rerank_invoke_url(
            "nemotron-ranking-vl-ms:8000",
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
        )

        assert url == "http://nemotron-ranking-vl-ms:8000/v1/ranking"


class TestNVIDIAVLMRerank:
    """Focused tests for the requests-based multimodal reranker."""

    def test_build_payload_attaches_image_when_available(self):
        """Payloads should include passage images when enable_image_input is True."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
            enable_image_input=True,
        )
        doc = Document(
            page_content="A chart about tax rates.",
            metadata={
                "content_metadata": {
                    "type": "image",
                    "page_number": 1,
                    "location": [0, 0, 1, 1],
                },
                "collection_name": "demo",
                "source": {
                    "source_id": "sample.pdf",
                    "source_location": "s3://default-bucket/demo/page.png",
                },
            },
        )

        with patch.object(ranker, "_build_image_data_url", return_value="data:image/png;base64,abc"):
            payload = ranker._build_payload("What does the chart show?", [doc])

        assert payload["query"] == {"text": "What does the chart show?"}
        assert payload["passages"] == [
            {
                "text": "A chart about tax rates.",
                "image": "data:image/png;base64,abc",
            }
        ]

    def test_build_payload_omits_image_when_flag_disabled(self):
        """Payloads should be text-only when enable_image_input is False (default)."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
        )
        doc = Document(
            page_content="A chart about tax rates.",
            metadata={
                "content_metadata": {"type": "image", "page_number": 1, "location": [0, 0, 1, 1]},
                "collection_name": "demo",
                "source": {"source_id": "sample.pdf", "source_location": "s3://default-bucket/demo/page.png"},
            },
        )

        with patch.object(ranker, "_build_image_data_url", return_value="data:image/png;base64,abc") as mock_builder:
            payload = ranker._build_payload("What does the chart show?", [doc])

        mock_builder.assert_not_called()
        assert payload["passages"] == [{"text": "A chart about tax rates."}]

    def test_compress_documents_returns_ranked_documents_in_api_order(self):
        """Returned documents should follow the API ranking order."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
            top_n=2,
        )
        docs = [
            Document(page_content="doc-0", metadata={}),
            Document(page_content="doc-1", metadata={}),
            Document(page_content="doc-2", metadata={}),
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "rankings": [
                {"index": 2, "logit": 3.5},
                {"index": 0, "logit": 1.2},
            ]
        }
        mock_response.raise_for_status.return_value = None
        ranker._session = Mock()
        ranker._session.post.return_value = mock_response

        reranked = ranker.compress_documents(docs, "test query")

        assert [doc.page_content for doc in reranked] == ["doc-2", "doc-0"]
        assert reranked[0].metadata["relevance_score"] == 3.5
        assert reranked[1].metadata["relevance_score"] == 1.2

    def test_compress_documents_rejects_out_of_range_index(self):
        """Invalid API indices should raise a clear runtime error."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
        )
        mock_response = Mock()
        mock_response.json.return_value = {"rankings": [{"index": 99, "logit": 1.0}]}
        mock_response.raise_for_status.return_value = None
        ranker._session = Mock()
        ranker._session.post.return_value = mock_response

        with pytest.raises(
            RuntimeError, match="invalid response from VLM reranker: index out of range"
        ):
            ranker.compress_documents([Document(page_content="doc", metadata={})], "query")


class TestGetRankingModelVLM:
    """Factory tests specific to the multimodal reranker path."""

    @patch("nvidia_rag.utils.reranker.NVIDIAVLMRerank")
    def test_get_ranking_model_uses_vlm_reranker_for_vl_model(self, mock_vlm_rerank):
        """VL model ids should select the requests-based VLM reranker."""
        mock_config = MagicMock()
        mock_config.ranking.model_engine = "nvidia-ai-endpoints"
        mock_config.ranking.get_api_key.return_value = "test-api-key"
        mock_reranker = Mock()
        mock_vlm_rerank.return_value = mock_reranker

        result = _get_ranking_model(
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
            "nemotron-ranking-vl-ms:8000",
            5,
            config=mock_config,
        )

        mock_vlm_rerank.assert_called_once_with(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000/v1",
            api_key="test-api-key",
            top_n=5,
            default_headers={"source": "rag-blueprint"},
            config=mock_config,
        )
        assert result == mock_reranker

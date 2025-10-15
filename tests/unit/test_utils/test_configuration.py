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

"""Unit tests for the application configuration classes."""

import json
import os
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest
import yaml

from nvidia_rag.utils.configuration import (
    AppConfig,
    VectorStoreConfig,
    LLMConfig,
    ModelParametersConfig,
    QueryRewriterConfig,
    TextSplitterConfig,
    EmbeddingConfig,
    RankingConfig,
    RetrieverConfig,
    TracingConfig,
    VLMConfig,
    MinioConfig,
    SummarizerConfig,
    NvIngestConfig,
)


class TestVectorStoreConfig:
    """Test cases for VectorStoreConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig.from_dict({})

        assert config.name == "milvus"
        assert config.url == "http://localhost:19530"
        assert config.nlist == 64
        assert config.nprobe == 16
        assert config.index_type == "GPU_CAGRA"
        assert config.enable_gpu_index is True
        assert config.enable_gpu_search is True
        assert config.search_type == "dense"
        assert config.default_collection_name == "multimodal_data"


    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_custom_names(self):
        """Test custom environment variable names."""
        env_vars = {
            "COLLECTION_NAME": "test_collection",
        }

        with patch.dict(os.environ, env_vars):
            config = VectorStoreConfig.from_dict({})

            assert config.default_collection_name == "test_collection"

class TestLLMConfig:
    """Test cases for LLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig.from_dict({})

        assert config.server_url == ""
        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert isinstance(config.parameters, ModelParametersConfig)
        assert config.parameters.max_tokens == 32768
        assert config.parameters.temperature == 0
        assert config.parameters.top_p == 1.0


    def test_get_model_parameters_default(self):
        """Test get_model_parameters with default model (nemotron pattern)."""
        config = LLMConfig.from_dict({})
        params = config.get_model_parameters()

        # Default model contains "llama-3.3-nemotron-super-49b" so it triggers nemotron logic
        expected = {
            "max_tokens": 32768,
            "temperature": 0,
            "top_p": 1.0
        }
        assert params == expected

    def test_get_model_parameters_generic(self):
        """Test get_model_parameters with a generic model (no special patterns)."""
        config = LLMConfig.from_dict({"modelName": "meta/llama-3.1-8b-instruct"})
        params = config.get_model_parameters()

        # Generic model should use the base parameter values
        expected = {
            "max_tokens": 32768,
            "temperature": 0,
            "top_p": 1.0
        }
        assert params == expected


class TestQueryRewriterConfig:
    """Test cases for QueryRewriterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QueryRewriterConfig.from_dict({})

        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.server_url == ""
        assert config.enable_query_rewriter is False



class TestTextSplitterConfig:
    """Test cases for TextSplitterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TextSplitterConfig.from_dict({})

        assert config.model_name == "Snowflake/snowflake-arctic-embed-l"
        assert config.chunk_size == 510
        assert config.chunk_overlap == 200


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig.from_dict({})

        assert config.model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert config.dimensions == 2048
        assert config.server_url == ""


class TestRankingConfig:
    """Test cases for RankingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RankingConfig.from_dict({})

        assert config.model_name == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert config.server_url == ""
        assert config.enable_reranker is True


class TestRetrieverConfig:
    """Test cases for RetrieverConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetrieverConfig.from_dict({})

        assert config.top_k == 10
        assert config.vdb_top_k == 100
        assert config.score_threshold == 0.25
        assert config.nr_url == "http://retrieval-ms:8000"
        assert config.nr_pipeline == "ranked_hybrid"


class TestMinioConfig:
    """Test cases for MinioConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MinioConfig.from_dict({})

        assert config.endpoint == "localhost:9010"
        assert config.access_key == "minioadmin"
        assert config.secret_key == "minioadmin"


class TestSummarizerConfig:
    """Test cases for SummarizerConfig."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_values(self):
        """Test default configuration values."""
        config = SummarizerConfig.from_dict({})

        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.server_url == ""
        assert config.max_chunk_length == 50000
        assert config.chunk_overlap == 200

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_custom_names(self):
        """Test custom environment variable names."""
        env_vars = {
            "SUMMARY_LLM": "custom/summarizer-model",
            "SUMMARY_LLM_SERVERURL": "http://summarizer:8080",
            "SUMMARY_LLM_MAX_CHUNK_LENGTH": "75000",
            "SUMMARY_CHUNK_OVERLAP": "300"
        }

        with patch.dict(os.environ, env_vars):
            config = SummarizerConfig.from_dict({})

            assert config.model_name == "custom/summarizer-model"
            assert config.server_url == "http://summarizer:8080"
            assert config.max_chunk_length == 75000
            assert config.chunk_overlap == 300


class TestNvIngestConfig:
    """Test cases for NvIngestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NvIngestConfig.from_dict({})

        assert config.message_client_hostname == "localhost"
        assert config.message_client_port == 7670
        assert config.extract_text is True
        assert config.extract_infographics is False
        assert config.extract_tables is True
        assert config.extract_charts is True
        assert config.extract_images is False
        assert config.pdf_extract_method == "None"
        assert config.text_depth == "page"
        assert config.tokenizer == "intfloat/e5-large-unsupervised"
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 150
        assert config.caption_model_name == "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
        assert config.caption_endpoint_url == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert config.enable_pdf_splitter is True


class TestAppConfig:
    """Test cases for the main AppConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AppConfig.from_dict({})

        # Test that all nested configs are properly initialized
        assert isinstance(config.vector_store, VectorStoreConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.query_rewriter, QueryRewriterConfig)
        assert isinstance(config.text_splitter, TextSplitterConfig)
        assert isinstance(config.embeddings, EmbeddingConfig)
        assert isinstance(config.ranking, RankingConfig)
        assert isinstance(config.retriever, RetrieverConfig)
        assert isinstance(config.nv_ingest, NvIngestConfig)
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.vlm, VLMConfig)
        assert isinstance(config.minio, MinioConfig)
        assert isinstance(config.summarizer, SummarizerConfig)

        # Test top-level boolean flags
        assert config.enable_guardrails is False
        assert config.enable_citations is True
        assert config.enable_vlm_inference is False
        assert config.temp_dir == "./tmp-data"

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_top_level(self):
        """Test top-level environment variables."""
        env_vars = {
            "ENABLE_GUARDRAILS": "true",
            "ENABLE_CITATIONS": "false",
            "ENABLE_VLM_INFERENCE": "true",
            "TEMP_DIR": "/custom/temp"
        }

        with patch.dict(os.environ, env_vars):
            config = AppConfig.from_dict({})

            assert config.enable_guardrails is True
            assert config.enable_citations is False
            assert config.enable_vlm_inference is True
            assert config.temp_dir == "/custom/temp"

    @patch.dict(os.environ, {}, clear=True)
    def test_nested_environment_variables(self):
        """Test that nested configuration environment variables work."""
        env_vars = {
            "APP_VECTORSTORE_NAME": "custom_vectorstore",
            "APP_LLM_MODELNAME": "custom/llm-model",
            "ENABLE_RERANKER": "false",
            "MINIO_ENDPOINT": "custom-minio:9000"
        }

        with patch.dict(os.environ, env_vars):
            config = AppConfig.from_dict({})

            assert config.vector_store.name == "custom_vectorstore"
            assert config.llm.model_name == "custom/llm-model"
            assert config.ranking.enable_reranker is False
            assert config.minio.endpoint == "custom-minio:9000"

    def test_from_dict_nested_structure(self):
        """Test loading from dictionary with nested structure."""
        data = {
            "vectorStore": {
                "name": "elasticsearch",
                "url": "http://es:9200"
            },
            "llm": {
                "modelName": "custom/model",
                "serverUrl": "http://llm:8080"
            },
            "enableGuardrails": True,
            "enableCitations": False,
            "tempDir": "/custom/temp"
        }

        config = AppConfig.from_dict(data)

        assert config.vector_store.name == "elasticsearch"
        assert config.vector_store.url == "http://es:9200"
        assert config.llm.model_name == "custom/model"
        assert config.llm.server_url == "http://llm:8080"
        assert config.enable_guardrails is True
        assert config.enable_citations is False
        assert config.temp_dir == "/custom/temp"

class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""

    @patch.dict(os.environ, {}, clear=True)
    def test_complex_environment_scenario(self):
        """Test a complex scenario with multiple environment variables."""
        # Set up a comprehensive environment
        env_vars = {
            # Vector store config
            "APP_VECTORSTORE_NAME": "elasticsearch",
            "COLLECTION_NAME": "test_collection",

            # LLM config
            "APP_LLM_MODELNAME": "custom/llm-model",
            "APP_LLM_SERVERURL": "http://llm:8080",

            # Feature flags
            "ENABLE_GUARDRAILS": "true",
            "ENABLE_CITATIONS": "false",
            "ENABLE_RERANKER": "false",
            "ENABLE_VLM_INFERENCE": "true",

            # Minio config
            "MINIO_ENDPOINT": "minio.example.com:9000",
            "MINIO_ACCESSKEY": "test_key",
            "MINIO_SECRETKEY": "test_secret",

            # Other configs
            "TEMP_DIR": "/custom/temp",
            "VECTOR_DB_TOPK": "50"
        }

        with patch.dict(os.environ, env_vars):
            config = AppConfig.from_dict({})

            # Verify all environment variables are applied correctly
            assert config.vector_store.name == "elasticsearch"
            assert config.vector_store.default_collection_name == "test_collection"
            assert config.llm.model_name == "custom/llm-model"
            assert config.llm.server_url == "http://llm:8080"
            assert config.enable_guardrails is True
            assert config.enable_citations is False
            assert config.ranking.enable_reranker is False
            assert config.enable_vlm_inference is True
            assert config.minio.endpoint == "minio.example.com:9000"
            assert config.minio.access_key == "test_key"
            assert config.minio.secret_key == "test_secret"
            assert config.temp_dir == "/custom/temp"
            assert config.retriever.vdb_top_k == 50
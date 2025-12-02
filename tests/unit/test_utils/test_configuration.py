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
    EmbeddingConfig,
    LLMConfig,
    MinioConfig,
    ModelParametersConfig,
    NvidiaRAGConfig,
    NvIngestConfig,
    QueryRewriterConfig,
    RankingConfig,
    RetrieverConfig,
    SearchType,
    SummarizerConfig,
    TextSplitterConfig,
    TracingConfig,
    VectorStoreConfig,
    VLMConfig,
)


class TestVectorStoreConfig:
    """Test cases for VectorStoreConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig()

        assert config.name == "milvus"
        assert config.url == "http://localhost:19530"
        assert config.nlist == 64
        assert config.nprobe == 16
        assert config.index_type == "GPU_CAGRA"
        assert config.search_type == SearchType.DENSE
        assert config.default_collection_name == "multimodal_data"

    def test_search_type_enum_default(self):
        """Test that search_type default is SearchType.DENSE enum."""
        config = VectorStoreConfig()

        # Verify it's the correct enum type
        assert isinstance(config.search_type, SearchType)
        assert config.search_type == SearchType.DENSE
        # StrEnum also supports string comparison
        assert config.search_type == "dense"

    @patch.dict(os.environ, {}, clear=True)
    def test_search_type_enum_from_env_hybrid(self):
        """Test that search_type can be set via environment variable to hybrid."""
        with patch.dict(os.environ, {"APP_VECTORSTORE_SEARCHTYPE": "hybrid"}):
            config = VectorStoreConfig()

            assert isinstance(config.search_type, SearchType)
            assert config.search_type == SearchType.HYBRID
            assert config.search_type == "hybrid"

    @patch.dict(os.environ, {}, clear=True)
    def test_search_type_enum_from_env_dense(self):
        """Test that search_type can be set via environment variable to dense."""
        with patch.dict(os.environ, {"APP_VECTORSTORE_SEARCHTYPE": "dense"}):
            config = VectorStoreConfig()

            assert isinstance(config.search_type, SearchType)
            assert config.search_type == SearchType.DENSE

    @patch.dict(os.environ, {}, clear=True)
    def test_search_type_enum_invalid_value_raises_error(self):
        """Test that invalid search_type value raises validation error."""
        from pydantic import ValidationError

        with patch.dict(os.environ, {"APP_VECTORSTORE_SEARCHTYPE": "invalid_type"}):
            with pytest.raises(ValidationError) as exc_info:
                VectorStoreConfig()

            # Verify the error message mentions the valid options
            error_message = str(exc_info.value)
            assert "search_type" in error_message
            assert "dense" in error_message or "hybrid" in error_message

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_custom_names(self):
        """Test custom environment variable names."""
        env_vars = {
            "COLLECTION_NAME": "test_collection",
        }

        with patch.dict(os.environ, env_vars):
            config = VectorStoreConfig()

            assert config.default_collection_name == "test_collection"


class TestLLMConfig:
    """Test cases for LLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.server_url == ""
        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert isinstance(config.parameters, ModelParametersConfig)
        assert config.parameters.max_tokens == 32768
        assert config.parameters.temperature == 0
        assert config.parameters.top_p == 1.0

    def test_get_model_parameters_default(self):
        """Test get_model_parameters with default model (nemotron pattern)."""
        config = LLMConfig()
        params = config.get_model_parameters()

        # Default model contains "llama-3.3-nemotron-super-49b" so it triggers nemotron logic
        expected = {
            "min_tokens": 0,
            "ignore_eos": False,
            "max_tokens": 32768,
            "min_thinking_tokens": 1,
            "max_thinking_tokens": 8192,
            "temperature": 0,
            "top_p": 1.0,
        }
        assert params == expected

    def test_get_model_parameters_generic(self):
        """Test get_model_parameters with a generic model (no special patterns)."""
        config = LLMConfig(model_name="meta/llama-3.1-8b-instruct")
        params = config.get_model_parameters()

        # Generic model should use the base parameter values
        expected = {
            "min_tokens": 0,
            "ignore_eos": False,
            "max_tokens": 32768,
            "min_thinking_tokens": 1,
            "max_thinking_tokens": 8192,
            "temperature": 0,
            "top_p": 1.0,
        }
        assert params == expected


class TestQueryRewriterConfig:
    """Test cases for QueryRewriterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QueryRewriterConfig()

        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.server_url == ""
        assert config.enable_query_rewriter is False


class TestTextSplitterConfig:
    """Test cases for TextSplitterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TextSplitterConfig()

        assert config.model_name == "Snowflake/snowflake-arctic-embed-l"
        assert config.chunk_size == 510
        assert config.chunk_overlap == 200


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert config.dimensions == 2048
        assert config.server_url == ""


class TestRankingConfig:
    """Test cases for RankingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RankingConfig()

        assert config.model_name == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        assert config.model_engine == "nvidia-ai-endpoints"
        assert config.server_url == ""
        assert config.enable_reranker is True


class TestRetrieverConfig:
    """Test cases for RetrieverConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetrieverConfig()

        assert config.top_k == 10
        assert config.vdb_top_k == 100
        assert config.score_threshold == 0.25
        assert config.nr_url == "http://retrieval-ms:8000"
        assert config.nr_pipeline == "ranked_hybrid"


class TestMinioConfig:
    """Test cases for MinioConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MinioConfig()

        assert config.endpoint == "localhost:9010"
        assert config.access_key.get_secret_value() == "minioadmin"
        assert config.secret_key.get_secret_value() == "minioadmin"


class TestSummarizerConfig:
    """Test cases for SummarizerConfig."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_values(self):
        """Test default configuration values."""
        config = SummarizerConfig()

        assert config.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert config.server_url == ""
        assert config.max_chunk_length == 9000
        assert config.chunk_overlap == 400
        assert config.temperature == 0.0
        assert config.top_p == 1.0

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_custom_names(self):
        """Test custom environment variable names."""
        env_vars = {
            "SUMMARY_LLM": "custom/summarizer-model",
            "SUMMARY_LLM_SERVERURL": "http://summarizer:8080",
            "SUMMARY_LLM_MAX_CHUNK_LENGTH": "75000",
            "SUMMARY_CHUNK_OVERLAP": "300",
            "SUMMARY_LLM_TEMPERATURE": "0.5",
            "SUMMARY_LLM_TOP_P": "0.9",
        }

        with patch.dict(os.environ, env_vars):
            config = SummarizerConfig()

            assert config.model_name == "custom/summarizer-model"
            assert config.server_url == "http://summarizer:8080"
            assert config.max_chunk_length == 75000
            assert config.chunk_overlap == 300
            assert config.temperature == 0.5
            assert config.top_p == 0.9


class TestNvIngestConfig:
    """Test cases for NvIngestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NvIngestConfig()

        assert config.message_client_hostname == "localhost"
        assert config.message_client_port == 7670
        assert config.extract_text is True
        assert config.extract_infographics is False
        assert config.extract_tables is True
        assert config.extract_charts is True
        assert config.extract_images is False
        assert config.pdf_extract_method is None
        assert config.text_depth == "page"
        assert config.tokenizer == "intfloat/e5-large-unsupervised"
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 150
        assert config.caption_model_name == "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
        assert (
            config.caption_endpoint_url
            == "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        assert config.enable_pdf_splitter is True

    @pytest.mark.parametrize(
        "input_value",
        [
            "None",   # String "None"
            "none",   # Lowercase
            "NONE",   # Uppercase
            "null",   # YAML null string
            "NULL",   # Uppercase null
            "",       # Empty string
        ],
    )
    def test_pdf_extract_method_normalizes_none_strings_to_none(self, input_value):
        """Test that string representations of None are normalized to Python None."""
        config = NvIngestConfig(pdf_extract_method=input_value)
        assert config.pdf_extract_method is None

    def test_pdf_extract_method_preserves_valid_values(self):
        """Test that valid extraction method strings are preserved."""
        config = NvIngestConfig(pdf_extract_method="pdfium")
        assert config.pdf_extract_method == "pdfium"

        config = NvIngestConfig(pdf_extract_method="tesseract")
        assert config.pdf_extract_method == "tesseract"

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.parametrize(
        "env_value",
        ["None", "none", "null", ""],
    )
    def test_pdf_extract_method_normalizes_none_from_env_var(self, env_value):
        """Test that string None values from environment variables are normalized."""
        with patch.dict(os.environ, {"APP_NVINGEST_PDFEXTRACTMETHOD": env_value}):
            config = NvIngestConfig()
            assert config.pdf_extract_method is None

    @patch.dict(os.environ, {}, clear=True)
    def test_pdf_extract_method_preserves_valid_value_from_env_var(self):
        """Test that valid extraction method from environment variable is preserved."""
        with patch.dict(os.environ, {"APP_NVINGEST_PDFEXTRACTMETHOD": "pdfium"}):
            config = NvIngestConfig()
            assert config.pdf_extract_method == "pdfium"

    def test_pdf_extract_method_none_in_full_config(self):
        """Test pdf_extract_method normalization through NvidiaRAGConfig."""
        # Test via dict (simulates YAML loading)
        config = NvidiaRAGConfig.from_dict({
            "nv_ingest": {"pdf_extract_method": "None"}
        })
        assert config.nv_ingest.pdf_extract_method is None

        config = NvidiaRAGConfig.from_dict({
            "nv_ingest": {"pdf_extract_method": "null"}
        })
        assert config.nv_ingest.pdf_extract_method is None

        # Valid value should be preserved
        config = NvidiaRAGConfig.from_dict({
            "nv_ingest": {"pdf_extract_method": "pdfium"}
        })
        assert config.nv_ingest.pdf_extract_method == "pdfium"


class TestNvidiaRAGConfig:
    """Test cases for the main NvidiaRAGConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NvidiaRAGConfig.from_dict({})

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
            "TEMP_DIR": "/custom/temp",
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig.from_dict({})

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
            "MINIO_ENDPOINT": "custom-minio:9000",
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig.from_dict({})

            assert config.vector_store.name == "custom_vectorstore"
            assert config.llm.model_name == "custom/llm-model"
            assert config.ranking.enable_reranker is False
            assert config.minio.endpoint == "custom-minio:9000"

    def test_from_dict_nested_structure(self):
        """Test loading from dictionary with nested structure."""
        data = {
            "vector_store": {"name": "elasticsearch", "url": "http://es:9200"},
            "llm": {"model_name": "custom/model", "server_url": "http://llm:8080"},
            "enable_guardrails": True,
            "enable_citations": False,
            "temp_dir": "/custom/temp",
        }

        config = NvidiaRAGConfig.from_dict(data)

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
            "VECTOR_DB_TOPK": "50",
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig.from_dict({})

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
            assert config.minio.access_key.get_secret_value() == "test_key"
            assert config.minio.secret_key.get_secret_value() == "test_secret"
            assert config.temp_dir == "/custom/temp"
            assert config.retriever.vdb_top_k == 50

    @patch.dict(os.environ, {}, clear=True)
    def test_quoted_boolean_environment_variables(self):
        """Test that quoted boolean values from Docker Compose are handled correctly."""
        # Simulate Docker Compose setting boolean values as quoted strings
        env_vars = {
            "APP_TRACING_ENABLED": '"False"',  # Docker Compose style: "False"
            "ENABLE_GUARDRAILS": '"True"',  # Docker Compose style: "True"
            "ENABLE_CITATIONS": '"false"',  # lowercase with quotes
            "ENABLE_RERANKER": '"true"',  # lowercase with quotes
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify that quoted boolean strings are correctly parsed as booleans
            assert config.tracing.enabled is False
            assert config.enable_guardrails is True
            assert config.enable_citations is False
            assert config.ranking.enable_reranker is True

    @patch.dict(os.environ, {}, clear=True)
    def test_quoted_string_environment_variables(self):
        """Test that quoted string values from Docker Compose are handled correctly."""
        # Simulate Docker Compose setting string values as quoted strings
        env_vars = {
            "APP_VECTORSTORE_NAME": '"milvus"',  # Docker Compose style with quotes
            "APP_VECTORSTORE_URL": '"http://milvus:19530"',
            "APP_LLM_MODELNAME": '"nvidia/llama-3.3-nemotron-super-49b-v1.5"',
            "COLLECTION_NAME": '"test_collection"',
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify that quoted strings are correctly stripped
            assert config.vector_store.name == "milvus"
            assert config.vector_store.url == "http://milvus:19530"
            assert config.llm.model_name == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
            assert config.vector_store.default_collection_name == "test_collection"

    @patch.dict(os.environ, {}, clear=True)
    def test_mixed_quote_types_environment_variables(self):
        """Test handling of both single and double quotes in environment variables."""
        env_vars = {
            "APP_VECTORSTORE_NAME": "'elasticsearch'",  # Single quotes
            "APP_VECTORSTORE_URL": '"http://es:9200"',  # Double quotes
            "COLLECTION_NAME": ' "test_collection" ',  # Quotes with whitespace
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify both quote types and whitespace are handled
            assert config.vector_store.name == "elasticsearch"
            assert config.vector_store.url == "http://es:9200"
            assert config.vector_store.default_collection_name == "test_collection"

    @patch.dict(os.environ, {}, clear=True)
    def test_secretstr_from_environment_variables(self):
        """Test that environment variables are automatically converted to SecretStr."""
        from pydantic import SecretStr
        
        env_vars = {
            "APP_VECTORSTORE_PASSWORD": "my_secret_password",
            "APP_VECTORSTORE_APIKEY": "my_api_key_123",
            "MINIO_ACCESSKEY": "minio_user",
            "MINIO_SECRETKEY": "minio_pass_456",
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify automatic conversion to SecretStr
            assert isinstance(config.vector_store.password, SecretStr)
            assert config.vector_store.password.get_secret_value() == "my_secret_password"
            
            assert isinstance(config.vector_store.api_key, SecretStr)
            assert config.vector_store.api_key.get_secret_value() == "my_api_key_123"
            
            assert isinstance(config.minio.access_key, SecretStr)
            assert config.minio.access_key.get_secret_value() == "minio_user"
            
            assert isinstance(config.minio.secret_key, SecretStr)
            assert config.minio.secret_key.get_secret_value() == "minio_pass_456"

    @patch.dict(os.environ, {}, clear=True)
    def test_secretstr_with_quoted_environment_variables(self):
        """Test that SecretStr works with quoted environment variables (Docker Compose style)."""
        env_vars = {
            "APP_VECTORSTORE_PASSWORD": '"quoted_password"',  # Double quotes
            "MINIO_SECRETKEY": "'single_quoted_secret'",  # Single quotes
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify quotes are stripped and converted to SecretStr
            assert config.vector_store.password.get_secret_value() == "quoted_password"
            assert config.minio.secret_key.get_secret_value() == "single_quoted_secret"

    @patch.dict(os.environ, {}, clear=True)
    def test_secretstr_string_representation_masked(self):
        """Test that SecretStr masks values in string representation."""
        env_vars = {
            "APP_VECTORSTORE_PASSWORD": "secret123",
            "MINIO_ACCESSKEY": "access456",
        }

        with patch.dict(os.environ, env_vars):
            config = NvidiaRAGConfig()

            # Verify string representation is masked
            password_str = str(config.vector_store.password)
            access_key_str = str(config.minio.access_key)
            
            assert "secret123" not in password_str
            assert "access456" not in access_key_str
            assert "*" in password_str
            assert "*" in access_key_str

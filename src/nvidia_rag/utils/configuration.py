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
"""Simple configuration for NVIDIA RAG."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base class with shared configuration - inherit once, use everywhere
class _ConfigBase(BaseSettings):
    """Base configuration class with shared settings."""
    
    model_config = SettingsConfigDict(
        extra="allow",  # Allow extra fields for forward compatibility
        validate_assignment=True,  # Validate when fields are set
    )


class VectorStoreConfig(_ConfigBase):
    """Vector Store configuration.
    
    Environment variables:
        APP_VECTORSTORE_NAME, APP_VECTORSTORE_URL, APP_VECTORSTORE_INDEXTYPE,
        APP_VECTORSTORE_SEARCHTYPE, COLLECTION_NAME, etc.
    """

    # Simple env specification: just list the env var name
    name: str = Field(default="milvus", env="APP_VECTORSTORE_NAME")
    url: str = Field(default="http://localhost:19530", env="APP_VECTORSTORE_URL")
    nlist: int = Field(default=64, env="APP_VECTORSTORE_NLIST")
    nprobe: int = Field(default=16, env="APP_VECTORSTORE_NPROBE")
    index_type: str = Field(default="GPU_CAGRA", env="APP_VECTORSTORE_INDEXTYPE")
    enable_gpu_index: bool = Field(default=True, env="APP_VECTORSTORE_ENABLEGPUINDEX")
    enable_gpu_search: bool = Field(default=True, env="APP_VECTORSTORE_ENABLEGPUSEARCH")
    search_type: str = Field(default="dense", env="APP_VECTORSTORE_SEARCHTYPE")
    default_collection_name: str = Field(default="multimodal_data", env="COLLECTION_NAME")
    ef: int = Field(default=100, env="APP_VECTORSTORE_EF")
    username: str = Field(default="", env="APP_VECTORSTORE_USERNAME")
    password: str = Field(default="", env="APP_VECTORSTORE_PASSWORD")

    # API key authentication for vector store (used by Elasticsearch)
    api_key: str = configfield(
        "api_key",
        env_name="APP_VECTORSTORE_APIKEY",
        default="",
        help_txt="API key for vector store authentication (base64 form 'id:secret')",
    )
    api_key_id: str = configfield(
        "api_key_id",
        env_name="APP_VECTORSTORE_APIKEY_ID",
        default="",
        help_txt="API key ID for vector store authentication",
    )
    api_key_secret: str = configfield(
        "api_key_secret",
        env_name="APP_VECTORSTORE_APIKEY_SECRET",
        default="",
        help_txt="API key secret for vector store authentication",
    )


class NvIngestConfig(_ConfigBase):
    """NV-Ingest configuration."""

    message_client_hostname: str = Field(default="localhost", env="APP_NVINGEST_MESSAGECLIENTHOSTNAME")
    message_client_port: int = Field(default=7670, env="APP_NVINGEST_MESSAGECLIENTPORT")
    extract_text: bool = Field(default=True, env="APP_NVINGEST_EXTRACTTEXT")
    extract_infographics: bool = Field(default=False, env="APP_NVINGEST_EXTRACTINFOGRAPHICS")
    extract_tables: bool = Field(default=True, env="APP_NVINGEST_EXTRACTTABLES")
    extract_charts: bool = Field(default=True, env="APP_NVINGEST_EXTRACTCHARTS")
    extract_images: bool = Field(default=False, env="APP_NVINGEST_EXTRACTIMAGES")
    extract_page_as_image: bool = Field(default=False, env="APP_NVINGEST_EXTRACTPAGEASIMAGE")
    structured_elements_modality: str = Field(default="", env="STRUCTURED_ELEMENTS_MODALITY")
    image_elements_modality: str = Field(default="", env="IMAGE_ELEMENTS_MODALITY")
    pdf_extract_method: str = Field(default="None", env="APP_NVINGEST_PDFEXTRACTMETHOD")
    text_depth: str = Field(default="page", env="APP_NVINGEST_TEXTDEPTH")
    tokenizer: str = Field(default="intfloat/e5-large-unsupervised", env="APP_NVINGEST_TOKENIZER")
    chunk_size: int = Field(default=1024, env="APP_NVINGEST_CHUNKSIZE")
    chunk_overlap: int = Field(default=150, env="APP_NVINGEST_CHUNKOVERLAP")
    caption_model_name: str = Field(
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", env="APP_NVINGEST_CAPTIONMODELNAME"
    )
    caption_endpoint_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        env="APP_NVINGEST_CAPTIONENDPOINTURL",
    )
    enable_pdf_splitter: bool = Field(default=True, env="APP_NVINGEST_ENABLEPDFSPLITTER")
    segment_audio: bool = Field(default=False, env="APP_NVINGEST_SEGMENTAUDIO")
    save_to_disk: bool = Field(default=False, env="APP_NVINGEST_SAVETODISK")
    # Batch processing configuration
    enable_batch_mode: bool = Field(default=True, env="ENABLE_NV_INGEST_BATCH_MODE")
    files_per_batch: int = Field(default=16, env="NV_INGEST_FILES_PER_BATCH")
    enable_parallel_batch_mode: bool = Field(default=True, env="ENABLE_NV_INGEST_PARALLEL_BATCH_MODE")
    concurrent_batches: int = Field(default=4, env="NV_INGEST_CONCURRENT_BATCHES")


class ModelParametersConfig(_ConfigBase):
    """Model parameters configuration."""

    max_tokens: int = Field(default=32768, env="LLM_MAX_TOKENS")
    min_tokens: int = Field(default=0, env="LLM_MIN_TOKENS")
    ignore_eos: bool = Field(default=False, env="LLM_IGNORE_EOS")
    temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    top_p: float = Field(default=1.0, env="LLM_TOP_P")


class LLMConfig(_ConfigBase):
    """LLM configuration."""

    server_url: str = Field(default="", env="APP_LLM_SERVERURL")
    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5", env="APP_LLM_MODELNAME"
    )
    model_engine: str = Field(default="nvidia-ai-endpoints", env="APP_LLM_MODELENGINE")
    parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)

    def get_model_parameters(self) -> dict:
        """Return model parameters as dict."""
        return {
            "min_tokens": self.parameters.min_tokens,
            "ignore_eos": self.parameters.ignore_eos,
            "max_tokens": self.parameters.max_tokens,
            "temperature": self.parameters.temperature,
            "top_p": self.parameters.top_p,
        }


class QueryRewriterConfig(_ConfigBase):
    """Query Rewriter configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5", env="APP_QUERYREWRITER_MODELNAME"
    )
    server_url: str = Field(default="", env="APP_QUERYREWRITER_SERVERURL")
    enable_query_rewriter: bool = Field(default=False, env="ENABLE_QUERYREWRITER")


class FilterExpressionGeneratorConfig(_ConfigBase):
    """Filter Expression Generator configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
    )
    server_url: str = Field(default="", env="APP_FILTEREXPRESSIONGENERATOR_SERVERURL")
    enable_filter_generator: bool = Field(default=False, env="ENABLE_FILTER_GENERATOR")
    temperature: float = Field(default=0.0, env="APP_FILTEREXPRESSIONGENERATOR_TEMPERATURE")
    top_p: float = Field(default=1.0, env="APP_FILTEREXPRESSIONGENERATOR_TOPP")
    max_tokens: int = Field(default=32768, env="APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS")


class TextSplitterConfig(_ConfigBase):
    """Text Splitter configuration."""

    model_name: str = Field(
        default="Snowflake/snowflake-arctic-embed-l", env="APP_TEXTSPLITTER_MODELNAME"
    )
    chunk_size: int = Field(default=510, env="APP_TEXTSPLITTER_CHUNKSIZE")
    chunk_overlap: int = Field(default=200, env="APP_TEXTSPLITTER_CHUNKOVERLAP")


class EmbeddingConfig(_ConfigBase):
    """Embedding configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.2-nv-embedqa-1b-v2", env="APP_EMBEDDINGS_MODELNAME"
    )
    model_engine: str = Field(default="nvidia-ai-endpoints", env="APP_EMBEDDINGS_MODELENGINE")
    dimensions: int = Field(default=2048, env="APP_EMBEDDINGS_DIMENSIONS")
    server_url: str = Field(default="", env="APP_EMBEDDINGS_SERVERURL")


class RankingConfig(_ConfigBase):
    """Ranking configuration."""


    model_name: str = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2", env="APP_RANKING_MODELNAME"
    )
    model_engine: str = Field(default="nvidia-ai-endpoints", env="APP_RANKING_MODELENGINE")
    server_url: str = Field(default="", env="APP_RANKING_SERVERURL")
    enable_reranker: bool = Field(default=True, env="ENABLE_RERANKER")


class RetrieverConfig(_ConfigBase):
    """Retriever configuration."""


    top_k: int = Field(default=10, env="APP_RETRIEVER_TOPK")
    vdb_top_k: int = Field(default=100, env="VECTOR_DB_TOPK")
    score_threshold: float = Field(default=0.25, env="APP_RETRIEVER_SCORETHRESHOLD")
    nr_url: str = Field(default="http://retrieval-ms:8000", env="APP_RETRIEVER_NRURL")
    nr_pipeline: str = Field(default="ranked_hybrid", env="APP_RETRIEVER_NRPIPELINE")


class TracingConfig(_ConfigBase):
    """Tracing configuration."""


    enabled: bool = Field(default=False, env="APP_TRACING_ENABLED")
    otlp_http_endpoint: str = Field(default="", env="APP_TRACING_OTLPHTTPENDPOINT")
    otlp_grpc_endpoint: str = Field(default="", env="APP_TRACING_OTLPGRPCENDPOINT")
    prometheus_multiproc_dir: str = Field(default="/tmp/prom_data", env="PROMETHEUS_MULTIPROC_DIR")


class VLMConfig(_ConfigBase):
    """VLM configuration."""


    server_url: str = Field(default="http://localhost:8000/v1", env="APP_VLM_SERVERURL")
    model_name: str = Field(
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", env="APP_VLM_MODELNAME"
    )
    enable_vlm_response_reasoning: bool = Field(
        default=False, env="ENABLE_VLM_RESPONSE_REASONING"
    )
    max_total_images: int = Field(default=4, env="APP_VLM_MAX_TOTAL_IMAGES")
    max_query_images: int = Field(default=1, env="APP_VLM_MAX_QUERY_IMAGES")
    max_context_images: int = Field(default=1, env="APP_VLM_MAX_CONTEXT_IMAGES")
    vlm_response_as_final_answer: bool = Field(
        default=False, env="APP_VLM_RESPONSE_AS_FINAL_ANSWER"
    )


class MinioConfig(_ConfigBase):
    """Minio configuration."""


    endpoint: str = Field(default="localhost:9010", env="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", env="MINIO_ACCESSKEY")
    secret_key: str = Field(default="minioadmin", env="MINIO_SECRETKEY")


class SummarizerConfig(_ConfigBase):
    """Summarizer configuration."""


    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5", env="SUMMARY_LLM"
    )
    server_url: str = Field(default="", env="SUMMARY_LLM_SERVERURL")
    max_chunk_length: int = Field(default=50000, env="SUMMARY_LLM_MAX_CHUNK_LENGTH")
    chunk_overlap: int = Field(default=200, env="SUMMARY_CHUNK_OVERLAP")
    temperature: float = Field(default=0.0, env="SUMMARY_LLM_TEMPERATURE")
    top_p: float = Field(default=1.0, env="SUMMARY_LLM_TOP_P")


class MetadataConfig(_ConfigBase):
    """Metadata configuration."""


    max_array_length: int = Field(default=1000, env="APP_METADATA_MAXARRAYLENGTH")
    max_string_length: int = Field(default=65535, env="APP_METADATA_MAXSTRINGLENGTH")
    allow_partial_filtering: bool = Field(default=False, env="APP_METADATA_ALLOWPARTIALFILTERING")


class QueryDecompositionConfig(_ConfigBase):
    """Query Decomposition configuration."""


    enable_query_decomposition: bool = Field(default=False, env="ENABLE_QUERY_DECOMPOSITION")
    recursion_depth: int = Field(default=3, env="MAX_RECURSION_DEPTH")


class ReflectionConfig(_ConfigBase):
    """Reflection configuration for context relevance and response groundedness."""

    enable_reflection: bool = Field(default=False, env="ENABLE_REFLECTION")
    max_loops: int = Field(default=3, env="MAX_REFLECTION_LOOP")
    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5", env="REFLECTION_LLM"
    )
    server_url: str = Field(default="", env="REFLECTION_LLM_SERVERURL")
    context_relevance_threshold: int = Field(default=1, env="CONTEXT_RELEVANCE_THRESHOLD")
    response_groundedness_threshold: int = Field(default=1, env="RESPONSE_GROUNDEDNESS_THRESHOLD")


class NvidiaRAGConfig(_ConfigBase):
    """Main NVIDIA RAG configuration.
    
    Priority order (highest to lowest):
    1. Config file values (YAML/JSON)
    2. Environment variables  
    3. Default values
    """

    model_config = SettingsConfigDict(extra="allow", env_nested_delimiter="__")

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    query_rewriter: QueryRewriterConfig = Field(default_factory=QueryRewriterConfig)
    filter_expression_generator: FilterExpressionGeneratorConfig = Field(
        default_factory=FilterExpressionGeneratorConfig
    )
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    nv_ingest: NvIngestConfig = Field(default_factory=NvIngestConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    query_decomposition: QueryDecompositionConfig = Field(default_factory=QueryDecompositionConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)

    # Top-level flags
    enable_guardrails: bool = Field(default=False, env="ENABLE_GUARDRAILS")
    enable_citations: bool = Field(default=True, env="ENABLE_CITATIONS")
    enable_vlm_inference: bool = Field(default=False, env="ENABLE_VLM_INFERENCE")
    default_confidence_threshold: float = Field(default=0.0, env="RERANKER_CONFIDENCE_THRESHOLD")
    temp_dir: str = Field(default="./tmp-data", env="TEMP_DIR")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NvidiaRAGConfig":
        """Create config from dictionary.
        
        Priority: dict values > env vars > defaults
        
        Args:
            data: Configuration dictionary
            
        Returns:
            NvidiaRAGConfig instance
        """
        return cls(**data)

    @classmethod
    def from_yaml(cls, filepath: str) -> "NvidiaRAGConfig":
        """Create config from YAML file.
        
        Priority: YAML values > env vars > defaults
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            NvidiaRAGConfig instance
        """
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "NvidiaRAGConfig":
        """Create config from JSON file.
        
        Priority: JSON values > env vars > defaults
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            NvidiaRAGConfig instance
        """
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __str__(self) -> str:
        """Return formatted config as YAML-like string for easy reading."""
        import yaml
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)

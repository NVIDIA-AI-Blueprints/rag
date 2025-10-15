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
"""The definition of the application configuration."""

import os

from .configuration_wizard import ConfigWizard, configclass, configfield


@configclass
class VectorStoreConfig(ConfigWizard):
    """Configuration class for the Vector Store connection.

    :cvar name: Name of vector store
    :cvar url: URL of Vector Store
    """

    name: str = configfield(
        "name",
        default="milvus",
        help_txt="The name of vector store",  # supports "milvus", "elasticsearch"
    )
    url: str = configfield(
        "url",
        default="http://localhost:19530",
        help_txt="The host of the machine running Vector Store DB",
    )
    nlist: int = configfield(
        "nlist",
        default=64,
        help_txt="Number of cluster units",  # IVF Flat milvus
    )
    nprobe: int = configfield(
        "nprobe",
        default=16,
        help_txt="Number of units to query",  # IVF Flat milvus
    )
    index_type: str = configfield(
        "index_type",
        default="GPU_CAGRA",
        help_txt="Index of the vector db",  # IVF Flat for milvus
    )

    enable_gpu_index: bool = configfield(
        "enable_gpu_index",
        default=True,
        help_txt="Flag to control GPU indexing",
    )

    enable_gpu_search: bool = configfield(
        "enable_gpu_search",
        default=True,
        help_txt="Flag to control GPU search",
    )

    search_type: str = configfield(
        "search_type",
        default="dense",  # dense or hybrid
        help_txt="Flag to control search type - 'dense' retrieval or 'hybrid' retrieval",
    )

    default_collection_name: str = configfield(
        "default_collection_name",
        default="multimodal_data",
        env_name="COLLECTION_NAME",
        help_txt="Default collection name for vector store",
    )

    ef: int = configfield(
        "ef",
        default=100,
        help_txt="Parameter controlling query time/accuracy trade-off. Higher ef leads to more accurate but slower search.",
    )


@configclass
class NvIngestConfig(ConfigWizard):
    """
    Configuration for NV-Ingest.
    """

    # NV-Ingest Runtime Connectivity Configuration parameters
    message_client_hostname: str = configfield(
        "message_client_hostname",
        default="localhost",  # TODO
        help_txt="NV Ingest Message Client Host Name",
    )

    message_client_port: int = configfield(
        "message_client_port",
        default=7670,
        help_txt="NV Ingest Message Client Port",
    )

    # Extraction Configuration Parameters (Add additional parameters here)
    extract_text: bool = configfield(
        "extract_text",
        default=True,
        help_txt="Enable extract text for nv-ingest extraction",
    )

    extract_infographics: bool = configfield(
        "extract_infographics",
        default=False,
        help_txt="Enable extract infographics for nv-ingest extraction",
    )

    extract_tables: bool = configfield(
        "extract_tables",
        default=True,
        help_txt="Enable extract tables for nv-ingest extraction",
    )

    extract_charts: bool = configfield(
        "extract_charts",
        default=True,
        help_txt="Enable extract charts for nv-ingest extraction",
    )

    extract_images: bool = configfield(
        "extract_images",
        default=False,
        help_txt="Enable extract images for nv-ingest extraction",
    )

    extract_page_as_image: bool = configfield(
        "extract_page_as_image",
        default=False,
        help_txt="Enable extract page as image for nv-ingest extraction",
    )

    structured_elements_modality: str = configfield(
        "structured_elements_modality",
        default="",
        help_txt="Modality of structured elements",
        env_name="STRUCTURED_ELEMENTS_MODALITY",
    )

    image_elements_modality: str = configfield(
        "image_elements_modality",
        default="",
        help_txt="Modality of image elements",
        env_name="IMAGE_ELEMENTS_MODALITY",
    )

    pdf_extract_method: str = configfield(
        "pdf_extract_method",
        default="None",  # Literal['pdfium','nemoretriever_parse','None']
        help_txt="Extract method 'pdfium', 'nemoretriever_parse', 'None'",
    )

    text_depth: str = configfield(
        "text_depth",
        default="page",  # Literal['page', 'document']
        help_txt="Extract text by 'page' or 'document'",
    )

    # Splitting Configuration Parameters (Add additional parameters here)
    tokenizer: str = configfield(
        "tokenizer",
        default="intfloat/e5-large-unsupervised",
        # Literal["intfloat/e5-large-unsupervised" , "meta-llama/Llama-3.2-1B"]
        help_txt="Tokenizer for text splitting.",
    )

    chunk_size: int = configfield(
        "chunk_size",
        default=1024,
        help_txt="Chunk size for text splitting.",
    )

    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=150,
        help_txt="Chunk overlap for text splitting.",
    )

    # Captioning Configuration Parameters
    caption_model_name: str = configfield(
        "caption_model_name",
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        help_txt="NV Ingest Captioning model name",
    )

    caption_endpoint_url: str = configfield(
        "caption_endpoint_url",
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        help_txt="NV Ingest Captioning model Endpoint URL",
    )

    enable_pdf_splitter: bool = configfield(
        "enable_pdf_splitter",
        default=True,
        help_txt="Enable post chunk split for NV Ingest",
    )

    segment_audio: bool = configfield(
        "segment_audio",
        default=False,
        help_txt="Enable audio segmentation for NV Ingest",
    )

    save_to_disk: bool = configfield(
        "save_to_disk",
        default=False,
        help_txt="Enable saving results to disk for NV Ingest",
    )


@configclass
class ModelParametersConfig(ConfigWizard):
    """Configuration class for model parameters based on model name.

    This defines default parameters for different LLM models.
    """

    max_tokens: int = configfield(
        "max_tokens",
        env_name="LLM_MAX_TOKENS",
        default=32768,
        help_txt="The maximum number of tokens to generate in any given call.",
    )

    temperature: float = configfield(
        "temperature",
        env_name="LLM_TEMPERATURE",
        default=0,
        help_txt="The sampling temperature to use for text generation.",
    )

    top_p: float = configfield(
        "top_p",
        env_name="LLM_TOP_P",
        default=1.0,
        help_txt="The top-p sampling mass used for text generation.",
    )


@configclass
class LLMConfig(ConfigWizard):
    """Configuration class for the llm connection.

    :cvar server_url: The location of the llm server hosting the model.
    :cvar model_name: The name of the hosted model.
    """

    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The location of the Triton server hosting the llm model.",
    )
    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help_txt="The name of the hosted model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are nvidia-ai-endpoints",
    )
    # Add model parameters configuration
    parameters: ModelParametersConfig = configfield(
        "parameters",
        help_txt="Model-specific parameters for generation.",
        default=ModelParametersConfig(),
    )

    def get_model_parameters(self) -> dict:
        """Return appropriate parameters based on the model name.

        Returns a dictionary with max_tokens, temperature, and top_p
        adjusted according to the model name.
        """
        params = {
            "max_tokens": self.parameters.max_tokens,
            "temperature": self.parameters.temperature,
            "top_p": self.parameters.top_p,
        }
        return params


@configclass
class QueryRewriterConfig(ConfigWizard):
    """Configuration class for the Query Rewriter."""

    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help_txt="The llm name of the query rewriter model",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The location of the query rewriter model.",
    )
    enable_query_rewriter: bool = configfield(
        "enable_query_rewriter",
        env_name="ENABLE_QUERYREWRITER",
        default=False,
        help_txt="Enable query rewriter",
    )
    # TODO: Add temperature, top_p, max_tokens


@configclass
class FilterExpressionGeneratorConfig(ConfigWizard):
    """Configuration class for the Filter Expression Generator."""

    model_name: str = configfield(
        "model_name",
        env_name="APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help_txt="The llm name of the filter expression generator model",
    )
    server_url: str = configfield(
        "server_url",
        env_name="APP_FILTEREXPRESSIONGENERATOR_SERVERURL",
        default="",
        help_txt="The location of the filter expression generator model.",
    )
    enable_filter_generator: bool = configfield(
        "enable_filter_generator",
        env_name="ENABLE_FILTER_GENERATOR",
        default=False,
        help_txt="Enable filter expression generator",
    )
    temperature: float = configfield(
        "temperature",
        default=0,
        help_txt="The sampling temperature for filter expression generation.",
    )
    top_p: float = configfield(
        "top_p",
        default=1.0,
        help_txt="The top-p sampling mass for filter expression generation.",
    )
    max_tokens: int = configfield(
        "max_tokens",
        default=32768,
        help_txt="The maximum number of tokens for filter expression generation.",
    )


@configclass
class TextSplitterConfig(ConfigWizard):
    """Configuration class for the Text Splitter.

    :cvar chunk_size: Chunk size for text splitter. Tokens per chunk in token-based splitters.
    :cvar chunk_overlap: Text overlap in text splitter.
    """

    model_name: str = configfield(
        "model_name",
        default="Snowflake/snowflake-arctic-embed-l",
        help_txt="The name of Sentence Transformer model used for SentenceTransformer TextSplitter.",
    )
    chunk_size: int = configfield(
        "chunk_size",
        default=510,
        help_txt="Chunk size for text splitting.",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=200,
        help_txt="Overlapping text length for splitting.",
    )


@configclass
class EmbeddingConfig(ConfigWizard):
    """Configuration class for the Embeddings.

    :cvar model_name: The name of the huggingface embedding model.
    """

    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.2-nv-embedqa-1b-v2",
        help_txt="The name of huggingface embedding model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are hugginface",
    )
    dimensions: int = configfield(
        "dimensions",
        default=2048,
        help_txt="The required dimensions of the embedding model. Currently utilized for vector DB indexing.",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The url of the server hosting nemo embedding model",
    )


@configclass
class RankingConfig(ConfigWizard):
    """Configuration class for the Re-ranking.

    :cvar model_name: The name of the Ranking model.
    """

    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        help_txt="The name of Ranking model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are nvidia-ai-endpoints",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The url of the server hosting nemo Ranking model",
    )
    enable_reranker: bool = configfield(
        "enable_reranker",
        env_name="ENABLE_RERANKER",
        default=True,
        help_txt="Enable reranking",
    )


@configclass
class RetrieverConfig(ConfigWizard):
    """Configuration class for the Retrieval pipeline.

    :cvar top_k: Number of relevant results to retrieve.
    :cvar score_threshold: The minimum confidence score for the retrieved values to be considered.
    """

    top_k: int = configfield(
        "top_k",
        default=10,
        help_txt="Number of relevant results to retrieve",
    )
    vdb_top_k: int = configfield(
        "vdb_top_k",
        env_name="VECTOR_DB_TOPK",
        default=100,
        help_txt="Number of relevant results to retrieve from vector db",
    )
    score_threshold: float = configfield(
        "score_threshold",
        default=0.25,
        help_txt="The minimum confidence score for the retrieved values to be considered",
    )
    nr_url: str = configfield(
        "nr_url",
        default="http://retrieval-ms:8000",
        help_txt="The nemo retriever microservice url",
    )
    nr_pipeline: str = configfield(
        "nr_pipeline",
        default="ranked_hybrid",
        help_txt="The name of the nemo retriever pipeline one of ranked_hybrid or hybrid",
    )


@configclass
class TracingConfig(ConfigWizard):
    """Configuration class for Open Telemetry Tracing."""

    enabled: bool = configfield(
        "enabled",
        default=False,
        help_txt="Enable Open Telemetry Tracing",
    )
    otlp_http_endpoint: str = configfield(
        "otlp_http_endpoint",
        env_name="APP_TRACING_OTLPHTTPENDPOINT",
        default="",
        help_txt="HTTP endpoint for OpenTelemetry trace export",
    )
    otlp_grpc_endpoint: str = configfield(
        "otlp_grpc_endpoint",
        env_name="APP_TRACING_OTLPGRPCENDPOINT",
        default="",
        help_txt="gRPC endpoint for OpenTelemetry trace export",
    )
    prometheus_multiproc_dir: str = configfield(
        "prometheus_multiproc_dir",
        env_name="PROMETHEUS_MULTIPROC_DIR",
        default="/tmp/prom_data",
        help_txt="Directory to store Prometheus multi-process metrics",
    )


@configclass
class VLMConfig(ConfigWizard):
    """Configuration class for the VLM."""

    server_url: str = configfield(
        "server_url",
        default="http://localhost:8000/v1",
        help_txt="The url of the server hosting the VLM model",
    )
    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        help_txt="The name of the VLM model",
    )
    enable_vlm_response_reasoning: bool = configfield(
        "enable_vlm_response_reasoning",
        env_name="ENABLE_VLM_RESPONSE_REASONING",
        default=False,
        help_txt="Enable reasoning gate on VLM responses before adding them to the prompt",
    )
    max_total_images: int = configfield(
        "max_total_images",
        env_name="APP_VLM_MAX_TOTAL_IMAGES",
        default=4,
        help_txt="Maximum total images sent to VLM per request (query + context).",
    )
    max_query_images: int = configfield(
        "max_query_images",
        env_name="APP_VLM_MAX_QUERY_IMAGES",
        default=1,
        help_txt="Maximum number of query images included in the VLM request.",
    )
    max_context_images: int = configfield(
        "max_context_images",
        env_name="APP_VLM_MAX_CONTEXT_IMAGES",
        default=1,
        help_txt="Maximum number of context images included in the VLM request.",
    )
    vlm_response_as_final_answer: bool = configfield(
        "vlm_response_as_final_answer",
        env_name="APP_VLM_RESPONSE_AS_FINAL_ANSWER",
        default=False,
        help_txt="If enabled, use the VLM's response as the final answer instead of further LLM reasoning.",
    )


@configclass
class MinioConfig(ConfigWizard):
    """Configuration class for the Minio."""

    endpoint: str = configfield(
        "endpoint",
        env_name="MINIO_ENDPOINT",
        default="localhost:9010",
        help_txt="The endpoint of the minio server",
    )
    # TODO: Hide secret keys so it's not visible when showing config
    access_key: str = configfield(
        "access_key",
        env_name="MINIO_ACCESSKEY",
        default="minioadmin",
        help_txt="The access key of the minio server",
    )
    secret_key: str = configfield(
        "secret_key",
        env_name="MINIO_SECRETKEY",
        default="minioadmin",
        help_txt="The secret key of the minio server",
    )


@configclass
class SummarizerConfig(ConfigWizard):
    """Configuration class for the Summarizer."""

    model_name: str = configfield(
        "model_name",
        env_name="SUMMARY_LLM",
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help_txt="The name of the summarizer model",
    )
    server_url: str = configfield(
        "server_url",
        env_name="SUMMARY_LLM_SERVERURL",
        default="",
        help_txt="The url of the server hosting the summarizer model",
    )
    max_chunk_length: int = configfield(
        "max_chunk_length",
        env_name="SUMMARY_LLM_MAX_CHUNK_LENGTH",
        default=50000,
        help_txt="Maximum chunk size in characters for the summarizer model",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        env_name="SUMMARY_CHUNK_OVERLAP",
        default=200,
        help_txt="Overlap between chunks for iterative summarization (in characters)",
    )


@configclass
class MetadataConfig(ConfigWizard):
    """Configuration for metadata handling and validation.
    - All type/format constants are referenced from metadata_validation.py
    """

    max_array_length: int = configfield(
        "max_array_length",
        default=1000,
        help_txt="Maximum length for array metadata fields",
    )
    max_string_length: int = configfield(
        "max_string_length",
        default=65535,
        help_txt="Maximum length for string metadata fields",
    )
    allow_partial_filtering: bool = configfield(
        "allow_partial_filtering",
        default=False,
        help_txt="Allow partial filtering across collections. When True, only collections that fully support the filter are used. When False, all collections must support the filter or the request fails.",
    )


@configclass
class QueryDecompositionConfig(ConfigWizard):
    """Configuration class for the Query Decomposition."""

    enable_query_decomposition: bool = configfield(
        "enable_query_decomposition",
        env_name="ENABLE_QUERY_DECOMPOSITION",
        default=False,
        help_txt="Enable query decomposition",
    )
    recursion_depth: int = configfield(
        "recursion_depth",
        env_name="MAX_RECURSION_DEPTH",
        default=3,
        help_txt="Maximum recursion depth for query decomposition",
    )


@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar vector_store: The configuration of the vector db connection.
    :type vector_store: VectorStoreConfig
    :cvar llm: The configuration of the backend llm server.
    :type llm: LLMConfig
    :cvar text_splitter: The configuration for text splitter
    :type text_splitter: TextSplitterConfig
    :cvar embeddings: The configuration for huggingface embeddings
    :type embeddings: EmbeddingConfig
    :cvar prompts: The Prompts template for RAG and Chat
    :type prompts: PromptsConfig
    :cvar metadata: The configuration for metadata handling.
    :type metadata: MetadataConfig
    """

    vector_store: VectorStoreConfig = configfield(
        "vector_store",
        env=False,
        help_txt="The configuration of the vector db connection.",
        default=VectorStoreConfig(),
    )
    llm: LLMConfig = configfield(
        "llm",
        env=False,
        help_txt="The configuration for the server hosting the Large Language Models.",
        default=LLMConfig(),
    )
    query_rewriter: QueryRewriterConfig = configfield(
        "query_rewriter",
        env=False,
        help_txt="The configuration for the query rewriter.",
        default=QueryRewriterConfig(),
    )
    filter_expression_generator: FilterExpressionGeneratorConfig = configfield(
        "filter_expression_generator",
        env=False,
        help_txt="The configuration for the filter expression generator.",
        default=FilterExpressionGeneratorConfig(),
    )
    text_splitter: TextSplitterConfig = configfield(
        "text_splitter",
        env=False,
        help_txt="The configuration for text splitter.",
        default=TextSplitterConfig(),
    )
    embeddings: EmbeddingConfig = configfield(
        "embeddings",
        env=False,
        help_txt="The configuration of embedding model.",
        default=EmbeddingConfig(),
    )
    ranking: RankingConfig = configfield(
        "ranking",
        env=False,
        help_txt="The configuration of ranking model.",
        default=RankingConfig(),
    )
    retriever: RetrieverConfig = configfield(
        "retriever",
        env=False,
        help_txt="The configuration of the retriever pipeline.",
        default=RetrieverConfig(),
    )
    nv_ingest: NvIngestConfig = configfield(
        "nv_ingest",
        env=False,
        help_txt="The configuration for nv-ingest.",
        default=NvIngestConfig(),
    )
    tracing: TracingConfig = configfield(
        "tracing", env=False, help_txt="", default=TracingConfig()
    )
    enable_guardrails: bool = configfield(
        "enable_guardrails",
        env_name="ENABLE_GUARDRAILS",
        default=False,
        help_txt="Enable guardrails",
    )
    enable_citations: bool = configfield(
        "enable_citations",
        env_name="ENABLE_CITATIONS",
        default=True,
        help_txt="Enable citations",
    )
    enable_vlm_inference: bool = configfield(
        "enable_vlm_inference",
        env_name="ENABLE_VLM_INFERENCE",
        default=False,
        help_txt="Enable VLM inference",
    )
    default_confidence_threshold: float = configfield(
        "default_confidence_threshold",
        env_name="RERANKER_CONFIDENCE_THRESHOLD",
        default=0.0,
        help_txt="Default confidence threshold for filtering documents by reranker relevance scores (0.0 to 1.0). Only documents with scores >= this threshold are included.",
    )
    vlm: VLMConfig = configfield(
        "vlm",
        env=False,
        help_txt="The configuration for the VLM.",
        default=VLMConfig(),
    )
    minio: MinioConfig = configfield(
        "minio",
        env=False,
        help_txt="The configuration of the minio server.",
        default=MinioConfig(),
    )
    temp_dir: str = configfield(
        "temp_dir",
        env_name="TEMP_DIR",
        default="./tmp-data",
        help_txt="The temporary directory for the application.",
    )
    summarizer: SummarizerConfig = configfield(
        "summarizer",
        env=False,
        help_txt="The configuration for the summarizer.",
        default=SummarizerConfig(),
    )

    metadata: MetadataConfig = configfield(
        "metadata",
        env=False,
        help_txt="The configuration for metadata handling.",
        default=MetadataConfig(),
    )
    query_decomposition: QueryDecompositionConfig = configfield(
        "query_decomposition",
        env=False,
        help_txt="The configuration for query decomposition.",
        default=QueryDecompositionConfig(),
    )

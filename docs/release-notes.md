<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Release Notes for NVIDIA RAG Blueprint

This documentation contains the release notes for [NVIDIA RAG Blueprint](readme.md).



## Release 2.4.0 (26-02-TBD)

This release adds new features to the RAG pipeline for supporting agent workflows and enhances generations with VLMs augmenting multimodal input.

## Release 2.3.2 (2025-12-25)

This release is a hotfix for RAG v2.3.0, and includes the following changes:

- Bump embedqa version to 1.10.1 and nim-llm to version 1.14.0.
- Align Helm values and any referenced tags with the new embedqa and nim-llm versions.



### Highlights 

This release contains the following key changes:

- Added support for non-NIM models including OpenAI, models hosted on AWS and Azure, OSS models, and others. Supported through service-specific API keys. For details, refer to [Get an API Key](api-key.md).
- The RAG Blueprint now uses [nemoretriever-ocr-v1](https://build.nvidia.com/nvidia/nemoretriever-ocr-v1/modelcard) as the default OCR model. For details, refer to [NeMo Retriever OCR Configuration Guide](nemoretriever-ocr.md).
- The Vision-Language Model (VLM) inference feature now uses the model [nemotron-nano-12b-v2-vl](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard). For details, refer to [VLM for Generation](vlm.md).
- User interface improvements including catalog display, image and text query, and others. For details, refer to [User Interface](user-interface.md).
- Added ingestion metrics endpoint support with OpenTelemetry (OTEL) for monitoring document uploads, elements ingested, and pages processed. For details, refer to [Observability](observability.md).
- Support image and text as input query. For details, refer to [Multimodal Query Support](multimodal-query.md).
- Now support using thinking budget control to keep balance between accuracy and performance. For details, refer to [Enable Reasoning](enable-nemotron-thinking.md).
- Vector Database enhancements including secure database access. For details, refer to [Milvus Configuration](milvus-configuration.md) and [Elasticsearch Configuration](change-vectordb.md).
- You can now access RAG functionality from a Model Context Protocol (MCP) server for tool integration. For details, refer to [MCP Server and Client Usage](nvidia-rag-mcp.md).
- Added OpenAI-compatible search endpoint for integration with OpenAI tools. For details, refer to [API - RAG Server Schema](api-rag.md).
- Added support for collection-level data catalog, descriptions, and metadata. For details, refer to [Data Catalog](data-catalog.md).
- Enhanced `/status` endpoint publishing ingestion metrics and status information. For details, refer to the [ingestion notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb).
- Multi-turn conversation support is no longer the default for either retrieval or generation stage in the pipeline. Refer to [Multi-Turn Conversation Support](./multiturn.md) for details.
- Improved document processing and element extraction.
- Enhancements to RAG library mode including the following. For details, refer to [Use the NVIDIA RAG Blueprint Python Package](python-client.md).
  - Independent multi-instance support for the RAG Server and the ingestion server
  - Configuration support through function arguments
  - Async interface for RAG methods
  - Compatibility with the [NVIDIA NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
- Summarization enhancements including the following. For details, refer to [Documentation Summarization Customization Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/summarization.ipynb).
  - Shallow summarization support
  - Easy model switches and dedicated configurations
  - Ease of prompt changes



### Fixed Known Issues

The following are the known issues that are fixed in this version:

- Fixed issue in NIM LLM for automatic profile selection. For details, refer to [Model Profiles](model-profiles.md).

For the full list of known issues, refer to [Known Issues](#all-known-issues).



## All Known Issues

The following are the known issues for the NVIDIA RAG Blueprint:

- Optional features reflection and image captioning are not available in Helm-based deployment.
- Currently, Helm-based deployment is not supported for [NeMo Guardrails](nemo-guardrails.md).
- The Blueprint responses can have significant latency when using [NVIDIA API Catalog cloud hosted models](deploy-docker-nvidia-hosted.md).
- The accuracy of the pipeline is optimized for certain file types like `.pdf`, `.txt`, `.docx`. The accuracy may be poor for other file types supported by NV-Ingest, since image captioning is disabled by default.
- When updating model configurations in Kubernetes `values.yaml` (for example, changing from 70B to 8B models), the RAG UI automatically detects and displays the new model configuration from the backend. No container rebuilds are required - simply redeploy the Helm chart with updated values and refresh the UI to see the new model settings in the Settings panel.
- The NeMo LLM microservice can take 5-6 minutes to start for every deployment.
- B200 GPUs are not supported for the following advanced features. For these features, use H100 or A100 GPUs instead.
  - Image captioning support for ingested documents
  - NeMo Guardrails for guardrails at input/output
  - VLM-based inferencing in RAG
  - PDF extraction with Nemoretriever Parse
- Sometimes when HTTP cloud NIM endpoints are used from `deploy/compose/.env`, the `nv-ingest-ms-runtime` still logs gRPC environment variables. Following log entries can be ignored.
- For MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing This affects the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.
- Individual file uploads are limited to a maximum size of 400 MB during ingestion. Files exceeding this limit are rejected and must be split into smaller segments before ingesting.
- `llama-3.3-nemotron-super-49b-v1.5` model provides more verbose responses in non-reasoning mode compared to v1.0. For some queries the LLM model may respond with information not available in given context. Also for out of domain queries the model may provide responses based on it's own knowledge. Developers are strongly advised to [tune the prompt](prompt-customization.md) for their use cases to avoid these scenarios.
- Slow VDB upload is observed in Helm deployments for Elasticsearch.



## Release Notes for Previous Versions

| [2.3.0](https://docs.nvidia.com/rag/2.3.0/release-notes.html) | [2.2.1](https://docs.nvidia.com/rag/2.3.0/release-notes.html#release-2-2-1-2025-07-22) | [2.2.0](https://docs.nvidia.com/rag/2.3.0/release-notes.html#release-2-2-0-2025-07-08) | [2.1.0](https://docs.nvidia.com/rag/2.3.0/release-notes.html#release-2-1-0-2025-05-13) | [2.0.0](https://docs.nvidia.com/rag/2.3.0/release-notes.html#release-2-0-0-2025-03-18) | [1.0.0](https://docs.nvidia.com/rag/2.3.0/release-notes.html#release-1-0-0-2025-01-15) |



## Related Topics

- [Known Issues and Troubleshooting the RAG UI](user-interface.md#known-issues-and-troubleshooting-the-rag-ui)
- [Troubleshoot NVIDIA RAG Blueprint](troubleshooting.md)
- [Migration Guide](migration_guide.md)
- [Get Started with NVIDIA RAG Blueprint](deploy-docker-self-hosted.md)

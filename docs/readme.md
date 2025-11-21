<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
:orphan:
# NVIDIA RAG Blueprint Documentation


Welcome to the NVIDIA RAG Blueprint documentation. You can learn more here, including how to get started with the RAG Blueprint, how to customize the RAG Blueprint, and how to troubleshoot the RAG Blueprint.

- To view this documentation on docs.nvidia.com, browse to [NVIDIA RAG Blueprint Documentation](https://docs.nvidia.com/rag/latest/).
- To view this documentation on GitHub, browse to [NVIDIA RAG Blueprint Documentation](readme.md).


## Release Notes

For the release notes, refer to the [Changelog](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CHANGELOG.md).


## Support Matrix

For hardware requirements and other information, refer to the [Support Matrix](support-matrix.md).


## Get Started With RAG Blueprint

- Use the procedures in [Get Started](deploy-docker-self-hosted.md) to get started quickly with the NVIDIA RAG Blueprint.
- Experiment and test in the [Web User Interface](user-interface.md).
- Explore the notebooks that demonstrate how to use the APIs. For details refer to [Notebooks](notebooks.md).



## Deployment Options for RAG Blueprint

You can deploy the RAG Blueprint with Docker, Helm, or NIM Operator, and target dedicated hardware or a Kubernetes cluster. 
Use the following documentation to deploy the blueprint.

- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy on Kubernetes with Helm](deploy-helm.md)
- [Deploy on Kubernetes with Helm from the repository](deploy-helm-from-repo.md)
- [Deploy on Kubernetes with Helm and MIG Support](mig-deployment.md)
- [Deploy on Kubernetes with NIM Operator](deploy-nim-operator.md)



## Developer Guide

After you deploy the RAG blueprint, you can customize it for your use cases.

- Common configurations

    - [Best Practices for Common Settings](accuracy_perf.md)
    - [Change the LLM or Embedding Model](change-model.md)
    - [Customize LLM Parameters at Runtime](llm-params.md)
    - [Customize Prompts](prompt-customization.md)
    - [Model Profiles for Hardware Configurations](model-profiles.md)
    - [Multi-Collection Retrieval](multi-collection-retrieval.md)
    - [Multi-Turn Conversation Support](multiturn.md)
    - [Query rewriting to improve the accuracy of multi-turn conversations](query_rewriter.md)
    - [Reasoning in Nemotron LLM model](enable-nemotron-thinking.md)
    - [Self-reflection to improve accuracy](self-reflection.md)
    - [Summarization](summarization.md)


- Data Ingestion & Processing

    - [Audio Ingestion Support](audio_ingestion.md)
    - [Custom metadata Support](custom-metadata.md)
    - [File System Access to Extraction Results](mount-ingestor-volume.md)
    - [Multimodal Embedding Support (Early Access)](vlm-embed.md)
    - [NeMo Retriever OCR for Enhanced Text Extraction (Early Access)](nemoretriever-ocr.md)
    - [PDF Extraction with Nemoretriever Parse](nemoretriever-parse-extraction.md)
    - [Text-Only Ingestion](text_only_ingest.md)


- Vector Database and Retrieval

    - [Change the Vector Database](change-vectordb.md)
    - [Hybrid Search](hybrid_search.md)
    - [Milvus Configuration](milvus-configuration.md)
    - [Query Decomposition](query_decomposition.md)


- Multimodal and Advanced Generation

    - [Image captioning support for ingested documents](image_captioning.md)
    - [VLM based inferencing in RAG](vlm.md)


- Governance

    - [NeMo Guardrails for input/output](nemo-guardrails.md)


- Observability and Telemetry

    - [Observability](observability.md)



## Troubleshoot RAG Blueprint

- [Troubleshoot](troubleshooting.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Migrate from a Previous Version](migration_guide.md)



## Reference

- [Use the Python Package](python-client.md)
- [Milvus Collection Schema Requirements](milvus-schema.md)
- [API - Ingestor Server Schema](api_reference/openapi_schema_ingestor_server.json)
- [API - RAG Server Schema](api_reference/openapi_schema_rag_server.json)



## Blog Posts

- [NVIDIA NeMo Retriever Delivers Accurate Multimodal PDF Data Extraction 15x Faster](https://developer.nvidia.com/blog/nvidia-nemo-retriever-delivers-accurate-multimodal-pdf-data-extraction-15x-faster/)
- [Finding the Best Chunking Strategy for Accurate AI Responses](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# NVIDIA RAG Blueprint Documentation


Welcome to the [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/README.md) documentation. You can learn more here, including how to get started with the RAG Blueprint, how to customize the RAG Blueprint, and how to troubleshoot the RAG Blueprint.

- To view this documentation on docs.nvidia.com, browse to [NVIDIA RAG Blueprint Documentation](https://docs.nvidia.com/rag/latest/).
- To view this documentation on GitHub, browse to [NVIDIA RAG Blueprint Documentation](readme.md).



```{toctree}
   :name: NVIDIA RAG Blueprint
   :caption: NVIDIA RAG Blueprint
   :maxdepth: 1
   :hidden:

   Overview <https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/README.md>
   Release Notes <https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CHANGELOG.md>
   Support Matrix <support-matrix.md>
```


```{toctree}
   :name: Get Started
   :caption: Get Started
   :maxdepth: 1
   :hidden:

   Get an API Key <api-key.md>
   Deploy with Docker (Self-Hosted Models) <deploy-docker-self-hosted.md>
   Web User Interface <user-interface.md>
   Notebooks <notebooks.md>
```


```{toctree}
   :name: Deployment Options for RAG Blueprint
   :caption: Deployment Options for RAG Blueprint
   :maxdepth: 1
   :hidden:

   Deploy with Docker (NVIDIA-Hosted Models) <deploy-docker-nvidia-hosted.md>
   Deploy on Kubernetes with Helm <deploy-helm.md>
   Deploy on Kubernetes with Helm from the repository <deploy-helm-from-repo.md>
   Deploy on Kubernetes with Helm and MIG Support <mig-deployment.md>
   Deploy on Kubernetes with NIM Operator <deploy-nim-operator.md>
```


```{toctree}
   :name: Common configurations
   :caption: Common configurations
   :maxdepth: 1
   :hidden:

   Best Practices for Common Settings <accuracy_perf.md>
   Change the LLM or Embedding Model <change-model.md>
   Customize LLM Parameters at Runtime <llm-params.md>
   Customize Prompts <prompt-customization.md>
   Model Profiles for Hardware Configurations <model-profiles.md>
   Multi-Collection Retrieval <multi-collection-retrieval.md>
   Multi-Turn Conversation Support <multiturn.md>
   Query rewriting to improve the accuracy of multi-turn conversations <query_rewriter.md>
   Reasoning in Nemotron LLM model <enable-nemotron-thinking.md>
   Self-reflection to improve accuracy <self-reflection.md>
   Summarization <summarization.md>
```


```{toctree}
   :name: Data Ingestion & Processing
   :caption: Data Ingestion & Processing
   :maxdepth: 1
   :hidden:

   Audio Ingestion Support <audio_ingestion.md>
   Custom metadata Support <custom-metadata.md>
   File System Access to Extraction Results <mount-ingestor-volume.md>
   Multimodal Embedding Support (Early Access) <vlm-embed.md>
   NeMo Retriever OCR for Enhanced Text Extraction (Early Access) <nemoretriever-ocr.md>
   PDF Extraction with Nemoretriever Parse <nemoretriever-parse-extraction.md>
   Text-Only Ingestion <text_only_ingest.md>
   Deploy NV-Ingest Standalone <nv-ingest-standalone.md>
```


```{toctree}
   :name: Vector Database and Retrieval
   :caption: Vector Database and Retrieval
   :maxdepth: 1
   :hidden:

   Change the Vector Database <change-vectordb.md>
   Hybrid Search <hybrid_search.md>
   Milvus Configuration <milvus-configuration.md>
   Query Decomposition <query_decomposition.md>
```


```{toctree}
   :name: Multimodal and Advanced Generation
   :caption: Multimodal and Advanced Generation
   :maxdepth: 1
   :hidden:

   Image captioning support for ingested documents <image_captioning.md>
   VLM based inferencing in RAG <vlm.md>
```


```{toctree}
   :name: Governance
   :caption: Governance
   :maxdepth: 1
   :hidden:

   NeMo Guardrails for input/output <nemo-guardrails.md>
```


```{toctree}
   :name: Observability and Telemetry
   :caption: Observability and Telemetry
   :maxdepth: 1
   :hidden:

   Observability <observability.md>
```


```{toctree}
   :name: Troubleshoot RAG Blueprint
   :caption: Troubleshoot RAG Blueprint
   :maxdepth: 1
   :hidden:

   Troubleshoot <troubleshooting.md>
   RAG Pipeline Debugging Guide <debugging.md>
   Migrate from a Previous Version <migration_guide.md>
```


```{toctree}
   :name: Reference
   :caption: Reference
   :maxdepth: 1
   :hidden:

   Use the Python Package <python-client.md>
   Milvus Collection Schema Requirements <milvus-schema.md>
   API - Ingestor Server Schema <api-ingestor.md>
   API - RAG Server Schema <api-rag.md>
```

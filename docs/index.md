<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# NVIDIA RAG Blueprint Documentation

Welcome to the NVIDIA RAG Blueprint documentation. 
You can learn more here, including how to get started with the RAG Blueprint, how to customize the RAG Blueprint, and how to troubleshoot the RAG Blueprint.

- To view this documentation on docs.nvidia.com, browse to [NVIDIA RAG Blueprint Documentation](https://docs.nvidia.com/rag/latest/index.html).
- To view this documentation on GitHub, browse to [NVIDIA RAG Blueprint Documentation](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/readme.md).


## Release Notes

For the release notes, refer to [Release Notes](release-notes.md).


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
    - [Custom Metadata Support](custom-metadata.md)
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


- Evaluation

    - [Evaluate Your NVIDIA RAG Blueprint System](evaluate.md)

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



```{toctree}
   :name: NVIDIA RAG Blueprint
   :caption: NVIDIA RAG Blueprint
   :maxdepth: 1
   :hidden:

   Release Notes <release-notes.md>
   Support Matrix <support-matrix.md>
```


```{toctree}
   :name: Get Started
   :caption: Get Started
   :maxdepth: 1
   :hidden:

   Get an API Key <api-key.md>
   Get Started with the RAG Blueprint <deploy-docker-self-hosted.md>
   Web User Interface <user-interface.md>
   Notebooks <notebooks.md>
```


```{toctree}
   :name: Deployment Options for RAG Blueprint
   :caption: Deployment Options for RAG Blueprint
   :maxdepth: 1
   :hidden:

   Deploy with Docker (Self-Hosted Models) <deploy-docker-self-hosted.md>
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
   Change the Model <change-model.md>
   Customize Parameters <llm-params.md>
   Customize Prompts <prompt-customization.md>
   Model Profiles <model-profiles.md>
   Multi-Collection Retrieval <multi-collection-retrieval.md>
   Multi-Turn Conversation Support <multiturn.md>
   Query Rewriting<query_rewriter.md>
   Reasoning <enable-nemotron-thinking.md>
   Self-reflection <self-reflection.md>
   Summarization <summarization.md>
```


```{toctree}
   :name: Data Ingestion and Processing
   :caption: Data Ingestion and Processing
   :maxdepth: 1
   :hidden:

   Audio Ingestion Support <audio_ingestion.md>
   Custom metadata Support <custom-metadata.md>
   Enhanced PDF Extraction <nemoretriever-parse-extraction.md>
   File System Access to Results <mount-ingestor-volume.md>
   Multimodal Embedding Support (Early Access) <vlm-embed.md>
   NeMo Retriever OCR (Early Access) <nemoretriever-ocr.md>
   Standalone NV-Ingest <nv-ingest-standalone.md>
   Text-Only Ingestion <text_only_ingest.md>
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

   Image Captioning <image_captioning.md>
   VLM-based Inferencing <vlm.md>
```


```{toctree}
   :name: Evaluation
   :caption: Evaluation
   :maxdepth: 1
   :hidden:

   Evaluate Your RAG System <evaluate.md>
```


```{toctree}
   :name: Governance
   :caption: Governance
   :maxdepth: 1
   :hidden:

   NeMo Guardrails <nemo-guardrails.md>
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
   Migration Guide <migration_guide.md>
```


```{toctree}
   :name: Reference
   :caption: Reference
   :maxdepth: 1
   :hidden:

   Use the Python Client <python-client.md>
   Milvus Collection Schema <milvus-schema.md>
   API - Ingestor Server Schema <api-ingestor.md>
   API - RAG Server Schema <api-rag.md>
```

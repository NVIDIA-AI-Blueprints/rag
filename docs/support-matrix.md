<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Minimum System Requirements for NVIDIA RAG Blueprint

The following are the system requirements for the NVIDIA RAG Blueprint.

## OS Requirements
Ubuntu 22.04 OS

## Deployment Options
- [Docker](quickstart.md#deploy-with-docker-compose)
- [Kubernetes](quickstart.md#deploy-with-helm-chart)

## Driver versions

- GPU Driver -  530.30.02 or later
- CUDA version - 12.6 or later

## Hardware Requirements
By default, this blueprint deploys the referenced NIM microservices locally. For this, you will require a minimum of:
 - 2xH100
 - 2xB200
 - 3xA100
The blueprint can be also modified to use NIM microservices hosted by NVIDIA in [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

Following are the hardware requirements for each component.
The reference code in the solution (glue code) is referred to as as the "pipeline".

The overall hardware requirements depend on whether you
[Deploy With Docker Compose](quickstart.md#deploy-with-docker-compose) or [Deploy With Helm Chart](quickstart.md#deploy-with-helm-chart) or [Interact using native python package](../notebooks/rag_library_usage.ipynb).


## Hardware requirements for self hosting all NVIDIA NIM microservices

The NIM and hardware requirements only need to be met if you are self-hosting them with default settings of RAG.
See [Using self-hosted NVIDIA NIM microservices](quickstart.md#deploy-with-docker-compose).

- **Pipeline operation**: 1x L40 GPU or similar recommended. It is needed for Milvus vector store database, as GPU acceleration is enabled by default.
- **LLM NIM**: [NVIDIA llama-3.3-nemotron-super-49b-v1](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#id83)
  - For improved paralleled performance, we recommend 8x or more H100s/A100s/B200s for LLM inference.
- **Embedding NIM**: [Llama-3.2-NV-EmbedQA-1B-v2 Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html#llama-3-2-nv-embedqa-1b-v2)
  - The pipeline can share the GPU with the Embedding NIM, but it is recommended to have a separate GPU for the Embedding NIM for optimal performance.
- **Reranking NIM**: [llama-3_2-nv-rerankqa-1b-v2 Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html#llama-3-2-nv-rerankqa-1b-v2)
- **NVIDIA NIM for Image OCR**: [baidu/paddleocr](https://docs.nvidia.com/nim/ingestion/table-extraction/latest/support-matrix.html#supported-hardware)
- **NVIDIA NIMs for Object Detection**:
  - [NeMo Retriever Page Elements v2](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-page-elements-v2)
  - [NeMo Retriever Graphic Elements v1](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-graphic-elements-v1)
  - [NeMo Retriever Table Structure v1](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-table-structure-v1)

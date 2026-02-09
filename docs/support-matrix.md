<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Minimum System Requirements for NVIDIA RAG Blueprint

This documentation contains the system requirements for the [NVIDIA RAG Blueprint](readme.md).

:::{important}
You can deploy the RAG Blueprint with Docker, Helm, or NIM Operator, and target dedicated hardware or a Kubernetes cluster. 
Some requirements are different depending on your target system and deployment method. 
:::

## Disk Space Requirements

:::{important}
Ensure that you have at least 200GB of available disk space before you deploy the RAG Blueprint. This space is required for the following:

- NIM model downloads and caching (largest component, ~100-150GB)
- Container images (~20-30GB)
- Vector database data and indices
- Application logs and temporary files

Insufficient disk space causes deployment failures during model downloads or runtime operations.
:::

## Operating System

For the RAG Blueprint you need the following operating system:

- Ubuntu 22.04 OS


## Driver Versions

For the RAG Blueprint you need the following drivers:

- GPU Driver -  560 or later
- CUDA version - 12.9 or later

For details, see [NVIDIA NIM for LLMs Software](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#software).


## Hardware Requirements (Docker)

By default, the RAG Blueprint deploys the NIM microservices locally ([self-hosted](deploy-docker-self-hosted.md)). You need one of the following:

 - 2 x H100
 - 3 x B200
 - 3 x A100 SXM
 - 2 x RTX PRO 6000

:::{tip}
You can also modify the RAG Blueprint to use [NVIDIA-hosted](deploy-docker-nvidia-hosted.md) NIM microservices.
:::

:::{tip}
**No GPU Available?** Try the [Containerless Deployment (Lite Mode)](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_lite_usage.ipynb) which requires no GPU hardware and uses NVIDIA cloud APIs for all processing.
:::

## Hardware Requirements (Kubernetes)

To install the RAG Blueprint on Kubernetes, you need one of the following:

- 8 x H100-80GB
- 9 x B200
- 9 x A100-80GB SXM
- 8 x RTX PRO 6000
- 4 x H100 (with [Multi-Instance GPU](./mig-deployment.md)



## Hardware requirements for self-hosting all NVIDIA NIM microservices

The following are requirements and recommendations for the individual components of the RAG Blueprint:

- **Pipeline operation** – 1x L40 GPU or similar recommended. This is needed for the Milvus vector database, as GPU acceleration is enabled by default.
- **LLM NIM (llama-3.3-nemotron-super-49b-v1.5)** – Refer to the [Support Matrix](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#llama-3-3-nemotron-super-49b-v1-5).
- **Embedding NIM (Llama-3.2-NV-EmbedQA-1B-v2 )** – Refer to the [Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html#llama-3-2-nv-embedqa-1b-v2).
- **Reranking NIM (llama-3_2-nv-rerankqa-1b-v2 )**: Refer to the [Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html#llama-3-2-nv-rerankqa-1b-v2).
- **NeMo Retriever OCR (Default)**: Refer to the [Support Matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/1.2.0/support-matrix.html).
- **NVIDIA NIM for Image OCR (baidu/paddleocr - Legacy)**: Refer to the [Support Matrix](https://docs.nvidia.com/nim/ingestion/table-extraction/latest/support-matrix.html#supported-hardware).
- **NVIDIA NIMs for Object Detection**:
  - NeMo Retriever Page Elements v3 [Support Matrix](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-page-elements-v3)
  - NeMo Retriever Graphic Elements v1 [Support Matrix](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-graphic-elements-v1)
  - NeMo Retriever Table Structure v1 [Support Matrix](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#nemo-retriever-table-structure-v1)

:::{tip}
NeMo Retriever OCR is now the default OCR service. To use the legacy Paddle OCR instead, see [OCR Configuration Guide](nemoretriever-ocr.md).
:::


## Related Topics

- [Model Profiles](model-profiles.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Deploy with Helm and MIG Support](mig-deployment.md)

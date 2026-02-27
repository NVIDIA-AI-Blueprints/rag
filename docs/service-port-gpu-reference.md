<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Service Port and GPU Reference

The following table provides a comprehensive reference of all services, their port mappings, and GPU assignments used in the self-hosted deployment.

## Core Application Services

| Service | Container Name | Host Port(s) | Container Port(s) | Default GPU ID | Notes |
|---------|---------------|--------------|-------------------|----------------|-------|
| RAG Server | `rag-server` | 8081 | 8081 | N/A (CPU) | Main RAG API endpoint |
| Ingestor Server | `ingestor-server` | 8082 | 8082 | N/A (CPU) | Document ingestion API |
| RAG Frontend | `rag-frontend` | 8090 | 3000 | N/A (CPU) | Web UI |
| NV-Ingest Runtime | `nv-ingest-ms-runtime` | 7670, 7671, 8265 | 7670, 7671, 8265 | N/A (CPU) | Main orchestrator (Ray dashboard: 8265) |

## NIM Microservices

| Service | Container Name | Host Port(s) | Container Port(s) | Default GPU ID | Environment Variable | Notes |
|---------|---------------|--------------|-------------------|----------------|---------------------|-------|
| LLM | `nim-llm-ms` | 8999 | 8000 | 1 | `LLM_MS_GPU_ID` | Main language model |
| Embedding | `nemotron-embedding-ms` | 9080 | 8000 | 0 | `EMBEDDING_MS_GPU_ID` | Text embeddings |
| VLM Embedding | `nemotron-vlm-embedding-ms` | 9081 | 8000 | 0 | `VLM_EMBEDDING_MS_GPU_ID` | Vision-language embeddings (opt-in, profile: vlm-embed) |
| Ranking | `nemotron-ranking-ms` | 1976 | 8000 | 0 | `RANKING_MS_GPU_ID` | Reranking model |
| VLM | `nemo-vlm-microservice` | 1977 | 8000 | 5 | `VLM_MS_GPU_ID` | Vision-language model (opt-in, profile: vlm-only, vlm-generation) |
| Nemotron Parse | `compose-nemotron-parse-1` | 8015, 8016, 8017 | 8000, 8001, 8002 | 1 | `NEMOTRON_PARSE_MS_GPU_ID` | PDF parsing (opt-in, profile: nemotron-parse) |
| RIVA ASR | `compose-audio-1` | 8021, 8022 | 50051, 9000 | 0 | `AUDIO_MS_GPU_ID` | Audio speech recognition (opt-in, profile: audio) |
| Page Elements | `compose-page-elements-1` | 8000, 8001, 8002 | 8000, 8001, 8002 | 0 | `YOLOX_MS_GPU_ID` | Object detection for pages |
| Graphic Elements | `compose-graphic-elements-1` | 8003, 8004, 8005 | 8000, 8001, 8002 | 0 | `YOLOX_GRAPHICS_MS_GPU_ID` | Graphics detection |
| Table Structure | `compose-table-structure-1` | 8006, 8007, 8008 | 8000, 8001, 8002 | 0 | `YOLOX_TABLE_MS_GPU_ID` | Table structure detection |
| NeMo Retriever OCR | `compose-nemoretriever-ocr-1` | 8012, 8013, 8014 | 8000, 8001, 8002 | 0 | `OCR_MS_GPU_ID` | OCR service (default) |

## Vector Database and Infrastructure

| Service | Container Name | Host Port(s) | Container Port(s) | Default GPU ID | Environment Variable | Notes |
|---------|---------------|--------------|-------------------|----------------|---------------------|-------|
| Milvus | `milvus-standalone` | 19530, 9091 | 19530, 9091 | 0 | `VECTORSTORE_GPU_DEVICE_ID` | Vector database |
| Milvus MinIO | `milvus-minio` | 9010, 9011 | 9010, 9011 | N/A (CPU) | N/A | Object storage |
| Milvus etcd | `milvus-etcd` | N/A | 2379 | N/A (CPU) | N/A | Metadata storage |
| Redis | `compose-redis-1` | 6379 | 6379 | N/A (CPU) | N/A | Task queue |
| Elasticsearch | `elasticsearch` | 9200 | 9200 | N/A (CPU) | N/A | Profile: elasticsearch |

:::{note}
**Opt-in NIM Services:**

The following NIM services are opt-in and require explicit Docker Compose profile activation:
- **VLM Embedding** (`nemotron-vlm-embedding-ms`): Use profile `vlm-embed` for vision-language embeddings
- **VLM** (`nemo-vlm-microservice`): Use profile `vlm-only` or `vlm-generation` for vision-language model
- **Nemotron Parse** (`compose-nemotron-parse-1`): Use profile `nemotron-parse` for advanced PDF parsing
- **RIVA ASR** (`compose-audio-1`): Use profile `audio` for audio speech recognition

To activate these services, add `--profile <profile-name>` when launching services. For example:
```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile nemotron-parse up -d
```
:::

:::{tip}
**Customizing GPU Allocations:**

- Set GPU IDs using environment variables in `deploy/compose/.env` before launching services.
- For services using multiple ports (e.g., page-elements: 8000, 8001, 8002), these correspond to HTTP API, gRPC, and metrics endpoints respectively.
- Services marked with "Profile:" only start when that Docker Compose profile is specified using `--profile <name>`.
- Multiple services can share the same GPU (e.g., embedding, ranking, and ingestion services default to GPU 0).
- For multi-GPU setups on A100 SXM or B200, see step 3 in the deployment procedure.
:::

:::{note}
**Port Conflict Resolution:**

If you have port conflicts with existing services:
1. Stop conflicting services, or
2. Modify port mappings in the respective Docker Compose YAML files (e.g., change `"8081:8081"` to `"8181:8081"` to expose on host port 8181).
3. Update corresponding environment variables that reference these ports (e.g., `APP_VECTORSTORE_URL`).
:::

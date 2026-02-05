<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multimodal Query Support for NVIDIA RAG Blueprint

The multimodal query feature in the [NVIDIA RAG Blueprint](readme.md) enables you to query your knowledge base using both text and images. This is particularly useful for use cases where visual context enhances the query, such as:

- **Product identification**: "What is the price of this item?" + product image
- **Document lookup**: "Find documents related to this chart" + chart image
- **Visual Q&A**: "What material is this made of?" + product image

This feature combines:
- **VLM Embeddings**: `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` for creating multimodal embeddings that understand both text and images
- **Vision-Language Model**: `nvidia/nemotron-nano-12b-v2-vl` for generating intelligent responses based on visual and textual context



## Prerequisites

Before enabling multimodal query support, ensure you have:

1. [Obtained an API Key](api-key.md)
2. [Deployed the NVIDIA RAG Blueprint](readme.md#deployment-options-for-rag-blueprint)
3. An NVIDIA H100 or A100 GPU for on-prem deployments



## Self-Hosted (On-Prem) Deployment

Use this section to deploy multimodal query support with locally hosted NVIDIA NIMs.

### 1. Start the Vector Database

Start the Milvus vector database service:

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

### 2. Deploy the VLM and VLM Embedding NIMs

Deploy the Vision-Language Model and multimodal embedding services.

Set your NGC API key (replace with your actual key):

```bash
export NGC_API_KEY="nvapi-..."
```

Then run the deployment commands:

```bash
# Create the model cache directory
mkdir -p ~/.cache/model-cache
export MODEL_DIRECTORY=~/.cache/model-cache

# (Optional) Select a specific GPU for the VLM Microservice
# Use `nvidia-smi` to check available GPUs and set the desired GPU ID
export VLM_MS_GPU_ID=1  # Default is GPU 5; change to use a different GPU

# Deploy NIMs with VLM and VLM embedding profiles
USERID=$(id -u) docker compose --profile vlm-ingest --profile vlm-only -f deploy/compose/nims.yaml up -d
```

:::{warning}
The first deployment may take 10-20 minutes as models download (~10GB+). Subsequent deployments will be faster as models are cached.
:::

Monitor the deployment status:

```bash
watch -n 5 'docker ps --format "table {{.Names}}\t{{.Status}}"'
```

Wait until the services show as `healthy`:

### 3. Configure Environment Variables

Set the model names and service URLs for the RAG pipeline:

```bash
# VLM (Vision-Language Model) configuration
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"
export APP_LLM_SERVERURL=""

# Multimodal embedding model configuration
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
export APP_EMBEDDINGS_SERVERURL="nemoretriever-vlm-embedding-ms:8000/v1"
export ENABLE_VLM_INFERENCE="true"
export VLM_TO_LLM_FALLBACK="false"
```

### 4. Configure Image Extraction for Ingestion

Enable image extraction and storage during document ingestion:

```bash
# Configure image extraction
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"
export APP_NVINGEST_EXTRACTIMAGES="True"

# Disable reranker (not supported with multimodal queries)
export ENABLE_RERANKER="false"
export APP_RANKING_SERVERURL=""
```

### 5. Start the Ingestor Server

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

Verify the service is healthy

### 6. Start the RAG Server

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
```

Verify the service is healthy


### 7. Verify All Services Are Running

Check the status of all deployed containers:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Confirm all the containers are running and healthy


## NVIDIA-Hosted (Cloud) Deployment

Use this section to deploy multimodal query support using NVIDIA-hosted API endpoints.

:::{note}
When using NVIDIA-hosted endpoints, you might encounter rate limiting with larger file ingestions (>10 files). For details, see [Troubleshoot](troubleshooting.md).
:::

### 1. Start the Vector Database

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

### 2. Configure Environment Variables

#### a. Open `deploy/compose/.env` and uncomment the section `Endpoints for using cloud NIMs`. Then set the environment variables by running the following code.

```bash
source deploy/compose/.env
```

#### b. Set the environment variables to use NVIDIA-hosted endpoints for VLM models:

Set your NGC API key (replace with your actual key):

```bash
export NGC_API_KEY="nvapi-..."
```

Then set the VLM configuration:

```bash
# VLM (Vision-Language Model) configuration - cloud hosted
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com"
export APP_LLM_SERVERURL=""

# Multimodal embedding model configuration - cloud hosted
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
export APP_EMBEDDINGS_SERVERURL="https://integrate.api.nvidia.com/v1"
export ENABLE_VLM_INFERENCE="true"
export VLM_TO_LLM_FALLBACK="false"
```

### 3. Configure Image Extraction for Ingestion

```bash
# Configure image extraction
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"
export APP_NVINGEST_EXTRACTIMAGES="True"

# Disable reranker (not supported with multimodal queries)
export ENABLE_RERANKER="false"
export APP_RANKING_SERVERURL=""
```

### 4. Start the Ingestor Server

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

Verify the ingestor server is healthy

### 5. Start the RAG Server

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
```

Verify the RAG server is healthy

### 6. Verify All Services Are Running

Check the status of all deployed containers

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

You should see output similar to the following:

```output
NAMES                                   STATUS
compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
ingestor-server                         Up 5 minutes
compose-redis-1                         Up 5 minutes
rag-frontend                            Up 9 minutes
rag-server                              Up 9 minutes
milvus-standalone                       Up 36 minutes (healthy)
milvus-minio                            Up 35 minutes (healthy)
milvus-etcd                             Up 35 minutes (healthy)
```



## Helm Chart Deployment

Use this section to deploy multimodal query support on Kubernetes using Helm charts.

:::{note}
This configuration disables the default LLM NIM and text embedding NIM, replacing them with VLM NIM and VLM embedding NIM. The GPU resources previously used by the disabled services will be available for the VLM services.
If MIG slicing is enabled on the cluster, ensure to assign a dedicated slice to the VLM. Check [mig-deployment.md](./mig-deployment.md) for more information.
:::

### 1. Modify values.yaml

Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable multimodal query support:

```yaml
# Multimodal Query Configuration
# This replaces the default LLM and text embedding NIMs with VLM variants

# Enable VLM NIM for multimodal generation
nim-vlm:
  enabled: true

# Enable VLM embedding NIM for multimodal embeddings
nvidia-nim-llama-32-nemoretriever-1b-vlm-embed-v1:
  enabled: true
  image:
    repository: nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1
    tag: "1.7.0"

# Optional: disable the default text embedding NIM
nvidia-nim-llama-32-nv-embedqa-1b-v2:
  enabled: false

# Disable LLM NIM (VLM handles generation)
nim-llm:
  enabled: false

# Configure environment variables
envVars:
  # VLM inference settings
  ENABLE_VLM_INFERENCE: "true"
  VLM_TO_LLM_FALLBACK: "false"
  APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
  APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

  # VLM embedding settings
  APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000/v1"
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

  # Disable reranker (not supported with multimodal queries)
  ENABLE_RERANKER: "False"
  APP_RANKING_SERVERURL: ""

ingestor-server:
  envVars:
    # Image extraction settings
    APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY: ""
    APP_NVINGEST_IMAGE_ELEMENTS_MODALITY: "image"
    APP_NVINGEST_EXTRACTIMAGES: "True"

    # VLM embedding settings for ingestor
    APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000/v1"
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

nv-ingest:
  envVars:
    EMBEDDING_NIM_ENDPOINT: "http://nemoretriever-vlm-embedding-ms:8000/v1"
    EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
```

### 2. Deploy or Upgrade the Chart

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### 3. Verify the Deployment

Verify the VLM pods are running:

```bash
kubectl get pods -n rag | grep -E "(vlm|embedding)"
```

Expected output:
```
nim-vlm-f4c446cbf-ffzm7                              1/1     Running   0          22m
nemoretriever-vlm-embedding-ms-...                   1/1     Running   0          22m
```

:::{note}
It may take several minutes for the VLM pods to initialize and download the model weights.
:::



## Using Multimodal Queries

After deployment, you can start querying your knowledge base with both text and images.

:::{important}
**You must select a collection before querying.** Multimodal queries require a knowledge base to search against. Before performing any query (including visual Q&A, product identification, or document lookup), ensure you have:

1. **Created a collection**: Use the Web UI, Python client, or API to create a new collection
2. **Ingested documents**: Upload documents (PDFs, images, etc.) to your collection
3. **Selected the collection**: When querying, explicitly specify the collection name

Queries without a selected collection will not return relevant results from your knowledge base.
:::

### Web UI

Access the RAG frontend at `http://localhost:8090` to experiment with multimodal queries through the user interface.

1. In the sidebar, select your collection from the **Collection** dropdown
2. Upload an image and/or enter your text query
3. Click **Send** to get responses based on your knowledge base

For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).

### Python Client

When using the Python client, always specify `collection_names` in your query:

```python
# Example: Multimodal query with collection specified
await rag.generate(
    messages=[{"role": "user", "content": "What is this product?"}],
    use_knowledge_base=True,
    collection_names=["your_collection_name"],  # Required: specify your collection
)
```

For details, see [NVIDIA RAG Blueprint Python Package](python-client.md).

### Interactive Notebook

For a step-by-step guide with code examples covering collection creation, document ingestion, and querying with images, see the [Multimodal Query Notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/image_input.ipynb).

## Limitations

- **Reranker not supported**: The reranker must be disabled (`enable_reranker: False`) for multimodal queries.
- **Single-page retrieval for image queries**: When an image is included in the query, the retrieval results are constrained to content from a single page per document. Multi-page context retrieval is not supported for image-based queries.
- **Summary generation not supported**: The multimodal query pipeline replaces the LLM with a VLM for response generation, and summary generation does not work with VLMs. If you need summary generation alongside multimodal queries, you must deploy a separate LLM dedicated to `summary generation. For details, see [Summarization](summarization.md).`
- **Elasticsearch not supported**: Multimodal queries are only supported with Milvus as the vector database. Elasticsearch is not supported for multimodal query workflows.





## Related Topics

- [Vision-Language Model (VLM) for Generation](vlm.md)
- [VLM Embedding for Ingestion](vlm-embed.md)
- [Image Captioning Support](image_captioning.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)


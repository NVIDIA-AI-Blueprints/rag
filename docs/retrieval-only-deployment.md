<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy Retrieval-Only Mode for NVIDIA RAG Blueprint

This guide explains how to deploy the [NVIDIA RAG Blueprint](readme.md) for retrieval-only use cases without deploying the LLM generation components. This deployment mode is ideal when you only need document search and retrieval capabilities, saving GPU resources by not running the LLM NIM.

## Overview

In retrieval-only mode, you deploy:
- **Embedding NIM** - For converting queries to vectors
- **Reranking NIM** - For reordering retrieved results by relevance
- **Vector Database** - For storing and searching document embeddings
- **RAG Server** - For handling `/search` API requests

You skip deploying:
- **LLM NIM** (`nim-llm-ms`) - Not needed for retrieval-only workflows

This configuration allows you to use the `/search` API endpoint to retrieve relevant documents without generating LLM responses, significantly reducing GPU memory requirements.

## Use Cases

Retrieval-only deployments are useful for:

- **Search Applications**: Building document search systems without answer generation
- **Retrieval Pipelines**: Integrating with your own LLM or downstream processing
- **Resource-Constrained Environments**: When GPU resources are limited
- **Custom Generation**: Using retrieved documents with an external LLM service
- **Testing and Development**: Validating retrieval quality before adding generation


## Prerequisites

:::{important}
Before you deploy the RAG Blueprint, consider the following:

- For self-hosted NIMs, ensure that you have at least 50-80GB of available disk space for embedding and reranking model caches (significantly less than full deployment).
- First-time deployment takes 5-10 minutes for self-hosted NIMs, or 2-3 minutes for NVIDIA-hosted models.
- Model downloads do not show progress bars.

For monitoring deployment progress, refer to [Deploy on Kubernetes with Helm](./deploy-helm.md#verify-a-deployment).
:::

1. [Get an API Key](api-key.md).

2. Install Docker Engine and Docker Compose. Ensure Docker Compose version is 2.29.1 or later.

3. Authenticate Docker with NGC:

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

4. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Deploy Retrieval-Only Mode with Docker Compose

### Step 1: Set Up Environment

1. Create a directory to cache the models:

   ```bash
   mkdir -p ~/.cache/model-cache
   export MODEL_DIRECTORY=~/.cache/model-cache
   ```

2. Export the required environment variables:

   ```bash
   # For self-hosted NIMs
   source deploy/compose/.env

   # For NVIDIA-hosted NIMs
   source deploy/compose/nvdev.env
   ```

### Step 2: Start Retrieval NIMs Only

Choose one of the following options based on your deployment preference.

#### Option A: Self-Hosted NIMs

Instead of starting all NIMs, use the `text-embed` profile to start only the embedding service:

```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d nemoretriever-ranking-ms nemoretriever-embedding-ms
```

:::{note}
The `text-embed` profile starts only `nemoretriever-embedding-ms`, which is sufficient for retrieval operations. The LLM NIM (`nim-llm-ms`) is not started, saving significant GPU memory.
:::

Wait for the services to become healthy:

```bash
watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
```

Expected output:

```output
NAMES                          STATUS
nemoretriever-ranking-ms       Up 5 minutes (healthy)
nemoretriever-embedding-ms     Up 5 minutes (healthy)
```

#### Option B: NVIDIA-Hosted NIMs

For an even lighter deployment, use [NVIDIA-hosted NIMs](deploy-docker-nvidia-hosted.md) for embedding and reranking while running only the RAG server locally:

```bash
# Configure to use NVIDIA-hosted endpoints
export APP_EMBEDDINGS_SERVERURL=""
export APP_RANKING_SERVERURL=""
```

:::{note}
When `APP_EMBEDDINGS_SERVERURL` and `APP_RANKING_SERVERURL` are empty, the RAG server uses NVIDIA-hosted API endpoints (requires valid `NGC_API_KEY`).
:::

### Step 3: Start the Vector Database

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

### Step 4: Start the RAG Server

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d rag-server
```

Verify the RAG server is running:

```bash
curl -X 'GET' 'http://localhost:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
```

### Step 5: (Optional) Start the Ingestion Server

If you need to ingest documents, start the ingestion server:

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d ingestor-server
```

:::{tip}
If you already have documents ingested from a previous deployment, you can skip this step and use the existing collections.
:::


## Using the Search API

The `/search` endpoint retrieves relevant documents without LLM generation. This is the primary API for retrieval-only mode.

### Basic Search Request

```python
import requests

url = "http://localhost:8081/v1/search"
payload = {
    "query": "What are the key features of the product?",
    "collection_names": ["my_collection"],
    "enable_reranker": True
}

response = requests.post(url, json=payload)
results = response.json()

# Process retrieved documents
for doc in results.get("citations", []):
    print(f"Source: {doc['source']}")
    print(f"Content: {doc['content'][:200]}...")
    print(f"Score: {doc.get('score', 'N/A')}")
    print("---")
```

### Search with Metadata Filtering

```python
payload = {
    "query": "What are the key features of the product?",
    "collection_names": ["my_collection"],
    "enable_reranker": True,
    # Filter by custom metadata
    "filter_expr": 'content_metadata["category"] == "electronics"'
}
```

### Using the CLI Script

You can also use the provided CLI script for search operations:

```bash
# Basic search
python scripts/retriever_api_usage.py --mode search "Tell me about the product features"

# Search with specific collection
python scripts/retriever_api_usage.py \
    --mode search \
    --payload-json '{"collection_names":["my_collection"], "reranker_top_k": 5}' \
    "What is the return policy?"

# Save results to file
python scripts/retriever_api_usage.py \
    --mode search \
    --output-json results.json \
    "Technical specifications"
```

## Deploy with Helm (Kubernetes)

For Kubernetes deployments, configure the Helm chart to disable the LLM NIM:

```bash
helm upgrade --install rag nvidia-blueprint-rag \
  --namespace rag \
  --set nimOperator.nim-llm.enabled=false \
  --set nimOperator.nvidia-nim-llama-32-nv-embedqa-1b-v2.enabled=true \
  --set nimOperator.nvidia-nim-llama-32-nv-rerankqa-1b-v2.enabled=true \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

Or modify `values.yaml`:

```yaml
# Disable LLM NIM for retrieval-only deployment
nimOperator:
  nim-llm:
    enabled: false

  # Keep embedding and reranking NIMs enabled
  nvidia-nim-llama-32-nv-embedqa-1b-v2:
    enabled: true

  nvidia-nim-llama-32-nv-rerankqa-1b-v2:
    enabled: true
```


## Integration with External LLMs

After retrieving documents, you can send them to your own LLM for generation:

```python
import requests

# Step 1: Retrieve relevant documents
search_url = "http://localhost:8081/v1/search"
search_payload = {
    "query": "What are the key features of the product?",
    "reranker_top_k": 5,
    "collection_names": ["my_collection"],
    "enable_reranker": True
}
search_response = requests.post(search_url, json=search_payload)
citations = search_response.json().get("citations", [])

# Step 2: Format context from retrieved documents
context = "\n\n".join([
    f"[Source: {doc['source']}]\n{doc['content']}"
    for doc in citations
])

# Step 3: Send to your LLM
prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: Tell me more about the feature XYZ of the product?

Answer:"""

# Use your preferred LLM API (OpenAI, Claude, local model, etc.)
llm_response = your_llm_client.generate(prompt)
```


## GPU Resource Comparison

| Deployment Mode | Required GPUs | Memory Usage |
|----------------|---------------|--------------|
| Full RAG (with LLM) | 2-4 GPUs | ~160GB+ |
| Retrieval-Only | 1 GPU | ~24GB |
| Cloud-Hosted NIMs | 0 GPUs | N/A |

:::{note}
GPU requirements depend on the specific embedding and reranking models used. The values above are estimates for the default models.
:::


## Troubleshooting

### Generate endpoint returns error

This is expected behavior in retrieval-only mode. The `/generate` endpoint requires an LLM, which is not deployed. Use the `/search` endpoint instead.

### Embedding service not healthy

Check the embedding NIM logs:

```bash
docker logs nemoretriever-embedding-ms
```

Ensure the model cache directory has proper permissions:

```bash
chmod -R 755 ~/.cache/model-cache
```

### Search returns empty results

1. Verify documents are ingested in the collection:

   ```bash
   curl -X GET "http://localhost:8082/v1/documents?collection_name=my_collection"
   ```

2. Check that the collection name in the search request matches the ingested collection.

3. Try increasing `vdb_top_k` to retrieve more candidates.


## Shut Down Services

To stop all retrieval-only services:

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml down
docker compose -f deploy/compose/vectordb.yaml down
docker compose -f deploy/compose/nims.yaml down
```


## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Get Started With Docker Compose](deploy-docker-self-hosted.md)
- [API - RAG Server Schema](api-rag.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [Enable Text-Only Ingestion](text_only_ingest.md)
- [Notebooks](notebooks.md)

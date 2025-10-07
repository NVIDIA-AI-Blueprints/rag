<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multi-Collection Retrieval for NVIDIA RAG Blueprint

This document describes how to use the [NVIDIA RAG Blueprint](readme.md) to retrieve and generate responses from multiple vector collections simultaneously.

## Overview

The RAG system supports retrieving chunks from multiple collections in both search and generate endpoints. When using multiple collections:

1. Documents/Chunks are retrieved from all specified collections
2. A reranker is used to rank chunks across collections
3. The top-ranked chunks are used for response generation

## Limitations

Multi-collection retrieval has the following limitations:

1. Multi-collection retrieval is only supported when reranking is enabled. The reranker is mandatory to properly rank and merge chunks from different collections.

2. Currently limited to a maximum of 5 collections per query. Exceeding this limit may result in performance degradation or re-ranker context-length errors.

## Prerequisites

Multi-collection retrieval requires reranking to be enabled. The reranker service and environment variables are enabled by default in Docker and Helm deployments. Ensure the reranking microservice is deployed and accessible at the configured URL.

### For Docker Compose Deployment

The reranker settings are configured in `deploy/compose/docker-compose-rag-server.yaml`. Ensure the following environment variables are set before deploying (these are enabled by default):

```bash
# Note: These export statements are only needed if reranking is disabled (its enabled by default)
# and you want to enable it for multi-collection retrieval.

# Enable reranker (default: True)
export ENABLE_RERANKER=True

# Set reranker model (default is already configured)
export APP_RANKING_MODELNAME="nvidia/llama-3.2-nv-rerankqa-1b-v2"

# Reranker service URL (default is already configured)
export APP_RANKING_SERVERURL="nemoretriever-ranking-ms:8000"
```

### For Helm Deployment

The reranker settings are configured in `deploy/helm/nvidia-blueprint-rag/values.yaml`. Ensure the following settings are enabled (these are enabled by default):

```yaml
envVars:
  # Enable reranker (default: "True")
  ENABLE_RERANKER: "True"
  
  # Reranker model name (default is already configured)
  APP_RANKING_MODELNAME: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
  
  # Reranker service URL (default is already configured)
  APP_RANKING_SERVERURL: "nemoretriever-reranking-ms:8000"
```

## Multiple Collection Setup during Ingestion

Before using multi-collection retrieval, you must first ingest documents into each collection separately. Use the ingestion API (see [Ingestion API Usage](../notebooks/ingestion_api_usage.ipynb)) to create and populate each collection individually. For example:

```python
# Create and ingest into first collection
await create_collection(collection_name="collection1")
await upload_documents(collection_name="collection1", filepaths=[...])

# Create and ingest into second collection
await create_collection(collection_name="collection2")
await upload_documents(collection_name="collection2", filepaths=[...])
```

## API Usage

The [Retriever API Usage Notebook](../notebooks/retriever_api_usage.ipynb) demonstrates basic API usage. To enable multi-collection retrieval, modify the notebook examples by:

1. Adding multiple collection names in the `collection_names` array
2. Setting `enable_reranker` to `true`

### Search API

To search across multiple collections, specify the collection names in the `collection_names` array and set `enable_reranker` to `true`:

```python
payload = {
    "query": "Your search query",
    "collection_names": ["collection1", "collection2", "collection3"],
    "enable_reranker": True,
    "reranker_top_k": 10,  # Number of documents to return after reranking
    "vdb_top_k": 100,      # Number of documents to retrieve from each collection before reranking
    # ... other parameters ...
}
```

### Generate API

Similarly for generation, specify multiple collections and enable reranking:

```python
payload = {
    "messages": [{"role": "user", "content": "Your question"}],
    "collection_names": ["collection1", "collection2", "collection3"],
    "enable_reranker": True,
    "reranker_top_k": 10,  # Number of chunks to use in generation
    "vdb_top_k": 100,      # Number of chunks to retrieve from each collection
    "use_knowledge_base": True,
    # ... other parameters ...
}
```

## Important Notes

1. The reranker is mandatory for multi-collection retrieval to ensure proper ranking across collections
2. `vdb_top_k` determines how many chunks are retrieved from each collection before reranking
3. `reranker_top_k` determines the final number of chunks used after reranking across all collections
4. The reranking process helps ensure the most relevant chunks are selected regardless of their source collection and compress the number to be passed into LLM context window

## Use Cases

Multi-collection retrieval is useful in scenarios where you need to maintain separate collections while enabling unified search:

1. **Company-Specific Collections**: Organize documents by company (e.g., `company_a_docs`, `company_b_docs`) to compare policies or find industry trends across companies.

2. **Domain-Specific Collections**: Maintain separate collections for different domains (e.g., `technical_docs`, `marketing_docs`, `legal_docs`) while enabling cross-domain search.

In all cases, the reranker ensures the most relevant chunks are selected from across collections, providing a unified search experience.

## Troubleshooting

When using multi-collection retrieval with multiple collections and high `vdb_top_k` values, the reranking context length may increase significantly due to the large number of chunks retrieved (calculated as `vdb_top_k × number_of_collections`). This can lead to potential context length limit errors.

**Solution**: Reduce the `VECTOR_DB_TOPK` value in your deployment configuration:

**For Docker Compose**: Edit `deploy/compose/docker-compose-rag-server.yaml`:
```yaml
environment:
  # Reduce from default 100 to a lower value (e.g., 50 or 25)
  VECTOR_DB_TOPK: ${VECTOR_DB_TOPK:-50}
```

**For Helm**: Edit `deploy/helm/nvidia-blueprint-rag/values.yaml`:
```yaml
envVars:
  # Reduce from default "100" to a lower value
  VECTOR_DB_TOPK: "50"
```

Start with a lower value (e.g., 25-50) and adjust based on your retrieval quality requirements and system performance.


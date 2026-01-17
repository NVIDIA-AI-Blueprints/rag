<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Hybrid Search Support for NVIDIA RAG Blueprint

You can enable hybrid search for [NVIDIA RAG Blueprint](readme.md). Hybrid search enables higher accuracy for documents having more domain specific technical jargons. It combines sparse and dense representations to leverage the strengths of both retrieval methods—sparse models (e.g., BM25) excel at keyword matching, while dense embeddings (e.g., vector-based search) capture semantic meaning. This allows hybrid search to retrieve relevant documents even when technical jargon or synonyms are used.

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), to enable hybrid search support for Milvus Vector Database, developers can follow below steps:

## Steps

1. Set the search type to `hybrid`
   ```bash
   export APP_VECTORSTORE_SEARCHTYPE="hybrid"
   ```

2. Relaunch the rag and ingestion services
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

## Weighted Hybrid Search

By default, hybrid search uses RRF (Reciprocal Rank Fusion) for combining results. You can also use weighted hybrid search, which allows you to control the relative importance of dense (semantic) and sparse (keyword) search results.

### Docker Compose

To enable weighted hybrid search with Docker Compose:

1. Set the ranker type to `weighted` and configure the weights:
   ```bash
   export APP_VECTORSTORE_SEARCHTYPE="hybrid"
   export APP_VECTORSTORE_RANKER_TYPE="weighted"
   export APP_VECTORSTORE_DENSE_WEIGHT="0.5"   # Weight for semantic search (default: 0.5)
   export APP_VECTORSTORE_SPARSE_WEIGHT="0.5"  # Weight for keyword search (default: 0.5)
   ```

2. Relaunch the rag and ingestion services:
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

:::{note}
You can adjust the weights based on your use case. For example, if your queries rely more on semantic understanding, increase the dense weight (e.g., 0.7 dense, 0.3 sparse). For more keyword-based search, increase the sparse weight.
:::


## Helm

To enable hybrid search using Helm deployment:

Modify the following values in your `values.yaml` file:

```yaml
envVars:
  APP_VECTORSTORE_SEARCHTYPE: "hybrid"

ingestor-server:
  envVars:
    APP_VECTORSTORE_SEARCHTYPE: "hybrid"
```

### Weighted Hybrid Search with Helm

To use weighted hybrid search instead of RRF, modify the following values in your `values.yaml` file:

```yaml
envVars:
  APP_VECTORSTORE_SEARCHTYPE: "hybrid"
  APP_VECTORSTORE_RANKER_TYPE: "weighted"
  APP_VECTORSTORE_DENSE_WEIGHT: "0.5"   # Weight for semantic search (default: 0.5)
  APP_VECTORSTORE_SPARSE_WEIGHT: "0.5"  # Weight for keyword search (default: 0.5)

ingestor-server:
  envVars:
    APP_VECTORSTORE_SEARCHTYPE: "hybrid"
    APP_VECTORSTORE_RANKER_TYPE: "weighted"
    APP_VECTORSTORE_DENSE_WEIGHT: "0.5"
    APP_VECTORSTORE_SPARSE_WEIGHT: "0.5"
```

Redeploy the chart with the updated configuration:

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc2.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

:::{note}
**Elasticsearch RRF Limitation**: RRF (Reciprocal Rank Fusion) is not supported in the open-source version of Elasticsearch. If you're using open-source Elasticsearch with hybrid search, you must use the `weighted` ranker type instead of `rrf`. Attempting to use RRF with open-source Elasticsearch will result in the following error:
RRF (Reciprocal Rank Fusion) requires an Elasticsearch commercial license.
:::

:::{note}
Preexisting collections in Milvus created using search type `dense` won't work, when the search type is changed to `hybrid`. If you are switching the search type, ensure you are creating new collection and re-uploading documents before doing retrieval.
:::
<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Hybrid Search Support for NVIDIA RAG Blueprint

You can enable hybrid search for [NVIDIA RAG Blueprint](readme.md). Hybrid search enables higher accuracy for documents having more domain specific technical jargons. It combines sparse and dense representations to leverage the strengths of both retrieval methods—sparse models (e.g., BM25) excel at keyword matching, while dense embeddings (e.g., vector-based search) capture semantic meaning. This allows hybrid search to retrieve relevant documents even when technical jargon or synonyms are used.

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), to enable hybrid search support for Milvus Vector Database, developers can follow below steps:

# Steps

1. Set the search type to `hybrid`
   ```bash
   export APP_VECTORSTORE_SEARCHTYPE="hybrid"
   ```

2. Relaunch the rag and ingestion services
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```


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

Redeploy the chart with the updated configuration:

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

:::{note}
Preexisting collections in Milvus created using search type `dense` won't work, when the search type is changed to `hybrid`. If you are switching the search type, ensure you are creating new collection and re-uploading documents before doing retrieval.
:::
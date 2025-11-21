<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Text-Only Ingestion Support in Docker for NVIDIA RAG Blueprint

You can enable text-only ingestion for the [NVIDIA RAG Blueprint](readme.md). For ingesting text only files, developers do not need to deploy the complete pipeline with all NIMs connected. If your use case requires extracting text from files, follow steps below to deploy just the necessary components.

1. Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. Set the environment variables to enable text-only extraction mode:

   ```bash
   export APP_NVINGEST_EXTRACTTEXT=True
   export APP_NVINGEST_EXTRACTINFOGRAPHICS=False
   export APP_NVINGEST_EXTRACTTABLES=False
   export APP_NVINGEST_EXTRACTCHARTS=False
   ```

   Then deploy the ingestor-server:

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d ingestor-server
   ```

3. While deploying the NIMs in step 4, selectively deploy just the NIMs necessary for rag-server and ingestion in text-only mode.

   ```bash
   USERID=$(id -u) docker compose --profile rag -f deploy/compose/nims.yaml up -d
   ```

   Confirm all the below mentioned NIMs are running and the one's specified below are in healthy state before proceeding further. Make sure to allocate GPUs according to your hardware (2xH100, 2xB200 or 4xA100 to `nim-llm-ms` based on your deployment GPU profile) as stated in the quickstart guide.

   ```bash
   watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
   ```

   ```output
      NAMES                                   STATUS

      nemoretriever-ranking-ms                Up 14 minutes (healthy)
      nemoretriever-embedding-ms              Up 14 minutes (healthy)
      nim-llm-ms                              Up 14 minutes (healthy)
   ```

4. Continue following the rest of steps in deployment guide to deploy the rag-server containers.

5. Once the ingestion and rag servers are deployed, open the [ingestion notebook](../notebooks/ingestion_api_usage.ipynb) and follow the steps. While trying out the the `Upload Document Endpoint` set the payload to below.
   ```bash
       data = {
        "vdb_endpoint": "http://milvus:19530",
        "collection_name": collection_name,
        "split_options": {
            "chunk_size": 1024,
            "chunk_overlap": 150
        }
    }
   ```

6. After ingestion completes, you can try out the queries relevant to the text in the documents using [retrieval notebook](../notebooks/retriever_api_usage.ipynb).

:::{note}
In case you are [interacting with cloud hosted models](deploy-docker-nvidia-hosted.md) and want to enable text only mode, then in step 2, just export these specific environment variables as shown below:
   ```bash
   export APP_EMBEDDINGS_SERVERURL=""
   export APP_LLM_SERVERURL=""
   export APP_RANKING_SERVERURL=""
   export YOLOX_HTTP_ENDPOINT="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
   export YOLOX_INFER_PROTOCOL="http"
   ```
:::

# Enable text only ingestion support in Helm


To ingest text-only files, you do not need to deploy the complete pipeline with all NIMs connected.
If your scenario requires only text extraction from files, use the following steps to deploy only the necessary components using Helm.

When you install the Helm chart, enable only the following services that are required for text ingestion:

- `rag-server`
- `ingestor-server`
- `nv-ingest`
- `nvidia-nim-llama-32-nv-embedqa-1b-v2`
- `text-reranking-nim`
- `nim-llm`
- `milvus`
- `minio`

Additionally, ensure that **table extraction**, **chart extraction**, and **image extraction** are disabled.

1. First, modify the environment variables in the `values.yaml` file to enable text-only extraction:

   In the `nv-ingest.envVars` section, set the following values:
   ```yaml
   APP_NVINGEST_EXTRACTTEXT: "True"
   APP_NVINGEST_EXTRACTINFOGRAPHICS: "False"
   APP_NVINGEST_EXTRACTTABLES: "False"
   APP_NVINGEST_EXTRACTCHARTS: "False"
   ```

2. Then use the modified `values.yaml` file in your Helm upgrade command:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --values deploy/helm/nvidia-blueprint-rag/values.yaml \
  --set nim-llm.enabled=true \
  --set nvidia-nim-llama-32-nv-embedqa-1b-v2.enabled=true \
  --set nvidia-nim-llama-32-nv-rerankqa-1b-v2.enabled=true \
  --set ingestor-server.enabled=true \
  --set nv-ingest.enabled=true \
  --set nv-ingest.nemoretriever-page-elements-v2.deployed=false \
  --set nv-ingest.nemoretriever-graphic-elements-v1.deployed=false \
  --set nv-ingest.nemoretriever-table-structure-v1.deployed=false \
  --set nv-ingest.paddleocr-nim.deployed=false \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

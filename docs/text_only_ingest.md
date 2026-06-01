<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Text-Only Ingestion Support in Docker for NVIDIA RAG Blueprint

You can enable text-only ingestion for the [NVIDIA RAG Blueprint](readme.md). For ingesting text only files, developers do not need to deploy the complete pipeline with all NIMs connected. If your use case requires extracting text from files, follow steps below to deploy just the necessary components.

1. Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. Set the environment variables to enable text-only extraction mode. `COMPONENTS_TO_READY_CHECK` must be set to an empty string so the nv-ingest readiness probe does not wait for the disabled extraction NIMs (the compose default in [docker-compose-ingestor-server.yaml](../deploy/compose/docker-compose-ingestor-server.yaml) is `ALL`):

   ```bash
   export APP_NVINGEST_EXTRACTTEXT=True
   export APP_NVINGEST_EXTRACTINFOGRAPHICS=False
   export APP_NVINGEST_EXTRACTTABLES=False
   export APP_NVINGEST_EXTRACTCHARTS=False
   export COMPONENTS_TO_READY_CHECK=""
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

      nemotron-ranking-ms                Up 14 minutes (healthy)
      nemotron-embedding-ms              Up 14 minutes (healthy)
      nim-llm-ms                              Up 14 minutes (healthy)
   ```

4. Continue following the rest of steps in deployment guide to deploy the rag-server containers.

5. Once the ingestion and rag servers are deployed, open the [ingestion notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb) and follow the steps. While trying out the `Upload Document Endpoint` set the payload to below.
   ```bash
       data = {
        "vdb_endpoint": "http://elasticsearch:9200",
        "collection_name": collection_name,
        "split_options": {
            "chunk_size": 1024,
            "chunk_overlap": 150
        }
    }
   ```

6. After ingestion completes, you can try out the queries relevant to the text in the documents using [retrieval notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/retriever_api_usage.ipynb).

:::{note}
In case you are [interacting with cloud hosted models](deploy-docker-nvidia-hosted.md) and want to enable text only mode, then in step 2, just export these specific environment variables as shown below:
   ```bash
   export APP_EMBEDDINGS_SERVERURL=""
   export APP_LLM_SERVERURL=""
   export APP_RANKING_SERVERURL=""
   export YOLOX_HTTP_ENDPOINT="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
   export YOLOX_INFER_PROTOCOL="http"
   ```
:::

## Enable text only ingestion support in Helm


To ingest text-only files, you do not need to deploy the complete pipeline with all NIMs connected.
If your scenario requires only text extraction from files, use the following steps to deploy only the necessary components using Helm.

In the v2.6.0 chart, the **VLM embedder** (`nvidia-nim-llama-nemotron-embed-vl-1b-v2`) is enabled by default and the **text embedder** (`nvidia-nim-llama-nemotron-embed-1b-v2`) is disabled. For text-only ingestion, flip the two flags and repoint the embedding env vars at the text endpoint. Keep the following enabled:

- `rag-server`
- `ingestor-server`
- `nv-ingest`
- `nvidia-nim-llama-nemotron-embed-1b-v2` (text embedder)
- `nvidia-nim-llama-nemotron-rerank-1b-v2` (text reranker)
- `nim-llm`
- `eck-elasticsearch`
- `seaweedfs`

Additionally, disable **table extraction**, **chart extraction**, and **image extraction**.

1. Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to (a) swap the embedder NIMs, (b) repoint embedding env vars at the text endpoint, and (c) turn off image/table/chart extraction:

   ```yaml
   nimOperator:
     # Disable VLM embedder, enable text embedder
     nvidia-nim-llama-nemotron-embed-vl-1b-v2:
       enabled: false
     nvidia-nim-llama-nemotron-embed-1b-v2:
       enabled: true

   # rag-server: point at the text embedder
   envVars:
     APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-1b-v2"
     APP_EMBEDDINGS_SERVERURL: "nemotron-embedding-ms:8000/v1"

   ingestor-server:
     envVars:
       APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-1b-v2"
       APP_EMBEDDINGS_SERVERURL: "nemotron-embedding-ms:8000/v1"

   nv-ingest:
     envVars:
       # Embedding target for the nv-ingest runtime
       EMBEDDING_NIM_ENDPOINT: "http://nemotron-embedding-ms:8000/v1"
       EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-nemotron-embed-1b-v2"

       # Text-only extraction mode
       APP_NVINGEST_EXTRACTTEXT: "True"
       APP_NVINGEST_EXTRACTINFOGRAPHICS: "False"
       APP_NVINGEST_EXTRACTTABLES: "False"
       APP_NVINGEST_EXTRACTCHARTS: "False"

       # Health check: skip readiness on disabled extraction NIMs.
       # The chart default in values.yaml is "ALL"; with table / chart / image
       # extraction turned off, nv-ingest readiness would otherwise wait
       # indefinitely for NIMs that are not deployed.
       COMPONENTS_TO_READY_CHECK: ""
   ```

2. Apply the chart with the modified values, disabling the nv-ingest CV NIMs you no longer need:

   ```bash
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.6.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --values deploy/helm/nvidia-blueprint-rag/values.yaml \
     --set nv-ingest.nimOperator.page_elements.enabled=false \
     --set nv-ingest.nimOperator.graphic_elements.enabled=false \
     --set nv-ingest.nimOperator.table_structure.enabled=false \
     --set nv-ingest.nimOperator.ocr.enabled=false \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY
   ```

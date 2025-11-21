<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Use Multimodal (VLM) Embedding for Ingestion for NVIDIA RAG Blueprint (Early Access)

This guide shows how to enable and use the multimodal embedding model `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` with the [NVIDIA RAG Blueprint](readme.md) ingestion pipeline.

In this documentation you do the following:

- Start the VLM embedding microservice
- Configure ingestion to embed content as text or images using env vars
- Point the ingestor to the VLM embedding service and model

Requirements: An NVIDIA GPU and a valid `NGC_API_KEY`.

:::{note}
**Early Access**: Currently, `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` is in early access preview.
:::

:::{note}
**PDF Support Only**: The VLM embedding feature is currently only supported for PDF documents. Other document formats (Word, PowerPoint, etc.) are not supported with VLM embedding.
:::

## Limitations

- The VLM embedding feature is experimental and responses may not be accurate.
- Summary generation doesn't work when this feature is enabled.

## 1. Start the VLM Embedding NIM locally

We provide a dedicated compose profile that starts only the VLM embedding service so the text embedding service does not start.
You can skip this step if you are interested in using cloud hosted endpoints.

```bash
export USERID=$(id -u)
export NGC_API_KEY=<your_ngc_api_key>
# Optionally select a GPU for the VLM embed service
export VLM_EMBEDDING_MS_GPU_ID=<gpu_id_or_leave_default>

# Start only the VLM embedding microservice
docker compose -f deploy/compose/nims.yaml --profile vlm-embed up -d

# Verify the service is healthy
docker ps --filter "name=nemoretriever-vlm-embedding-ms" --format "table {{.Names}}\t{{.Status}}"
```

Service details (from `deploy/compose/nims.yaml`):
- Service name: `nemoretriever-vlm-embedding-ms`
- Default port mapping: `9081:8000` (internal NIM port `8000`)

## 2. Point the Ingestor to the VLM Embedding Model

Set the ingestor’s embedding endpoint and model to the VLM service and model. These env vars are read by `ingestor-server` and are also propagated to `nv-ingest-ms-runtime` so both components use the VLM embedding model. You can choose to use cloud hosted model endpoint as well by using the commented line.

```bash
# Point to the required VLM embedding endpoint
export APP_EMBEDDINGS_SERVERURL="nemoretriever-vlm-embedding-ms:8000" # For on-prem deployed
# export APP_EMBEDDINGS_SERVERURL="" # For cloud hosted NIM
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

# Launch or restart the ingestor server so the new env vars take effect
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

## 3. Configure How Content Is Embedded (text vs image)

You can control what gets embedded as text or as images using these env vars:
- `APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY`: set to `image` to embed extracted tables/charts as images (keep text as text)
- `APP_NVINGEST_IMAGE_ELEMENTS_MODALITY`: set to `image` to embed page images as images
- `APP_NVINGEST_EXTRACTPAGEASIMAGE`: set to `True` to treat each page as a single image (experimental)

Below are common configurations.

### Baseline: All extracted content embedded as text

Extractor collects text, tables, and charts as textual content; embedder treats all content as text.

```bash
export APP_NVINGEST_EXTRACTTEXT="True"
export APP_NVINGEST_EXTRACTTABLES="True"
export APP_NVINGEST_EXTRACTCHARTS="True"
export APP_NVINGEST_EXTRACTIMAGES="False"
# Do not set structured/image modalities (or set them empty) so everything embeds as text
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY=""
export APP_NVINGEST_EXTRACTPAGEASIMAGE="False"

# Apply by restarting ingestor-server
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

### Embed structured elements (tables, charts) as images

Extractor collects text, tables, and charts; embedder treats standard text as text while embedding tables and charts as images via `APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY="image"`.

```bash
export APP_NVINGEST_EXTRACTTEXT="True"
export APP_NVINGEST_EXTRACTTABLES="True"
export APP_NVINGEST_EXTRACTCHARTS="True"
export APP_NVINGEST_EXTRACTIMAGES="False"
# Use the VLM model to capture spatial/structural info for tables and charts
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY="image"
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY=""
export APP_NVINGEST_EXTRACTPAGEASIMAGE="False"

docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

### Embed entire pages as images (experimental)

Extractor captures each page as a single image (`APP_NVINGEST_EXTRACTPAGEASIMAGE="True"`); embedder processes page images via `APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"`. Other extraction types are disabled to avoid duplicating content.

:::{note}
Citations don't work in the `generate` and `search` APIs of the RAG server with this configuration.
:::

```bash
# Treat each page as a single image (turn off other extractors)
export APP_NVINGEST_EXTRACTTEXT="False"
export APP_NVINGEST_EXTRACTTABLES="False"
export APP_NVINGEST_EXTRACTCHARTS="False"
export APP_NVINGEST_EXTRACTIMAGES="False"
export APP_NVINGEST_EXTRACTPAGEASIMAGE="True"
# Ensure page images are embedded as images
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""

docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

## Quick Reference
- **Start only VLM embedding service**: `docker compose -f deploy/compose/nims.yaml --profile vlm-embed up -d`
- **Point ingestor to VLM embedding**:
  - `APP_EMBEDDINGS_SERVERURL=nemoretriever-vlm-embedding-ms:8000`
  - `APP_EMBEDDINGS_MODELNAME=nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1`
- **Modality env vars**:
  - `APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY`: `image` or empty
  - `APP_NVINGEST_IMAGE_ELEMENTS_MODALITY`: `image` or empty
  - `APP_NVINGEST_EXTRACTPAGEASIMAGE`: `True` or `False`

If you use a `.env` file, add the variables there instead of exporting them, then rerun the compose commands.


## Using Helm chart deployment

To deploy the VLM embedding service with Helm, update the image and model settings, set the corresponding environment variables, and then apply the chart with your updated `values.yaml`.

1. Update `deploy/helm/nvidia-blueprint-rag/values.yaml`:

```yaml
# Enable VLM embedding NIM and set its image
nvidia-nim-llama-32-nemoretriever-1b-vlm-embed-v1:
  enabled: true
  image:
    repository: nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1
    tag: "1.7.0"

# Optional: disable the default text embedding NIM
nvidia-nim-llama-32-nv-embedqa-1b-v2:
  enabled: false

# Point services to the VLM embedding endpoint and model
envVars:
  APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

ingestor-server:
  envVars:
    APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

nv-ingest:
  envVars:
    EMBEDDING_NIM_ENDPOINT: "http://nemoretriever-vlm-embedding-ms:8000/v1"
    EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
```

2. Deploy the chart with the updated values:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```



## Where to update extraction and embedding env vars

Set extraction-related variables under `envVars` and `ingestor-server.envVars`, and set embedding service variables under `nv-ingest.envVars` in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml).

```yaml
envVars:
  APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
ingestor-server:
  envVars:
    # Extraction toggles
    APP_NVINGEST_EXTRACTTEXT: "True"
    APP_NVINGEST_EXTRACTTABLES: "True"
    APP_NVINGEST_EXTRACTCHARTS: "True"
    APP_NVINGEST_EXTRACTIMAGES: "False"
    APP_NVINGEST_EXTRACTPAGEASIMAGE: "False"
    # Embedding modality controls
    APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY: ""   # set to "image" to embed tables/charts as images
    APP_NVINGEST_IMAGE_ELEMENTS_MODALITY: ""        # set to "image" to embed page images as images
    # Ingestor-side embedding target
    APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

nv-ingest:
  envVars:
    # NV-Ingest runtime embedding target
    EMBEDDING_NIM_ENDPOINT: "http://nemoretriever-vlm-embedding-ms:8000/v1"
    EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
```



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

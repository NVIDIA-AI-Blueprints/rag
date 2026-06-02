<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multimodal Retriever (VLM Embedding & VLM Reranker) for NVIDIA RAG Blueprint

The multimodal retriever has two independently switchable components that together let the [NVIDIA RAG Blueprint](readme.md) embed and re-rank documents with awareness of their visual content rather than text alone:

1. **VLM Embedding for Ingestion** — uses the default `nvidia/llama-nemotron-embed-vl-1b-v2` embedder so text passages, PDF pages, tables, charts, and image elements can be embedded by a multimodal model.
2. **VLM Reranker** — replaces the text reranker with `nvidia/llama-nemotron-rerank-vl-1b-v2` so retrieved passages are scored using both their text and the cited images.

Both components plug into the same retrieval pipeline and can be enabled independently or together. Pair them with [VLM-based generation](vlm.md) for a fully multimodal RAG pipeline; see [Enabling Full VLM Multimodal RAG Pipeline](vlm.md#enabling-full-vlm-multimodal-rag-pipeline) for the end-to-end picture, and [Multimodal Query Support](multimodal-query.md) for the user-facing image+text query flow.

Requirements: an NVIDIA GPU per enabled component (H100/A100 recommended) and a valid `NGC_API_KEY`.

---

# Part 1 — VLM Embedding for Ingestion

The multimodal embedding model `nvidia/llama-nemotron-embed-vl-1b-v2` is the default embedding model in v2.6.0. The setup steps in this section are useful when you need to start only the VLM embedding service, confirm the active endpoint, switch back from the optional text-only embedder, or enable image-modality ingestion.

In this section you do the following:

- Start the VLM embedding microservice
- Configure ingestion to embed content as text or images using env vars
- Point the ingestor to the VLM embedding service and model

:::{note}
**Image-modality PDF support:** The default v2.6.0 configuration uses the VLM embedding service while keeping extracted text, tables, and charts in text modality. Advanced image-modality ingestion, such as embedding structured elements or whole pages as images, is currently supported for PDF workflows.
:::

## Limitations

- Advanced image-modality ingestion is experimental and responses may not be accurate.
- Summary generation does not work with image-modality ingestion configurations such as whole-page image extraction.

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
docker ps --filter "name=nemotron-vlm-embedding-ms" --format "table {{.Names}}\t{{.Status}}"
```

Service details (from `deploy/compose/nims.yaml`):
- Service name: `nemotron-vlm-embedding-ms`
- Default port mapping: `9081:8000` (internal NIM port `8000`)

## 2. Point the Ingestor to the VLM Embedding Model

Set the ingestor's embedding endpoint and model to the VLM service and model. These env vars are read by `ingestor-server` and are also propagated to `nv-ingest-ms-runtime` so both components use the VLM embedding model. You can choose to use a cloud-hosted model endpoint as well by using the commented line.

```bash
# Point to the required VLM embedding endpoint
export APP_EMBEDDINGS_SERVERURL="nemotron-vlm-embedding-ms:8000/v1" # For on-prem deployed
# export APP_EMBEDDINGS_SERVERURL="https://integrate.api.nvidia.com/v1" # For cloud hosted NIM
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-nemotron-embed-vl-1b-v2"

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

## VLM Embedding Quick Reference

- **Start only VLM embedding service**: `docker compose -f deploy/compose/nims.yaml --profile vlm-embed up -d`
- **Point ingestor to VLM embedding**:
  - `APP_EMBEDDINGS_SERVERURL=nemotron-vlm-embedding-ms:8000/v1`
  - `APP_EMBEDDINGS_MODELNAME=nvidia/llama-nemotron-embed-vl-1b-v2`
- **Modality env vars**:
  - `APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY`: `image` or empty
  - `APP_NVINGEST_IMAGE_ELEMENTS_MODALITY`: `image` or empty
  - `APP_NVINGEST_EXTRACTPAGEASIMAGE`: `True` or `False`

If you use a `.env` file, add the variables there instead of exporting them, then rerun the compose commands.

## VLM Embedding via Helm

To deploy the VLM embedding service with Helm, update the image and model settings, set the corresponding environment variables, and then apply the chart with your updated values.yaml.

1. Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable VLM embedding:

   ```yaml
   # Enable VLM embedding NIM and set its image
   nvidia-nim-llama-nemotron-embed-vl-1b-v2:
     enabled: true
     image:
       repository: nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2
       tag: "1.12.0"

   # Optional: disable the default text embedding NIM
   nvidia-nim-llama-nemotron-embed-1b-v2:
     enabled: false

   # Point services to the VLM embedding endpoint and model
   envVars:
     APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
     APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"

   ingestor-server:
     envVars:
       APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
       APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"

   nv-ingest:
     envVars:
       EMBEDDING_NIM_ENDPOINT: "http://nemotron-vlm-embedding-ms:8000/v1"
       EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-nemotron-embed-vl-1b-v2"
   ```

2. After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

   For detailed Helm deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### Additional Helm Configuration: Extraction and Embedding Modalities

To configure how content is extracted and embedded (similar to the Docker configurations shown above), you can add extraction and modality settings to your [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml):

- Set extraction-related variables under `envVars` and `ingestor-server.envVars`
- Set embedding service variables under `nv-ingest.envVars`

**Example with extraction and modality settings:**

```yaml
envVars:
  APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"
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
    APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"

nv-ingest:
  envVars:
    # NeMo Retriever Library runtime embedding target
    EMBEDDING_NIM_ENDPOINT: "http://nemotron-vlm-embedding-ms:8000/v1"
    EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-nemotron-embed-vl-1b-v2"
```

---

# Part 2 — VLM Reranker

The VLM reranker uses a vision-language reranking model — `nvidia/llama-nemotron-rerank-vl-1b-v2` — to re-rank retrieved passages **with awareness of the cited images**, not just the surrounding text. This produces better ordering for image-heavy corpora (PDFs with charts, diagrams, scanned tables) where the most relevant chunk is signalled by its visual content rather than its text.

The VLM reranker is a drop-in replacement for the default text reranker (`nvidia/llama-nemotron-rerank-1b-v2`). When the image-input flag is enabled, the rag-server fetches the base64 image data for each retrieved image/structured chunk from object storage and attaches it to the reranking request alongside the chunk's text.

## How It Works

1. **Retrieval** runs as usual against the vector database and returns the top-K candidate chunks.
2. The rag-server builds a reranking request whose `passages` carry each chunk's text **and** (when enabled) a PNG-base64 image data URL fetched from object storage for `image` and `structured` chunks.
3. The VLM reranker scores each passage with multimodal context and the rag-server keeps the top-N.

The image-attachment behaviour is gated by the `ENABLE_VLM_RERANKER_IMAGE_INPUT` flag. With the flag off, the VLM reranker behaves like a text-only reranker — it still uses a multimodal model, but no image content is passed in the request.

## The `ENABLE_VLM_RERANKER_IMAGE_INPUT` Flag

| Flag | Default | Purpose |
|------|---------|---------|
| `ENABLE_VLM_RERANKER_IMAGE_INPUT` | `False` | When `True`, base64 image data for retrieved `image`/`structured` chunks is included in the reranking request. When `False`, only chunk text is sent. |

**When to set it to `True`:**
- Your corpus contains images, charts, diagrams, or tables ingested via VLM Embedding (Part 1) in image modality.
- Reranking quality on image queries is poor because the text caption alone doesn't disambiguate the right chunk.
- You're running the [full VLM multimodal pipeline](vlm.md#enabling-full-vlm-multimodal-rag-pipeline).

**When to leave it `False`:**
- Your corpus is text-only or you only ingest text modality.
- Latency is critical — fetching images from object storage and round-tripping them to the reranker adds time per request.
- The reranker model is the text variant (`nvidia/llama-nemotron-rerank-1b-v2`). The flag is only honoured by `nvidia/llama-nemotron-rerank-vl-1b-v2`.

## Enable VLM Reranker with Docker Compose

The VLM reranker NIM is provided as the `nemotron-ranking-vl-ms` service in [`deploy/compose/nims.yaml`](../deploy/compose/nims.yaml) under the `vlm-rerank` and `vlm-rag` profiles. Image: `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0`. Docker Compose publishes this service on host port `1979`; Docker-internal callers should continue to use `nemotron-ranking-vl-ms:8000`.

1. Start the VLM reranker NIM (and disable the text reranker if it was running):

   ```bash
   export USERID=$(id -u)
   export NGC_API_KEY="nvapi-..."
   # Optional: pin the GPU for the VLM reranker
   export RANKING_VL_MS_GPU_ID=0

   # Start the VLM reranker (and any other services on the vlm-rerank profile)
   docker compose -f deploy/compose/nims.yaml --profile vlm-rerank up -d
   ```

   Use the `vlm-rag` profile if you also want VLM generation and VLM embedding to come up with the same command.

2. Point the rag-server at the VLM reranker and enable image input:

   ```bash
   export APP_RANKING_MODELNAME="nvidia/llama-nemotron-rerank-vl-1b-v2"
   export APP_RANKING_SERVERURL="nemotron-ranking-vl-ms:8000"
   export ENABLE_RERANKER="True"
   export ENABLE_VLM_RERANKER_IMAGE_INPUT="True"

   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   - `APP_RANKING_MODELNAME` must contain the substring `rerank-vl` for the rag-server to route through the multimodal reranker code path.
   - `APP_RANKING_SERVERURL` points to the VLM reranker NIM service. For NVIDIA-hosted endpoints, set it to `https://ai.api.nvidia.com` (or leave unset to use the default cloud URL).

3. Restart the rag-server so the new flag takes effect.

### Use the NVIDIA-Hosted VLM Reranker (Optional)

```bash
export APP_RANKING_MODELNAME="nvidia/llama-nemotron-rerank-vl-1b-v2"
export APP_RANKING_SERVERURL=""   # empty = use NVIDIA-hosted default
export ENABLE_VLM_RERANKER_IMAGE_INPUT="True"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## Enable VLM Reranker with Helm

The VLM reranker NIM is defined in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) as `nimOperator.nvidia-nim-llama-nemotron-rerank-vl-1b-v2` (disabled by default). Service name `nemotron-ranking-vl-ms`, image `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0`.

1. In `values.yaml`, enable the VLM reranker NIM and disable the text reranker:

   ```yaml
   nimOperator:
     nvidia-nim-llama-nemotron-rerank-vl-1b-v2:
       enabled: true
     # Optional: disable the text reranker NIM to free up its GPU slot
     nvidia-nim-llama-nemotron-rerank-1b-v2:
       enabled: false
   ```

2. Update the rag-server `envVars` to point at the VLM reranker and turn on image input:

   ```yaml
   envVars:
     ENABLE_RERANKER: "True"
     APP_RANKING_MODELNAME: "nvidia/llama-nemotron-rerank-vl-1b-v2"
     APP_RANKING_SERVERURL: "nemotron-ranking-vl-ms:8000"
     ENABLE_VLM_RERANKER_IMAGE_INPUT: "True"
   ```

3. Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

4. Verify the VLM reranker pod is running:

   ```bash
   kubectl get pods -n rag | grep nemotron-ranking-vl
   ```

## Hardware

`nvidia/llama-nemotron-rerank-vl-1b-v2` requires **1× NVIDIA GPU** (H100 or A100 recommended). When running alongside VLM generation and VLM embedding for a fully multimodal pipeline, plan for at least **3 GPUs** total: one each for the VLM, the VLM embedder, and the VLM reranker. With MIG slicing on H100, smaller slices may be sufficient — see [MIG Deployment](mig-deployment.md).

## VLM Reranker Limitations

- **Only the VL reranker model honours the image-input flag.** Setting `ENABLE_VLM_RERANKER_IMAGE_INPUT=True` while `APP_RANKING_MODELNAME` is the text reranker has no effect — the rag-server only follows the multimodal code path when the model name contains `rerank-vl`.
- **Image queries bypass the reranker entirely.** When the user query itself contains an image, the rag-server skips reranking (text or VLM) and returns the vector-DB results directly. This is independent of the flag.
- **Latency.** Each image-bearing passage requires an object-store fetch and a base64 round-trip to the reranker. Expect ~50–200 ms of additional reranking latency depending on `vdb_top_k` and image sizes.
- **Object-store availability.** If the rag-server cannot reach object storage (`OBJECTSTORE_ENDPOINT`), it logs a warning and falls back to text-only passages for that chunk.

---

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [VLM-based Generation](vlm.md)
- [Multimodal Query Support](multimodal-query.md)
- [Change the LLM, Embedding Model, or Reranker](change-model.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# VLM Reranker for NVIDIA RAG Blueprint

The VLM reranker uses a vision-language reranking model — `nvidia/llama-nemotron-rerank-vl-1b-v2` — to re-rank retrieved passages **with awareness of the cited images**, not just the surrounding text. This produces better ordering for image-heavy corpora (PDFs with charts, diagrams, scanned tables) where the most relevant chunk is signalled by its visual content rather than its text.

The VLM reranker is a drop-in replacement for the default text reranker (`nvidia/llama-nemotron-rerank-1b-v2`). When the image-input flag is enabled, the rag-server fetches the base64 image data for each retrieved image/structured chunk from object storage and attaches it to the reranking request alongside the chunk's text.

:::{tip}
Use the VLM reranker together with [VLM Embedding for Ingestion](vlm-embed.md) and [VLM-based Generation](vlm.md) for a fully multimodal RAG pipeline. See [Enabling Full VLM Multimodal RAG Pipeline](vlm.md#enabling-full-vlm-multimodal-rag-pipeline) for the end-to-end setup.
:::

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
- Your corpus contains images, charts, diagrams, or tables ingested via [VLM Embedding](vlm-embed.md) in image modality.
- Reranking quality on image queries is poor because the text caption alone doesn't disambiguate the right chunk.
- You're running the [full VLM multimodal pipeline](vlm.md#enabling-full-vlm-multimodal-rag-pipeline).

**When to leave it `False`:**
- Your corpus is text-only or you only ingest text modality.
- Latency is critical — fetching images from object storage and round-tripping them to the reranker adds time per request.
- The reranker model is the text variant (`nvidia/llama-nemotron-rerank-1b-v2`). The flag is only honoured by `nvidia/llama-nemotron-rerank-vl-1b-v2`.

## Enable with Docker Compose

The VLM reranker NIM is provided as the `nemotron-ranking-vl-ms` service in [`deploy/compose/nims.yaml`](../deploy/compose/nims.yaml) under the `vlm-rerank` and `vlm-rag` profiles. Image: `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0`.

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

## Enable with Helm

The VLM reranker NIM is defined in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) as `nimOperator.nvidia-nim-llama-nemotron-rerank-vl-1b-v2` (disabled by default). Service name `nemotron-ranking-vl-ms`, image `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0`.

1. In `values.yaml`, enable the VLM reranker NIM and disable the text reranker:

   ```yaml
   nimOperator:
     nvidia-nim-llama-nemotron-rerank-vl-1b-v2:
       enabled: true
     # Optional: disable the text reranker NIM to free up its GPU slot
     nvidia-nim-llama-32-nv-rerankqa-1b-v2:
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

## Limitations

- **Latency.** Each image-bearing passage requires an object-store fetch and a base64 round-trip to the reranker. Expect ~50–200 ms of additional reranking latency depending on `vdb_top_k` and image sizes.

## Related Topics

- [Vision-Language Model (VLM) for Generation](vlm.md)
- [VLM Embedding for Ingestion](vlm-embed.md)
- [Multimodal Query Support](multimodal-query.md)
- [Change the LLM or Embedding Model](change-model.md)
- [Best Practices for Common Settings](accuracy_perf.md)

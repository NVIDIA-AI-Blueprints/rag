<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Vision-Language Model (VLM) for Generation for NVIDIA RAG Blueprint

The Vision-Language Model (VLM) inference feature in the [NVIDIA RAG Blueprint](readme.md) enhances the system's ability to understand and reason about visual content. Unlike traditional image upload systems, this feature operates on image citations that are internally discovered during the retrieval process.

:::{note}
B200 GPUs are not supported for VLM based inferencing in RAG. For this feature, use H100 or A100 GPUs instead.
:::

**Key use cases for VLM**

- Documents with charts and graphs: Financial reports, scientific papers, business analytics
- Technical diagrams: Engineering schematics, architectural plans, flowcharts
- Visual data representations: Infographics, tables with visual elements, dashboards
- Mixed content documents: PDFs containing both text and images
- Image-heavy content: Catalogs, product documentation, visual guides

**Key benefits of VLM**

- **Integrated multimodal experience** – Users do not need to manually upload images; visual content is automatically discovered and analyzed from images embedded in documents.
- **Improved accuracy** – Enhanced response quality for documents containing images, charts, diagrams, and visual data.
- **Quality assurance** – Internal reasoning ensures only relevant visual insights are used.
- **Contextual understanding** – Visual analysis is performed in the context of the user's specific question.
- **Fallback handling** – System gracefully handles cases where images are insufficient or irrelevant.

:::{warning}
Enabling VLM inference increases response latency from additional image processing and VLM model inference time. Consider this trade-off between accuracy and speed based on your requirements.
:::

## How VLM Works in the RAG Pipeline

When VLM inference is enabled, the **VLM replaces the traditional LLM** in the RAG pipeline for generation tasks.

1. **Automatic Image Discovery**: When a user query is processed, the RAG system retrieves relevant documents from the vector database. If any of these documents contain images (charts, diagrams, photos, and so on), they are automatically identified.
2. **Image Captioning at Ingestion**: During ingestion, images are extracted and captioned so they can be indexed and later cited for question answering.
3. **VLM Answer Generation**: At query time, the RAG server sends the user question, conversation history, and cited images to a Vision-Language Model. The VLM directly generates the final answer for the user, taking the place of the traditional LLM.

**What users experience**: Users interact with the system normally. The VLM processing happens transparently:

1. User asks a question about content that may have visual elements.
2. System retrieves relevant documents including any images.
3. VLM analyzes images and text context if present and relevant.
4. User receives a single, coherent answer generated directly by the VLM.

## Prompt Customization

The VLM feature uses predefined prompts that can be customized in [`src/nvidia_rag/rag_server/prompt.yaml`](../src/nvidia_rag/rag_server/prompt.yaml) under the `vlm_template` section. The `vlm_template` controls how the question, textual context, and cited images are presented to the VLM.

**VLM reasoning compared to non-reasoning mode**: Nemotron Omni supports two modes controlled by the `APP_VLM_ENABLE_THINKING` environment variable:

- **Reasoning mode (default)**: `APP_VLM_ENABLE_THINKING=true`. The model produces a chain-of-thought trace before the final answer. Default parameters: `APP_VLM_TEMPERATURE=0.6`, `APP_VLM_TOP_P=0.95`, `APP_VLM_MAX_TOKENS=32768`, `APP_VLM_THINKING_TOKEN_BUDGET=16384`.
- **Non-reasoning mode**: `APP_VLM_ENABLE_THINKING=false`. The model skips the reasoning trace and returns only the final answer.

**What reaches the streaming client** is structured by field:

- Reasoning is filtered out of user-facing `content` and surfaced in
  `reasoning_content` when the model emits it.
- The final answer streams through `content`.
- `VLM_FILTER_THINK_TOKENS` is retained as a compatibility setting; streamed
  reasoning is not wrapped or concatenated into `content`.

Set these parameters using environment variables in your deployment configuration (for example in `docker-compose-rag-server.yaml` or Helm `values.yaml`).

## Enable VLM with Docker Compose

NVIDIA RAG uses the **Nemotron Omni** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) Vision-Language Model by default, provided as the `vlm-ms` service in `deploy/compose/nims.yaml`.

The `vlm-generation` profile in `deploy/compose/nims.yaml` is designed for VLM-based generation on **2xH100 GPUs**. It skips the NIM LLM deployment (VLM replaces LLM), deploys the VLM service (`vlm-ms`), and deploys embedding and reranker microservices. The `ingest` profile (combined below) additionally starts the ingestion-extraction NIMs (`page-elements`, `graphic-elements`, `table-structure`, `nemotron-ocr`) and the captioning VLM (`vlm-captioning-ms`): without these, ingestion of PDFs and image-bearing documents will fail to extract tables, charts, and OCR text.

**GPU allocation for 2xH100**: GPU 0 for Embedding and Reranker; GPU 1 for VLM (replaces LLM). You must set `VLM_MS_GPU_ID=1`.

1. Set the VLM GPU assignment and start the VLM generation services together with the ingestion-extraction NIMs (skips `nim-llm`):

   ```bash
   export VLM_MS_GPU_ID=1
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation --profile ingest up -d
   ```

   Combining `--profile vlm-generation` with `--profile ingest` is equivalent to "start everything in `nims.yaml` except `nim-llm`": the LLM container is intentionally omitted because VLM replaces it. You can confirm the exact service set with `docker compose -f deploy/compose/nims.yaml --profile vlm-generation --profile ingest config --services`.

   :::{warning}
   Only change `VLM_MS_GPU_ID` for systems with 3+ GPUs.
   :::

   For systems with 3+ GPUs, you can assign VLM to a different GPU (for example, GPU 3):

   ```bash
   export VLM_MS_GPU_ID=3
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation --profile ingest up -d
   ```

2. Enable image extraction and captioning for ingestion. In `deploy/compose/docker-compose-ingestor-server.yaml`, under the `ingestor-server` service, set `APP_NVINGEST_EXTRACTIMAGES` to `True` so images are extracted and stored (disabled by default). Image captioning is enabled by default: `APP_NVINGEST_CAPTIONMODELNAME` is set to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` and `APP_NVINGEST_CAPTIONENDPOINTURL` points to the `vlm-ms` service. Override using environment variables if needed:

   ```bash
   export APP_NVINGEST_EXTRACTIMAGES=True
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

3. Configure the RAG server to use VLM. Set the following environment variables in [docker-compose-rag-server.yaml](../deploy/compose/docker-compose-rag-server.yaml), then restart the rag-server:

   ```bash
   export ENABLE_VLM_INFERENCE="true"
   export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
   export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   - `ENABLE_VLM_INFERENCE`: Enables VLM inference in the RAG server.
   - `APP_VLM_MODELNAME`: The name of the VLM model to use.
   - `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote).

   :::{note}
   When using the `vlm-generation` profile, there is no LLM service running. The VLM handles all generation tasks. Optional fallback is controlled by `VLM_TO_LLM_FALLBACK` (see [VLM to LLM Fallback](#vlm-to-llm-fallback-optional)).
   :::

4. Continue with the rest of the steps in [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md) to deploy the ingestion-server and rag-server containers.

### Using a Remote NVIDIA-Hosted NIM (Optional)

To use a remote NVIDIA-hosted NIM for VLM inference, set `APP_VLM_SERVERURL` to the remote endpoint:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com/v1/"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

Continue with [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md) as needed.

## Enable VLM with Helm

:::{note}
**GPU requirements for Helm**: VLM uses the same GPU normally assigned to LLM (GPU 1). With MIG slicing, assign a dedicated MIG slice to the VLM—see [mig-deployment.md](mig-deployment.md) and [values-mig-h100.yaml](../deploy/helm/mig-slicing/values-mig-h100.yaml) or [values-mig-rtx6000.yaml](../deploy/helm/mig-slicing/values-mig-rtx6000.yaml). To run both VLM and LLM simultaneously, an additional GPU is required.
:::

1. In [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml), under the `rag-server` `envVars` section, set:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
   APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"
   ```

2. Enable image extraction and captioning for ingestion. Image captioning is recommended when running VLM generation so that ingested images are indexed with their captions and surface as citations at query time. The captioning model is served by a dedicated `nim-vlm-captioning` NIM (see [Separate VLMs for Generation and Captioning](#separate-vlms-for-generation-and-captioning)). Under `nimOperator` and `ingestor-server.envVars`, set:

   ```yaml
   nimOperator:
     nim-vlm-captioning:
       enabled: true

   ingestor-server:
     envVars:
       APP_NVINGEST_EXTRACTIMAGES: "True"
       APP_NVINGEST_CAPTIONENDPOINTURL: "http://nim-vlm-captioning:8000/v1/chat/completions"
       APP_NVINGEST_CAPTIONMODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
   ```

3. Enable `nim-vlm` and disable `nim-llm` (VLM replaces LLM for generation):

   ```yaml
   nimOperator:
     nim-vlm:
       enabled: true
     nim-llm:
       enabled: false
   ```

   :::{important}
   By disabling `nim-llm` and enabling `nim-vlm`, the VLM uses the GPU resources normally allocated to the LLM, so no additional hardware is required.
   :::

4. Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment). For full steps, see [Deploy with Helm](deploy-helm.md).

5. Verify the VLM pod is running. A pod with the name `nim-vlm-*` will start (the `nim-llm` pod will not be created when it is disabled). Example status:

   ```text
   rag       nim-vlm-f4c446cbf-ffzm7       1/1     Running   0          22m
   ```

:::{note}
**Service architecture**: With VLM enabled and LLM disabled, the RAG pipeline uses VLM for all generation tasks. The embedding and reranking services remain active for document retrieval. For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`. For remote endpoints, the `NGC_API_KEY` is required for authentication.
:::

## Configuration

- **Image limits**: `APP_VLM_MAX_TOTAL_IMAGES` (default: 5) is the maximum total images (from query, history, and context) included in the VLM prompt. Set using environment variables and restart the rag-server to apply.

   Example (Docker Compose):

   ```bash
   export ENABLE_VLM_INFERENCE="true"
   export APP_VLM_MAX_TOTAL_IMAGES="5"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

- **Context limitations**: The VLM receives the current user query, a truncated conversation history, and a textual summary of retrieved documents, together with any cited images. The effective context window of the VLM is limited, so very long conversations or large document contexts may be truncated.

   :::{warning}
   Keep user questions as self-contained as possible, especially in long-running conversations. Use retrieval and prompt tuning to focus the most relevant context for the VLM.
   :::

## Separate VLMs for Generation and Captioning

This blueprint uses two distinct VLM NIMs by default: one for chat / RAG answering and one for ingestion-time image captioning. The two services are configured independently so each can be scaled, replaced, or pointed at a different endpoint without affecting the other.

| Role | Default model | Compose service | Helm chart key |
|------|---------------|-----------------|----------------|
| Generation (chat / RAG answering) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `vlm-ms` | `nimOperator.nim-vlm` |
| Ingestion-time image captioning | `nvidia/nemotron-nano-12b-v2-vl` | `vlm-captioning-ms` | `nimOperator.nim-vlm-captioning` |

### Configure the generation VLM

Set the generation VLM using `APP_VLM_MODELNAME` and `APP_VLM_SERVERURL`. With Docker Compose:

```bash
export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

With Helm, in `values.yaml`:

```yaml
envVars:
  APP_VLM_MODELNAME: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
  APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

nimOperator:
  nim-vlm:
    enabled: true
```

### Configure the captioning VLM

The captioning model is consumed by both the ingestor-server and the upstream `nv-ingest-ms-runtime`, so set both pairs of env vars to the same values.

With Docker Compose:

```bash
export APP_NVINGEST_CAPTIONMODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_NVINGEST_CAPTIONENDPOINTURL="http://vlm-captioning-ms:8000/v1/chat/completions"
export VLM_CAPTION_MODEL_NAME="nvidia/nemotron-nano-12b-v2-vl"
export VLM_CAPTION_ENDPOINT="http://vlm-captioning-ms:8000/v1/chat/completions"
docker compose -f deploy/compose/nims.yaml --profile vlm-rag up -d   # starts vlm-captioning-ms
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

With Helm, in `values.yaml`:

```yaml
nimOperator:
  nim-vlm-captioning:
    enabled: true

ingestor-server:
  envVars:
    APP_NVINGEST_CAPTIONMODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
    APP_NVINGEST_CAPTIONENDPOINTURL: "http://nim-vlm-captioning:8000/v1/chat/completions"

nv-ingest:
  envVars:
    VLM_CAPTION_MODEL_NAME: nvidia/nemotron-nano-12b-v2-vl
    VLM_CAPTION_ENDPOINT: http://nim-vlm-captioning:8000/v1/chat/completions
```

### Use a single shared VLM for both roles

If you want to run only one VLM container and use it for both generation and captioning, point the captioning endpoints at the generation service and disable the dedicated captioning NIM.

With Docker Compose:

```bash
export APP_NVINGEST_CAPTIONENDPOINTURL="http://vlm-ms:8000/v1/chat/completions"
export VLM_CAPTION_ENDPOINT="http://vlm-ms:8000/v1/chat/completions"
# Bring up generation only (skip the vlm-captioning-ms profile)
docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

With Helm, in `values.yaml`:

```yaml
nimOperator:
  nim-vlm-captioning:
    enabled: false

ingestor-server:
  envVars:
    APP_NVINGEST_CAPTIONENDPOINTURL: "http://nim-vlm:8000/v1/chat/completions"

nv-ingest:
  envVars:
    VLM_CAPTION_ENDPOINT: http://nim-vlm:8000/v1/chat/completions
```

In this mode set `APP_NVINGEST_CAPTIONMODELNAME` and `VLM_CAPTION_MODEL_NAME` to whatever model the generation NIM serves.

### NVIDIA-hosted endpoints

To use NVIDIA-hosted endpoints for either role, point the corresponding `*_SERVERURL` / `*_ENDPOINTURL` at `https://integrate.api.nvidia.com/v1` (chat completions: `https://integrate.api.nvidia.com/v1/chat/completions`). Both `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` and `nvidia/nemotron-nano-12b-v2-vl` are available there.

## VLM to LLM Fallback (Optional)

By default, with VLM enabled, the RAG server uses VLM for all generation tasks. The `VLM_TO_LLM_FALLBACK` environment variable controls behavior for text-only queries (no images in query, messages, or retrieved context).

- **Default (no fallback)**: `VLM_TO_LLM_FALLBACK="false"`. The VLM handles all queries. Recommended for the 2xH100 setup.
- **Enable fallback**: `VLM_TO_LLM_FALLBACK="true"`. Text-only queries use a traditional LLM. You must deploy an LLM service alongside the VLM.

**GPU requirements for fallback**: Minimum **3xH100 GPUs**: GPU 0: Embedding and Reranker; GPU 1: VLM; GPU 2: LLM.

**Docker Compose with fallback**: Start both VLM and LLM (do not use the `vlm-generation` profile):

```bash
export VLM_MS_GPU_ID=1
export LLM_MS_GPU_ID=2
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d

export ENABLE_VLM_INFERENCE="true"
export VLM_TO_LLM_FALLBACK="true"
export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

:::{warning}
Do not use the `vlm-generation` profile when fallback is enabled; it skips the LLM. Using `VLM_TO_LLM_FALLBACK="true"` with that profile will cause errors for text-only queries.
:::

**Helm with fallback**: In `values.yaml`, set `VLM_TO_LLM_FALLBACK: "true"` and keep both `nim-vlm` and `nim-llm` enabled:

```yaml
envVars:
  ENABLE_VLM_INFERENCE: "true"
  VLM_TO_LLM_FALLBACK: "true"
  APP_VLM_MODELNAME: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
  APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

nimOperator:
  nim-vlm:
    enabled: true
  nim-llm:
    enabled: true
```

## Enabling Full VLM Multimodal RAG Pipeline

The VLM generation path covered above can be combined with the multimodal embedder and a multimodal reranker for a **fully VLM-powered ingestion + retrieval + generation pipeline**. This is the recommended setup when your corpus is image-heavy (PDFs with charts, diagrams, scanned tables) or when end-user queries themselves carry images.

The pipeline has three independently switchable components, each with its own dedicated guide:

### 1. VLM Embedding for ingestion (image modality)

The default embedder is **`nvidia/llama-nemotron-embed-vl-1b-v2`**, so PDF pages, tables, charts, and image elements are embedded by a multimodal model. The same model embeds text + image queries at retrieval time, so no extra rag-server config is needed beyond pointing `APP_EMBEDDINGS_*` at the VLM embedding NIM.

Setup, modality switches (text-only, structured-as-image, page-as-image), and Docker/Helm flows: see [Multimodal Retriever: Part 1: VLM Embedding for Ingestion](multimodal-retriever.md#part-1--vlm-embedding-for-ingestion).

### 2. VLM Reranker (image-aware reranking)

Swap the default text reranker for **`nvidia/llama-nemotron-rerank-vl-1b-v2`** and turn on `ENABLE_VLM_RERANKER_IMAGE_INPUT=True` so the reranker scores passages with awareness of the cited images, not just the surrounding text. This noticeably improves ordering when the most relevant chunk is signalled by its image content.

What the flag does, when to enable it, and Docker/Helm flows: see [Multimodal Retriever: Part 2: VLM Reranker](multimodal-retriever.md#part-2--vlm-reranker).

### 3. VLM Generation

Covered earlier on this page ([Enable VLM with Docker Compose](#enable-vlm-with-docker-compose), [Enable VLM with Helm](#enable-vlm-with-helm)). With reasoning mode enabled, the VLM produces a chain-of-thought before the final answer; the rag-server streams either just the answer or both, depending on `VLM_FILTER_THINK_TOKENS`.

### Putting it all together

For an end-to-end deployment that wires up multimodal *queries* (image + text from the user) on top of all three components, see [Multimodal Query Support](multimodal-query.md). It walks through the combined Docker Compose and Helm setups, including the trade-offs around reranking on image queries.


## Troubleshooting

- Ensure the VLM NIM is running and reachable at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure `NGC_API_KEY` is valid and has access to the model.
- Check rag-server logs for VLM inference or API authentication errors.
- Verify that images are ingested, captioned, and indexed in your knowledge base.

## Related Topics

- [Multimodal Retriever (VLM Embedding & VLM Reranker)](multimodal-retriever.md)
- [Multimodal Query Support](multimodal-query.md)
- [Change the LLM, Embedding Model, or Reranker](change-model.md)
- [Release Notes](release-notes.md)
- [Debugging](debugging.md)
- [Troubleshoot NVIDIA RAG Blueprint](troubleshooting.md)

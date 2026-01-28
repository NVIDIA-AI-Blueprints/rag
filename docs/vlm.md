<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Vision-Language Model (VLM) for Generation for NVIDIA RAG Blueprint

The Vision-Language Model (VLM) inference feature in the [NVIDIA RAG Blueprint](readme.md) enhances the system's ability to understand and reason about visual content. 
Unlike traditional image upload systems, this feature operates on image citations that are internally discovered during the retrieval process. 

:::{note}
B200 GPUs are not supported for VLM based inferencing in RAG.
For this feature, use H100 or A100 GPUs instead.
:::


- Key Use Cases for VLM

  - **Documents with charts and graphs**: Financial reports, scientific papers, business analytics.
  - **Technical diagrams**: Engineering schematics, architectural plans, flowcharts
  - **Visual data representations**: Infographics, tables with visual elements, dashboards
  - **Mixed content documents**: PDFs containing both text and images
  - **Image-heavy content**: Catalogs, product documentation, visual guides

- Key Benefits of VLM

  - **Seamless Multimodal Experience** – Users don't need to manually upload images; visual content is automatically discovered and analyzed from images embedded in documents.
  - **Improved Accuracy** – Enhanced response quality for documents containing images, charts, diagrams, and visual data.
  - **Quality Assurance** – Internal reasoning ensures only relevant visual insights are used.
  - **Contextual Understanding** – Visual analysis is performed in the context of the user's specific question.
  - **Fallback Handling** – System gracefully handles cases where images are insufficient or irrelevant.

:::{warning}
Enabling VLM inference increases response latency from additional image processing and VLM model inference time. Consider this trade-off between accuracy and speed based on your requirements.
:::



## How VLM Works in the RAG Pipeline

When VLM inference is enabled, the **VLM replaces the traditional LLM** in the RAG pipeline for generation tasks.

The VLM feature follows this flow:

1. **Automatic Image Discovery**: When a user query is processed, the RAG system retrieves relevant documents from the vector database. If any of these documents contain images (charts, diagrams, photos, etc.), they are automatically identified.
2. **Image Captioning at Ingestion**: During ingestion, images are extracted and captioned so they can be indexed and later cited for question answering.
3. **VLM Answer Generation**: At query time, the RAG server sends the user question, conversation history, and cited images to a Vision-Language Model. The **VLM directly generates the final answer** for the user, taking the place of the traditional LLM.



## Prompt customization

The VLM feature uses predefined prompts that can be customized to suit your specific needs:

- **VLM Analysis Prompt**: Located in [`src/nvidia_rag/rag_server/prompt.yaml`](../src/nvidia_rag/rag_server/prompt.yaml) under the `vlm_template` section.

To customize this prompt, follow the steps outlined in the [prompt.yaml file](../src/nvidia_rag/rag_server/prompt.yaml) for modifying prompt templates. The `vlm_template` controls how the question, textual context, and cited images are presented to the VLM.

### **VLM reasoning vs. non-reasoning mode**

The VLM model supports two modes that are controlled entirely via the `vlm_template`:

- **Non-reasoning mode (default)**:
  - Template path ends with `/no_think`.
  - Default generation parameters:
    - `APP_VLM_TEMPERATURE=0.1`
    - `APP_VLM_TOP_P=1.0`
    - `APP_VLM_MAX_TOKENS=8192`
- **Reasoning mode (chain-of-thought style)**:
  - Change the route in `vlm_template` from `/no_think` to `/think`.
  - Recommended generation parameters:
    - `APP_VLM_TEMPERATURE=0.3`
    - `APP_VLM_TOP_P=0.91`
    - `APP_VLM_MAX_TOKENS=8192`

You can set these parameters via environment variables (for example in `docker-compose-rag-server.yaml`) or directly through your deployment configuration.

### What Users Experience

Users interact with the system normally - they ask questions and receive responses. The VLM processing happens transparently in the background:

1. **User asks a question** about content that may have visual elements
2. **System retrieves relevant documents** including any images
3. **VLM analyzes images and text context** if present and relevant
4. **User receives a single, coherent answer** generated directly by the VLM



## Accuracy Improvement Example

The following example that uses the Ragbattle dataset demonstrates the accuracy improvement from enabling VLM.

Using the [Deloitte's Tax transformation trends survey from May 2021](https://www.deloitte.com/content/dam/assets-shared/en_gb/legacy/docs/research/2022/Deloitte-tax-operations-transformation-trends-survey-2021.pdf) 
and the following question:

```text
What is the percentage of companies with NextGen ERP systems/Advanced that said the tax team was highly effective in advising the business on emerging compliance issues?
```

Before enabling VLM, the system answers 38% with an accuracy score of 0.0. 
After enabling VLM, the system answers 64% with an accuracy score of 1.0. 
The answer is found on page 21 of the PDF (page 20 of the document).

The following table shows some approximate accuracy improvements from enabling VLM.

| Query                                                                                    | Correct Answer      | Answer Without VLM (Score) | Answer With VLM (Score)   | Reason for Improvement |
|------------------------------------------------------------------------------------------|---------------------|----------------------------|---------------------------|------------------------|
| Percentage for "…NextGen ERP system/Advanced" on "Effectiveness of the tax team…" graph. | "64%"               | "38%" (0.0)                | "64%" (1.0)               | Precise reading of a charted percentage. |
| Are Business development companies more or less flexible than Mezzanine funds?           | "less flexible"     | "more flexible" (0.0)      | "less flexible" (1.0)     | Correct comparative interpretation from a structured source. |
| Estimated cost of capital range for business development companies.                      | "SOFR+600 to 1,000" | "12-16%" (0.25)            | "SOFR+600 to 1,000" (1.0) | Extracted the correct range from a structured chart. |



## Start the VLM NIM Service (Local)

NVIDIA RAG uses the [**nemotron-nano-12b-v2-vl**](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl) Vision-Language Model by default, provided as the `vlm-ms` service in `deploy/compose/nims.yaml`.

### VLM-Generation Profile: VLM Replaces LLM

The `vlm-generation` profile in `deploy/compose/nims.yaml` is specifically designed for VLM-based generation on **2xH100 GPUs**. This profile:

- **Skips the NIM LLM deployment entirely** - VLM replaces the LLM service
- Deploys the VLM service (`vlm-ms`) as a replacement for the NIM LLM service
- Deploys embedding and reranker microservices

### GPU Assignment for 2xH100 Deployment

VLM must use GPU 1 (the GPU normally assigned to LLM):

**GPU Allocation**:
- GPU 0: Embedding and Reranker services
- GPU 1: VLM service (replaces LLM)

**Required Environment Variable**:
```bash
export VLM_MS_GPU_ID=1  # REQUIRED
```

### Quick Start: Deploy VLM

```bash
# Step 1: Set VLM GPU assignment
export VLM_MS_GPU_ID=1

# Step 2: Start VLM and supporting services (skips nim-llm)
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

### Customizing GPU Usage (Advanced)

For systems with 3+ GPUs, you can assign VLM to a different GPU:

```bash
export VLM_MS_GPU_ID=3  # Example: Use GPU 3 on multi-GPU systems
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

:::{warning}
Only change `VLM_MS_GPU_ID` for systems with 3+ GPUs.
:::



## Enable image extraction and captioning for VLM

For VLM-based generation to work correctly, images must be extracted and captioned during ingestion:

- In `deploy/compose/docker-compose-ingestor-server.yaml`, under the `ingestor-server` service, ensure:
  - `APP_NVINGEST_EXTRACTIMAGES` is set to `True` so images are extracted and stored.
  - Image captioning is enabled (by default, `APP_NVINGEST_CAPTIONMODELNAME` is set to `nvidia/nemotron-nano-12b-v2-vl` and `APP_NVINGEST_CAPTIONENDPOINTURL` points to the `vlm-ms` service).

When running with Docker Compose you can override these via environment variables, for example:

```bash
export APP_NVINGEST_EXTRACTIMAGES=True

docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

This ensures that images are available as citations and can be sent to the VLM at query time.

---

## Enable VLM Inference in RAG Server

After starting the VLM NIM service with the `vlm-generation` profile, configure the RAG server to use VLM for generation.

Set the following environment variables in [docker-compose-rag-server.yaml](../deploy/compose/docker-compose-rag-server.yaml):

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

**Environment Variable Descriptions**:
- `ENABLE_VLM_INFERENCE`: Enables VLM inference in the RAG server.
- `APP_VLM_MODELNAME`: The name of the VLM model to use.
- `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote).

:::{note}
When using the `vlm-generation` profile, there is no LLM service running. The VLM handles all generation tasks, with optional fallback behavior controlled by `VLM_TO_LLM_FALLBACK` (described later).
:::


Continue following the rest of the steps in [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md) to deploy the ingestion-server and rag-server containers.



## Using a Remote NVIDIA-Hosted NIM Endpoint (Optional)

To use a remote NVIDIA-hosted NIM for VLM inference:

1. Set the `APP_VLM_SERVERURL` environment variable to the remote endpoint provided by NVIDIA:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com/v1/"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

Continue following the rest of the steps in [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md) to deploy the ingestion-server and rag-server containers.



## Using Helm Chart Deployment

:::{note}
**GPU Requirements for Helm Deployment**:
- VLM uses GPU 1 (the same GPU normally assigned to LLM)
- **With MIG**: If MIG slicing is enabled, assign a dedicated MIG slice to the VLM. Refer to [mig-deployment.md](./mig-deployment.md) and [values-mig.yaml](../deploy/helm/mig-slicing/values-mig.yaml) for configuration details.
- **Additional GPU**: If you want to run both VLM and LLM simultaneously (not typical), an additional GPU is required.
:::

To enable VLM inference in Helm-based deployments, follow these steps:

1. Set VLM environment variables in `values.yaml`

   In your [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, under the `envVars` section, set the following environment variables:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
   APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"  # Local VLM NIM endpoint
   ```

2. Enable the `nim-vlm` helm chart and disable `nim-llm` since VLM replaces LLM:

   ```yaml
   nim-vlm:
     enabled: true
   
   nim-llm:
     enabled: false  # VLM replaces LLM for generation
   ```

   :::{important}
   **GPU Assignment**: By disabling `nim-llm` and enabling `nim-vlm`, the VLM will use the GPU resources normally allocated to the LLM, enabling VLM deployment without additional hardware.
   :::

3. Apply the updated Helm chart

   Run the following command to upgrade or install your deployment:

   ```
   helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc2.1.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```

4. Check if the VLM pod has come up

   A pod with the name `rag-nim-vlm-0` will start, this pod corresponds to the VLM model deployment. The `nim-llm` pod will NOT be created since it's disabled.

   ```
     rag       rag-nim-vlm-0       0/1     ContainerCreating   0          6m37s
   ```


:::{note}
**Service Architecture**:
- With VLM enabled and LLM disabled, the RAG pipeline uses VLM for all generation tasks
- The embedding and reranking services remain active for document retrieval
- For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`
- For remote endpoints, the `NGC_API_KEY` is required for authentication
:::


## Configure VLM image limits

Control how many images are sent to the VLM per request:
- `APP_VLM_MAX_TOTAL_IMAGES` (default: 5): Maximum total images (from the query, conversation history, and retrieved context) that are included in the VLM prompt. The pipeline will never exceed this.

Example (docker compose):

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MAX_TOTAL_IMAGES="5"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## Conversation history and context limitations

:::{warning}
The VLM receives the **current user query**, a truncated **conversation history**, and a textual summary of retrieved documents, together with any cited images. The effective context window of the VLM is limited, so very long conversations or large document contexts may be truncated.
:::

Mitigations:
- Keep user questions as self-contained as possible, especially in long-running conversations.
- Use retrieval and prompt tuning to focus the most relevant context for the VLM.



## VLM to LLM Fallback (Optional)

By default, once VLM inference is enabled, the RAG server uses VLM for **all** generation tasks, regardless of whether images are present. The `VLM_TO_LLM_FALLBACK` environment variable controls what happens for text-only queries (when no images are present in the query, messages, or retrieved context).

### Default Behavior (No Fallback)

```bash
VLM_TO_LLM_FALLBACK="false"  # Default
```

With the default setting, the VLM handles all queries, even text-only queries without any images. This is the recommended configuration for the 2xH100 setup described earlier.

### Enabling Fallback to LLM

If you want text-only queries to use a traditional LLM instead of the VLM:

```bash
VLM_TO_LLM_FALLBACK="true"
```

**Important**: When fallback is enabled, you **must also deploy an LLM service** alongside the VLM.

#### GPU Requirements for Fallback Configuration

To run both VLM and LLM services simultaneously, you need a minimum of **3xH100 GPUs**:

- **GPU 0**: Embedding and Reranker services
- **GPU 1**: VLM service
- **GPU 2**: LLM service (for fallback)

#### Deploying with LLM Fallback (Docker Compose)

To enable fallback with Docker Compose, you need to start both the VLM and LLM services:

```bash
# Set GPU assignments
export VLM_MS_GPU_ID=1
export LLM_MS_GPU_ID=2  # LLM on GPU 2

# Start all services (do NOT use vlm-generation profile)
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d

# Enable VLM inference with fallback
export ENABLE_VLM_INFERENCE="true"
export VLM_TO_LLM_FALLBACK="true"
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"

docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

:::{warning}
Do **NOT** use the `vlm-generation` profile when enabling fallback, as it skips the LLM deployment entirely. Using `VLM_TO_LLM_FALLBACK="true"` with the `vlm-generation` profile will cause errors for text-only queries.
:::

#### Deploying with LLM Fallback (Helm)

For Helm deployments with fallback, enable both `nim-vlm` and `nim-llm`:

```yaml
# In values.yaml
ENABLE_VLM_INFERENCE: "true"
VLM_TO_LLM_FALLBACK: "true"
APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

nim-vlm:
  enabled: true

nim-llm:
  enabled: true  # Keep LLM enabled for fallback
```


## Troubleshooting

- Ensure the VLM NIM is running and accessible at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure your `NGC_API_KEY` is valid and has access to the requested model.
- Check rag-server logs for errors related to VLM inference or API authentication.
- Verify that images are properly ingested, captioned, and indexed in your knowledge base.


## Related Topics

- [VLM Embedding for Ingestion](vlm-embed.md)
- [Multimodal Query Support](multimodal-query.md)
- [Release Notes](release-notes.md)
- [Debugging](debugging.md)
- [Troubleshoot NVIDIA RAG Blueprint](troubleshooting.md)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Vision-Language Model (VLM) for Generation for NVIDIA RAG Blueprint

The Vision-Language Model (VLM) inference feature in the [NVIDIA RAG Blueprint](readme.md) enhances the system's ability to understand and reason about visual content that is **automatically retrieved from the knowledge base**. Unlike traditional image upload systems, this feature operates on **image citations** that are internally discovered during the retrieval process.


:::{warning}
B200 GPUs are not supported for VLM based inferencing in RAG.
For this feature, use H100 or A100 GPUs instead.
:::



## **How VLM Works in the RAG Pipeline**

The VLM feature follows this flow:

1. **Automatic Image Discovery**: When a user query is processed, the RAG system retrieves relevant documents from the vector database. If any of these documents contain images (charts, diagrams, photos, etc.), they are automatically identified.
2. **Image Captioning at Ingestion**: During ingestion, images are extracted and captioned so they can be indexed and later cited for question answering.
3. **VLM Answer Generation**: At query time, the RAG server sends the user question, conversation history, and cited images to a Vision-Language Model. The **VLM directly generates the final answer** for the user.

There is no separate LLM reasoning step that post-processes the VLM output—once VLM inference is enabled, the VLM is responsible for generating the response (with optional fallback behavior described below).

## **Key Benefits**

- **Seamless Multimodal Experience**: Users don't need to manually upload images; visual content is automatically discovered and analyzed from images embedded in documents
- **Improved Accuracy**: Enhanced response quality for documents containing images, charts, diagrams, and visual data
- **Contextual Understanding**: Visual analysis is performed in the context of the user's specific question and retrieved document snippets
- **Configurable Fallback**: When no images are present, you can choose whether to keep using the VLM or fall back to the standard text-only LLM RAG flow

---

## When to Use VLM

The VLM feature is particularly beneficial when your knowledge base contains:

- **Documents with charts and graphs**: Financial reports, scientific papers, business analytics
- **Technical diagrams**: Engineering schematics, architectural plans, flowcharts
- **Visual data representations**: Infographics, tables with visual elements, dashboards
- **Mixed content documents**: PDFs containing both text and images
- **Image-heavy content**: Catalogs, product documentation, visual guides

:::{note}
**Latency Impact**: Enabling VLM inference will increase response latency due to additional image processing and VLM model inference time. Consider this trade-off between accuracy and speed based on your use case requirements.
:::

---

## **Prompt customization**

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

### **What Users Experience**

Users interact with the system normally - they ask questions and receive responses. The VLM processing happens transparently in the background:

1. **User asks a question** about content that may have visual elements
2. **System retrieves relevant documents** including any images
3. **VLM analyzes images and text context** if present and relevant
4. **User receives a single, coherent answer** generated directly by the VLM

---

## Start the VLM NIM Service (Local)

NVIDIA RAG uses the [**nemotron-nano-12b-v2-vl**](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl) Vision-Language Model by default, provided as the `vlm-ms` service in `deploy/compose/nims.yaml`.

To start the local VLM NIM service and the other NIMs required for VLM-based generation, run:

```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

This will launch the `vlm-ms` container (serving the model on port 1977, internal port 8000) together with the embedding and reranker microservices used by the RAG server.

### Customizing GPU Usage for VLM Service (Optional)

By default, the `vlm-ms` service uses GPU ID 5. You can customize which GPU to use by setting the `VLM_MS_GPU_ID` environment variable before starting the service:

```bash
export VLM_MS_GPU_ID=2  # Use GPU 2 instead of GPU 5
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

Alternatively, you can modify the `nims.yaml` file directly to change the GPU assignment:

```yaml
# In deploy/compose/nims.yaml, locate the vlm-ms service and modify:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['${VLM_MS_GPU_ID:-5}']  # Change 5 to your desired GPU ID
          capabilities: [gpu]
```

:::{note}
Ensure the specified GPU is available and has sufficient memory for the VLM model.
:::

---

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

### Enable VLM Inference in RAG Server

Start only the required NIM services (VLM, Embedding, Reranker) using the `vlm-generation` profile defined in `deploy/compose/nims.yaml`:

```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
```

This profile starts the following services and skips `nim-llm`:

- nemoretriever-embedding-ms
- nemoretriever-ranking-ms
- vlm-ms

Set the following environment variables in [docker-compose-rag-server.yaml](../deploy/compose/docker-compose-rag-server.yaml) to enable VLM inference in RAG server:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

- `ENABLE_VLM_INFERENCE`: Enables VLM inference in the RAG server.
- `APP_VLM_MODELNAME`: The name of the VLM model to use (default: `nvidia/nemotron-nano-12b-v2-vl`).
- `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote).

Once `ENABLE_VLM_INFERENCE` is set, the RAG server uses the VLM to generate the final answer. The `VLM_TO_LLM_FALLBACK` flag controls what happens when no images are available, as described later.

---

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
On prem deployment of the VLM model requires an additional 1xH100 or 1xB200 GPU in default deployment configuration.
If MIG slicing is enabled on the cluster, ensure to assign a dedicated slice to the VLM. Check [mig-deployment.md](./mig-deployment.md) and  [values-mig.yaml](../deploy/helm/mig-slicing/values-mig.yaml) for more information.
:::

To enable VLM inference in Helm-based deployments, follow these steps:

1. **Set VLM environment variables in `values.yaml`**

   In your [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, under the `envVars` section, set the following environment variables:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
   APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"  # Local VLM NIM endpoint
   ```

  Also enable the `nim-vlm` helm chart
  ```yaml
  nim-vlm:
    enabled: true
  ```

2. **Apply the updated Helm chart**

   Run the following command to upgrade or install your deployment:

   ```
   helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```

3. **Check if the VLM pod has come up**

  A pod with the name `rag-0` will start, this pod corresponds to the VLM model deployment.

    ```
      rag       rag-0       0/1     ContainerCreating   0          6m37s
    ```


:::{note}
For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`. For remote endpoints, the `NGC_API_KEY` is required for authentication.
:::


### **When VLM Processing Occurs**

VLM processing is triggered when:
- `ENABLE_VLM_INFERENCE` is set to `true`
- The VLM service is accessible and responding

Once VLM inference is enabled, the RAG server uses the VLM to generate the final answer. The `VLM_TO_LLM_FALLBACK` flag controls behavior **only when no images are present** in the query, messages, or retrieved context:

- If `VLM_TO_LLM_FALLBACK="false"` (default): the pipeline **still routes generation through the VLM**, even for text-only queries with no images.
- If `VLM_TO_LLM_FALLBACK="true"`: text-only queries (with no images in the query, messages, or context) **fall back to the regular LLM-based RAG flow** instead of calling the VLM.


## Troubleshooting

- Ensure the VLM NIM is running and accessible at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure your `NGC_API_KEY` is valid and has access to the requested model.
- Check rag-server logs for errors related to VLM inference or API authentication.
- Verify that images are properly ingested, captioned, and indexed in your knowledge base.


### Configure VLM image limits

Control how many images are sent to the VLM per request:
- `APP_VLM_MAX_TOTAL_IMAGES` (default: 5): Maximum total images (from the query, conversation history, and retrieved context) that are included in the VLM prompt. The pipeline will never exceed this.

Example (docker compose):

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MAX_TOTAL_IMAGES="5"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```


**Important Notes:**
- When this flag is enabled and images are provided as input (either from context or query), the VLM response will always be used as the final answer
- This mode is useful when you want pure visual analysis without additional text interpretation or reasoning
- The response will be based solely on what the VLM can extract from the images, without incorporating textual context from retrieved documents

#### Use VLM response as the final answer (Helm)

To enable final-answer mode with Helm (skip `nim-llm` and return the VLM output directly):

1) In your `values.yaml` for the chart at `deploy/helm/nvidia-blueprint-rag/values.yaml`, set the following under `envVars`:

```yaml
ENABLE_VLM_INFERENCE: "true"
```

2) Enable the VLM NIM and disable the LLM NIM:

```yaml
nim-vlm:
  enabled: true

nim-llm:
  enabled: false
```

3) (Optional, recommended) Ensure features that depend on the LLM remain disabled:

```yaml
ENABLE_QUERYREWRITER: "False"
ENABLE_REFLECTION: "False"
```

4) Apply or upgrade the release:

```bash
helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev-rc1.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

:::{note}
In this mode, the RAG server will use the VLM output as the final response. Keep the embedding and reranker services enabled as in the default chart configuration. If you use a local VLM, also set `APP_VLM_SERVERURL` (for example, `http://nim-vlm:8000/v1`) and enable the `nim-vlm` subchart as shown above.
:::

### Conversation history and context limitations

:::{warning}
The VLM receives the **current user query**, a truncated **conversation history**, and a textual summary of retrieved documents, together with any cited images. The effective context window of the VLM is limited, so very long conversations or large document contexts may be truncated.
:::

Mitigations:
- Keep user questions as self-contained as possible, especially in long-running conversations.
- Use retrieval and prompt tuning to focus the most relevant context for the VLM.
<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Vision-Language Model (VLM) for Generation for NVIDIA RAG Blueprint

The Vision-Language Model (VLM) inference feature in the [NVIDIA RAG Blueprint](readme.md) enhances the system's ability to understand and reason about visual content that is **automatically retrieved from the knowledge base**. Unlike traditional image upload systems, this feature operates on **image citations** that are internally discovered during the retrieval process.


> [!WARNING]
>
> B200 GPUs are not supported for VLM based inferencing in RAG.
> For this feature, use H100 or A100 GPUs instead.



## **How VLM Works in the RAG Pipeline**

The VLM feature follows this sophisticated flow:

1. **Automatic Image Discovery**: When a user query is processed, the RAG system retrieves relevant documents from the vector database. If any of these documents contain images (charts, diagrams, photos, etc.), they are automatically identified.

2. **VLM Analysis**: Up to 4 relevant images are sent to a Vision-Language Model for analysis, along with the user's question.

3. **Intelligent Reasoning**: The VLM's response is **not directly returned to the user**. Instead, it undergoes an internal reasoning process where another LLM evaluates whether the visual insights should be incorporated into the final response.

4. **Conditional Integration**: Only if the reasoning determines the VLM response is relevant and valuable, it gets augmented into the LLM's final prompt as additional context.

5. **Unified Response**: The user receives a single, coherent response that seamlessly incorporates both textual and visual understanding.

## **Key Benefits**

- **Seamless Multimodal Experience**: Users don't need to manually upload images; visual content is automatically discovered and analyzed from images embedded in documents
- **Improved Accuracy**: Enhanced response quality for documents containing images, charts, diagrams, and visual data
- **Quality Assurance**: Internal reasoning ensures only relevant visual insights are used
- **Contextual Understanding**: Visual analysis is performed in the context of the user's specific question
- **Fallback Handling**: System gracefully handles cases where images are insufficient or irrelevant

---

## When to Use VLM

The VLM feature is particularly beneficial when your knowledge base contains:

- **Documents with charts and graphs**: Financial reports, scientific papers, business analytics
- **Technical diagrams**: Engineering schematics, architectural plans, flowcharts
- **Visual data representations**: Infographics, tables with visual elements, dashboards
- **Mixed content documents**: PDFs containing both text and images
- **Image-heavy content**: Catalogs, product documentation, visual guides

> [!Note]
> **Latency Impact**: Enabling VLM inference will increase response latency due to additional image processing and VLM model inference time. Consider this trade-off between accuracy and speed based on your use case requirements.

---

## **Prompt customization**

The VLM feature uses predefined prompts that can be customized to suit your specific needs:

- **VLM Analysis Prompt**: Located in [`src/nvidia_rag/rag_server/prompt.yaml`](../src/nvidia_rag/rag_server/prompt.yaml) under the `vlm_template` section
- **Response Reasoning Prompt**: Located in the same file under the `vlm_response_reasoning_template` section

To customize these prompts, follow the steps outlined in the [prompt.yaml file](../src/nvidia_rag/rag_server/prompt.yaml) for modifying prompt templates. The significance of these two prompts are explained below.

The VLM feature employs a sophisticated two-step process where these prompts are utilized:

1. **VLM Analysis Step**:
   - Images are sent to the Vision-Language Model using the `vlm_template` prompt
   - The VLM analyzes the visual content and generates a response based solely on the images
   - If images lack sufficient information, the VLM returns: *"The provided images do not contain enough information to answer this question."*

2. **Response Verification Step**:
   - The VLM's response is then sent to an LLM using the `vlm_response_reasoning_template` prompt
   - This LLM evaluates whether the VLM response should be incorporated into the final response
   - The reasoning LLM considers relevance, consistency with textual context, and whether the response adds valuable information
   - Only if the reasoning returns "USE" does the VLM response get integrated into the final prompt

This two-step process ensures that visual insights are only used when they genuinely enhance the response quality and relevance.


### **What Users Experience**

Users interact with the system normally - they ask questions and receive responses. The VLM processing happens transparently in the background:

1. **User asks a question** about content that may have visual elements
2. **System retrieves relevant documents** including any images
3. **VLM analyzes images** if present and relevant
4. **System generates unified response** that incorporates visual insights when beneficial
5. **User receives a single, coherent answer** that seamlessly blends textual and visual understanding

---

## Start the VLM NIM Service (Local)

NVIDIA RAG uses the [**llama-3.1-nemotron-nano-vl-8b-v1**](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1) VLM model by default, provided as the `vlm-ms` service in `nims.yaml`.

To start the local VLM NIM service, run:

```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm up -d
```

This will launch the `vlm-ms` container, which serves the model on port 1977 (internal port 8000).

### Customizing GPU Usage for VLM Service (Optional)

By default, the `vlm-ms` service uses GPU ID 5. You can customize which GPU to use by setting the `VLM_MS_GPU_ID` environment variable before starting the service:

```bash
export VLM_MS_GPU_ID=2  # Use GPU 2 instead of GPU 5
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm up -d
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

> [!Note]
> Ensure the specified GPU is available and has sufficient memory for the VLM model.

---

### Enable VLM Inference in RAG Server

Set the following environment variables to enable VLM inference:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

- `ENABLE_VLM_INFERENCE`: Enables VLM inference in the RAG server
- `APP_VLM_MODELNAME`: The name of the VLM model to use (default: Llama Cosmos Nemotron 8b)
- `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote)

---

Continue following the rest of the steps in [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md) to deploy the ingestion-server and rag-server containers.

## Using a Remote NVIDIA-Hosted NIM Endpoint (Optional)

To use a remote NVIDIA-hosted NIM for VLM inference:

1. Set the `APP_VLM_SERVERURL` environment variable to the remote endpoint provided by NVIDIA:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com/v1/"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

Continue following the rest of the steps in [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md) to deploy the ingestion-server and rag-server containers.


## Using Helm Chart Deployment

> [!Note]
> On prem deployment of the VLM model requires an additional 1xH100 or 1xB200 GPU in default deployment configuration.
> If MIG slicing is enabled on the cluster, ensure to assign a dedicated slice to the VLM. Check [mig-deployment.md](./mig-deployment.md) and  [values-mig.yaml](../deploy/helm/mig-slicing/values-mig.yaml) for more information.

To enable VLM inference in Helm-based deployments, follow these steps:

1. **Set VLM environment variables in `values.yaml`**

   In your [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, under the `envVars` section, set the following environment variables:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
   APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"  # Local VLM NIM endpoint
   ```

  Also enable the `nim-vlm` service.
  ```yaml
  nim-vlm:
    enabled: true
  ```

2. **Apply the updated Helm chart**

   Run the following command to upgrade or install your deployment:

   ```
   helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
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


> [!Note]
> For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`. For remote endpoints, the `NGC_API_KEY` is required for authentication.



### **When VLM Processing Occurs**

VLM processing is triggered when:
- `ENABLE_VLM_INFERENCE` is set to `true`
- Retrieved documents contain images (identified by `content_metadata.type`)
- Images are successfully extracted from MinIO storage
- The VLM service is accessible and responding



## Troubleshooting

- Ensure the VLM NIM is running and accessible at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure your `NGC_API_KEY` is valid and has access to the requested model.
- Check rag-server logs for errors related to VLM inference or API authentication.
- Verify that images are properly ingested and indexed in your knowledge base.
- Monitor VLM response reasoning logs to understand when visual insights are being used or skipped.

### VLM response reasoning (optional)

By default, VLM response reasoning is disabled. If you observe incorrect or low-quality VLM outputs being incorporated, you can enable a reasoning gate so an LLM verifies whether to include the VLM response.

- Enable at runtime (docker compose):

  ```bash
  export ENABLE_VLM_INFERENCE="true"
  export ENABLE_VLM_RESPONSE_REASONING="true"

  # Apply by restarting rag-server
  docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
  ```

- Programmatic/config toggle:
  - Set `vlm.enable_vlm_response_reasoning` to `true` (maps to `ENABLE_VLM_RESPONSE_REASONING`).

When enabled, the LLM checks the VLM output and includes it only when deemed relevant and helpful. Disable it by setting `ENABLE_VLM_RESPONSE_REASONING="false"` if it filters out too aggressively during experimentation.

### Configure VLM image limits

Control how many images are sent to the VLM per request:

- `APP_VLM_MAX_TOTAL_IMAGES` (default: 4): Maximum total images included (query + context). The pipeline will never exceed this.
- `APP_VLM_MAX_QUERY_IMAGES` (default: 1): Maximum number of query images (e.g., screenshots supplied alongside the question).
- `APP_VLM_MAX_CONTEXT_IMAGES` (default: 1): Maximum number of context images (extracted from citations).

Notes:
- If `APP_VLM_MAX_QUERY_IMAGES + APP_VLM_MAX_CONTEXT_IMAGES > APP_VLM_MAX_TOTAL_IMAGES`, the server logs a warning and truncates to the total limit.
- Query images are added first (up to `APP_VLM_MAX_QUERY_IMAGES`), then remaining slots are filled with context images up to `APP_VLM_MAX_CONTEXT_IMAGES` while respecting the total cap.

Example (docker compose):

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MAX_TOTAL_IMAGES="4"
export APP_VLM_MAX_QUERY_IMAGES="1"
export APP_VLM_MAX_CONTEXT_IMAGES="3"

# Apply by restarting rag-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### VLM response as final answer

By default, the VLM's response is added to the LLM as additional context, and the LLM generates the final response incorporating both textual and visual insights. You can configure the system to use the VLM's response directly as the final answer, bypassing the LLM reasoning step entirely.

#### Use VLM response as the final answer with a dedicated profile

If you want the VLM output to be returned as the final answer and avoid starting the `nim-llm` service, use the dedicated compose profile that runs only the VLM, embedding, and reranker microservices.

1) Set the environment variables for the RAG server:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_RESPONSE_AS_FINAL_ANSWER="true"
```

2) Start only the required NIM services (VLM, Embedding, Reranker) using the `vlm-final-answer` profile defined in `deploy/compose/nims.yaml`:

```bash
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm-final-answer up -d
```

This profile starts the following services and skips `nim-llm`:

- nemoretriever-embedding-ms
- nemoretriever-ranking-ms
- vlm-ms

- `APP_VLM_RESPONSE_AS_FINAL_ANSWER` (default: false): When enabled, the VLM's response becomes the final answer without additional LLM processing.

Enable at runtime (docker compose):

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_RESPONSE_AS_FINAL_ANSWER="true"

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
APP_VLM_RESPONSE_AS_FINAL_ANSWER: "true"
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
helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

> [!Note]
> In this mode, the RAG server will use the VLM output as the final response. Keep the embedding and reranker services enabled as in the default chart configuration. If you use a local VLM, also set `APP_VLM_SERVERURL` (for example, `http://nim-vlm:8000/v1`) and enable the `nim-vlm` subchart as shown above.


### Conversation history and context limitations

> [!Warning]
> Conversation history is not passed to the VLM. The VLM receives only the current prompt and the cited image(s), and its effective context window is limited. When `APP_VLM_RESPONSE_AS_FINAL_ANSWER` is set to `true` and the user query depends on prior turns or broader textual context, the VLM will not decontextualize the query and may produce incomplete or off-target answers.

Mitigations:
- Rephrase or rewrite the user query to be self-contained before sending to the VLM (e.g., enable query rewriting upstream).
- Keep `APP_VLM_RESPONSE_AS_FINAL_ANSWER` set to `false` so the LLM composes the final answer, optionally incorporating the VLM output as additional context.
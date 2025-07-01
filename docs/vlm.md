<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Vision-Language Model (VLM) for generation in NVIDIA RAG

## Overview

The Vision-Language Model (VLM) inference feature in NVIDIA RAG enhances the system's ability to understand and reason about visual content that is **automatically retrieved from the knowledge base**. Unlike traditional image upload systems, this feature operates on **image citations** that are internally discovered during the retrieval process.

### **How VLM Works in the RAG Pipeline**

The VLM feature follows this sophisticated flow:

1. **Automatic Image Discovery**: When a user query is processed, the RAG system retrieves relevant documents from the vector database. If any of these documents contain images (charts, diagrams, photos, etc.), they are automatically identified.

2. **VLM Analysis**: Up to 4 relevant images are sent to a Vision-Language Model for analysis, along with the user's question.

3. **Intelligent Reasoning**: The VLM's response is **not directly returned to the user**. Instead, it undergoes an internal reasoning process where another LLM evaluates whether the visual insights should be incorporated into the final response.

4. **Conditional Integration**: Only if the reasoning determines the VLM response is relevant and valuable, it gets augmented into the LLM's final prompt as additional context.

5. **Unified Response**: The user receives a single, coherent response that seamlessly incorporates both textual and visual understanding.

### **Key Benefits**

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
```

- `ENABLE_VLM_INFERENCE`: Enables VLM inference in the RAG server
- `APP_VLM_MODELNAME`: The name of the VLM model to use (default: Llama Cosmos Nemotron 8b)
- `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote)

---

Continue following the rest of steps [in quickstart](quickstart.md) to deploy the ingestion-server and rag-server containers.

## Using a Remote NVIDIA-Hosted NIM Endpoint (Optional)

To use a remote NVIDIA-hosted NIM for VLM inference:

1. Set the `APP_VLM_SERVERURL` environment variable to the remote endpoint provided by NVIDIA:

```bash
export ENABLE_VLM_INFERENCE="true"
export APP_VLM_MODELNAME="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com/v1/"
```

Continue following the rest of steps [in quickstart](quickstart.md) to deploy the ingestion-server and rag-server containers.

---

## Using Helm Chart Deployment

> [!Note]
> On prem deployment of the VLM model requires an additional 1xH100 or 1xB200 GPU in default deployment configuration.
> If MIG slicing is enabled on the cluster, ensure to assign a dedicated slice to the VLM. Check [mig-deployment.md](./mig-deployment.md) and  [values-mig.yaml](../deploy/helm/mig-slicing/values-mig.yaml) for more information. 

To enable VLM inference in Helm-based deployments, follow these steps:

1. **Set VLM environment variables in `values.yaml`**

   In your `rag-server/values.yaml` file, under the `envVars` section, set the following environment variables:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
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
   helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.2.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f rag-server/values.yaml
   ```

3. **Check if the VLM pod has come up**

  A pod with the name `rag-0` will start, this pod corresponds to the VLM model deployment.
    
    ```
      rag       rag-0       0/1     ContainerCreating   0          6m37s
    ```


> [!Note]
> For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`. For remote endpoints, the `NGC_API_KEY` is required for authentication.

---

### **When VLM Processing Occurs**

VLM processing is triggered when:
- `ENABLE_VLM_INFERENCE` is set to `true`
- Retrieved documents contain images (identified by `content_metadata.type`)
- Images are successfully extracted from MinIO storage
- The VLM service is accessible and responding

---

## Troubleshooting

- Ensure the VLM NIM is running and accessible at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure your `NGC_API_KEY` is valid and has access to the requested model.
- Check rag-server logs for errors related to VLM inference or API authentication.
- Verify that images are properly ingested and indexed in your knowledge base.
- Monitor VLM response reasoning logs to understand when visual insights are being used or skipped.
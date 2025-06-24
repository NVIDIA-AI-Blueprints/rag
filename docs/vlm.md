<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Vision-Language Model (VLM) Inference in NVIDIA RAG

## Overview

The Vision-Language Model (VLM) inference feature in NVIDIA RAG enables the system to analyze images alongside text, providing richer, multimodal responses. When enabled, the RAG pipeline can:
- Analyze up to 4 images per query using a VLM (e.g., NIM for VLMs).
- Merge and process images from document context.
- Use an LLM to reason about and validate VLM-generated responses.

This is especially useful for scenarios where visual context is important, such as document Q&A, visual search, or multimodal chatbots.

---

## Step 1: Start the VLM NIM Service (Local)

NVIDIA RAG uses the **Llama Cosmos Nemotron 8b** VLM model by default, provided as the `vlm-ms` service in `nims.yaml`.

To start the local VLM NIM service, run:

```bash
docker compose -f deploy/compose/nims.yaml --profile vlm up -d
```

This will launch the `vlm-ms` container, which serves the model on port 1977 (internal port 8000).

---

## Step 2: Enable VLM Inference in RAG Server

In your `docker-compose-rag-server.yaml`, ensure the following environment variable is set to enable VLM inference:

```yaml
environment:
  ENABLE_VLM_INFERENCE: "true"
```

This tells the RAG server to invoke the VLM for image analysis when relevant.

---

## Step 3: Configure VLM Model and Endpoint via Environment Variables

The VLM model name and server URL are configured using environment variables in `docker-compose-rag-server.yaml`:

```yaml
environment:
  APP_VLM_MODELNAME: "nvidia/llama-cosmos-nemotron-8b-instruct"  # Default VLM model
  APP_VLM_SERVERURL: "http://vlm-ms:8000/v1"  # Local VLM NIM endpoint
```

- `APP_VLM_MODELNAME`: The name of the VLM model to use (default: Llama Cosmos Nemotron 8b).
- `APP_VLM_SERVERURL`: The URL of the VLM NIM server (local or remote).

You can override these values as needed for your deployment.

---

## Step 4: Continue following the rest of steps in quickstart to deploy the ingestion-server and rag-server containers.


## Step 5: Using a Remote NVIDIA-Hosted NIM Endpoint (Optional)

To use a remote NVIDIA-hosted NIM for VLM inference:

1. Set the `APP_VLM_SERVERURL` environment variable to the remote endpoint provided by NVIDIA.
    `https://integrate.api.nvidia.com/v1/`

2. Set the `NGC_API_KEY` environment variable with your NVIDIA API key:

```bash
export NGC_API_KEY="<your-nvidia-api-key>"
```

or in `docker-compose-rag-server.yaml`:

```yaml
environment:
  NGC_API_KEY: "<your-nvidia-api-key>"
```

**Note:** The API key is required for authentication with NVIDIA-hosted NIM endpoints.

---

## Using Helm Chart Deployment

To enable VLM inference in Helm-based deployments, follow these steps:

1. **Set VLM environment variables in `values.yaml`**

   In your `rag-server/values.yaml` file, under the `envVars` section, set the following environment variables:

   ```yaml
   ENABLE_VLM_INFERENCE: "true"
   APP_VLM_MODELNAME: "nvidia/llama-cosmos-nemotron-8b-instruct"
   APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"  # Local VLM NIM endpoint
   ```

   For remote NVIDIA-hosted NIM endpoints, set:

   ```yaml
   APP_VLM_SERVERURL: "https://integrate.api.nvidia.com/v1/"
   ```

  Also enable the `nim-vlm` helm chart
  ```yaml
  nim-vlm:
    enabled: true
  ```

2. **Set the NVIDIA API Key for remote endpoints**

   In your `values.yaml`, ensure the `ngcApiSecret.password` is set to your NVIDIA API key:

   ```yaml
   ngcApiSecret:
     name: "ngc-api"
     password: "<your-nvidia-api-key>"
     create: true
   ```

   The deployment will automatically mount this key as the `NVIDIA_API_KEY` environment variable in the container.

3. **Apply the updated Helm chart**

   Run the following command to upgrade or install your deployment:

   ```bash
   helm upgrade --install rag -n <namespace> https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.2.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f rag-server/values.yaml
   ```

> [!Note]
> For local VLM inference, ensure the VLM NIM service is running and accessible at the configured `APP_VLM_SERVERURL`. For remote endpoints, the `NVIDIA_API_KEY` is required for authentication.

---

## Example: Minimal Configuration for Remote VLM

```yaml
environment:
  ENABLE_VLM_INFERENCE: "true"
  APP_VLM_MODELNAME: "nvidia/llama-cosmos-nemotron-8b-instruct"
  APP_VLM_SERVERURL: "https://integrate.api.nvidia.com/v1/"
```

---

## Troubleshooting
- Ensure the VLM NIM is running and accessible at the configured `APP_VLM_SERVERURL`.
- For remote endpoints, ensure your `NGC_API_KEY` is valid and has access to the requested model.
- Check logs for errors related to VLM inference or API authentication. 
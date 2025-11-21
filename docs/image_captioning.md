<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Image Captioning Support for NVIDIA RAG Blueprint

You can enable image captioning support for [NVIDIA RAG Blueprint](readme.md). Enabling image captioning will yield higher accuracy for querstions relevant to images in the ingested documents at the cost of higher ingestion latency.

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), to enable image captioning support, you have the following options:

  - [Using on-prem VLM model (Recommended)](#using-on-prem-vlm-model-recommended)
  - [Using cloud hosted VLM model](#using-cloud-hosted-vlm-model)
  - [Using Helm chart deployment (On-prem only)](#using-helm-chart-deployment-on-prem-only)

:::{warning}
B200 GPUs are not supported for image captioning support for ingested documents.
For this feature, use H100 or A100 GPUs instead.
:::


## Using on-prem VLM model (Recommended)
1. Deploy the VLM model on-prem. You need a H100 or A100 or B200 GPU to deploy this model.
   ```bash
   export VLM_MS_GPU_ID=<AVAILABLE_GPU_ID>
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile vlm up -d
   ```

2. Make sure the vlm container is up and running
   ```bash
   docker ps --filter "name=nemo-vlm-microservice" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   nemo-vlm-microservice                   Up 5 minutes (healthy)
   ```

3. Enable image captioning
   Export the below environment variable and relaunch the ingestor-server container.
   ```bash
   export APP_NVINGEST_EXTRACTIMAGES="True"
   export APP_NVINGEST_CAPTIONENDPOINTURL="http://vlm-ms:8000/v1/chat/completions"
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

## Using cloud hosted VLM model
1. Set caption endpoint and model to API catalog
   ```bash
   export APP_NVINGEST_CAPTIONENDPOINTURL="https://integrate.api.nvidia.com/v1/chat/completions"
   export APP_NVINGEST_CAPTIONMODELNAME="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
   ```

2. Enable image captioning
   Export the below environment variable and relaunch the ingestor-server container.
   ```bash
   export APP_NVINGEST_EXTRACTIMAGES="True"
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

:::{tip}
You can change the model name and model endpoint in case of an externally hosted VLM model by setting these two environment variables and restarting the ingestion services
:::

```bash
export APP_NVINGEST_CAPTIONMODELNAME="<vlm_nim_http_endpoint_url>"
export APP_NVINGEST_CAPTIONMODELNAME="<model_name>"
```

## Using Helm chart deployment (On-prem only)

To enable image captioning in Helm-based deployments by using an on-prem VLM model, use the following procedure.


1. In the `values.yaml` file, in the `ingestor-server.envVars` section, set the following environment variables.

   ```yaml
   APP_NVINGEST_EXTRACTIMAGES: "True"
   APP_NVINGEST_CAPTIONENDPOINTURL: "http://nim-vlm:8000/v1/chat/completions"
   APP_NVINGEST_CAPTIONMODELNAME: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
   ```

2. Enable the VLM image captioning model in your `values.yaml` file.

   ```yaml
   nim-vlm:
      enabled: true
   ```

3. Apply the updated Helm chart by running the following code.

   ```bash
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
   --username '$oauthtoken' \
   --password "${NGC_API_KEY}" \
   --set imagePullSecret.password=$NGC_API_KEY \
   --set ngcApiSecret.password=$NGC_API_KEY \
   -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```

:::{note}
Enabling the on-prem VLM model increases the total GPU requirement to 9xH100 GPUs.
:::

:::{warning}
With [image captioning enabled](image_captioning.md), uploaded files will fail to get ingested, if they do not contain any graphs, charts, tables or plots. This is currently a known limitation.
:::
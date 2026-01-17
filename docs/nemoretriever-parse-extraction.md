<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable PDF extraction with Nemotron Parse for NVIDIA RAG Blueprint

For enhanced PDF extraction capabilities, particularly for scanned documents or documents with complex layouts, you can use the Nemotron Parse service with the **NVIDIA RAG Blueprint** This service provides higher-accuracy text extraction and improved PDF parsing compared to the default PDF extraction method.

:::{warning}
Nemotron Parse is not supported on NVIDIA B200 GPUs or RTX Pro 6000 GPUs.
For this feature, use H100 or A100 GPUs instead.
:::



## Using Docker Compose

### Using On-Prem Models

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. Deploy the Nemotron Parse service along with other required NIMs:
   ```bash
   USERID=$(id -u) docker compose --profile rag --profile nemotron-parse -f deploy/compose/nims.yaml up -d
   ```

3. Configure the ingestor-server to use Nemotron Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

### Using NVIDIA Hosted API Endpoints

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-nvidia-hosted.md) up to and including the step labelled "Start the vector db containers from the repo root."


2. Export the following variables to use nemotron parse API endpoints:

   ```bash
   export NEMOTRON_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
   export NEMOTRON_PARSE_MODEL_NAME=nvidia/nemotron-parse
   export NEMOTRON_PARSE_INFER_PROTOCOL=http
   ```

3. Configure the ingestor-server to use Nemotron Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

:::{note}
When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).
:::

## Using Helm

To enable PDF extraction with Nemotron Parse using Helm, you need to enable the Nemotron Parse service and configure the ingestor-server to use it.

### Prerequisites
- Ensure you have sufficient GPU resources. Nemotron Parse requires a dedicated GPU.

### Deployment Command

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc2.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set nv-ingest.nimOperator.nemotron_parse.enabled=true \
  --set ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemotron_parse"
```

:::{note}
**Key Configuration Changes:**
- `nv-ingest.nimOperator.nemotron_parse.enabled=true` - Enables Nemotron Parse NIM
- `ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemotron_parse"` - Configures ingestor to use Nemotron Parse
:::

## Limitations and Requirements

When using Nemotron Parse for PDF extraction, consider the following:

- Nemotron Parse only supports PDF format documents, not image files. Attempting to process non-PDF files will lead them to be extracted using the default extraction method.
- The service requires GPU resources and must run on a dedicated GPU. Make sure you have sufficient GPU resources available before enabling this feature.
- The extraction quality may vary depending on the PDF structure and content.
- Nemotron Parse is not supported on NVIDIA B200 GPUs or RTX Pro 6000 GPUs.

For detailed information about hardware requirements and supported GPUs for all NeMo Retriever extraction NIMs, refer to the [Nemotron Parse Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-parse).

## Available PDF Extraction Methods

The `APP_NVINGEST_PDFEXTRACTMETHOD` environment variable supports the following values:

- `nemotron_parse`: Uses the Nemotron Parse service for enhanced PDF extraction (recommended for scanned documents or documents with complex layouts)
- `pdfium`: Uses the default PDFium-based extraction
- `None`: Uses the default extraction method

:::{note}
The Nemotron Parse service requires GPU resources and must run on a dedicated GPU. Make sure you have sufficient GPU resources available before enabling this feature.
:::

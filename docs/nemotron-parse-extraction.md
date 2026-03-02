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

   :::{note}
   **Nemotron-parse-only:** To use only Nemotron Parse for PDF and table extraction (no OCR, page-elements, graphic-elements, or table-structure NIMs), additionally set **`APP_NVINGEST_EXTRACTTABLESMETHOD: "nemotron_parse"`** in ingestor-server environment variables.
   In addition, set **`COMPONENTS_TO_READY_CHECK`** to an empty value in the **nv-ingest** service environment (e.g. in the compose file where nv-ingest runs) so that nv-ingest readiness passes when other NIMs are not running.
   :::

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb).

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

   :::{note}
   **Nemotron-parse-only:** To use only Nemotron Parse for PDF and table extraction (no other ingest NIMs required), additionally set **`APP_NVINGEST_EXTRACTTABLESMETHOD: "nemotron_parse"`** in ingestor-server environment variables.
   If you run only the Nemotron Parse NIM, also set **`COMPONENTS_TO_READY_CHECK`** to an empty value in the **nv-ingest** service environment so nv-ingest readiness passes without other NIMs.
   :::

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb).

:::{note}
When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).
:::

## Using Helm

To enable PDF extraction with Nemotron Parse using Helm, enable the Nemotron Parse service and configure the ingestor-server to use it.

### Prerequisites
- Ensure you have sufficient GPU resources. Nemotron Parse requires a dedicated GPU.

### Deployment Steps

To deploy with Nemotron Parse enabled:

Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable Nemotron Parse and, for nemotron-parse-only, configure nv-ingest readiness:

```yaml
# Enable Nemotron Parse NIM
nv-ingest:
  nimOperator:
    nemotron_parse:
      enabled: true

# Configure ingestor-server to use Nemotron Parse
ingestor-server:
  envVars:
    APP_NVINGEST_PDFEXTRACTMETHOD: "nemotron_parse"
```

:::{note}
**Nemotron-parse-only:** To use only Nemotron Parse for PDF and table extraction (no other ingest NIMs), additionally set **`APP_NVINGEST_EXTRACTTABLESMETHOD: "nemotron_parse"`** in ingestor-server environment variables.
If you want only the Nemotron Parse NIM to run, also set **`COMPONENTS_TO_READY_CHECK`** to an empty string in **nv-ingest** env vars (`nv-ingest.envVars.COMPONENTS_TO_READY_CHECK: ""`), and optionally disable the other ingest NIMs as in step 2 below.
:::

**(Optional) Nemotron-parse-only pipeline** — If you want only the Nemotron Parse NIM to run (no OCR, page-elements, graphic-elements, or table-structure NIMs), do the following:

   - **Disable the other ingest NIMs** under `nv-ingest.nimOperator` in `values.yaml` by setting `enabled: false` for each. The NIM keys and structure are defined in [`deploy/helm/nvidia-blueprint-rag/values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) (see the `nv-ingest.nimOperator` section). For example:

   ```yaml
   nv-ingest:
     nimOperator:
       nemotron_parse:
         enabled: true
       nemoretriever_ocr_v1:
         enabled: false
       graphic_elements:
         enabled: false
       page_elements:
         enabled: false
       table_structure:
         enabled: false
   ```

   - **Set nv-ingest readiness so it does not wait for other NIMs** — add or set `COMPONENTS_TO_READY_CHECK` to an empty string for nv-ingest so the readiness probe passes when other NIMs are disabled:

   ```yaml
   nv-ingest:
     nimOperator:
       nemotron_parse:
         enabled: true
       # ... other NIMs set to enabled: false as above ...
     envVars:
       COMPONENTS_TO_READY_CHECK: ""
   ```

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

:::{note}
**Key Configuration Changes:**
- `nv-ingest.nimOperator.nemotron_parse.enabled=true` - Enables Nemotron Parse NIM
- `nv-ingest.envVars.COMPONENTS_TO_READY_CHECK=""` - For nemotron-parse-only: nv-ingest readiness passes without other NIMs
- `ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemotron_parse"` - Configures ingestor to use Nemotron Parse for PDF extraction
- `ingestor-server.envVars.APP_NVINGEST_EXTRACTTABLESMETHOD="nemotron_parse"` - Configures ingestor to use Nemotron Parse for table extraction (nemotron-parse-only pipeline)
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

**Table extraction method:** The `APP_NVINGEST_EXTRACTTABLESMETHOD` environment variable controls how tables are extracted. Set it to `nemotron_parse` to use Nemotron Parse for table extraction (recommended for a nemotron-parse-only pipeline). The default is `yolox`, which uses the YOLOX-based table NIMs.

:::{note}
The Nemotron Parse service requires GPU resources and must run on a dedicated GPU. Make sure you have sufficient GPU resources available before enabling this feature.
:::

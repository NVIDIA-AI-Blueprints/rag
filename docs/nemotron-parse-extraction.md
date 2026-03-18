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

Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable Nemotron Parse and configure the ingestor-server:

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

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

:::{note}
**Key Configuration Changes:**
- `nv-ingest.nimOperator.nemotron_parse.enabled=true` - Enables Nemotron Parse NIM
- `ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemotron_parse"` - Configures ingestor to use Nemotron Parse for PDF extraction
:::

## Experimental: Nemotron-parse-only extraction

:::{note}
The steps in this section describe a nemotron-parse-only pipeline. For production use, the default pipeline (Nemotron Parse with page-elements and table-structure NIMs) is recommended for better accuracy.
:::

The **default** Nemotron Parse pipeline uses the **page-elements** and **table-structure** NIMs together with the Nemotron Parse NIM in the extraction pipeline. This combination provides better accuracy for PDF and table extraction. 
To **experiment** with a nemotron-parse-only extraction pipeline (using only the Nemotron Parse NIM, without OCR, page-elements, graphic-elements, or table-structure NIMs), use the following steps.

### Key configuration

- **PDF extraction method** — Set `APP_NVINGEST_PDFEXTRACTMETHOD` to `nemotron_parse` so the ingestor uses Nemotron Parse for PDF text extraction.
- **Table extraction method** — Set `APP_NVINGEST_EXTRACTTABLESMETHOD` to `nemotron_parse` so the ingestor uses Nemotron Parse for table extraction instead of the default YOLOX-based table NIMs. This is required for a nemotron-parse-only pipeline.
- **nv-ingest health check** — Set `COMPONENTS_TO_READY_CHECK` to an empty string (`""`) in the **nv-ingest** service environment. By default, nv-ingest readiness waits for other ingest NIMs. With only Nemotron Parse running, the readiness probe would otherwise never pass. Emptying this value allows nv-ingest to become ready when only Nemotron Parse is available.

### Using Docker Compose (nemotron-parse-only)

#### On-prem models

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. Start only the Nemotron Parse service (and any other non-ingest services your setup needs):
   ```bash
   USERID=$(id -u) docker compose --profile rag --profile nemotron-parse -f deploy/compose/nims.yaml up -d
   ```
  You can skip the OCR, page-elements, graphic-elements, or table-structure NIMs if you want a nemotron-parse-only pipeline.

3. Configure the ingestor-server and nv-ingest for nemotron-parse-only. Set these environment variables:

   **Ingestor-server** (ingestor-server environment):
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse
   export APP_NVINGEST_EXTRACTTABLESMETHOD=nemotron_parse
   ```

   **nv-ingest** (nv-ingest service environment, e.g. in the compose file where nv-ingest runs):
   ```bash
   export COMPONENTS_TO_READY_CHECK=""
   ```
   This ensures the nv-ingest readiness probe passes when other ingest NIMs are not running.

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. Ingest PDFs using the [ingestion API usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb).

#### NVIDIA hosted API endpoints

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-nvidia-hosted.md) up to and including the step labelled "Start the vector db containers from the repo root."

2. Export variables for the Nemotron Parse API:
   ```bash
   export NEMOTRON_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
   export NEMOTRON_PARSE_MODEL_NAME=nvidia/nemotron-parse
   export NEMOTRON_PARSE_INFER_PROTOCOL=http
   ```

3. Configure the ingestor-server and nv-ingest for nemotron-parse-only:

   **Ingestor-server**:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse
   export APP_NVINGEST_EXTRACTTABLESMETHOD=nemotron_parse
   ```

   **nv-ingest** (so readiness passes without other NIMs):
   ```bash
   export COMPONENTS_TO_READY_CHECK=""
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. Ingest PDFs using the [ingestion API usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/ingestion_api_usage.ipynb).

:::{note}
When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).
:::

### Using Helm (nemotron-parse-only)

To run only Nemotron Parse for PDF and table extraction with Helm:

1. **Prerequisites**: Ensure you have sufficient GPU resources. Nemotron Parse requires a dedicated GPU.

2. Edit [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml):

   - **Enable Nemotron Parse** and **disable the other ingest NIMs** under `nv-ingest.nimOperator`:

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
     envVars:
       COMPONENTS_TO_READY_CHECK: ""
   ```

   - **Configure the ingestor-server** to use Nemotron Parse for both PDF and table extraction:

   ```yaml
   ingestor-server:
     envVars:
       APP_NVINGEST_PDFEXTRACTMETHOD: "nemotron_parse"
       APP_NVINGEST_EXTRACTTABLESMETHOD: "nemotron_parse"
   ```

3. Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

4. For full Helm deployment steps, see the [Helm Deployment Guide](deploy-helm.md).

**Summary of nemotron-parse-only Helm settings:**

| Setting | Purpose |
|---------|---------|
| `nv-ingest.nimOperator.nemotron_parse.enabled: true` | Enable the Nemotron Parse NIM. |
| `nv-ingest.nimOperator.<other_nims>.enabled: false` | Disable OCR, page-elements, graphic-elements, and table-structure NIMs. |
| `nv-ingest.envVars.COMPONENTS_TO_READY_CHECK: ""` | nv-ingest health check: readiness passes without other NIMs. |
| `ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD: "nemotron_parse"` | Use Nemotron Parse for PDF extraction. |
| `ingestor-server.envVars.APP_NVINGEST_EXTRACTTABLESMETHOD: "nemotron_parse"` | Use Nemotron Parse for table extraction. |

## Limitations and Requirements

When using Nemotron Parse for PDF extraction, consider the following:

- Nemotron Parse only supports PDF format documents, not image files. Attempting to process non-PDF files will lead them to be extracted using the default extraction method.
- The service requires GPU resources and must run on a dedicated GPU. Make sure you have sufficient GPU resources available before enabling this feature.
- The extraction quality may vary depending on the PDF structure and content.
- Nemotron Parse is not supported on NVIDIA B200 GPUs or RTX Pro 6000 GPUs.

For detailed information about hardware requirements and supported GPUs for extraction NIMs used by NeMo Retriever Library, refer to the [Nemotron Parse Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-parse).

## Available PDF Extraction Methods

The `APP_NVINGEST_PDFEXTRACTMETHOD` environment variable supports the following values:

- `nemotron_parse`: Uses the Nemotron Parse service for enhanced PDF extraction (recommended for scanned documents or documents with complex layouts)
- `pdfium`: Uses the default PDFium-based extraction
- `None`: Uses the default extraction method

**Table extraction method:** The `APP_NVINGEST_EXTRACTTABLESMETHOD` environment variable controls how tables are extracted. Set it to `nemotron_parse` to use Nemotron Parse for table extraction (recommended for a nemotron-parse-only pipeline). The default is `yolox`, which uses the YOLOX-based table NIMs.

:::{note}
The Nemotron Parse service requires GPU resources and must run on a dedicated GPU. Make sure you have sufficient GPU resources available before enabling this feature.
:::

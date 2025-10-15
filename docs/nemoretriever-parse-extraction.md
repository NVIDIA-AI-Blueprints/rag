<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable PDF extraction with Nemoretriever Parse for NVIDIA RAG Blueprint

For enhanced PDF extraction capabilities, you can use the Nemoretriever Parse service with the [NVIDIA RAG Blueprint](readme.md). This service provides improved PDF parsing and structure understanding compared to the default PDF extraction method.

> [!WARNING]
>
> B200 GPUs are not supported for PDF extraction with Nemoretriever Parse.
> For this feature, use H100 or A100 GPUs instead.



## Using Docker Compose

### Using On-Prem Models

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. Deploy the Nemoretriever Parse service along with other required NIMs:
   ```bash
   USERID=$(id -u) docker compose --profile rag --profile nemoretriever-parse -f deploy/compose/nims.yaml up -d
   ```

3. Configure the ingestor-server to use Nemoretriever Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemoretriever_parse
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

### Using NVIDIA Hosted API Endpoints

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-nvidia-hosted.md) up to and including the step labelled "Start the vector db containers from the repo root."


2. Export the following variables to use nemoretriever parse API endpoints:

   ```bash
   export NEMORETRIEVER_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
   export NEMORETRIEVER_PARSE_MODEL_NAME=nvidia/nemoretriever-parse
   export NEMORETRIEVER_PARSE_INFER_PROTOCOL=http
   ```

3. Configure the ingestor-server to use Nemoretriever Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemoretriever_parse
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the deployment guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

> [!Note]
> When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).

## Using Helm

To enable PDF extraction with Nemoretriever Parse using Helm, we need to enable the Nemoretriever Parse service `nv-ingest.nim-vlm-text-extraction.deployed=true` and update the PDF extract method `ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemoretriever_parse"`.

Update the deployment to enable Nemoretriever Parse with the following command.

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set nv-ingest.nim-vlm-text-extraction.deployed=true \
  --set ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemoretriever_parse"
```

## Limitations and Requirements

When using Nemoretriever Parse for PDF extraction, consider the following:

- Nemoretriever Parse only supports PDF format documents. Attempting to process non-PDF files will result in extraction errors.
- The service requires GPU resources. Make sure you have sufficient GPU resources available before enabling this feature.
- The extraction quality may vary depending on the PDF structure and content.
- Nemoretriever Parse is currently not supported on NVIDIA B200 GPUs.

For detailed information about hardware requirements and supported GPUs for all NeMo Retriever extraction NIMs, refer to the [NeMo Retriever Extraction Support Matrix](https://docs.nvidia.com/nemo/retriever/extraction/support-matrix/).

## Available PDF Extraction Methods

The `APP_NVINGEST_PDFEXTRACTMETHOD` environment variable supports the following values:

- `nemoretriever_parse`: Uses the Nemoretriever Parse service for enhanced PDF extraction
- `pdfium`: Uses the default PDFium-based extraction
- `None`: Uses the default extraction method

> [!Note]
> The Nemoretriever Parse service requires GPU resources. Make sure you have sufficient GPU resources available before enabling this feature.

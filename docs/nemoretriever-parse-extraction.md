<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable PDF extraction with Nemoretriever Parse

For enhanced PDF extraction capabilities, you can use the Nemoretriever Parse service. This service provides improved PDF parsing and structure understanding compared to the default PDF extraction method.

## Using Docker Compose

### Using On-Prem Models

1. Follow steps outlined in the [quickstart guide](quickstart.md#start-using-on-prem-models) till step 4. Deploy all the deployed NIMs for ingestion.

2. Deploy the Nemoretriever Parse service along with other required NIMs:
   ```bash
   USERID=$(id -u) docker compose --profile rag --profile nemoretriever-parse -f deploy/compose/nims.yaml up -d
   ```

3. Configure the ingestor-server to use Nemoretriever Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemoretriever_parse
   ```

4. Deploy the ingestion-server and rag-server containers following the remaining steps in the quickstart guide.

5. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

### Using NVIDIA Hosted API Endpoints

1. Follow steps outlined in the [quickstart guide](quickstart.md#start-using-nvidia-hosted-models) till step 2. Export the following variables to use nemoretriever parse API endpoints:

   ```bash
   export NEMORETRIEVER_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
   export NEMORETRIEVER_PARSE_MODEL_NAME=nvidia/nemoretriever-parse
   export NEMORETRIEVER_PARSE_INFER_PROTOCOL=http
   ```

2. Configure the ingestor-server to use Nemoretriever Parse by setting the environment variable:
   ```bash
   export APP_NVINGEST_PDFEXTRACTMETHOD=nemoretriever_parse
   ```

3. Deploy the ingestion-server and rag-server containers following the remaining steps in the quickstart guide.

4. You can now ingest PDF files using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

> [!Note]
> When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).

## Using Helm

To enable PDF extraction with Nemoretriever Parse using Helm, you need to enable the Nemoretriever Parse service along with other required services:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.2.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set nim-llm.enabled=true \
  --set nvidia-nim-llama-32-nv-embedqa-1b-v2.enabled=true \
  --set text-reranking-nim.enabled=true \
  --set ingestor-server.enabled=true \
  --set ingestor-server.nv-ingest.nemoretriever-page-elements-v2.deployed=true \
  --set ingestor-server.nv-ingest.nemoretriever-graphic-elements-v1.deployed=true \
  --set ingestor-server.nv-ingest.nemoretriever-table-structure-v1.deployed=true \
  --set ingestor-server.nv-ingest.paddleocr-nim.deployed=true \
  --set ingestor-server.nv-ingest.nim-vlm-text-extraction.deployed=true \
  --set ingestor-server.envVars.APP_NVINGEST_PDFEXTRACTMETHOD="nemoretriever_parse" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

## Limitations and Requirements

When using Nemoretriever Parse for PDF extraction, please note the following:

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

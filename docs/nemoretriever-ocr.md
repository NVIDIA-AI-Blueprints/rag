<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# NeMo Retriever OCR Configuration Guide for NVIDIA RAG Blueprint (Early Access)

You can enable NeMO Retriever OCR for your [NVIDIA RAG Blueprint](readme.md). NeMo Retriever OCR is an advanced optical character recognition service that provides enhanced text extraction capabilities for document processing workflows. It serves as a high-performance alternative to the default Paddle OCR service, offering significant improvements in speed and resource efficiency.

For more information about NeMo Retriever OCR, refer to [NeMo Retriever OCR v1 Container](https://build.nvidia.com/nvidia/nemoretriever-ocr-v1).

> [!Note]
> **Early Access**: Currently, the NeMo Retriever OCR v1 container is in early access preview.


## Key Benefits of NeMo Retriever OCR

- **Performance**: Nemo-retriever OCR based pipeline is more than 2x faster than Paddle OCR for PDF ingestion tasks. NeMo Retriever OCR can be used for High-volume document processing in batch ingestion workflows.

## Considerations for NeMo Retriever OCR

- **GPU Requirements**: Check the [NeMo Retriever OCR Support Matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html) for detailed hardware requirements and supported GPUs
  - **GPU Memory**: Requires approximately **~4.1GB of GPU memory** [Triton server (~2.3GB) + Python backend processes(~1.8GB)] - ensure sufficient GPU memory is available
- **Quality Variance**: Extraction quality may vary based on image quality and text complexity
- **Early Access**: Currently in preview - monitor for updates and stability improvements


## How to Enable NeMo Retriever OCR

### Docker Compose Deployment for NeMo Retriever OCR

#### Self-Hosted Deployment Configuration

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. **Configure Environment Variables**:
   ```bash
   export OCR_GRPC_ENDPOINT=nemoretriever-ocr:8001
   export OCR_HTTP_ENDPOINT=http://nemoretriever-ocr:8000/v1/infer
   export OCR_INFER_PROTOCOL=grpc
   export OCR_MODEL_NAME=scene_text_ensemble
   ```

   > [!Warning]
   > **Critical Health Check Requirement**: Even when using gRPC protocol (`OCR_INFER_PROTOCOL=grpc`), you must also export the `OCR_HTTP_ENDPOINT` because the health check from nv-ingest uses HTTP.

3. **Stop Paddle OCR deployment if already running**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml down paddle
   ```

4. **Deploy NeMo Retriever OCR Service**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile nemoretriever-ocr up -d
   ```

5. **Verify Service Status**:
   ```bash
   watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
   ```

6. **Restart Ingestor Server**:
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

7. **Test Document Ingestion**: Use the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb) to verify functionality.

#### NVIDIA-Hosted Deployment Configuration

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-nvidia-hosted.md) up to and including the step labelled "Start the vector db containers from the repo root."

2. **Configure API Endpoints**:
   ```bash
   export OCR_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr
   export OCR_INFER_PROTOCOL=http
   export OCR_MODEL_NAME=scene_text_ensemble
   ```

3. **Deploy Services**: Continue with the remaining steps in the deployment guide to deploy ingestion-server and rag-server containers.

4. **Test Document Ingestion**: Use the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb) to verify functionality.

> [!Note]
> **Default Behavior**: Paddle OCR is the default OCR service and runs automatically when you start the NIMs. To use NeMo Retriever OCR instead, you must explicitly start it with the `--profile nemoretriever-ocr` flag.

### Helm Deployment to Enable NeMo Retriever OCR

To enable NeMo Retriever OCR using Helm, we need to set the `nv-ingest.nemoretriever-ocr.deployed=true` and `nv-ingest.envVars.OCR_MODEL_NAME="scene_text_ensemble"`.

Update the deployment to enable NeMo Retriever OCR and disable Paddle OCR for resource optimization with the following command.

```bash
# Apply to a fresh deployment (recommended to uninstall existing deployments first)
# helm uninstall rag -n rag
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set nv-ingest.paddleocr-nim.deployed=false \
  --set nv-ingest.nemoretriever-ocr.deployed=true \
  --set nv-ingest.envVars.OCR_MODEL_NAME="scene_text_ensemble" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

## NeMo Retriever OCR Configuration Options

### Required Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OCR_GRPC_ENDPOINT` | gRPC endpoint for OCR service | `nemoretriever-ocr:8001` | Yes (on-premises) |
| `OCR_HTTP_ENDPOINT` | HTTP endpoint for OCR service | `http://nemoretriever-ocr:8000/v1/infer` | Yes |
| `OCR_INFER_PROTOCOL` | Communication protocol | `grpc` | Yes |
| `OCR_MODEL_NAME` | OCR model to use | `scene_text_ensemble` | Yes |

### Available OCR Service Options

The system supports two OCR service options:

- **NeMo Retriever OCR**: Enhanced text extraction optimized for document processing
- **Paddle OCR**: Default OCR service for general text extraction

### Hardware Requirements and Support Matrix

For detailed information about hardware requirements and supported GPUs, refer to the [NeMo Retriever OCR Support Matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html).

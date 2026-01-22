<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# OCR Configuration Guide for NVIDIA RAG Blueprint

This guide explains the OCR (Optical Character Recognition) services available in the [NVIDIA RAG Blueprint](readme.md), including configuration and deployment options.


## Overview

The NVIDIA RAG Blueprint supports two OCR services:

1. **NeMo Retriever OCR** (Default) - High-performance OCR service offering 2x+ faster performance
2. **Paddle OCR** (Legacy) - General-purpose OCR service maintained for compatibility

:::{tip}
**NeMo Retriever OCR is now the default OCR service** and is recommended for all new deployments due to its superior performance and efficiency.
:::


## NeMo Retriever OCR (Default)

NeMo Retriever OCR is the default and recommended OCR service for the NVIDIA RAG Blueprint, providing:

- **2x+ faster performance** compared to Paddle OCR
- Optimized text extraction from documents and images
- Enhanced accuracy for modern document layouts
- Better resource efficiency


### Key Features

- High-performance optical character recognition
- Support for various document types and image formats
- GPU-accelerated processing
- Production-ready with model control capabilities

### Default Configuration

By default, the NVIDIA RAG Blueprint is configured to use NeMo Retriever OCR with the following settings:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `OCR_GRPC_ENDPOINT` | `nemoretriever-ocr:8001` | gRPC endpoint for OCR service |
| `OCR_HTTP_ENDPOINT` | `http://nemoretriever-ocr:8000/v1/infer` | HTTP endpoint for OCR service |
| `OCR_INFER_PROTOCOL` | `grpc` | Communication protocol (grpc or http) |
| `OCR_MODEL_NAME` | `scene_text_ensemble` | OCR model to use |

### Hardware Requirements

For detailed hardware requirements and GPU support, refer to the [NeMo Retriever OCR Support Matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/1.2.0/support-matrix.html).

### Docker Configuration

The NeMo Retriever OCR service is configured in the Docker Compose file with the following key settings:

- **Image**: `nvcr.io/nim/nvidia/nemoretriever-ocr-v1:1.2.0`
- **GPU Memory**: 8192 MB (default)
- **Max Batch Size**: 32 (default)
- **Ports**: 8012 (HTTP), 8013 (gRPC), 8014 (Metrics)

You can customize the GPU allocation by setting:

```bash
export OCR_MS_GPU_ID=0  # Specify which GPU to use
export OCR_CUDA_MEMORY_POOL_MB=8192  # Adjust memory allocation
export OCR_BATCH_SIZE=32  # Configure batch size
export OCR_OMP_NUM_THREADS=8  # Set OpenMP threads
```


## Paddle OCR (Legacy)

Paddle OCR is maintained as a legacy option for compatibility with existing workflows. While still functional, it is recommended to migrate to NeMo Retriever OCR for better performance.

### When to Use Paddle OCR

Consider using Paddle OCR if you:
- Need compatibility with existing Paddle OCR-based workflows
- Have specific requirements that work better with Paddle OCR
- Are migrating from an older deployment

### Hardware Requirements

For detailed hardware requirements, refer to the [Paddle OCR Support Matrix](https://docs.nvidia.com/nim/ingestion/table-extraction/latest/support-matrix.html#supported-hardware).

### Docker Configuration

The Paddle OCR service configuration:

- **Image**: `nvcr.io/nim/baidu/paddleocr:1.5.0`
- **GPU Memory**: 3072 MB (default)
- **Ports**: 8009 (HTTP), 8010 (gRPC), 8011 (Metrics)

:::{note}
**Legacy Service**: Paddle OCR is maintained as a legacy option. For new deployments, we recommend using the default NeMo Retriever OCR service for better performance.
:::


## Deployment Instructions

### Docker Compose Deployment

#### Using NeMo Retriever OCR (Default)

NeMo Retriever OCR is deployed by default when you follow the standard deployment guide. No additional configuration is required.

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) for standard setup.

2. **Start Services**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d
   ```

   :::{tip}
   NeMo Retriever OCR is included in the default profile and will start automatically.
   :::

3. **Verify Service Status**:
   ```bash
   watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
   ```

#### Switching to Paddle OCR

If you need to use Paddle OCR instead:

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-self-hosted.md) up to and including the step labelled "Start all required NIMs."

2. **Configure Environment Variables**:
   ```bash
   export OCR_GRPC_ENDPOINT=paddle:8001
   export OCR_HTTP_ENDPOINT=http://paddle:8000/v1/infer
   export OCR_INFER_PROTOCOL=grpc
   export OCR_MODEL_NAME=paddle
   ```

3. **Stop NeMo Retriever OCR if running**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml down nemoretriever-ocr
   ```

4. **Deploy Paddle OCR Service**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile paddle up -d
   ```

5. **Restart Ingestor Server and NV-Ingest Runtime**:
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

6. **Test Document Ingestion**: Use the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb) to verify functionality.


### NVIDIA-Hosted Deployment

#### Using NeMo Retriever OCR (Default)

Follow the standard [NVIDIA-hosted deployment guide](deploy-docker-nvidia-hosted.md) - NeMo Retriever OCR is the default configuration.

#### Using Paddle OCR with NVIDIA-Hosted Deployment

1. **Prerequisites**: Follow the [deployment guide](deploy-docker-nvidia-hosted.md) up to and including the step labelled "Start the vector db containers from the repo root."

2. **Configure API Endpoints**:
   ```bash
   export OCR_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/baidu/paddleocr
   export OCR_INFER_PROTOCOL=http
   export OCR_MODEL_NAME=paddle
   ```

3. **Deploy Services**: Continue with the remaining steps in the deployment guide to deploy ingestion-server and rag-server containers.

4. **Test Document Ingestion**: Use the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb) to verify functionality.


### Helm Deployment

#### Using NeMo Retriever OCR (Default)

NeMo Retriever OCR is deployed by default with Helm installations:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

#### Using Paddle OCR with Helm

To use Paddle OCR instead of the default NeMo Retriever OCR:

```bash
# Apply to a fresh deployment (recommended to uninstall existing deployments first)
# helm uninstall rag -n rag
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc2.1.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set nv-ingest.paddleocr-nim.deployed=true \
  --set nv-ingest.nemoretriever-ocr.deployed=false \
  --set nv-ingest.envVars.OCR_MODEL_NAME="paddle" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```


## OCR Configuration Reference

### Environment Variables

| Variable | Description | NeMo Retriever Default | Paddle Default | Required |
|----------|-------------|------------------------|----------------|----------|
| `OCR_GRPC_ENDPOINT` | gRPC endpoint for OCR service | `nemoretriever-ocr:8001` | `paddle:8001` | Yes (on-premises) |
| `OCR_HTTP_ENDPOINT` | HTTP endpoint for OCR service | `http://nemoretriever-ocr:8000/v1/infer` | `http://paddle:8000/v1/infer` | Yes |
| `OCR_INFER_PROTOCOL` | Communication protocol | `grpc` | `grpc` | Yes |
| `OCR_MODEL_NAME` | OCR model to use | `scene_text_ensemble` | `paddle` | Yes |
| `OCR_MS_GPU_ID` | GPU device ID to use | `0` | `0` | No |
| `OCR_CUDA_MEMORY_POOL_MB` | CUDA memory pool size | `8192` | `3072` | No |
| `OCR_BATCH_SIZE` | Max batch size (NeMo only) | `32` | N/A | No |
| `OCR_OMP_NUM_THREADS` | OpenMP thread count | `8` | `8` | No |

### Advanced Configuration

For deployments with NIMs on different workstations or outside the nvidia-rag docker network:

```bash
export OCR_GRPC_ENDPOINT="workstation_ip:8001"
```

Replace `workstation_ip` with the actual IP address of the machine running the OCR service.


## Switching Between OCR Services

### Migrating from Paddle OCR to NeMo Retriever OCR

To switch to the default NeMo Retriever OCR service:

1. **Stop Paddle OCR**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml down paddle
   ```

2. **Configure NeMo Retriever OCR environment variables**:
   ```bash
   export OCR_GRPC_ENDPOINT=nemoretriever-ocr:8001
   export OCR_HTTP_ENDPOINT=http://nemoretriever-ocr:8000/v1/infer
   export OCR_INFER_PROTOCOL=grpc
   export OCR_MODEL_NAME=scene_text_ensemble
   ```

3. **Start NeMo Retriever OCR**:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d nemoretriever-ocr
   ```

4. **Restart Ingestor Server**:
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

### Migrating from NeMo Retriever OCR to Paddle OCR

Follow the steps in [Switching to Paddle OCR](#switching-to-paddle-ocr) above.


## Performance Comparison

| Feature | NeMo Retriever OCR | Paddle OCR |
|---------|-------------------|------------|
| **Performance** | 2x+ faster | Baseline |
| **GPU Memory** | 8 GB (default) | 3 GB (default) |
| **Batch Processing** | Up to 32 | Limited |
| **Status** | **Recommended (Default)** | Legacy |
| **Use Case** | All new deployments | Legacy compatibility |


## Troubleshooting

### Common Issues

1. **OCR Service Not Starting**
   - Check GPU availability: `nvidia-smi`
   - Verify NGC API key is set correctly
   - Check logs: `docker logs <container-name>`

2. **Connection Errors**
   - Verify the OCR endpoint variables are set correctly
   - Ensure the OCR service is running: `docker ps`
   - Check network connectivity between services

3. **Performance Issues**
   - Consider increasing `OCR_CUDA_MEMORY_POOL_MB`
   - Adjust `OCR_BATCH_SIZE` for NeMo Retriever OCR
   - Verify GPU has sufficient memory

### Getting Logs

```bash
# NeMo Retriever OCR logs
docker logs nemoretriever-ocr

# Paddle OCR logs
docker logs paddle
```


## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Support Matrix](support-matrix.md)
- [Troubleshoot](troubleshooting.md)
- [Ingestion API Usage Notebook](../notebooks/ingestion_api_usage.ipynb)


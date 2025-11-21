<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Milvus Configuration for NVIDIA RAG Blueprint

You can configure how Milvus works with your [NVIDIA RAG Blueprint](readme.md).


## GPU to CPU Mode Switch

Milvus uses GPU acceleration by default for vector operations. Switch to CPU mode if you encounter:
- GPU memory constraints
- Development without GPU support

## Docker compose

### Configuration Steps

#### 1. Update Docker Compose Configuration (vectordb.yaml)

First, you need to modify the `deploy/compose/vectordb.yaml` file to disable GPU usage:

**Step 1: Comment Out GPU Reservations**
Comment out the entire deploy section that reserves GPU resources:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           capabilities: ["gpu"]
#           # count: ${INFERENCE_GPU_COUNT:-all}
#           device_ids: ['${VECTORSTORE_GPU_DEVICE_ID:-0}']
```

**Step 2: Change the Milvus Docker Image**
```yaml
# Change this line:
image: milvusdb/milvus:v2.6.2-gpu # milvusdb/milvus:v2.6.2 for CPU

# To this:
image: milvusdb/milvus:v2.6.2 # milvusdb/milvus:v2.6.2-gpu for GPU
```

#### 2. Set Environment Variables

Before starting any services, you must set these environment variables in your terminal. These variables tell the ingestor server to use CPU mode:

```bash
# Set these environment variables BEFORE starting the ingestor server
export APP_VECTORSTORE_ENABLEGPUSEARCH=False
export APP_VECTORSTORE_ENABLEGPUINDEX=False
```

#### 3. Restart Services

After making the configuration changes and setting environment variables, restart the services:

```bash
# 1. Stop existing services
docker compose -f deploy/compose/vectordb.yaml down

# 2. Start Milvus and dependencies
docker compose -f deploy/compose/vectordb.yaml up -d

# 3. Now start the ingestor server
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

## Switching Milvus to CPU Mode using Helm

To configure Milvus to run in CPU mode when deploying with Helm:

1. Disable GPU search and indexing by editing [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml).

    A. In the `envVars` and `ingestor-server.envVars` sections, set the following environment variables:

        ```yaml
        envVars:
        APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
        ingestor-server:
        envVars:
            APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
            APP_VECTORSTORE_ENABLEGPUINDEX: "False"
        ```

    B. Also, change the image under `milvus.image.all` to remove the `-gpu` tag.

        ```yaml
        milvus:
        image:
            all:
            repository: milvusdb/milvus
            tag: v2.5.17  # instead of v2.5.17-gpu
        ```

    C. (Optional) Remove or set GPU resource requests/limits to zero in the `milvus.standalone.resources` block.

        ```yaml
        milvus:
        standalone:
            resources:
            limits:
                nvidia.com/gpu: 0
        ```

2. After you modify values.yaml, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

## GPU Indexing with CPU Search

This mode uses the GPU to build indexes during ingestion while serving search on the CPU. It is useful when you want fast index construction but prefer CPU-based query serving for cost, capacity, or scheduling reasons.

For general GPU↔CPU switching instructions, see the [GPU to CPU Mode Switch](#gpu-to-cpu-mode-switch) section above.

### Environment Variables

Set the following before starting the ingestor server:

```bash
export APP_VECTORSTORE_ENABLEGPUSEARCH=False
export APP_VECTORSTORE_ENABLEGPUINDEX=True
```

With `APP_VECTORSTORE_ENABLEGPUSEARCH=False`, the client enables `adapt_for_cpu=true` automatically. `adapt_for_cpu` decides whether to use GPU for index-building and CPU for search. When this parameter is true, search requests must include the `ef` parameter.

### Docker Compose notes

- Keep Milvus running with a GPU-capable image if you want GPU index-building (for example: `milvusdb/milvus:v2.6.2-gpu`).
- Set the environment variables above before starting the ingestor server.
- For inference (search and generate) in `rag-server`, you can use either the GPU or CPU Docker image. Search will run on CPU for the Milvus collection built with GPU indexing when `APP_VECTORSTORE_ENABLEGPUSEARCH=False`.

Example sequence:

```bash
# Start/ensure Milvus is up (GPU image if you want GPU indexing)
docker compose -f deploy/compose/vectordb.yaml up -d

# Set env vars and start the ingestor (GPU indexing + CPU search)
export APP_VECTORSTORE_ENABLEGPUSEARCH=False
export APP_VECTORSTORE_ENABLEGPUINDEX=True
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d

# Start rag-server (either Milvus CPU or GPU image is fine)
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Helm notes

Set the environment variables in `values.yaml`:

```yaml
envVars:
  APP_VECTORSTORE_ENABLESEARCH: "True"
ingestor-server:
  envVars:
    APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
    APP_VECTORSTORE_ENABLEGPUINDEX: "True"
```

If you require GPU index-building, ensure the Milvus image variant supports GPU (for example, keep a `-gpu` tag where applicable). `rag-server` can be deployed with either CPU or GPU images for inference; search will be served on CPU for collections indexed with GPU when `APP_VECTORSTORE_ENABLEGPUSEARCH` is set to `False`.

Note: When `adapt_for_cpu` is in effect, your search requests must supply an `ef` parameter.


## (Optional) Customize the Milvus Endpoint

To use a custom Milvus endpoint, use the following procedure.

1. Update the `APP_VECTORSTORE_URL` and `MINIO_ENDPOINT` variables in both the RAG server and the ingestor server sections in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml). Your changes should look similar to the following.

   ```yaml
   env:
     # ... existing code ...
     APP_VECTORSTORE_URL: "http://your-custom-milvus-endpoint:19530"
     MINIO_ENDPOINT: "http://your-custom-minio-endpoint:9000"
     # ... existing code ...

   ingestor-server:
     env:
       # ... existing code ...
       APP_VECTORSTORE_URL: "http://your-custom-milvus-endpoint:19530"
       MINIO_ENDPOINT: "http://your-custom-minio-endpoint:9000"
       # ... existing code ...

   nv-ingest:
     envVars:
       # ... existing code ...
       MINIO_INTERNAL_ADDRESS: "http://your-custom-minio-endpoint:9000"
       # ... existing code ...
   ```

2. Disable the Milvus deployment. Set `milvusDeployed: false` in the `nv-ingest.milvusDeployed` section to prevent deploying the default Milvus instance. Your changes should look like the following.

   ```yaml
    nv-ingest:
      # ... existing code ...
      milvusDeployed: false
      # ... existing code ...
   ```

3. Redeploy the Helm chart by running the following code.

   ```sh
   helm upgrade rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz -f nvidia-blueprint-rag/values.yaml -n rag
   ```

## Troubleshooting

### GPU_CAGRA Error

If you encounter GPU_CAGRA errors that cannot be resolved by when switching to CPU mode, try the following:

1. Stop all running services:
   ```bash
   docker compose -f deploy/compose/vectordb.yaml down
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
   ```

2. Delete the Milvus volumes directory:
   ```bash
   rm -rf deploy/compose/volumes
   ```

3. Restart the services:
   ```bash
   docker compose -f deploy/compose/vectordb.yaml up -d
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

[!NOTE]
This will delete all existing vector data, so ensure you have backups if needed.



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

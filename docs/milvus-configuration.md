<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Milvus Configuration

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
image: milvusdb/milvus:v2.5.3-gpu # milvusdb/milvus:v2.5.3 for CPU

# To this:
image: milvusdb/milvus:v2.5.3 # milvusdb/milvus:v2.5.3-gpu for GPU
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

1. Edit values.yaml

Under the ingestor-server.envVars section, set the following environment variables:

```yaml
ingestor-server:
  envVars:
    APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
    APP_VECTORSTORE_ENABLEGPUINDEX: "False"
```

Also, change the Milvus image under milvus.image.all to remove the GPU tag:

```yaml
milvus:
  image:
    all:
      repository: milvusdb/milvus
      tag: v2.5.3  # instead of v2.5.3-gpu
```

Optionally, remove or set GPU resource requests/limits to zero in the milvus.standalone.resources block:

```yaml
milvus:
  standalone:
    resources:
      limits:
        nvidia.com/gpu: 0
```

2. Redeploy with Helm
   
After modifying values.yaml, apply the changes with:

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.2.0.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f rag-server/values.yaml
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


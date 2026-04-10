<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Milvus Configuration for NVIDIA RAG Blueprint

:::{note}
Milvus is an optional vector database for the NVIDIA RAG Blueprint. The default VDB is Elasticsearch. Use this guide if you want to switch to Milvus, or if you already use Milvus and need to tune GPU/CPU behavior, endpoints, authentication, or runtime API tokens. For enabling Milvus and wiring `APP_VECTORSTORE_*`, start with the **Switching to Milvus** section in [Vector database configuration](change-vectordb.md#switching-to-milvus).
:::

This document describes **optional Milvus-specific** settings. It does not replace the default Elasticsearch path—see [Vector database configuration](change-vectordb.md) for the standard vector database and for switching between backends.


## GPU to CPU Mode Switch

When Milvus is running, it uses GPU acceleration by default for vector operations. Switch to CPU mode if you encounter:
- GPU memory constraints
- Development without GPU support

## Docker compose

The commands below use the `milvus` Compose profile so the Milvus, etcd, and MinIO services start. Ensure `APP_VECTORSTORE_NAME` and `APP_VECTORSTORE_URL` target Milvus if you have not already switched from Elasticsearch ([Vector database configuration](change-vectordb.md)).

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
image: milvusdb/milvus:v2.6.5-gpu # milvusdb/milvus:v2.6.5-gpu for GPU

# To this:
image: milvusdb/milvus:v2.6.5 # milvusdb/milvus:v2.6.5 for CPU
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
# 1. Stop existing services (Milvus profile)
docker compose -f deploy/compose/vectordb.yaml --profile milvus down

# 2. Start Milvus and dependencies
docker compose -f deploy/compose/vectordb.yaml --profile milvus up -d

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
        nv-ingest:
          milvus:
            image:
              all:
                repository: docker.io/milvusdb/milvus
                tag: v2.6.5  # instead of v2.6.5-gpu
        ```

    C. (Optional) Remove or set GPU resource requests/limits to zero in the `milvus.standalone.resources` block.

        ```yaml
        nv-ingest:
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

Set the following before starting the ingestor and rag server:

```bash
export APP_VECTORSTORE_ENABLEGPUSEARCH=False
export APP_VECTORSTORE_ENABLEGPUINDEX=True
```

With `APP_VECTORSTORE_ENABLEGPUSEARCH=False`, the client enables `adapt_for_cpu=true` automatically. `adapt_for_cpu` decides whether to use GPU for index-building and CPU for search. When this parameter is true, search requests must include the `ef` parameter.

### Docker Compose notes

- Keep Milvus running with a GPU-capable image if you want GPU index-building (for example: `milvusdb/milvus:v2.6.5-gpu`).
- Set the environment variables above before starting the ingestor server.
- For inference (search and generate) in `rag-server`, you can use either the GPU or CPU Docker image. Search will run on CPU for the Milvus collection built with GPU indexing when `APP_VECTORSTORE_ENABLEGPUSEARCH=False`.

Example sequence:

```bash
# Start/ensure Milvus is up (GPU image if you want GPU indexing)
docker compose -f deploy/compose/vectordb.yaml --profile milvus up -d

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
  APP_VECTORSTORE_ENABLEGPUSEARCH: "False"

ingestor-server:
  envVars:
    APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
    APP_VECTORSTORE_ENABLEGPUINDEX: "True"
```

If you require GPU index-building, ensure the Milvus image variant supports GPU (for example, keep the `-gpu` tag in the Milvus image). The configuration should look like this:

```yaml
nv-ingest:
  milvus:
    image:
      all:
        repository: docker.io/milvusdb/milvus
        tag: v2.6.5-gpu  # GPU-enabled image for GPU indexing
```

`rag-server` can be deployed with either CPU or GPU images for inference; search will be served on CPU for collections indexed with GPU when `APP_VECTORSTORE_ENABLEGPUSEARCH` is set to `False`.

:::{note}
When `adapt_for_cpu` is in effect, your search requests must supply an `ef` parameter.
:::


## (Optional) Customize the Milvus Endpoint

Use this procedure when the RAG stack should talk to a Milvus instance you operate separately (outside the chart’s Milvus subchart), for example a shared cluster or a different namespace.

1. Update the `APP_VECTORSTORE_URL` and `MINIO_ENDPOINT` variables in both the RAG server and the ingestor server sections in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml). Your changes should look similar to the following.

   ```yaml
   envVars:
     # ... existing code ...
     APP_VECTORSTORE_URL: "http://your-custom-milvus-endpoint:19530"
     MINIO_ENDPOINT: "http://your-custom-minio-endpoint:9000"
     # ... existing code ...

   ingestor-server:
     envVars:
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

2. Turn off the in-chart Milvus deployment so Helm does not create a second Milvus alongside your external endpoint. Set `nv-ingest.milvusDeployed` to `false`:

   ```yaml
    nv-ingest:
      # ... existing code ...
      milvusDeployed: false
      # ... existing code ...
   ```

3. After you modify values.yaml, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).


## Milvus Authentication

Enable authentication for Milvus to secure the Milvus deployment you use with the blueprint (only applicable when Milvus is your configured vector database).

### Docker Compose

#### 1. Configure Milvus Authentication

Download the Milvus configuration file from the upstream release (matching the version used in `deploy/compose/vectordb.yaml`):
```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/v2.6.5/configs/milvus.yaml -O deploy/compose/milvus.yaml
```

Edit the downloaded `deploy/compose/milvus.yaml` to enable authentication:
```yaml
common:
  # ... existing milvus config ...
  security:
    authorizationEnabled: true
    defaultRootPassword: "your-secure-password"
```

Mount the configuration file in `deploy/compose/vectordb.yaml` by uncommenting the volume mount:
```yaml
volumes:
  - ${MILVUS_CONFIG_FILE:-./milvus.yaml}:/milvus/configs/milvus.yaml
```

(Optional) For more details on configuring Milvus with Docker, refer to the [Milvus Docker configuration](https://milvus.io/docs/configure-docker.md).

#### 2. Start Services

Start Milvus with authentication:
```bash
docker compose -f deploy/compose/vectordb.yaml --profile milvus up -d
```

Set authentication credentials and start RAG services:
```bash
export APP_VECTORSTORE_USERNAME="root"
export APP_VECTORSTORE_PASSWORD="your-secure-password"

docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

:::{important}
**Set the password before deployment** as it persists in the etcd volume. To change the password after deployment, stop the containers, remove the volumes, and restart:

```bash
docker compose -f deploy/compose/vectordb.yaml --profile milvus down
rm -rf deploy/compose/volumes/milvus deploy/compose/volumes/minio deploy/compose/volumes/etcd
docker compose -f deploy/compose/vectordb.yaml --profile milvus up -d
```
:::

:::{warning}
**Data Loss Warning:** Removing volumes deletes **all ingested data** (Milvus vectors and MinIO files). You must re-ingest all documents afterward. Ensure you have backups before proceeding.
:::

### Helm Chart

#### 1. Configure Milvus Authentication in Helm:

Configure user config:

The [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file includes the necessary configuration under `nv-ingest.milvus`:
```yaml
nv-ingest:
  milvus:
    extraConfigFiles:
      user.yaml: |+
        common:
          security:
            authorizationEnabled: true
            defaultRootPassword: your-secure-password
```

(Optional) For more details on configuring Milvus with Helm, refer to the [Milvus Helm configuration](https://milvus.io/docs/configure-helm.md).

:::{important}
Change the password before starting the Helm deployment. Once the deployment is started, the root password you set becomes persistent in the etcd volume. To change the password after deployment:

1. Uninstall the deployment:
   ```bash
   helm uninstall rag -n rag
   ```

2. Delete the Milvus and etcd PVCs (list PVCs first to confirm names):
   ```bash
   kubectl get pvc -n rag
   kubectl delete pvc milvus -n rag
   kubectl delete pvc data-rag-etcd-0 -n rag
   ```

3. Redeploy with the new password in  [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml). Refer to [Change a deployment](./deploy-helm.md#change-a-deployment) for redeploying the chart.

For additional instructions, refer to [Uninstall a Deployment](deploy-helm.md#uninstall-a-deployment).
:::

:::{warning}
**Data Loss Warning:** Deleting PVCs permanently removes **all ingested data** (vector embeddings and metadata). You must re-ingest all documents afterward. Ensure you have backups before proceeding.
:::

#### 2. Configure username and password in `deploy/helm/nvidia-blueprint-rag/values.yaml`:

```yaml
envVars:
  APP_VECTORSTORE_USERNAME: "root"
  APP_VECTORSTORE_PASSWORD: "your-secure-password"

ingestor-server:
  envVars:
    APP_VECTORSTORE_USERNAME: "root"
    APP_VECTORSTORE_PASSWORD: "your-secure-password"
```

#### 3. Deploy with Helm:

After you modify values.yaml with the authentication settings, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).


## Using VDB Auth Token at Runtime via APIs

When Milvus is the active vector database (`APP_VECTORSTORE_NAME=milvus`), NVIDIA RAG Blueprint servers accept a Vector DB (VDB) authentication token via the HTTP `Authorization` header at runtime. This header is forwarded to Milvus for auth-protected operations. If you use Elasticsearch as the default VDB, runtime bearer tokens are handled differently—refer to Using VDB Auth Token at Runtime via APIs in [Vector database configuration](change-vectordb.md).

This Milvus flow assumes you already know how to create Milvus users, roles, and privileges. After you configure those in Milvus, you can pass auth tokens in the `Authorization` header to enforce access control.

Access permissions are enforced based on the privileges assigned to each user:
- **Read operations**: Users without privileges such as `Load`, `Search`, and `Query` will not be able to read data from collections.
- **Write operations**: Users without privileges such as `Insert` and `Upsert` will not be able to create collections or write data to collections.

:::{note}
Ensure you create the reader and writer users in Milvus with appropriate roles and privileges before using this feature. For detailed guidance on enabling authentication, creating users, updating passwords, and related operations in Milvus, refer to the official Milvus documentation:
- [Authenticate User Access](https://milvus.io/docs/authenticate.md?tab=docker)
- [Milvus Users and Roles](https://milvus.io/docs/v2.4.x/users_and_roles.md)
:::

### Prerequisites

1. Ensure Milvus authentication is enabled so auth is enforced. In Milvus config this is `security.authorizationEnabled: true`. See the [Milvus Authentication](#milvus-authentication) section above for setup via Docker Compose or Helm.

2. Export the required environment variables before running the API examples:

```bash
# Service URLs
export INGESTOR_URL="http://localhost:8082"  # Adjust to your ingestor server URL
export RAG_URL="http://localhost:8081"        # Adjust to your RAG server URL
export APP_VECTORSTORE_URL="http://localhost:19530"  # Adjust to your Milvus endpoint

# Root/Admin credentials (use your configured root password)
export APP_VECTORSTORE_USERNAME="root"
export APP_VECTORSTORE_PASSWORD="your-secure-password"

# Reader user credentials (create these users in Milvus first)
# Note: You can use root credentials if reader user is not created, but creating separate users is recommended for best practices
export MILVUS_READER_USER="reader_user"
export MILVUS_READER_PASSWORD="reader_password"

# Writer user credentials (create these users in Milvus first)
# Note: You can use root credentials if writer user is not created, but creating separate users is recommended for best practices
export MILVUS_WRITER_USER="writer_user"
export MILVUS_WRITER_PASSWORD="writer_password"
```

### Header format
- Preferred: `Authorization: Bearer <token>`
- Also accepted: `Authorization: <token>`

For Milvus (with auth enabled), the token is typically the string `user:password`. For example:
- Admin/root: `${APP_VECTORSTORE_USERNAME}:${APP_VECTORSTORE_PASSWORD}` (use your configured root credentials)
- Reader user: `${MILVUS_READER_USER}:${MILVUS_READER_PASSWORD}`
- Writer user: `${MILVUS_WRITER_USER}:${MILVUS_WRITER_PASSWORD}`


### Ingestor Server examples

- List documents in a collection (reader token):

```bash
curl -G "$INGESTOR_URL/v1/documents" \
  -H "Authorization: Bearer ${MILVUS_READER_USER}:${MILVUS_READER_PASSWORD}" \
  --data-urlencode "collection_name=demo_collection"
```

- Delete a collection (writer token with DropCollection privilege):

```bash
curl -X DELETE "$INGESTOR_URL/v1/collections" \
  -H "Authorization: Bearer ${MILVUS_WRITER_USER}:${MILVUS_WRITER_PASSWORD}" \
  --data-urlencode "collection_names=demo_collection"
```

### RAG Server examples

- Search with reader token:

```bash
curl -X POST "$RAG_URL/v1/search" \
  -H "Authorization: Bearer ${MILVUS_READER_USER}:${MILVUS_READER_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hello",
    "use_knowledge_base": true,
    "collection_names": ["demo_collection"],
    "vdb_endpoint": "'"$APP_VECTORSTORE_URL"'",
    "reranker_top_k": 0,
    "vdb_top_k": 1
  }'
```

 - Generate with streaming (reader token):

```bash
curl -N -X POST "$RAG_URL/v1/generate" \
  -H "Authorization: Bearer ${MILVUS_READER_USER}:${MILVUS_READER_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello"}],
    "use_knowledge_base": true,
    "collection_names": ["demo_collection"],
    "vdb_endpoint": "'"$APP_VECTORSTORE_URL"'",
    "reranker_top_k": 0,
    "vdb_top_k": 1
  }'
```

### Notes and troubleshooting
- If a user lacks privileges on the target collection, the API will return an authorization error (non-200 status). Grant the appropriate collection privileges to the user/role in Milvus (e.g., `Query`, `Search`, `DescribeCollection`, `Load`, `DropCollection`).
- Header precedence: For Milvus, the VDB token provided at runtime via `Authorization` is used for the request. There is no need to configure `APP_VECTORSTORE_USERNAME`/`APP_VECTORSTORE_PASSWORD` for per-request auth when using headers.
- POST /collection API limitation: The `POST /collection` API does not support passing auth token via the `Authorization` header.

## Troubleshooting

### GPU_CAGRA Error

If you encounter GPU_CAGRA errors that cannot be resolved by when switching to CPU mode, try the following:

1. Stop all running services:
   ```bash
   docker compose -f deploy/compose/vectordb.yaml --profile milvus down
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
   ```

2. Delete the Milvus volumes directory:
   ```bash
   rm -rf deploy/compose/volumes
   ```

3. Restart the services:
   ```bash
   docker compose -f deploy/compose/vectordb.yaml --profile milvus up -d
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

:::{note}
This will delete all existing vector data, so ensure you have backups if needed.
:::


## Related Topics

- [Vector database configuration](change-vectordb.md) (Elasticsearch default, switching to Milvus)
- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)
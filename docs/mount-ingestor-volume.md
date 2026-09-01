<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Ingestor Server Volume Mounting for NVIDIA RAG Blueprint

You can mount a host directory to access extraction results from NeMo Retriever Library directly from the filesystem when you use the [NVIDIA RAG Blueprint](readme.md). Designed for advanced developers who need programmatic access to raw extraction results for custom processing pipelines or external vector database integration.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INGESTOR_SERVER_DATA_DIR` | `/data/` | Container internal path. Mapped to the `rag-vol-ingestor` Docker named volume. |
| `APP_NVINGEST_SAVETODISK` | `False` | Enable disk persistence |

### Setup

1. **Enable disk persistence:**
   ```bash
   # Enable disk persistence
   export APP_NVINGEST_SAVETODISK=True

   # (Optional) Override container internal path
   export INGESTOR_SERVER_DATA_DIR=/data/
   ```

   The ingestor-server compose file already mounts `rag-vol-ingestor` at `INGESTOR_SERVER_DATA_DIR`; nothing else needs to be configured to persist results.

## Accessing the Results from the Host

Docker named volumes are owned by `root` on the host, so use one of the following patterns to read the files:

```bash
# Copy a single result file out of the volume:
docker run --rm -v rag-vol-ingestor:/src:ro -v "$PWD":/dst alpine \
  cp /src/nv-ingest-results/<collection>/<file>.results.jsonl /dst/

# List the directory tree inside the volume:
docker run --rm -v rag-vol-ingestor:/src:ro alpine ls -la /src/nv-ingest-results

# Or copy directly from the running ingestor-server container:
docker cp ingestor-server:/data/nv-ingest-results ./nv-ingest-results
```

See [Manage Persistent Data Volumes](troubleshooting.md#manage-persistent-data-volumes) for backup, reset, and migration commands.

## Result Structure

Results are saved as `.jsonl` files with naming convention: `{original_filename}.results.jsonl`

```
rag-vol-ingestor:/
└── nv-ingest-results/
    ├── collection_name1/
    │   ├── document1.pdf.results.jsonl
    │   ├── presentation.pptx.results.jsonl
    │   └── spreadsheet.xlsx.results.jsonl
    └── collection_name2/
        ├── report.pdf.results.jsonl
        ├── analysis.docx.results.jsonl
        └── data.xlsx.results.jsonl
```

Each `.jsonl` file contains structured extraction metadata including text segments, document structure, images, tables, and chunk boundaries.

**Advanced Usage**: These `.jsonl` files can be used for storing data in vector databases or performing custom processing workflows as desired. This functionality is intended for advanced developers who need direct access to the structured extraction results.

---

:::{note}
This is an advanced feature for custom processing workflows. Standard RAG functionality stores results directly in the vector database.
:::

## Helm (Kubernetes)

### Overview

The Helm chart supports persisting ingestor-server data to a PersistentVolumeClaim (PVC). When enabled, the chart mounts a PVC at the same path used by `INGESTOR_SERVER_DATA_DIR` (default `/data/`). Set `APP_NVINGEST_SAVETODISK=True` to write extraction results to disk.

### Values

Edit [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) and set:

```yaml
ingestor-server:
  envVars:
    # Ensure results are written to disk inside the pod
    APP_NVINGEST_SAVETODISK: "True"
    # Directory inside the container where results will be written
    INGESTOR_SERVER_DATA_DIR: "/data/"

  # PVC configuration (created automatically unless existingClaim is set)
  persistence:
    enabled: true
    existingClaim: ""         # set to use an existing PVC; leave empty to create one
    storageClass: ""          # set if your cluster requires a specific class (e.g., "standard")
    accessModes:
      - ReadWriteOnce
    size: 50Gi
    # Optional: explicitly set the mount path (defaults to INGESTOR_SERVER_DATA_DIR)
    mountPath: "/data/"
    # Optional: mount a subPath within the PVC
    subPath: ""
```

Notes:

- If `existingClaim` is empty, the chart will create a PVC named `<appName>-data`. With the default `appName` of `ingestor-server`, the PVC name will be `ingestor-server-data`.
- The container writes results under `/data/` by default. Structure matches the compose example: `/data/nv-ingest-results/<collection>/file.results.jsonl`.

### Deploy the Changes

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### List and Access Files

List results inside the ingestor-server pod (default mount path `/data/`):

```bash
kubectl -n rag exec -it <ingestor-pod> -- ls -l /data/
```

Copy data from the pod to your local computer:

```bash
kubectl -n rag cp <ingestor-pod>:/data/ ./ingestor-data
```
<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Ingestor Server Volume Mounting for NVIDIA RAG Blueprint

You can mount a host directory to access NV-Ingest extraction results directly from the filesystem when you use the [NVIDIA RAG Blueprint](readme.md). Designed for advanced developers who need programmatic access to raw extraction results for custom processing pipelines or external vector database integration.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT` | `./volumes/ingestor-server` | Host filesystem path |
| `INGESTOR_SERVER_DATA_DIR` | `/data/` | Container internal path |
| `APP_NVINGEST_SAVETODISK` | `False` | Enable disk persistence |

### Setup

1. **Export environment variables:**
   ```bash
   # Enable disk persistence
   export APP_NVINGEST_SAVETODISK=True

   # Set host directory path (optional - customize as needed)
   export INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT=./volumes/ingestor-server

   # Set container internal path (optional - customize as needed)
   export INGESTOR_SERVER_DATA_DIR=/data/
   ```

## Troubleshooting

**Optional: Fix permissions issues**
If you encounter permission errors when accessing the volume:
```bash
sudo chown -R 1000:1000 ${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}
sudo chmod -R 755 ${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}
```

## Result Structure

Results are saved as `.jsonl` files with naming convention: `{original_filename}.results.jsonl`

```
${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}/
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

**Note**: This is an advanced feature for custom processing workflows. Standard RAG functionality stores results directly in the vector database.

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

### Install / Upgrade (On-prem only)

Ensure your NGC API key is available:

```bash
export NGC_API_KEY="<your-ngc-api-key>"
```

Using a custom values file:

```bash
helm upgrade --install rag -n rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

Or with inline overrides:

```bash
helm upgrade --install rag -n rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set ingestor-server.envVars.APP_NVINGEST_SAVETODISK=True \
  --set ingestor-server.envVars.INGESTOR_SERVER_DATA_DIR=/data/ \
  --set ingestor-server.persistence.enabled=true \
  --set ingestor-server.persistence.size=50Gi
```

### List and Access Files

List results inside the ingestor-server pod (default mount path `/data/`):

```bash
kubectl -n rag exec -it <ingestor-pod> -- ls -l /data/
```

Copy data from the pod to your local computer:

```bash
kubectl -n rag cp <ingestor-pod>:/data/ ./ingestor-data
```
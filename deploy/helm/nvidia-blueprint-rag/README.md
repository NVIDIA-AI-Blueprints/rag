# NVIDIA RAG Blueprint Helm Chart — OpenShift Deployment Guide

This Helm chart deploys the [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) on Red Hat OpenShift. The blueprint is a reference solution for buildin RAG pipelines with NVIDIA NIM microservices, enabling natural language question answering grounded in enterprise data.

## Table of Contents

- [What Gets Deployed](#what-gets-deployed)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Deployment Guide](#detailed-deployment-guide)
- [Configuration Reference](#configuration-reference)
- [Using NVIDIA-Hosted Models (No GPUs)](#using-nvidia-hosted-models-no-gpus)
- [Upgrading](#upgrading)
- [Uninstalling](#uninstalling)
- [OpenShift-Specific Challenges and Solutions](#openshift-specific-challenges-and-solutions)
- [Troubleshooting](#troubleshooting)

## What Gets Deployed

### Application Services

| Component | Description | Port | Purpose |
|-----------|-------------|------|---------|
| **RAG Server** | LangChain-based orchestration service | 8081 | Coordinates retrieval, generation, multi-turn conversations, query rewriting, and reflection |
| **Ingestor Server** | Document ingestion API | 8082 | Manages multimodal document ingestion (text, tables, charts, images, audio) via NV-Ingest |
| **Frontend** | Vite-based web UI | 3000 | User interface for chat, document upload, collection management, and settings |
| **NV-Ingest** | High-performance extraction pipeline | 7670 | GPU-accelerated content extraction from PDFs and other documents |

### NIM Microservices (via NIM Operator)

| Component | Image | GPU Memory | Purpose |
|-----------|-------|------------|---------|
| **NIM LLM** | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.14.0` | ~80 GB (A100-80GB / H100) | Response generation, query rewriting, summarization, reflection |
| **NIM Embedding** | `nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:1.13.0` | ~4 GB | Text embedding for vector similarity search |
| **NIM Reranking** | `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0` | ~4 GB | Reranking retrieved passages for accuracy |
| **NIM VLM Embedding** (optional) | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | ~4 GB | Multimodal (vision + text) embedding for image-aware retrieval |
| **NIM VLM** (optional) | `nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:1.6.0` | ~24 GB (A10G / L40S) | Vision-language model for multimodal inference and image captioning |

### NV-Ingest Extraction NIMs

| Component | Image | GPU Memory | Purpose |
|-----------|-------|------------|---------|
| **Nemotron OCR** | `nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0` | ~16 GB | Optical character recognition |
| **Page Elements** | `nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0` | ~4 GB | Document layout analysis and page element detection |
| **Graphic Elements** | `nvcr.io/nim/nvidia/nemotron-graphic-elements-v1:1.8.0` | ~4 GB | Chart and infographic detection |
| **Table Structure** | `nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0` | ~4 GB | Table detection and structure extraction |
| **Nemotron Parse** (optional) | `nvcr.io/nim/nvidia/nemotron-parse:1.5.0` | ~24 GB | VLM-based text extraction |
| **Audio** (optional) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.4.0` | ~4 GB | Audio transcription |

### Infrastructure Services

| Component | Description | Port | Purpose |
|-----------|-------------|------|---------|
| **Milvus** | GPU-accelerated vector database | 19530 | Stores and searches document embeddings with cuVS |
| **etcd** | Key-value store | 2379 | Metadata storage for Milvus |
| **MinIO** | Object storage | 9000 | Blob storage for Milvus data and multimodal content |
| **Redis** | In-memory data store | 6379 | Message queue for NV-Ingest and summary status tracking |
| **OpenTelemetry Collector** (optional) | Telemetry pipeline | 4317/4318 | Distributed tracing collection (disabled by default) |
| **Zipkin** (optional) | Trace visualization | 9411 | Trace storage and UI (disabled by default) |

### Additional Resources Created

- **ConfigMaps**: Prompt configuration, application settings
- **Secrets**: NGC API keys, image pull credentials, service-specific API keys
- **PersistentVolumeClaims**: Storage for NIM model caches, Milvus, etcd, MinIO, ingestor data
- **Services**: ClusterIP services for internal communication
- **ServiceAccount**: Dedicated service account for the RAG server
- **ServiceMonitor**: Prometheus metrics collection (optional)
- **NIMCache / NIMService**: Custom resources managed by the NIM Operator

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OpenShift Route (TLS)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│   Frontend (Vite UI) :3000  │   │   RAG Server (FastAPI) :8081         │
└─────────────────────────────┘   └──────────────────────────────────────┘
                                         │                    │
                    ┌────────────────────┘                    └────────────────────┐
                    ▼                                                              ▼
┌──────────────────────────────────┐                    ┌──────────────────────────────────┐
│   Ingestor Server :8082          │                    │   NIM LLM :8000                  │
│   (Document ingestion pipeline)  │                    │   (Nemotron Super 49B)           │
└──────────────────────────────────┘                    └──────────────────────────────────┘
          │                                                        │
          ▼                                           ┌────────────┼────────────┐
┌──────────────────────────────┐                      ▼            ▼            ▼
│   NV-Ingest :7670            │              ┌────────────┐ ┌──────────┐ ┌──────────┐
│   ┌────────┐  ┌───────────┐  │              │ NIM Embed  │ │ NIM Rank │ │ NIM VLM  │
│   │  OCR   │  │ Page Elem │  │              │   :8000    │ │  :8000   │ │  :8000   │
│   ├────────┤  ├───────────┤  │              └────────────┘ └──────────┘ └──────────┘
│   │ Tables │  │ Graphics  │  │                      │
│   └────────┘  └───────────┘  │                      ▼
└──────────────────────────────┘              ┌──────────────────────────┐
          │                                   │   Milvus (GPU) :19530   │
          ▼                                   └──────────────────────────┘
┌──────────────────────────┐                           │
│   Redis :6379            │                  ┌────────┴────────┐
└──────────────────────────┘                  ▼                 ▼
                                    ┌─────────────────┐  ┌─────────────────┐
                                    │   etcd :2379    │  │  MinIO :9000    │
                                    └─────────────────┘  └─────────────────┘
```

### Data Flow

1. **Document Ingestion**: Users upload documents via the Frontend or Ingestor Server API. NV-Ingest extracts text, tables, charts, and images using GPU-accelerated NIMs. Extracted content is embedded and stored in Milvus.
2. **User Query**: Users submit questions through the Frontend. The RAG Server processes the query, optionally rewriting it for better retrieval.
3. **Retrieval**: The query is embedded using NIM Embedding and matched against stored vectors in Milvus. NIM Reranking reorders results for precision.
4. **Generation**: Retrieved context is passed to the NIM LLM (Nemotron Super 49B) for response generation with citations. Optional reflection validates response groundedness.

### Total Resource Requirements

| Deployment Mode | GPUs Required | Notes |
|----------------|--------------|-------|
| Full (self-hosted NIMs) | 8–10 | All NIM models running in-cluster |
| Minimal (no VLM, no optional NIMs) | 7 | Core pipeline without VLM or audio |
| API-only (NVIDIA-hosted models) | 1 | Only Milvus GPU; NIM inference via [build.nvidia.com](https://build.nvidia.com/) |

## Prerequisites

- **OpenShift** 4.14 or later with cluster-admin access
- **Helm** 3.x installed
- **`oc` CLI** configured with cluster access
- **NVIDIA GPU Operator** installed and functional
- **NVIDIA NIM Operator** v3.0.2+ installed ([installation guide](https://docs.nvidia.com/nim-operator/latest/install.html))
- **NVIDIA NGC API key** from [build.nvidia.com](https://build.nvidia.com/)
- **StorageClass** with dynamic provisioning (e.g., `gp3-csi` on AWS)

### Minimum Hardware

| Node Role | GPU | Count | Notes |
|-----------|-----|-------|-------|
| GPU Workers (NIM LLM) | A100 80GB or H100 | 1 | 49B parameter model requires large VRAM |
| GPU Workers (NIM services) | A10G / L4 / L40S | 6–9 | Embedding, reranking, OCR, page/graphic/table elements, Milvus |
| CPU Workers | — | 2 | RAG server, ingestor, frontend, NV-Ingest, Redis |

### GPU Availability Check

```bash
# Verify GPU nodes exist
oc get nodes -l nvidia.com/gpu.present=true

# Check GPU allocatable capacity
oc describe node <gpu-node-name> | grep -A5 "Allocatable"

# Check GPU taint keys (needed for tolerations)
oc describe node <gpu-node-name> | grep Taints
```

### NGC License Acceptance

Each NIM container image on NGC requires individually accepting a license agreement before your API key can pull it. Without acceptance, image pulls fail with `412 Precondition Failed`. Accept licenses for each NIM at [build.nvidia.com](https://build.nvidia.com/) or [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/).

Verify access before deploying:

```bash
NGC_API_KEY="nvapi-..."
for IMAGE in \
  "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.14.0" \
  "nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:1.13.0" \
  "nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0" \
  "nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0" \
  "nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0" \
  "nvcr.io/nim/nvidia/nemotron-graphic-elements-v1:1.8.0" \
  "nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0"; do
  echo -n "${IMAGE}: "
  skopeo inspect --creds "\$oauthtoken:${NGC_API_KEY}" "docker://${IMAGE}" &>/dev/null \
    && echo "OK" || echo "FAILED — accept license at build.nvidia.com"
done
```

## Quick Start

```bash
# 1. Set your NGC API key
export NGC_API_KEY="nvapi-..."

# 2. Create namespace
oc create namespace <NAMESPACE>

# 3. Grant required SCCs (must be done before pod creation)
oc adm policy add-scc-to-user anyuid -z default -n <NAMESPACE>

# 4. Install the NIM Operator (if not already installed)
oc create namespace nim-operator
helm upgrade --install nim-operator nvidia/k8s-nim-operator \
  -n nim-operator --version=3.0.2

# 5. Install the chart
cd deploy/helm/nvidia-blueprint-rag
helm dependency build
helm upgrade --install rag -n <NAMESPACE> . \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY

# 6. Grant additional SCCs as pods are created
oc adm policy add-scc-to-user anyuid -z rag-zipkin -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-nv-ingest -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-server -n <NAMESPACE>
```

Or install from a published chart version on NGC (replace `<VERSION>` with the desired release):

```bash
helm upgrade --install rag -n <NAMESPACE> \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-<VERSION>.tgz \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

## Detailed Deployment Guide

### Step 1: Prepare Your Environment

```bash
# Verify OpenShift connectivity
oc whoami
oc cluster-info

# Verify Helm is installed
helm version

# Add the NVIDIA Helm repository (for NIM Operator)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --username='$oauthtoken' --password=$NGC_API_KEY
helm repo update
```

### Step 2: Create Namespace and Grant SCCs

OpenShift's default `restricted` SCC assigns random UIDs that conflict with many containers in this chart (NV-Ingest, etcd, Zipkin, NIM containers). You must grant the `anyuid` SCC before pods are created.

```bash
oc create namespace <NAMESPACE>

# Grant anyuid to the default service account
oc adm policy add-scc-to-user anyuid -z default -n <NAMESPACE>
```

> **Important**: This must be done *before* pod creation. If pods start with the wrong SCC, they cache the failure and may not recover cleanly — you would need to delete and recreate the pods.

### Step 3: Install the NIM Operator

The NIM Operator manages NIMCache and NIMService custom resources that handle model downloading, caching, and serving.

```bash
# Create the NIM Operator namespace
oc create namespace nim-operator

# Install the NIM Operator
helm upgrade --install nim-operator nvidia/k8s-nim-operator -n nim-operator --version=3.0.2
```

Verify the operator is running:

```bash
oc get pods -n nim-operator
```

### Step 4: Configure Values

Create a `values-openshift.yaml` file with your environment-specific overrides:

```yaml
imagePullSecret:
  password: "nvapi-your-key-here"

ngcApiSecret:
  password: "nvapi-your-key-here"

ingestor-server:
  persistence:
    storageClass: "gp3-csi"  # Your cluster's StorageClass

frontend:
  service:
    type: ClusterIP  # Use ClusterIP on OpenShift; expose via Route

nimOperator:
  nim-llm:
    tolerations:
      - key: "your-gpu-taint-key"
        operator: "Exists"
        effect: "NoSchedule"
    storage:
      pvc:
        storageClass: "gp3-csi"
  nvidia-nim-llama-32-nv-embedqa-1b-v2:
    tolerations:
      - key: "your-gpu-taint-key"
        operator: "Exists"
        effect: "NoSchedule"
    storage:
      pvc:
        storageClass: "gp3-csi"
  nvidia-nim-llama-32-nv-rerankqa-1b-v2:
    tolerations:
      - key: "your-gpu-taint-key"
        operator: "Exists"
        effect: "NoSchedule"
    storage:
      pvc:
        storageClass: "gp3-csi"

nv-ingest:
  milvus:
    standalone:
      tolerations:
        - key: "your-gpu-taint-key"
          operator: "Exists"
          effect: "NoSchedule"
  nimOperator:
    ocr:
      storage:
        pvc:
          storageClass: "gp3-csi"
      tolerations:
        - key: "your-gpu-taint-key"
          operator: "Exists"
          effect: "NoSchedule"
    graphic_elements:
      storage:
        pvc:
          storageClass: "gp3-csi"
      tolerations:
        - key: "your-gpu-taint-key"
          operator: "Exists"
          effect: "NoSchedule"
    page_elements:
      storage:
        pvc:
          storageClass: "gp3-csi"
      tolerations:
        - key: "your-gpu-taint-key"
          operator: "Exists"
          effect: "NoSchedule"
    table_structure:
      storage:
        pvc:
          storageClass: "gp3-csi"
      tolerations:
        - key: "your-gpu-taint-key"
          operator: "Exists"
          effect: "NoSchedule"
```

To discover your cluster's GPU node taint keys:

```bash
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns="NODE:.metadata.name,TAINTS:.spec.taints[*].key"
```

### Step 5: Install the Helm Chart

```bash
cd deploy/helm/nvidia-blueprint-rag

# Build chart dependencies
helm dependency build

# Install with your values file
helm upgrade --install rag -n <NAMESPACE> . \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f values-openshift.yaml
```

**Dry-run to preview resources:**

```bash
helm upgrade --install rag -n <NAMESPACE> . \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f values-openshift.yaml --dry-run --debug
```

### Step 6: Grant Additional SCCs

After the initial install, some pods will fail because they use dedicated service accounts that also need the `anyuid` SCC:

```bash
# Check which service accounts need SCC grants
oc get pods -n <NAMESPACE> -o wide | grep -E "CrashLoop|Error|Init"

# Grant SCCs to service accounts created by the chart
oc adm policy add-scc-to-user anyuid -z rag-zipkin -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-nv-ingest -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-server -n <NAMESPACE>
```

> **Tip**: After granting SCCs, you may need to delete the affected pods so they are recreated with the correct security context:
> ```bash
> oc delete pod -l app.kubernetes.io/instance=rag -n <NAMESPACE> --field-selector=status.phase!=Running
> ```

### Step 7: Create an OpenShift Route

Expose the frontend to external traffic via an OpenShift Route:

```bash
# Route for the frontend UI
oc create route edge rag-frontend \
  --service=rag-frontend \
  --port=3000 \
  -n <NAMESPACE>
```

### Step 8: Verify Deployment

```bash
# Check all pods
oc get pods -n <NAMESPACE>

# Check NIMCache and NIMService resources
oc get nimcache,nimservice -n <NAMESPACE>

# Get the application URL
echo "https://$(oc get route rag-frontend -n <NAMESPACE> -o jsonpath='{.spec.host}')"

# Get the API URL
echo "https://$(oc get route rag-api -n <NAMESPACE> -o jsonpath='{.spec.host}')"
```

**Expected pods (core deployment):**

```
NAME                                          READY   STATUS    AGE
rag-server-xxxxxxxxx-xxxxx                    1/1     Running   5m
ingestor-server-xxxxxxxxx-xxxxx               1/1     Running   5m
rag-frontend-xxxxxxxxx-xxxxx                  1/1     Running   5m
rag-nv-ingest-xxxxxxxxx-xxxxx                 1/1     Running   5m
milvus-standalone-xxxxxxxxx-xxxxx             1/1     Running   5m
rag-etcd-0                                    1/1     Running   5m
rag-minio-xxxxxxxxx-xxxxx                     1/1     Running   5m
rag-redis-master-0                            1/1     Running   5m
nim-llm-xxxxxxxxx-xxxxx                       1/1     Running   10m
nemoretriever-embedding-ms-xxxxxxxxx-xxxxx    1/1     Running   5m
nemoretriever-ranking-ms-xxxxxxxxx-xxxxx      1/1     Running   5m
nemotron-ocr-v1-xxxxxxxxx-xxxxx               1/1     Running   5m
nemotron-page-elements-v3-xxxxxxxxx-xxxxx     1/1     Running   5m
nemotron-graphic-elements-v1-xxxxxxxxx-xxxxx  1/1     Running   5m
nemotron-table-structure-v1-xxxxxxxxx-xxxxx   1/1     Running   5m
```

> **Note**: NIM pods may take 10–30 minutes to reach `Running` status as they download and load model weights. The NIM LLM (49B) can take significantly longer on first deployment.

### Step 9: Health Checks

```bash
API_HOST=$(oc get route rag-api -n <NAMESPACE> -o jsonpath='{.spec.host}')

# RAG Server health
curl -sk "https://${API_HOST}/health"
# Expected: {"status":"ok"}

# Test a query (after NIM LLM is ready)
curl -sk "https://${API_HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5"
  }'
```

## Configuration Reference

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `imagePullSecret.password` | NGC API key for pulling container images | `""` |
| `ngcApiSecret.password` | NGC API key for NIM model authentication | `""` |
| `image.tag` | RAG server image version | `"2.5.0"` |
| `envVars.APP_LLM_SERVERURL` | LLM endpoint (empty string = NVIDIA-hosted) | `"nim-llm:8000"` |
| `envVars.APP_LLM_MODELNAME` | LLM model name | `"nvidia/llama-3.3-nemotron-super-49b-v1.5"` |
| `envVars.APP_EMBEDDINGS_SERVERURL` | Embedding endpoint | `"nemotron-embedding-ms:8000/v1"` |
| `envVars.APP_RANKING_SERVERURL` | Reranking endpoint | `"nemotron-ranking-ms:8000"` |
| `envVars.APP_VECTORSTORE_NAME` | Vector database type | `"milvus"` |
| `envVars.ENABLE_RERANKER` | Enable reranking | `"True"` |
| `envVars.ENABLE_VLM_INFERENCE` | Enable VLM for multimodal queries | `"false"` |
| `envVars.ENABLE_GUARDRAILS` | Enable NeMo Guardrails | `"False"` |
| `envVars.ENABLE_REFLECTION` | Enable response reflection/validation | `"false"` |
| `envVars.ENABLE_MULTITURN` | Enable multi-turn conversation | `"True"` |
| `frontend.service.type` | Frontend service type | `"NodePort"` |
| `ingestor-server.persistence.storageClass` | StorageClass for ingestor PVC | `""` |
| `nimOperator.nim-llm.enabled` | Deploy self-hosted LLM NIM | `true` |
| `nv-ingest.enabled` | Deploy NV-Ingest pipeline | `true` |
| `opentelemetry-collector.enabled` | Deploy OpenTelemetry collector | `false` |
| `zipkin.enabled` | Deploy Zipkin tracing | `false` |

### NIM Service Versions (Upstream v2.5.0)

| NIM | Image | Tag |
|-----|-------|-----|
| LLM (Nemotron Super 49B) | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` | `1.14.0` |
| Embedding | `nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2` | `1.13.0` |
| Reranking | `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2` | `1.10.0` |
| VLM Embedding | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2` | `1.12.0` |
| VLM | `nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl` | `1.6.0` |
| OCR | `nvcr.io/nim/nvidia/nemotron-ocr-v1` | `1.3.0` |
| Page Elements | `nvcr.io/nim/nvidia/nemotron-page-elements-v3` | `1.8.0` |
| Graphic Elements | `nvcr.io/nim/nvidia/nemotron-graphic-elements-v1` | `1.8.0` |
| Table Structure | `nvcr.io/nim/nvidia/nemotron-table-structure-v1` | `1.8.0` |
| Nemotron Parse | `nvcr.io/nim/nvidia/nemotron-parse` | `1.5.0` |
| Audio (Parakeet) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` | `1.4.0` |

### Infrastructure Versions

| Component | Image | Tag |
|-----------|-------|-----|
| NV-Ingest | `nvcr.io/nvstaging/nim/nv-ingest` | `26.3.0-RC4` |
| Milvus | `docker.io/milvusdb/milvus` | `v2.6.5-gpu` |
| etcd | `milvusdb/etcd` | `3.5.23-r2` |
| MinIO | `docker.io/minio/minio` | `RELEASE.2025-09-07T16-13-09Z` |
| Redis | `redis` | `8.2.1` |

## Using NVIDIA-Hosted Models (No GPUs)

For clusters without GPUs (or with limited GPU capacity), you can use NVIDIA-hosted model endpoints at [build.nvidia.com](https://build.nvidia.com/). This requires only a Milvus GPU (or CPU-only Milvus).

Set the model server URLs to empty strings to use NVIDIA-hosted APIs:

```yaml
envVars:
  APP_LLM_SERVERURL: ""
  APP_EMBEDDINGS_SERVERURL: ""
  APP_RANKING_SERVERURL: ""

nimOperator:
  nim-llm:
    enabled: false
  nvidia-nim-llama-32-nv-embedqa-1b-v2:
    enabled: false
  nvidia-nim-llama-32-nv-rerankqa-1b-v2:
    enabled: false

ingestor-server:
  envVars:
    APP_EMBEDDINGS_SERVERURL: ""
```

Install with:

```bash
helm upgrade --install rag -n <NAMESPACE> . \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set envVars.APP_LLM_SERVERURL="" \
  --set envVars.APP_EMBEDDINGS_SERVERURL="" \
  --set envVars.APP_RANKING_SERVERURL="" \
  --set nimOperator.nim-llm.enabled=false \
  --set "nimOperator.nvidia-nim-llama-32-nv-embedqa-1b-v2.enabled=false" \
  --set "nimOperator.nvidia-nim-llama-32-nv-rerankqa-1b-v2.enabled=false"
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade rag -n <NAMESPACE> . -f values-openshift.yaml

# Upgrade with specific overrides
helm upgrade rag -n <NAMESPACE> . \
  --set image.tag="2.5.0" \
  --set ingestor-server.image.tag="2.5.0" \
  --set frontend.image.tag="2.5.0"

# Check upgrade history
helm history rag -n <NAMESPACE>

# Rollback if needed
helm rollback rag 1 -n <NAMESPACE>
```

## Uninstalling

```bash
# Uninstall the Helm release
helm uninstall rag -n <NAMESPACE>

# Clean up NIM Operator resources (NIMCache/NIMService persist after uninstall)
oc delete nimservice --all -n <NAMESPACE>
oc delete nimcache --all -n <NAMESPACE>

# Clean up PVCs (this deletes all data!)
oc delete pvc --all -n <NAMESPACE>

# Remove SCC grants
oc adm policy remove-scc-from-user anyuid -z default -n <NAMESPACE>
oc adm policy remove-scc-from-user anyuid -z rag-zipkin -n <NAMESPACE>
oc adm policy remove-scc-from-user anyuid -z rag-nv-ingest -n <NAMESPACE>
oc adm policy remove-scc-from-user anyuid -z rag-server -n <NAMESPACE>

# Delete the namespace
oc delete namespace <NAMESPACE>

# Uninstall NIM Operator (if no longer needed)
helm uninstall nim-operator -n nim-operator
oc delete namespace nim-operator
```

## OpenShift-Specific Challenges and Solutions

### Challenge 1: Security Context Constraints (SCC)

**Symptom**: Pods fail with `CrashLoopBackOff`. Logs show permission errors such as:
```
mkdir: cannot create directory '/opt/nim/.cache': Permission denied
```

**Why**: OpenShift's default `restricted` SCC assigns random UIDs. NIM containers and infrastructure services (etcd, NV-Ingest, Zipkin) expect to run as specific users.

**Fix**: Grant `anyuid` SCC to affected service accounts *before* deploying:
```bash
oc adm policy add-scc-to-user anyuid -z default -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-zipkin -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-nv-ingest -n <NAMESPACE>
oc adm policy add-scc-to-user anyuid -z rag-server -n <NAMESPACE>
```

### Challenge 2: GPU Node Scheduling and Tolerations

**Symptom**: NIM pods stay in `Pending` state indefinitely.

**Why**: GPU nodes typically have taints that prevent non-GPU workloads from landing on them. NIM workloads need matching tolerations.

**Fix**: Discover your taint keys and set tolerations in your values file:
```bash
oc get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns="NODE:.metadata.name,TAINTS:.spec.taints[*].key"
```

Set matching tolerations for each NIM component in your `values-openshift.yaml`.

### Challenge 3: StorageClass Configuration

**Symptom**: PVCs stay in `Pending` state.

**Why**: The default `values.yaml` may reference a StorageClass that doesn't exist on your cluster (e.g., `gp3-csi` on non-AWS clusters).

**Fix**: Set the correct StorageClass for all PVC-creating components:
```bash
# Check available StorageClasses
oc get storageclass
```

Update all `storageClass` fields in your values file, or set to `""` to use the cluster default.

### Challenge 4: Shared Memory (shm_size) for NIM Containers

**Symptom**: NIM model loading fails with out-of-memory errors:
```
RuntimeError: DataLoader worker is killed by signal: Bus error
```

**Why**: The default `/dev/shm` in a pod is only 64 MB. NIM containers need up to 16 GB of shared memory.

**Fix**: The chart configures `sharedMemorySizeLimit: "16Gi"` for NIM services via the NIM Operator. Verify this is set in your NIMService resources:
```bash
oc get nimservice nim-llm -n <NAMESPACE> -o yaml | grep sharedMemory
```

### Challenge 5: NIM LLM VRAM Requirements

**Symptom**: NIM LLM pod starts but crashes during model loading with `torch.OutOfMemoryError`.

**Why**: The Nemotron Super 49B model requires significant VRAM, especially at the default context length of 131072 tokens.

**Fix**: For GPUs with limited VRAM, reduce `NIM_MAX_MODEL_LEN`:
```yaml
nimOperator:
  nim-llm:
    env:
      - name: NIM_MAX_MODEL_LEN
        value: "40960"  # Reduce from default 131072
```

For A100-40GB GPUs, you may need a smaller model or FP8 quantization on SM89+ hardware (L40S, H100).

### Challenge 6: NGC Image Pull — 412 Precondition Failed

**Symptom**: Pods stuck in `ErrImagePull` with `412 Precondition Failed`.

**Why**: Each NIM image requires individually accepting a license on [build.nvidia.com](https://build.nvidia.com/). Having an NGC API key is not enough.

**Fix**: Visit each NIM's page on build.nvidia.com and accept the license. Verify with `skopeo inspect` as shown in [Prerequisites](#ngc-license-acceptance).

### Challenge 7: Frontend Service Type

**Symptom**: Frontend is not accessible externally.

**Why**: The default service type is `NodePort`, which may not work on all OpenShift clusters.

**Fix**: Use `ClusterIP` and expose via an OpenShift Route:
```yaml
frontend:
  service:
    type: ClusterIP
```

Then create a Route as shown in [Step 7](#step-7-create-an-openshift-route).

### Challenge 8: Long Request Timeouts

**Symptom**: Document ingestion or complex queries return `504 Gateway Timeout`.

**Why**: OpenShift's default Route timeout is 30 seconds. Document ingestion and large generation requests can take minutes.

**Fix**: Set the Route timeout annotation:
```bash
oc annotate route rag-api haproxy.router.openshift.io/timeout=300s -n <NAMESPACE>
oc annotate route rag-frontend haproxy.router.openshift.io/timeout=300s -n <NAMESPACE>
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status and events
oc get pods -n <NAMESPACE>
oc describe pod <pod-name> -n <NAMESPACE>
oc logs <pod-name> -n <NAMESPACE>
```

### Image pull errors

```bash
# Verify the image pull secret exists and is correct
oc get secret ngc-secret -n <NAMESPACE> -o yaml

# Test pulling an image manually
oc run test-pull --rm -it --restart=Never \
  --image=nvcr.io/nvidia/blueprint/rag-server:2.5.0 \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"ngc-secret"}]}}' \
  -n <NAMESPACE> -- echo "Pull successful"
```

### NIM pods not becoming ready

```bash
# Check NIMCache status (model downloading)
oc get nimcache -n <NAMESPACE>
oc describe nimcache nim-llm-cache -n <NAMESPACE>

# Check NIMService status
oc get nimservice -n <NAMESPACE>
oc describe nimservice nim-llm -n <NAMESPACE>

# Check NIM pod logs
oc logs -f $(oc get pod -l app=nim-llm -n <NAMESPACE> -o name | head -1) -n <NAMESPACE>
```

### Milvus issues

```bash
# Check Milvus standalone pod
oc logs deployment/milvus-standalone -n <NAMESPACE>

# Check etcd and MinIO
oc logs statefulset/rag-etcd -n <NAMESPACE>
oc logs deployment/rag-minio -n <NAMESPACE>

# Verify GPU is allocated to Milvus
oc describe pod -l app.kubernetes.io/name=milvus -n <NAMESPACE> | grep -A5 "nvidia.com/gpu"
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Pods stuck in `Pending` | Insufficient GPU resources or missing tolerations | Check node resources and GPU taints; update tolerations in values |
| `ImagePullBackOff` | Missing NGC secret or unaccepted NIM license | Verify `ngc-secret` exists; accept licenses at build.nvidia.com |
| `CrashLoopBackOff` | SCC restrictions or insufficient memory | Grant `anyuid` SCC; check resource limits |
| NIM LLM `OOMKilled` | Insufficient VRAM for model | Reduce `NIM_MAX_MODEL_LEN` or use a GPU with more VRAM |
| PVC `Pending` | StorageClass not found | Set correct `storageClass` in values or use `""` for default |
| `504 Gateway Timeout` | Route timeout too low | Annotate route with `haproxy.router.openshift.io/timeout=300s` |
| etcd `CrashLoopBackOff` | Permission denied on data directory | Grant `anyuid` SCC to `default` service account |
| NV-Ingest not processing | Redis or NV-Ingest service account lacks permissions | Grant `anyuid` SCC to `rag-nv-ingest` |
# RAG Shutdown

Stopping containers and processes does not require confirmation. Deleting data (volumes, cache, images) does.

## Step 1: Detect What Is Running

Detect all deployment modes — Docker, K8s, and library:

```bash
echo "=== DOCKER ===" && docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || echo "NO_DOCKER"; echo "=== LIBRARY ===" && ps aux | grep -E "(nvidia_rag|uvicorn|jupyter)" | grep -v grep || echo "NO_LIBRARY_PROCESSES"; echo "=== K8S ===" && kubectl get pods -n rag 2>/dev/null | head -10 || echo "NO_K8S"; echo "=== HELM ===" && helm list -n rag 2>/dev/null | grep rag || echo "NO_HELM_RELEASE"
```

Based on what's detected, execute the appropriate shutdown path below. If multiple modes are active (e.g., Docker + library), stop all of them.

## Step 2: Stop Services (Reverse Startup Order)

Stop in this order — reverse of deployment. Only stop what is actually running (detected in Step 1).

### 2a: Optional Services

Stop these first if they are running:

```bash
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml down 2>/dev/null; docker compose -f deploy/compose/observability.yaml down 2>/dev/null
```

### 2b: Application Services

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml down; docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
```

### 2c: Vector DB

```bash
docker compose -f deploy/compose/vectordb.yaml down
```

If a profile-specific vector DB stack was started and containers remain, include the profile explicitly:
```bash
docker compose -f deploy/compose/vectordb.yaml --profile elasticsearch down
```

### 2d: NIMs (Self-Hosted Only)

Only present if self-hosted deployment was used:

```bash
docker compose -f deploy/compose/nims.yaml down
```

This stops ALL NIM containers (LLM, embedding, ranking, OCR, detection, and any profile-specific NIMs like VLM, audio, nemotron-parse).

### 2e: Library Mode Processes

If library mode is active (detected Python processes): stop the running `nvidia_rag` / `uvicorn` RAG processes (identify their PIDs from the Step 1 detection output and terminate them), then bring down the backend containers:

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down 2>/dev/null; docker compose -f deploy/compose/vectordb.yaml down 2>/dev/null
```

### 2f: Kubernetes (Helm) Deployment

If K8s deployment was detected, use the release name and namespace from `helm list` output in step 1:

```bash
helm uninstall <release-name> -n <namespace> 2>/dev/null
```

To also clean up persistent data (only if the user requests full cleanup and confirms), delete the leftover `nimcache` and `pvc` resources in the namespace with `kubectl`.

## Step 3: Verify Everything Stopped

```bash
echo "=== REMAINING ===" && docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null; echo "=== K8S ===" && kubectl get pods -n rag 2>/dev/null | head -10 || echo "NOT_K8S"; helm list -n rag 2>/dev/null || true
```

If any RAG-related containers remain, list them, confirm with the user, then remove them individually by name:
```bash
docker ps -a --format "{{.Names}}" | grep -E "(rag|milvus|nim|ingest|redis|nemo|grafana|prometheus|embedding|ranking|vlm|ocr|page-elements|graphic-elements|table-structure)"
```
After the user confirms the list, stop each container and then delete the approved ones through the Docker CLI, one at a time by name.

If pods remain after `helm uninstall` and the user confirms, force-terminate them with `kubectl` in the `rag` namespace (zero grace period).

## Step 4: Optional Cleanup

Ask the user if they want to clean up data/volumes:

- **Remove Docker volumes** (deletes ingested data, vector DB indices, object-store data, and ingestor scratch). List the volumes and confirm with the user first:
  ```bash
  docker volume ls -q --filter "name=^rag-vol-"
  ```
  After the user approves specific volumes, delete only those (one at a time by name) using Docker's volume-management commands. These named volumes include Elasticsearch, Milvus/etcd, SeaweedFS, and ingestor scratch data — delete only the specific `rag-vol-*` volume(s) the user requested.

- **Remove model cache** (frees 100-200 GB for self-hosted). Only after the user confirms, delete the `~/.cache/model-cache/` directory with their preferred file-management tool.

- **Remove Docker images** (frees disk space). List the RAG images and confirm with the user first:
  ```bash
  docker images | grep -E "nvcr.io/nvidia|milvusdb"
  ```
  After the user approves, delete only the approved images by ID or tag through the Docker CLI.

Only perform cleanup if the user explicitly requests it.

## Quick One-Liner (All Docker Services)

If the user wants a fast full teardown:

```bash
cd "$(git rev-parse --show-toplevel)" && \
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml down 2>/dev/null; \
docker compose -f deploy/compose/observability.yaml down 2>/dev/null; \
docker compose -f deploy/compose/docker-compose-rag-server.yaml down 2>/dev/null; \
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down 2>/dev/null; \
docker compose -f deploy/compose/vectordb.yaml down 2>/dev/null; \
docker compose -f deploy/compose/nims.yaml down 2>/dev/null; \
echo "All RAG services stopped."
```

## Source Documentation
- `docs/troubleshooting.md` — if services won't stop or containers hang

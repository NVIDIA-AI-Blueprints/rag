# Deployment Reference

## Discovery

Run a non-mutating discovery pass before choosing a mode:

```bash
echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "NO_GPU"
echo "=== DOCKER ===" && docker --version 2>/dev/null || echo "NO_DOCKER"
echo "=== COMPOSE ===" && docker compose version 2>/dev/null || echo "NO_COMPOSE"
echo "=== NVIDIA_RUNTIME ===" && docker info 2>/dev/null | grep -i "runtimes.*nvidia" || echo "NO_NVIDIA_RUNTIME"
echo "=== PYTHON ===" && python3 --version 2>/dev/null || echo "NO_PYTHON"
echo "=== DISK ===" && df -h . 2>/dev/null | tail -1
echo "=== RAG ===" && docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(rag-server|ingestor-server|milvus|nim-)" || echo "NO_RAG_CONTAINERS"
```

Check key presence without printing values:

```bash
if [ -n "$NGC_API_KEY" ]; then echo "NGC_API_KEY_SET"; elif [ -n "$NVIDIA_API_KEY" ]; then echo "NVIDIA_API_KEY_SET"; else echo "NO_API_KEY_IN_ENV"; fi
```

## Routing

| User intent or host state | Route | Source docs |
|---|---|---|
| Local Docker with sufficient GPU and NVIDIA runtime | Docker self-hosted | `docs/deploy-docker-self-hosted.md`, `docs/support-matrix.md` |
| Docker available, local inference not suitable | Docker NVIDIA-hosted | `docs/deploy-docker-nvidia-hosted.md`, `docs/api-key.md` |
| Retrieval only, search only, no LLM | Retrieval-only Docker | `docs/retrieval-only-deployment.md` |
| Kubernetes, Helm, production chart | Helm | `docs/deploy-helm.md`, `docs/deploy-helm-from-repo.md` |
| MIG slicing | Helm MIG | `docs/mig-deployment.md` |
| Python library, no Docker, notebook flow | Library | `docs/python-client.md`, `notebooks/rag_library_usage.ipynb`, `notebooks/rag_library_lite_usage.ipynb` |

## Verification

Use the active host and port values from the deployment docs. Defaults are:

```bash
curl -s http://localhost:8081/v1/health?check_dependencies=true
curl -s http://localhost:8082/v1/health?check_dependencies=true
```

If health fails, route to `rag-troubleshoot-blueprint` and read
`docs/troubleshooting.md`.

## Shutdown

Before stopping or tearing down services, state the target deployment and ask for
confirmation. Use documented compose or Helm commands; do not remove volumes or
collections unless the user explicitly approves the data loss.


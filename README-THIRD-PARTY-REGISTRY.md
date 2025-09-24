# Third-Party Container Registry Support for NVIDIA RAG Blueprint

This branch adds support for deploying the NVIDIA RAG Blueprint using third-party container registries (GitLab, GitHub Container Registry, etc.) instead of requiring full NGC access.

### 1. Global Registry Configuration
Added `customRegistry` section in `values.yaml`:
```yaml
customRegistry:
  enabled: false
  registry: ""  # e.g., "ghcr.io" or "docker.io"
  namespace: ""  # e.g., "your-username/rag" or "your-org/your-repo"
  tag: "latest"  # Default tag for all custom images
```

### 2. Simplified Image Pull Secrets
- `imagePullSecretNgc` - For NGC registry access (NIMs, nv-ingest, etc.)
- `customRegistry` - Single configuration for custom registry (rag-server, ingestor-server, frontend)

### 3. Smart Image Resolution
Templates automatically use custom registry when `customRegistry.enabled=true`:
- **Custom registry enabled**: Uses `{registry}/{namespace}/{image}:{tag}`
- **Custom registry disabled**: Uses original NGC images

## Helm Deployment

### Step 1: Clone and Setup
```bash
git clone https://github.com/NVIDIA-AI-Blueprints/rag.git
cd rag
git checkout v2.3.0-draft-3rd-party-registry
```

### Step 2: Create NGC API Key
```bash
# Get your NGC API key from https://org.ngc.nvidia.com/setup/api-keys
export NGC_API_KEY="your-ngc-api-key"
```

### Step 3: Update Helm Dependencies

Add Helm Repositories with Authentication:
```bash
# Add nvidia-nemo-microservices repository
helm repo add nvidia-nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices --username='$oauthtoken' --password=$NGC_API_KEY

# Add nvidia-nim repository  
helm repo add nvidia-nim https://helm.ngc.nvidia.com/nim --username='$oauthtoken' --password=$NGC_API_KEY

# Add nvidia-nim-nvidia repository
helm repo add nvidia-nim-nvidia https://helm.ngc.nvidia.com/nim/nvidia --username='$oauthtoken' --password=$NGC_API_KEY
```

Update Helm Dependencies:
```bash
helm dependency update deploy/helm/nvidia-blueprint-rag
```

### Step 4: Build and Push Images
```bash
# Set your container registry configuration
export CONTAINER_REGISTRY=ghcr.io
export REGISTRY_NAMESPACE=your-username/rag
export CONTAINER_TAG=latest
export CONTAINER_REGISTRY_PATH=${CONTAINER_REGISTRY}/${REGISTRY_NAMESPACE}
export REGISTRY_USERNAME=your-username
export REGISTRY_PASSWORD=your-token

# Build images
docker compose -f deploy/compose/docker-compose-rag-server.yaml build
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml build

# Tag for your container registry
docker tag rag-server:2.3.0.rc2.2 ${CONTAINER_REGISTRY_PATH}/rag-server:${CONTAINER_TAG}
docker tag ingestor-server:2.3.0.rc2.2 ${CONTAINER_REGISTRY_PATH}/ingestor-server:${CONTAINER_TAG}
docker tag rag-frontend:2.3.0.rc2.2 ${CONTAINER_REGISTRY_PATH}/rag-frontend:${CONTAINER_TAG}

# Push to your container registry
docker push ${CONTAINER_REGISTRY_PATH}/rag-server:${CONTAINER_TAG}
docker push ${CONTAINER_REGISTRY_PATH}/ingestor-server:${CONTAINER_TAG}
docker push ${CONTAINER_REGISTRY_PATH}/rag-frontend:${CONTAINER_TAG}
```

### Step 5: Deploy with Custom Registry
```bash
# Create namespace
kubectl create namespace rag

# Deploy with custom registry (using the environment variables from Step 4)
helm upgrade --install rag -n rag deploy/helm/nvidia-blueprint-rag \
  --set customRegistry.enabled=true \
  --set customRegistry.registry="${CONTAINER_REGISTRY}" \
  --set customRegistry.namespace="${REGISTRY_NAMESPACE}" \
  --set customRegistry.tag="${CONTAINER_TAG}" \
  --set customRegistry.username="${REGISTRY_USERNAME}" \
  --set customRegistry.password="${REGISTRY_PASSWORD}" \
  --set customRegistry.create=true \
  --set imagePullSecretNgc.password="$NGC_API_KEY" \
  --set nvidia-nim-llama-32-nemoretriever-1b-vlm-embed-v1.enabled=false

# Verify deployment
kubectl get pods -n rag
```

## Configuration Options

### Complete Example: GitHub Container Registry with NIM Profile (or your own registry)
```bash
helm upgrade --install rag -n rag deploy/helm/nvidia-blueprint-rag \
  --set customRegistry.enabled=true \
  --set customRegistry.registry="${CONTAINER_REGISTRY}" \
  --set customRegistry.namespace="${REGISTRY_NAMESPACE}" \
  --set customRegistry.tag="${CONTAINER_TAG}" \
  --set customRegistry.username="${REGISTRY_USERNAME}" \
  --set customRegistry.password="${REGISTRY_PASSWORD}" \
  --set customRegistry.create=true \
  --set imagePullSecretNgc.password="$NGC_API_KEY" \
  --set nim-llm.env[0].name="NIM_MODEL_PROFILE" \
  --set nim-llm.env[0].value="tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-ed15592b3e4d174a719e8188493420073c39448d9b7ed742cfe614b96fecbdd9-1" \
  --set nvidia-nim-llama-32-nemoretriever-1b-vlm-embed-v1.enabled=false
```

**For Other GPU Types:**
```bash
# List available profiles for your system
curl -H "Authorization: Bearer $NGC_API_KEY" https://api.ngc.nvidia.com/v2/nim/profiles
```

**Note:** You can adapt this example for any container registry (Docker Hub, AWS ECR, etc.) by changing the `registry`, `namespace`, `username`, and `password` values accordingly.

**Reference:** See the [RAG Blueprint Quickstart Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/v2.3.0-draft/docs/quickstart.md#list-available-profiles-for-your-system) for more details on NIM profile configuration.

## Docker Compose Deployment

### Step 1: Clone Repository and Checkout Draft Branch
```bash
# Clone the RAG Blueprint repository
git clone https://github.com/NVIDIA-AI-Blueprints/rag.git
# Navigate to the repository directory
cd rag
# Checkout the v2.3.0-draft branch
git checkout v2.3.0-draft
```

### Step 2: Create NGC API Key
```bash
# Get your NGC API key from https://org.ngc.nvidia.com/setup/api-keys
export NGC_API_KEY="your-ngc-api-key"
```

**Reference:** [RAG Blueprint Quickstart Guide](docs/quickstart.md#obtain-an-api-key)

### Step 3: Prerequisites Setup
```bash
# 1. Install Docker Engine and Docker Compose (version 2.29.1 or later)
# 2. Authenticate with NGC registry:
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# 3. Install NVIDIA Container Toolkit for GPU-accelerated containers
# 4. Create model cache directory:
mkdir -p ~/.cache/model-cache
export MODEL_DIRECTORY=~/.cache/model-cache
```

### Step 4: Deploy with Docker Compose (Build Option)

**Note:** The container images for rag-server, ingestor-server, and rag-frontend are not publicly available. You can build them from source using the `--build` flag.

#### Starting required NIMs for RAG:
```bash
# Set environment variables:
source deploy/compose/.env  # For on-prem models (if available)

# Configure NIM profile for your GPU Type, For Example H100 SXM used here
export NIM_MODEL_PROFILE="tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-ed15592b3e4d174a719e8188493420073c39448d9b7ed742cfe614b96fecbdd9-1"

For reference: [RAG Blueprint Quickstart Guide](docs/quickstart.md#set-the-required-profile)

# Start all required NIMs:
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d

# Wait till the nemoretriever-ranking-ms, nemoretriever-embedding-ms and nim-llm-ms NIMs are in healthy state before proceeding further.
# The nemo LLM service may take up to 30 mins to start for the first time as the model is downloaded and cached.
```

#### Starting Vector Database, Ingestor-Server, Rag-Server and Rag-Playground:
```bash
# Start vector database:
docker compose -f deploy/compose/vectordb.yaml up -d

# Start ingestion services (build from source):
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build

# Start RAG services (build from source):
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build

# Verify all containers are running:
docker ps --format "table {{.Names}}\t{{.Status}}"
```

#### Access the application:
- Open web browser and go to `http://<IP of machine or localhost>:8090`

**Reference:** [RAG Blueprint Quickstart Guide](docs/quickstart.md#deploy-with-docker-compose)

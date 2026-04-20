<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Elasticsearch Configuration

This document describes **optional** GPU-accelerated vector indexing and authentication configuration for Elasticsearch in the [NVIDIA RAG Blueprint](readme.md).

For standard Elasticsearch usage (ports, switching backends, default deployment), see [Vector database configuration](change-vectordb.md).

---

## GPU Indexing

GPU indexing is part of **Elasticsearch Enterprise** (or a compatible Elastic license tier). You must obtain a GPU-enabled Elasticsearch image, apply a valid Elastic license, and configure your deployment accordingly.

### Prerequisites

- **Elastic license** – A subscription or trial that includes GPU vector indexing. Obtain the license JSON from Elastic (for example, a non-production or production stack license file).
- **NVIDIA GPU** – A supported NVIDIA GPU and driver stack on the Docker host or Kubernetes node.
- **NVIDIA Container Toolkit** – Configured so Docker (or the GPU Operator on Kubernetes) can schedule GPU devices into containers.

### Build a GPU-Enabled Elasticsearch Image

GPU vector indexing requires a custom Elasticsearch Docker image. Follow Elastic's official guide to build it:

[**Elasticsearch Docker image with GPU support**](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing#elasticsearch-docker-image-with-gpu-support)

After building, tag the image and push it to your registry if needed. Reference this tag in the Docker Compose or Helm configuration below.

### Docker Compose

The following steps apply to Docker Compose deployments using [`deploy/compose/vectordb.yaml`](../deploy/compose/vectordb.yaml).

#### 1. Set the GPU-Enabled Image

In [`deploy/compose/vectordb.yaml`](../deploy/compose/vectordb.yaml), update the `elasticsearch` service to reference your GPU-enabled image:

```yaml
elasticsearch:
  image: es-gpu   # Replace with your registry/tag if different
```

#### 2. Enable GPU Indexing

In the `elasticsearch` service `environment` block, uncomment the GPU indexing setting:

```yaml
environment:
  # ... existing variables ...
  - "vectors.indexing.use_gpu=true"
```

#### 3. Pass GPU Devices to the Container

Uncomment the `deploy` block to grant the container access to an NVIDIA GPU. Set `VECTORSTORE_GPU_DEVICE_ID` in [`.env`](../deploy/compose/.env) or your shell to pin a specific device:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: ["gpu"]
          device_ids: ["${VECTORSTORE_GPU_DEVICE_ID:-0}"]
```

#### 4. Set Up Data Volume Permissions

Elasticsearch data is bind-mounted under `deploy/compose/volumes/elasticsearch`. Ensure the directory exists and is owned by UID/GID **1000:1000**:

```bash
sudo mkdir -p deploy/compose/volumes/elasticsearch/
sudo chown -R 1000:1000 deploy/compose/volumes/elasticsearch/
```

#### 5. Start Elasticsearch

Bring up the Elasticsearch container:

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

#### 6. Apply the Elastic License

Once Elasticsearch is reachable on port 9200, install the license file:

```bash
curl -X PUT "http://localhost:9200/_license" \
  -H "Content-Type: application/json" \
  -d @/path/to/your-license.json
```

Verify the license is active:

```bash
curl -X GET "http://localhost:9200/_license"
```

Confirm the response shows an **Enterprise** (or applicable) tier with an active license state.

#### 7. Enable GPU Indexing in the Ingestor Server

Set `APP_VECTORSTORE_ENABLEGPUINDEX=True` before starting the ingestor server. This enables the GPU index strategy (`int8_hnsw`) used during document ingestion:

```bash
export APP_VECTORSTORE_ENABLEGPUINDEX=True

docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

:::{note}
For authentication configuration (xpack security, API keys), see [Elasticsearch Authentication](#elasticsearch-authentication).
:::

### Helm

The following steps apply to Helm deployments using [`deploy/helm/nvidia-blueprint-rag/values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml).

#### 1. Set the GPU-Enabled Image

In [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), uncomment and set the custom Elasticsearch image under the `eck-elasticsearch` section:

```yaml
eck-elasticsearch:
  enabled: true
  image: <your-es-gpu-image-here>   # Replace with your built or registry image tag
```

#### 2. Enable GPU Indexing and Resources

Under `eck-elasticsearch.nodeSets[0]`, uncomment `vectors.indexing.use_gpu: true` in the `config` block, and uncomment the `nvidia.com/gpu` resource requests and limits in the `podTemplate`:

```yaml
eck-elasticsearch:
  nodeSets:
  - name: default
    config:
      node.store.allow_mmap: false
      xpack.security.enabled: false
      xpack.security.http.ssl.enabled: false
      xpack.security.transport.ssl.enabled: false
      vectors.indexing.use_gpu: true          # Uncomment this line
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources:
            requests:
              memory: "4Gi"
              cpu: "500m"
              nvidia.com/gpu: 1               # Uncomment this line
            limits:
              memory: "4Gi"
              nvidia.com/gpu: 1               # Uncomment this line
```

#### 3. Enable GPU Indexing in the Ingestor Server

In [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), set `APP_VECTORSTORE_ENABLEGPUINDEX` to `"True"` in the `ingestor-server` section:

```yaml
ingestor-server:
  envVars:
    APP_VECTORSTORE_ENABLEGPUINDEX: "True"
```

#### 4. Deploy the Helm Chart

Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment). For a fresh install:

```bash
cd deploy/helm/

helm upgrade --install rag -n rag --create-namespace nvidia-blueprint-rag/ \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f nvidia-blueprint-rag/values.yaml
```

#### 5. Apply the Elastic License

After Elasticsearch is running, port-forward the service to access it from your local machine, then apply the license.

**In one terminal, start the tunnel:**

```bash
kubectl port-forward -n rag svc/rag-eck-elasticsearch-es-http 9200:9200
```

**In a second terminal, apply the license:**

```bash
curl -X PUT "http://localhost:9200/_license" \
  -H "Content-Type: application/json" \
  -d @/path/to/your-license.json
```

**Verify the license is active:**

```bash
curl -X GET "http://localhost:9200/_license"
```

Confirm the response shows an **Enterprise** (or applicable) tier with an active license state.

:::{note}
If xpack security is enabled, add `-u elastic:$ES_PASSWORD` to the curl commands. See [Elasticsearch Authentication (Helm)](#helm-chart) for retrieving the auto-generated password from the ECK secret.
:::

---

## Elasticsearch Authentication

Enable authentication for Elasticsearch to secure your vector database.

### Docker Compose

#### 1. Configure Elasticsearch Authentication (xpack)

Edit `deploy/compose/vectordb.yaml` to enable xpack security by setting `xpack.security.enabled` to true:
```yaml
environment:
  - xpack.security.enabled=true
```

Uncomment the username and password environment variables in the elasticsearch service in `deploy/compose/vectordb.yaml`:
```yaml
- ELASTIC_USERNAME=${APP_VECTORSTORE_USERNAME}
- ELASTIC_PASSWORD=${APP_VECTORSTORE_PASSWORD}
```

Add authentication in `healthcheck` in `deploy/compose/vectordb.yaml` by uncommenting the following:
```yaml
test: ["CMD", "curl", "-s", "-f", "-u", "${APP_VECTORSTORE_USERNAME}:${APP_VECTORSTORE_PASSWORD}", "http://localhost:9200/_cat/health"]
```
and commenting out
```yaml
test: ["CMD", "curl", "-s", "-f", "http://localhost:9200/_cat/health"]
```

#### 2. Start Elasticsearch Container with Credentials

Start the Elasticsearch container with username and password:

```bash
export APP_VECTORSTORE_USERNAME="elastic" # elastic recommended
export APP_VECTORSTORE_PASSWORD="your-secure-password"

docker compose -f deploy/compose/vectordb.yaml --profile elasticsearch up -d
```

#### 3. Generate Elasticsearch API Key (Optional but Recommended)

If you prefer to use API key authentication instead of username/password (recommended for production), generate an API key using curl. You need the username and password from the previous step.

```bash
# Either provide base64 apikey (base64 of "id:secret")
export APP_VECTORSTORE_APIKEY="base64-id-colon-secret"
# Or provide split ID/SECRET
export APP_VECTORSTORE_APIKEY_ID="your_id"
export APP_VECTORSTORE_APIKEY_SECRET="your_secret"

docker compose -f deploy/compose/vectordb.yaml --profile elasticsearch up -d
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Get an Elasticsearch API Key

If security is enabled, create an API key using curl. You need a user with permission to create API keys (e.g., the built-in `elastic` superuser in dev).

#### 1. Using curl (replace credentials and URL as appropriate):
```bash
# If running inside the cluster, port-forward first:
# kubectl -n rag port-forward svc/rag-eck-elasticsearch-es-http 9200:9200

curl -u elastic:your-secure-password \
  -X POST "http://127.0.0.1:9200/_security/api_key" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "rag-api-key",
    "role_descriptors": {}
  }'
```

Example response:
```json
{
  "id": "AbCdEfGhIj",
  "name": "rag-api-key",
  "expiration": null,
  "api_key": "ZyXwVuTsRq",
  "encoded": null
}
```

Convert the API key to base64:

```bash
# Base64 is computed over "<id>:<api_key>"
echo -n "AbCdEfGhIj:ZyXwVuTsRq" | base64
# Output example: QWJ...cXE=
```

#### 4. Set Environment Variables for Authentication

Choose ONE of the following authentication methods:

**Option A: API Key Authentication (Recommended)**

Set environment variables using the base64-encoded API key or split ID/SECRET:

```bash
# Either provide base64 apikey (base64 of "id:secret")
export APP_VECTORSTORE_APIKEY="QWJ...cXE="

# Or provide split ID/SECRET
export APP_VECTORSTORE_APIKEY_ID="AbCdEfGhIj"
export APP_VECTORSTORE_APIKEY_SECRET="ZyXwVuTsRq"
```

**Option B: Username/Password Authentication**

If you prefer to use username/password instead of API key:

```bash
export APP_VECTORSTORE_USERNAME="elastic"
export APP_VECTORSTORE_PASSWORD="your-secure-password"
```

#### 5. Start RAG Server and Ingestor Server

Start the RAG and Ingestor services with the authentication credentials:

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

:::{note}
API key authentication takes precedence over username/password when both are configured.
:::

### Helm Chart

Follow these steps to enable authentication for Elasticsearch in your Helm deployment.

#### 1. Enable Elasticsearch Authentication

Edit [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable X-Pack security. Set the following explicitly:

```yaml
eck-elasticsearch:
  nodeSets:
  - name: default
    config:
      node.store.allow_mmap: false
      xpack.security.enabled: true
      xpack.security.transport.ssl.enabled: true
```

:::{important}
**Key Configuration Flags:**
- `xpack.security.enabled: true` - Enables authentication (default user: `elastic`). Set this explicitly.
- `xpack.security.transport.ssl.enabled: true` - Enables SSL for node-to-node communication. Set this explicitly.
:::

#### 2. Replace Readiness Probe When Security Is Enabled

When X-Pack security is enabled, replace the current `readinessProbe` section under `eck-elasticsearch.nodeSets[0]` in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) with the ECK default probe (so the pod uses the readiness port script instead of an unauthenticated curl check). Ensure the following `podTemplate` is present under the same `nodeSets` entry:

```yaml
eck-elasticsearch:
  nodeSets:
  - name: default
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          readinessProbe:
            exec:
              command:
              - bash
              - -c
              - /mnt/elastic-internal/scripts/readiness-port-script.sh
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 5
            failureThreshold: 3
```

#### 3. Deploy with Authentication Enabled

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

Wait for Elasticsearch to restart:

```bash
# Monitor pod restart
kubectl get pods -n rag -w | grep elasticsearch

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l elasticsearch.k8s.elastic.co/cluster-name=rag-eck-elasticsearch -n rag --timeout=300s
```

#### 4. Retrieve Elasticsearch Password from Secret

When authentication is enabled, ECK automatically creates a Kubernetes secret containing the `elastic` user password:

```bash
# Find the Elasticsearch user secret
kubectl get secrets -n rag | grep elastic-user
# Expected: rag-eck-elasticsearch-es-elastic-user

# Retrieve the password
ES_PASSWORD=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user -n rag -o jsonpath='{.data.elastic}' | base64 -d)
echo "Elasticsearch password: $ES_PASSWORD"
```

:::{tip}
Save this password securely. The password is auto-generated by ECK and persists across pod restarts unless the secret is deleted.
:::

#### 5. Update Deployment with Credentials

Configure the RAG server and ingestor-server to use the retrieved credentials.

Edit [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) and set the following values for Elasticsearch authentication:

- **APP_VECTORSTORE_USERNAME:** set to `"elastic"` (the default Elasticsearch superuser).
- **APP_VECTORSTORE_PASSWORD:** set to the password retrieved in step 4.

Example (replace `your-retrieved-password` with your actual `$ES_PASSWORD`):

```yaml
# RAG Server configuration
envVars:
  APP_VECTORSTORE_URL: "http://rag-eck-elasticsearch-es-http:9200"
  APP_VECTORSTORE_NAME: "elasticsearch"
  APP_VECTORSTORE_USERNAME: "elastic"
  APP_VECTORSTORE_PASSWORD: "your-retrieved-password"   # use $ES_PASSWORD from step 4

# Ingestor Server configuration
ingestor-server:
  envVars:
    APP_VECTORSTORE_URL: "http://rag-eck-elasticsearch-es-http:9200"
    APP_VECTORSTORE_NAME: "elasticsearch"
    APP_VECTORSTORE_USERNAME: "elastic"
    APP_VECTORSTORE_PASSWORD: "your-retrieved-password"   # use $ES_PASSWORD from step 4
```

Then apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

#### 6. (Optional) Use API Key Authentication

For advanced use cases or production environments, you can use Elasticsearch API keys instead of username/password authentication.

**Generate an API Key:**

First, port-forward to access Elasticsearch:

```bash
kubectl port-forward -n rag svc/rag-eck-elasticsearch-es-http 9200:9200
```

Then generate an API key using the elastic user:

```bash
# Get the elastic password
ES_PASSWORD=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user -n rag -o jsonpath='{.data.elastic}' | base64 -d)

# Create an API key
curl -u elastic:$ES_PASSWORD \
  -X POST "http://localhost:9200/_security/api_key" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "rag-api-key",
    "role_descriptors": {}
  }'
```

Example response:
```json
{
  "id": "AbCdEfGhIj",
  "name": "rag-api-key",
  "api_key": "ZyXwVuTsRq"
}
```

**Encode the API Key:**

```bash
# Base64 encode the "id:api_key" format
echo -n "AbCdEfGhIj:ZyXwVuTsRq" | base64
# Output example: QWJDZEVmR2hJajpaeVh3VnVUc1Jx
```

**Configure with API Key in values.yaml:**

```yaml
# RAG Server configuration - Option 1: Base64 encoded API key
envVars:
  APP_VECTORSTORE_APIKEY: "QWJDZEVmR2hJajpaeVh3VnVUc1Jx"
  APP_VECTORSTORE_USERNAME: ""
  APP_VECTORSTORE_PASSWORD: ""

# Ingestor Server configuration - Option 1: Base64 encoded API key
ingestor-server:
  envVars:
    APP_VECTORSTORE_APIKEY: "QWJDZEVmR2hJajpaeVh3VnVUc1Jx"
    APP_VECTORSTORE_USERNAME: ""
    APP_VECTORSTORE_PASSWORD: ""
```

Or use split ID/SECRET format:

```yaml
# RAG Server configuration - Option 2: Split ID and secret
envVars:
  APP_VECTORSTORE_APIKEY_ID: "AbCdEfGhIj"
  APP_VECTORSTORE_APIKEY_SECRET: "ZyXwVuTsRq"
  APP_VECTORSTORE_USERNAME: ""
  APP_VECTORSTORE_PASSWORD: ""

# Ingestor Server configuration - Option 2: Split ID and secret
ingestor-server:
  envVars:
    APP_VECTORSTORE_APIKEY_ID: "AbCdEfGhIj"
    APP_VECTORSTORE_APIKEY_SECRET: "ZyXwVuTsRq"
    APP_VECTORSTORE_USERNAME: ""
    APP_VECTORSTORE_PASSWORD: ""
```

Then apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

:::{note}
**API Key vs Username/Password:**
- API keys are recommended for production environments and applications
- API keys can have specific permissions and expiration dates
- API keys can be rotated without changing the elastic user password
- **API key authentication takes precedence** when both username/password and API keys are configured
:::

#### 7. Verify Authentication

Test that the services can connect to Elasticsearch with authentication:

```bash
# Check ingestor-server logs for successful connection
kubectl logs -n rag -l app=ingestor-server --tail=20

# Test Elasticsearch connection manually
ES_PASSWORD=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user -n rag -o jsonpath='{.data.elastic}' | base64 -d)
kubectl exec -n rag rag-eck-elasticsearch-es-default-0 -- curl -s -u elastic:$ES_PASSWORD http://localhost:9200/_cluster/health
```

---

## Using VDB Auth Token at Runtime via APIs (Enterprise Feature)

When using Elasticsearch as the vector database, you can pass a per-request VDB authentication token via the HTTP `Authorization` header. The servers forward this token to Elasticsearch for that request. This enables per-user authentication or per-request scoping without changing server environment configuration.

Prerequisite:
- Ensure Elasticsearch authentication is enabled so security is enforced. In Elasticsearch this typically requires `xpack.security.enabled=true`. See the [Elasticsearch Authentication](#elasticsearch-authentication) section above for enabling security via Docker Compose or Helm and for obtaining API keys or setting credentials.

### Set Up Runtime Token and Endpoints

Before making API requests with authentication, export the required environment variables.

**1. Export service endpoints:**

```bash
export INGESTOR_URL="http://localhost:8082"
export RAG_URL="http://localhost:8081"
```

**2. Export authentication token:**

Runtime authentication via the `Authorization` header only supports Elasticsearch API keys. Export your API key token:

```bash
# Export your bearer token
export ES_VDB_TOKEN="your-bearer-token"
```

:::{note}
Bearer token authentication (OAuth/OIDC/SAML) is an enterprise support feature and not available in the free version of Elasticsearch. For most use cases, use Elasticsearch API keys as shown in [Get an Elasticsearch API Key](#get-an-elasticsearch-api-key) above.
:::

### Header Format

Use bearer authentication in your API requests:

```
Authorization: Bearer <token>
```

### Ingestor Server Examples

- List documents:

```bash
curl -G "$INGESTOR_URL/v1/documents" \
  -H "Authorization: Bearer ${ES_VDB_TOKEN}" \
  --data-urlencode "collection_name=es_demo_collection"
```

- Delete a collection:

```bash
curl -X DELETE "$INGESTOR_URL/v1/collections" \
  -H "Authorization: Bearer ${ES_VDB_TOKEN}" \
  --data-urlencode "collection_names=es_demo_collection"
```

:::{note}
You can also set `vdb_endpoint` in your request payload to override the configured `APP_VECTORSTORE_URL`.
:::

### RAG Server Examples

- Search:

```bash
curl -X POST "$RAG_URL/v1/search" \
  -H "Authorization: Bearer ${ES_VDB_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is vector search?",
    "use_knowledge_base": true,
    "collection_names": ["es_demo_collection"],
    "vdb_endpoint": "'"$APP_VECTORSTORE_URL"'",
    "reranker_top_k": 0,
    "vdb_top_k": 3
  }'
```

- Generate with streaming:

```bash
curl -N -X POST "$RAG_URL/v1/generate" \
  -H "Authorization: Bearer ${ES_VDB_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Give a short summary of vector databases"}],
    "use_knowledge_base": true,
    "collection_names": ["es_demo_collection"],
    "vdb_endpoint": "'"$APP_VECTORSTORE_URL"'",
    "reranker_top_k": 0,
    "vdb_top_k": 3
  }'
```

### Troubleshooting
- If you receive authentication/authorization errors from Elasticsearch, verify your token (API key validity, scopes, and expiration).
- Ensure the server is not also configured with conflicting credentials for the same request.
- Confirm that `APP_VECTORSTORE_NAME=elasticsearch` and `APP_VECTORSTORE_URL` are set correctly.

---

## Related Topics

- [Vector database configuration](change-vectordb.md) (default Elasticsearch setup, switching to Milvus)
- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [Troubleshoot](troubleshooting.md)

## Reference

- Elastic: [GPU vector indexing](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing)
- Blueprint: [Vector database configuration](change-vectordb.md)

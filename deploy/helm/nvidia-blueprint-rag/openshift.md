# OpenShift Deploy Runbook

## Prerequisites

- **OpenShift** 4.14+ with `oc` CLI configured
- **Helm** 3.x installed
- **NIM Operator** installed on the cluster (manages NIMCache / NIMService resources)
- **NGC API Key** — for pulling container images from `nvcr.io` and NIM model downloads. Get one at https://org.ngc.nvidia.com/setup/api-keys. The key must have access to `nvcr.io/nim/` (NIM models).
- **GPUs** — at least 4 GPUs (embedding 1 + reranking 1 + OCR 1 + page-elements 1). Disable graphic-elements and table-structure to save 2 GPUs, or enable them for full document processing (6 GPUs total).

## 1. Prepare

```bash
export NGC_API_KEY="nvapi-..."
export NAMESPACE="rag"
```

## 2. Build chart dependencies

The `nv-ingest` dependency is hosted on the NGC Helm repository:

```bash
cd deploy/helm/nvidia-blueprint-rag

helm repo add nvidia-nemo https://helm.ngc.nvidia.com/nvidia/nemo-microservices \
  --username '$oauthtoken' --password "$NGC_API_KEY"

helm dependency build
```

> **Note**: The upstream `Chart.yaml` may reference the `nvstaging` repo, which is restricted. If `helm dependency build` fails with 403, update the `nv-ingest` entry in `Chart.yaml` to use `https://helm.ngc.nvidia.com/nvidia/nemo-microservices` with version `26.3.0`.

## 3. Create namespace

```bash
oc new-project $NAMESPACE
```

## 4. Install

```bash
helm upgrade --install rag -n $NAMESPACE . \
  -f values-openshift.yaml \
  -f values-openshift-test.yaml \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY" \
  --timeout 15m
```

The chart automatically creates:
- **NGC image pull secret** and **NGC API secret** from the `--set` values
- **OpenShift Routes** for the frontend and rag-server (with edge TLS)
- **anyuid SCC RoleBinding** for all service accounts that need it

The `values-openshift-test.yaml` overlay adds:
- GPU tolerations for your cluster's taint keys (edit the toleration key to match your nodes)
- Reduced resource requests for ingestor-server and nv-ingest runtime
- Disabled self-hosted LLM NIM (uses NVIDIA API Catalog instead)
- Disabled observability stack (Elasticsearch, Zipkin, Prometheus)
- GA nv-ingest image override (from `nvstaging` to public registry)

## 5. Link pull secret to NIM Operator service account

The NIM Operator creates a `nim-cache-sa` ServiceAccount for model cache jobs. Link the NGC pull secret so it can pull NIM model images:

```bash
oc secrets link nim-cache-sa ngc-secret --for=pull -n $NAMESPACE
```

If NIMCache pods are stuck in `ImagePullBackOff`, delete them so the operator recreates them with the linked secret:

```bash
oc delete pod -l app.nvidia.com/nim-cache -n $NAMESPACE
```

## 6. Verify

```bash
# All pods
oc get pods -n $NAMESPACE

# NIMCache status (all should reach Ready)
oc get nimcache -n $NAMESPACE

# Routes
oc get routes -n $NAMESPACE

# Frontend URL
echo "https://$(oc get route rag-frontend -n $NAMESPACE -o jsonpath='{.spec.host}')"

# API health
API_HOST=$(oc get route rag-server -n $NAMESPACE -o jsonpath='{.spec.host}')
curl -sk "https://${API_HOST}/health"
```

NIMCache pods download model weights on first deploy and may take 5–10 minutes to reach `Ready`. NIMService pods start after their cache is ready.

## 7. Uninstall

```bash
helm uninstall rag -n $NAMESPACE

# NIMCache resources are retained by resource policy — delete manually
oc delete nimcache --all -n $NAMESPACE
oc delete pvc --all -n $NAMESPACE
```

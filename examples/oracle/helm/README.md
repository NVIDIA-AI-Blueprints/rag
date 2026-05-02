# Helm Deployment With Oracle 26ai

The Helm path is the primary Kubernetes deployment path for the Oracle branch.
By default it provisions a fresh Oracle Autonomous AI Database 26ai, bootstraps
`RAG_APP`, creates `oracle-creds`, and deploys the RAG Blueprint configured for
Oracle hybrid search.

## Prerequisites

- `kubectl` points at the target OKE cluster
- `helm` is installed
- OCI CLI config exists on the machine running Helm
- OCI identity has permission to inspect OKE/networking and create ADB
- NGC API key for self-hosted NIM image pulls and model downloads

Create the OCI config secret:

```bash
kubectl create secret generic oci-config \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem
```

## Create ADB Automatically

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --timeout 60m
```

If your cluster already has `ngc-api` and `ngc-secret`, use the same stock
secret names and disable creation:

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set imagePullSecret.create=false \
  --set imagePullSecret.name=ngc-secret \
  --set ngcApiSecret.create=false \
  --set ngcApiSecret.name=ngc-api \
  --timeout 60m
```

The provisioner auto-discovers the current OKE cluster, VCN, and a private
subnet from the active Kubernetes context. If discovery fails, provide
overrides:

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set oracle.createDatabase.compartmentId=<compartment_ocid> \
  --set oracle.createDatabase.subnetId=<private_subnet_ocid> \
  --timeout 60m
```

## Use Existing ADB

```bash
helm install rag <nvidia-rag-chart> \
  -f examples/oracle/helm/values.existing-adb.yaml \
  --timeout 60m
```

Before installing, create `oracle-creds` with `ORACLE_USER`,
`ORACLE_PASSWORD`, and `ORACLE_CS`.

## Change Models

The Oracle wrapper chart keeps model configuration as close to the stock
NVIDIA RAG chart as possible. Use the **same values keys** from
`deploy/helm/nvidia-blueprint-rag/values.yaml`, but nest them under `rag:`
because this chart wraps the stock chart as a dependency.

Stock RAG:

```yaml
envVars:
  APP_LLM_MODELNAME: "nvidia/nemotron-3-super-120b-a12b"
nimOperator:
  nim-llm:
    image:
      repository: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b
      tag: "latest"
```

Oracle wrapper:

```yaml
rag:
  envVars:
    APP_LLM_MODELNAME: "nvidia/nemotron-3-super-120b-a12b"
  nimOperator:
    nim-llm:
      image:
        repository: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b
        tag: "latest"
```

Example:

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  -f examples/oracle/helm/profiles/nemotron3-super-values.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --timeout 90m
```

See `profiles/README.md` for more model-switching examples.

## Uninstall Behavior

`helm uninstall rag` removes the RAG application, but does **not** delete the
ADB by default. This prevents accidental loss of enterprise data.

Use explicit cleanup tooling if the database should be deleted for a dev/test
environment.

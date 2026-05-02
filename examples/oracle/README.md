# Oracle 26ai Deployment Examples

This directory contains Oracle-first deployment examples for the NVIDIA RAG
Blueprint `oracle` branch.

## Default Experience

Create a fresh Oracle Autonomous AI Database 26ai automatically, bootstrap the
`RAG_APP` schema, create Kubernetes/Docker secrets, and deploy RAG configured to
use Oracle as the primary vector database.

Recommended order:

1. [Helm: create ADB](helm/README.md) — primary Kubernetes path
2. [Docker Compose: create ADB](docker/README.md) — primary local/dev path
3. [Existing ADB](values.existing-adb.yaml) — advanced/BYO path

## Deployment Modes

| Mode | Command | Creates ADB? | Intended user |
|---|---|---:|---|
| Helm create-ADB | `helm install rag examples/oracle/helm -f examples/oracle/helm/values.create-adb.yaml --set imagePullSecret.password=$NGC_API_KEY --set ngcApiSecret.password=$NGC_API_KEY` | yes | Kubernetes / OKE users |
| Docker create-ADB | `docker login nvcr.io -u '$oauthtoken' -p \"$NGC_API_KEY\" && docker compose -f examples/oracle/docker/docker-compose.create-adb.yaml up` | yes | local developers |
| Helm existing-ADB | `helm install rag ... -f examples/oracle/helm/values.existing-adb.yaml` | no | users with an existing ADB |
| Docker existing-ADB | `docker compose -f examples/oracle/docker/docker-compose.existing-adb.yaml up` | no | users with an existing ADB |

## Safety Defaults

- Fresh ADB creation is the default for Oracle branch examples.
- `helm uninstall` does **not** delete the ADB by default.
- The provisioner is idempotent: if the DB/user already exists, it repairs
  grants and secrets rather than dropping data.
- `bootstrap.sql` never drops `RAG_APP` or existing collections.

## OCI Authentication

Helm and Docker create-ADB examples require OCI credentials because they call
OCI APIs to create Autonomous Database.

For Helm, create a Kubernetes Secret:

```bash
kubectl create secret generic oci-config \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem
```

For Docker Compose, mount `~/.oci` into the provisioner container.

Production one-click deployments should use the OCI Resource Manager
Accelerator Pack, which runs Terraform inside OCI and does not require the
customer to install OCI CLI locally.

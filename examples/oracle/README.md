# Oracle Database 26ai adapter

This adapter connects the stock NVIDIA RAG Blueprint to Oracle Database 26ai
and Oracle Private AI Services (PAI) with cuVS. It keeps the NVIDIA Helm chart
unchanged: install the stock chart, create one short-lived Kubernetes Secret,
and apply one additional manifest.

The resulting Oracle-backed RAG and PAI services can be used by separately
deployed NVIDIA AI-Q and VSS Blueprint configurations. This adapter does not
install or modify AI-Q or VSS.

This version is for the NVIDIA RAG Blueprint 2.6.2 repository release and its
published `nvidia-blueprint-rag-v2.6.0.tgz` Helm chart. The standard path below
creates a new Oracle Base Database Service 26ai DB system with 2 OCPUs. It is a
billable OCI resource and is not deleted automatically.

## Prerequisites

Complete the [stock Helm prerequisites](../../docs/deploy-helm.md), including
the NVIDIA GPU and NIM operators, ECK, and a default storage class. The adapter
also requires:

- An AMD64 Oracle Kubernetes Engine (OKE) cluster that satisfies the stock RAG
  requirements and has one additional schedulable NVIDIA GPU for PAI.
- OCI API-key credentials with permission to discover the OKE network and
  create a Base Database Service DB system and its required network rules.
- Oracle Container Registry credentials with the terms accepted for the
  `database/private-ai` image.
- `helm` and `kubectl` configured for the target cluster.

You do not provide a DB system ID, GPU node, storage class, or application host
name for this standard path. The adapter discovers OKE placement, uses the
cluster's default scheduling and storage behavior, and creates the database.

## Deploy

Set the credential inputs below. The OCI config and profile use their standard
defaults; `OCI_PRIVATE_KEY_FILE` is the absolute path in the `key_file` entry of
the selected profile.

```bash
export NGC_API_KEY='<ngc-api-key>'
export ORACLE_REGISTRY_USERNAME='<oracle-registry-username>'
export ORACLE_REGISTRY_TOKEN='<oracle-registry-auth-token>'
export OCI_CONFIG_FILE="${OCI_CONFIG_FILE:-${HOME}/.oci/config}"
export OCI_PROFILE="${OCI_PROFILE:-DEFAULT}"
export OCI_PRIVATE_KEY_FILE='<absolute-path-to-oci-api-private-key>'
```

Install the unmodified stock NVIDIA chart in the required `rag` namespace.

```bash
helm upgrade --install rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.6.0.tgz \
  --namespace rag \
  --create-namespace \
  --username '$oauthtoken' \
  --password "$NGC_API_KEY" \
  --set-string imagePullSecret.password="$NGC_API_KEY",ngcApiSecret.password="$NGC_API_KEY"
```

Create the bootstrap Secret and apply the adapter once from the repository
root. The bootstrap deletes this input Secret after it has generated the
runtime credentials and completed successfully.

```bash
kubectl -n rag create secret generic oracle26ai-bootstrap-inputs \
  --from-file=config="$OCI_CONFIG_FILE" \
  --from-file=oci_api_key.pem="$OCI_PRIVATE_KEY_FILE" \
  --from-literal=OCI_PROFILE="$OCI_PROFILE" \
  --from-literal=ORACLE_REGISTRY_USERNAME="$ORACLE_REGISTRY_USERNAME" \
  --from-literal=ORACLE_REGISTRY_TOKEN="$ORACLE_REGISTRY_TOKEN"

kubectl -n rag apply -f examples/oracle/oracle26ai-rag-extension.yaml
```

No separate chart download or checksum command is needed: the adapter is
versioned in the same Git commit as these instructions. Pin that commit for a
repeatable production deployment.

## Verify

Database provisioning can take several hours. The reconciler deletes its own
Job only after PAI, the patched RAG services, and their health endpoints pass.
A failed reconciler Job remains in the namespace for inspection.

```bash
kubectl -n rag wait \
  --for=condition=complete job/oracle26ai-platform-bootstrap \
  --timeout=10h
kubectl -n rag wait \
  --for=delete job/oracle-dbcs-extension-reconcile \
  --timeout=8h
kubectl -n rag rollout status deployment/oracle26ai-pai --timeout=30m
kubectl -n rag rollout status deployment/rag-server --timeout=30m
kubectl -n rag rollout status deployment/ingestor-server --timeout=30m
kubectl -n rag get pods,svc
```

Use the stock RAG UI or API to create a collection, ingest a document, and ask
a question. Those unchanged workflows now store and search vectors in Oracle
Database 26ai and use PAI/cuVS for index offload.

## Remove the deployment

Record the DB system OCID before removing the Kubernetes resources:

```bash
kubectl -n rag get configmap oracle26ai-platform-state \
  -o jsonpath='{.data.dbSystemId}{"\n"}'
```

Terminate that DB system explicitly in OCI when its data is no longer needed;
deleting Kubernetes resources does not terminate it. Then remove the adapter
and the stock release:

```bash
kubectl -n rag delete -f examples/oracle/oracle26ai-rag-extension.yaml \
  --ignore-not-found
helm uninstall rag --namespace rag
kubectl delete namespace rag
```

This manifest redistributes the unmodified `python-oracledb` 3.4.2 wheel
(SHA-256 `b26a10f9c790bd141ffc8af68520803ed4a44a9258bf7d1eea9bfdd36bd6df7f`),
dual-licensed under UPL-1.0 or Apache-2.0. Copyright (c) 2016, 2025, Oracle
and/or its affiliates. The wheel retains its `LICENSE.txt`, `NOTICE.txt`, and
`THIRD_PARTY_LICENSES.txt` files byte-for-byte. No credentials, OCI identifiers,
database wallets, or private keys are included in this repository.

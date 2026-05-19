# NVIDIA RAG Blueprint with Oracle Database 26AI with cuVS

The Helm path is the primary Kubernetes deployment path for the Oracle branch.
A single `helm install` provisions a fresh Oracle Autonomous AI Database 26ai,
bootstraps `RAG_APP`, deploys the NVIDIA RAG Blueprint (with **Nemotron 3 Super
120B** as the default LLM) and Oracle hybrid search, **and** deploys the Oracle
Private AI Services container so that `CREATE VECTOR INDEX` builds run on the
GPU through NVIDIA cuVS instead of on the database CPU.

**No custom images needed.** Stock NGC images are used for rag-server and
ingestor-server; the Oracle Python deps (`oracledb`, `langchain-oracledb`)
are installed automatically at pod startup via init containers.

## What runs on the GPU

The chart defaults to **HNSW** vector indexes (not IVF) because Oracle 26ai's
`OFFLOAD_URL` parameter only applies to HNSW. When `oracle.gpuIndexOffload.enabled`
is true (the default), every HNSW `CREATE VECTOR INDEX` issued by `rag-server`
or `ingestor-server` includes
`PARAMETERS (TYPE HNSW, …, OFFLOAD_URL '…/v1/index')`, which causes the database
to stream vectors over HTTP/2 to the `oracle-pai-gpu-index` service, where cuVS
builds the graph on the GPU and returns it. Embedding generation, similarity
search, and the LLM still use the existing NVIDIA NIMs.

### How ADB reaches the GPU service

ADB is a managed OCI service in your VCN — it **cannot resolve cluster.local
DNS or route to a Kubernetes ClusterIP**. The chart therefore exposes the
`oracle-pai-gpu-index` Service as an **OCI internal LoadBalancer** (annotation
`service.beta.kubernetes.io/oci-load-balancer-internal: "true"`) which gets a
stable VCN-routable IP from the OKE service-LB subnet. A post-install Job
(`oracle-pai-verify`) waits for that LB IP, patches `oracle-creds` so
`ORACLE_PAI_INDEX_URL` is `http://<lb-ip>:8080/v1/index`, then rolls
`rag-server` and `ingestor-server` so they re-read `envFrom: oracle-creds`.
After that, the in-database `OFFLOAD_URL` points at a real, routable address.

### OCI networking (zero-touch when the provisioner has the right policy)

The provisioner Job opens the OCI security path automatically:

1. Reads the ADB private endpoint subnet CIDR from the ADB it just created.
2. Looks up the OKE cluster in the same VCN and reads its
   `service_lb_subnet_ids`.
3. Adds an idempotent ingress rule on the LB subnet's default security list:
   `TCP 8080 from <ADB CIDR>`.

If the dynamic group / OCI policy attached to the provisioner does not allow
`manage virtual-network-family`, the helper falls back to printing the exact
`oci network security-list update` command in the verify Job log. To enable
the auto-add path, ensure the provisioner identity has at least:

```
Allow dynamic-group rag-provisioner to manage autonomous-database-family in <compartment>
Allow dynamic-group rag-provisioner to use virtual-network-family in <compartment>
Allow dynamic-group rag-provisioner to read clusters in <compartment>
```

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

## Fetch Helm chart dependencies

The wrapper chart depends on `gpu-operator`, `k8s-nim-operator`, and the
`nvidia-blueprint-rag` sub-chart. **None of these tarballs are bundled
in this repo** — pull them from the listed registries before installing:

```bash
# Materialise the upstream RAG chart's deps (referenced via file:// from
# the wrapper chart; helm dep update is not recursive, so do this first).
helm dependency update deploy/helm/nvidia-blueprint-rag

# Then materialise the wrapper chart's own deps.
helm dependency update examples/oracle/helm
```

Re-run these whenever you bump a `version:` in `Chart.yaml`. The resulting
`charts/*.tgz` files are git-ignored.

## Create ADB Automatically

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set oracle.containerRegistry.username=$ORACLE_SSO_EMAIL \
  --set oracle.containerRegistry.password=$ORACLE_OCR_TOKEN \
  --timeout 60m
```

This single install:

- provisions a fresh Oracle Autonomous AI Database 26ai in the same VCN as
  your OKE cluster (pre-install hook auto-discovers the worker subnet),
- bootstraps `RAG_APP` and writes `oracle-creds`,
- deploys the NVIDIA RAG Blueprint configured for Oracle hybrid search,
- deploys the Oracle Private AI Services container (`gpu-index` variant) on a
  GPU node and exposes it as an OCI internal LoadBalancer, **and**
- runs a post-install Job (`oracle-pai-verify`) that:
  1. waits for the LoadBalancer ingress IP,
  2. health-checks PAI via the in-cluster sidecar Service,
  3. patches `oracle-creds` with `ORACLE_PAI_INDEX_URL=http://<lb-ip>:8080/v1/index`,
  4. rolls `rag-server` and `ingestor-server` to re-read the secret,
  5. fails the install if any of the above breaks, so a broken integration
     never silently degrades to a CPU index build.

Generate `$ORACLE_OCR_TOKEN` once at <https://container-registry.oracle.com>:

1. Sign in with your Oracle SSO.
2. Open **Database → Private AI** and accept the license (the pull is denied
   without acceptance, even with a valid token).
3. **Profile → Auth Token → Generate Secret Key**. Copy it once.

To opt out of GPU index offload (CPU-only HNSW build, much slower on large
corpora), add `--set oracle.gpuIndexOffload.enabled=false`.

By default the PAI pod schedules onto **any** GPU node (cuVS works on any
NVIDIA GPU with compute capability ≥ 7.5). To pin to a specific shape (for
example to keep H100s free for the LLM):

```bash
--set 'oracle.gpuIndexOffload.nodeSelector.nvidia\.com/gpu\.product=NVIDIA-A100-SXM4-80GB'
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

If you already have an Oracle Autonomous AI Database 26ai (or 23ai)
reachable from the OKE cluster, skip the provisioner and connect directly:

**1. Create the oracle-creds Secret:**

```bash
kubectl create secret generic oracle-creds \
  --from-literal=ORACLE_USER=RAG_APP \
  --from-literal=ORACLE_PASSWORD='<your-password>' \
  --from-literal=ORACLE_CS='(description=(address=(protocol=tcps)(port=1521)(host=<your-adb-host>))(connect_data=(service_name=<service>_medium.adb.oraclecloud.com))(security=(ssl_server_dn_match=no)))'
```

**2. Install:**

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.existing-adb.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set oracle.containerRegistry.username=$ORACLE_SSO_EMAIL \
  --set oracle.containerRegistry.password=$ORACLE_OCR_TOKEN \
  --set oracle.credentials.adminPassword='<admin-pw>' \
  --set oracle.credentials.appPassword='<app-pw>' \
  --timeout 60m
```

This skips the ADB provisioner Job but still deploys PAI cuVS, the RAG
blueprint, and runs the verify Job to wire `ORACLE_PAI_INDEX_URL`.

If your cluster already has `gpu-operator` and `k8s-nim-operator`, add:
`--set operators.gpu.enabled=false --set operators.nim.enabled=false`

## Cross-Region / Cross-Tenancy ADB

The RAG Blueprint can connect to **any** Oracle ADB 23ai/26ai regardless of
where it runs — same VCN, different region, different tenancy, or even
on-premises Oracle 26ai. The adapter is connection-string agnostic.

| ADB location | Connection mode | cuVS GPU offload |
|---|---|---|
| Same VCN, private endpoint | Walletless TLS (simplest) | YES — internal LB (default) |
| Same VCN, public endpoint | mTLS with wallet | YES — internal LB (default) |
| Different region | mTLS with wallet over internet | YES — set LB to public |
| Different tenancy | mTLS with wallet over internet | YES — set LB to public |
| On-premises Oracle 26ai | TLS or mTLS depending on network | YES — set LB to public |

cuVS GPU-accelerated index building works in **every** scenario. For
cross-region / cross-tenancy, expose the PAI LoadBalancer as public:
```bash
--set oracle.gpuIndexOffload.service.internal=false
```

For cross-region / cross-tenancy, use `values.existing-adb.yaml` with:

1. **Download the wallet** from the ADB's OCI Console (any tenancy).
2. Create the wallet Secret:
   ```bash
   unzip Wallet_test1.zip -d /tmp/wallet
   kubectl create secret generic oracle-wallet --from-file=/tmp/wallet/
   ```
3. Create oracle-creds with TNS alias + wallet password:
   ```bash
   kubectl create secret generic oracle-creds \
     --from-literal=ORACLE_USER=RAG_APP \
     --from-literal=ORACLE_PASSWORD='<password>' \
     --from-literal=ORACLE_CS='test1_medium' \
     --from-literal=ORACLE_WALLET_PASSWORD='<wallet download password>'
   ```
4. Uncomment the `oracle-wallet` volume sections in `values.existing-adb.yaml`.
5. Add `TNS_ADMIN: "/app/wallet"` to both `rag.envVars` and
   `rag.ingestor-server.envVars`.
6. Expose the PAI LB as public with authentication + HTTPS (enterprise
   security for data traversing the internet):
   ```bash
   --set oracle.gpuIndexOffload.service.internal=false
   --set oracle.gpuIndexOffload.auth=true
   --set oracle.gpuIndexOffload.https=true
   ```
7. Install:
   ```bash
   helm install rag examples/oracle/helm \
     -f examples/oracle/helm/values.existing-adb.yaml \
     --set operators.gpu.enabled=false \
     --set operators.nim.enabled=false \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     --set oracle.containerRegistry.username=$ORACLE_SSO_EMAIL \
     --set oracle.containerRegistry.password=$ORACLE_OCR_TOKEN \
     --set oracle.gpuIndexOffload.service.internal=false \
     --set oracle.gpuIndexOffload.auth=true \
     --set oracle.gpuIndexOffload.https=true \
     --timeout 60m
   ```

cuVS GPU-accelerated HNSW index building works end-to-end. The remote
ADB connects to the public LB IP via **HTTPS with authentication**
for `OFFLOAD_URL`. The verify Job patches `ORACLE_PAI_INDEX_URL` with
the public IP automatically. The `OFFLOAD_CREDENTIAL_NAME` on the
ADB side authenticates the offload connection to PAI.

**Security model for cross-region cuVS:**

| Layer | Protection |
|---|---|
| Transport | HTTPS/TLS 1.3 (PAI `PRIVATE_AI_HTTPS_ENABLED=true`) |
| Authentication | PAI API auth (`PRIVATE_AI_AUTHENTICATION_ENABLED=true`) + ADB `OFFLOAD_CREDENTIAL_NAME` |
| Network | OCI NSG / security list on the LB subnet restricts source CIDRs |
| At rest | Oracle ADB TDE encryption (platform-managed) |

For same-VCN deployments (internal LB), HTTP without auth is acceptable
because traffic never leaves the VCN and is governed by OCI security
lists / NSGs.

## Bringing your own data (BYO collections)

When `oracle.mode=existing`, the chart runs a post-install Job
(`oracle-byo-import`) that exposes pre-existing customer tables to the RAG
frontend the same way uploaded collections appear. Two passes:

**1. Discovery (always runs in existing-ADB mode).** The Job scans
`user_tab_columns` for any object with a `VECTOR` column. For each one
that already matches the RAG canonical shape — columns
`(id, text, vector, source, content_metadata)` — it idempotently registers
the table with `metadata_schema` and `document_info` so the UI lists it
with accurate counts and a `byo` tag. Tables that don't match the shape
are logged with the exact `values.yaml` snippet you can paste in to expose
them via a SQL view.

**2. Mapping (when `oracle.importExistingTables` is set).** For each
entry, the Job creates `CREATE OR REPLACE VIEW <collectionName> AS …` that
projects your columns into the RAG canonical shape. Views are read-only
by Oracle's DML rules, so your base tables are never mutated by the RAG
ingestor or delete-by-source flow.

```yaml
# values.existing-adb.yaml override
oracle:
  importExistingTables:
    - sourceTable: KB_USER.MY_DOCS    # schema-qualified or bare table
      collectionName: my_kb_view      # name to expose to the frontend
      columns:
        text: content                 # required: text content
        vector: embedding             # required: VECTOR column
        source: source_url            # optional: filename / URL
        sourceWrapJson: true          # wrap into JSON_OBJECT (default)
        contentMetadata: meta_json    # optional: JSON metadata column
        id: doc_id                    # optional; defaults to ROWID
```

After install:
* `GET /collections` lists every BYO collection alongside ingested ones.
* The frontend's collection picker shows them with no special handling.
* `read_only: true` is surfaced in the collection metadata so the UI can
  hide upload/delete buttons for SQL views and non-canonical tables.
* Search and chat work end-to-end against BYO collections (provided
  the embedding model matches the dimension of the existing vectors).

The Job never `DROP`s, `TRUNCATE`s, or `DELETE`s anything in the customer
schema; it only adds tracking rows and creates views.

## Change Models

The default LLM is **Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b`,
FP8, TP=2, 2 GPUs). To switch to a different model, override the values
nested under `rag:` (because this chart wraps the stock chart as a dependency).

**Switch to the smaller 49B model** (single GPU, faster startup):

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set rag.envVars.APP_LLM_MODELNAME=nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --set rag.envVars.APP_QUERYREWRITER_MODELNAME=nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --set rag.envVars.APP_FILTEREXPRESSIONGENERATOR_MODELNAME=nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --set rag.envVars.REFLECTION_LLM=nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --set rag.nimOperator.nim-llm.image.repository=nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --set rag.nimOperator.nim-llm.image.tag=1.14.0 \
  --set rag.nimOperator.nim-llm.model.tensorParallelism=1 \
  ...other flags...
```

**General pattern** — use the same values keys from the stock chart
(`deploy/helm/nvidia-blueprint-rag/values.yaml`), nested under `rag:`:

```yaml
rag:
  envVars:
    APP_LLM_MODELNAME: "nvidia/<your-model>"
  nimOperator:
    nim-llm:
      image:
        repository: nvcr.io/nim/nvidia/<your-model>
        tag: "<version>"
      model:
        tensorParallelism: "<tp>"
```

Pre-built profiles for common model swaps are in `profiles/`. See
`profiles/README.md` for details.

## Air-Gap / Restricted Egress Clusters

The init containers that install `oracledb` + `langchain-oracledb` into the
stock NGC images require outbound access to PyPI. For air-gap environments:

1. **Pre-download wheels** on a machine with internet:
   ```bash
   pip download --dest ./oracle-wheels oracledb==3.4.2 langchain-oracledb==1.3.0
   ```
2. Host the wheels on an internal PyPI mirror or HTTP server.
3. Override the init container args to point at your mirror:
   ```yaml
   --set rag.initContainers[0].args='{install,--target=/opt/oracle-deps,--no-cache-dir,--index-url=https://pypi.internal/simple,oracledb==3.4.2,langchain-oracledb==1.3.0}'
   ```

Alternatively, build custom images with `--extra oracle` baked in (see
`src/nvidia_rag/rag_server/Dockerfile`) and override `rag.image.repository`.

## Credential Rotation

To rotate the ADB password without a full redeploy:

1. Connect to ADB as ADMIN and run `ALTER USER RAG_APP IDENTIFIED BY "<new>";`
2. Update the Secret:
   ```bash
   kubectl patch secret oracle-creds -p '{"stringData":{"ORACLE_PASSWORD":"<new>"}}'
   ```
3. Roll the consumers:
   ```bash
   kubectl rollout restart deploy/rag-server deploy/ingestor-server
   ```

Pods re-read `envFrom: oracle-creds` on restart. No Helm upgrade needed.

## Supported Oracle Database Editions

The adapter works with **any** Oracle Database that has the `VECTOR` data
type (23ai or 26ai). Feature availability varies by edition:

| Deployment | cuVS GPU index offload | Hybrid search (Oracle Text) | Auto-provision via Helm |
|---|---|---|---|
| OCI ADB Serverless (default) | YES | YES | YES |
| OCI ADB Dedicated | YES | YES | NO — use existing-ADB mode |
| Oracle Base Database Service (OCI VM) | YES (Enterprise Ed.) | YES (Enterprise Ed.) | NO |
| Oracle Database@Azure / @AWS | YES (Enterprise Ed.) | YES (Enterprise Ed.) | NO |
| ExaDB-D / Exadata Cloud | YES | YES | NO |
| Containerized Oracle 26ai Enterprise (K8s) | YES | YES | NO |
| Containerized Oracle 23ai Free (K8s) | Dense search only | NO | NO |
| On-premises Oracle 26ai Enterprise | YES | YES | NO |

For any non-ADB deployment, use `values.existing-adb.yaml` and set
`ORACLE_CS` to the appropriate connection string:

- **In-cluster container**: `oracle-db.default.svc:1521/FREEPDB1` (no TLS, no wallet)
- **OCI Base DB / Exadata**: DSN descriptor with TLS or mTLS
- **Oracle Database@Azure**: connection string from the Azure portal
- **On-premises**: standard TNS descriptor or Easy Connect Plus

The auto-provisioning path (`values.create-adb.yaml`) is specific to OCI
ADB Serverless. All other editions use the existing-DB path, which is
the same adapter, same RAG pipeline, same cuVS — just a different
`ORACLE_CS`.

## Enterprise Security

| Concern | How this integration handles it |
|---|---|
| **Data at rest** | Oracle ADB uses Transparent Data Encryption (TDE) — all data, vectors, and indexes are encrypted on disk. Platform-managed keys by default; customer-managed keys (BYOK) optional via OCI Vault. |
| **Data in transit (DB)** | All connections use TLS 1.3 (walletless) or mTLS (wallet). No unencrypted DB traffic. |
| **Data in transit (cuVS)** | Same-VCN: HTTP on internal LB (traffic stays in VCN, governed by NSGs). Cross-region: HTTPS + PAI authentication enabled via `oracle.gpuIndexOffload.https=true` and `auth=true`. |
| **Credentials** | Stored in Kubernetes Secrets (`oracle-creds`, `oci-config`), never in ConfigMaps or logs. `provision_adb.py` explicitly avoids printing passwords. |
| **RBAC** | Provisioner and verify Jobs use dedicated ServiceAccounts with namespace-scoped Roles. No cluster-admin. Secret access limited to `oracle-creds` operations. |
| **Pod security** | All Oracle Jobs run with `runAsNonRoot: true`, `runAsUser: 1000`, `seccompProfile: RuntimeDefault`. No privileged containers. |
| **SQL injection** | Every table/collection name is validated against Oracle identifier rules before SQL interpolation. All user-supplied query text uses bind variables. BYO column mappings validated per-identifier. |
| **Network isolation** | PAI internal LB is VCN-scoped by default. ADB private endpoint restricts access to the VCN CIDR. NSG / security list rules limit ingress to TCP 1522 (ADB) and TCP 8080 (PAI) from known CIDRs. |
| **Credential rotation** | ADB password rotated via `ALTER USER` + `kubectl patch secret` + `kubectl rollout restart` — no full redeploy. See "Credential Rotation" above. |
| **Uninstall safety** | `helm uninstall` does NOT delete the ADB. Enterprise data is preserved. Explicit OCI CLI / Console deletion required. |
| **Supply chain** | Init container pip installs use pinned versions (`oracledb==3.4.2`, `langchain-oracledb==1.3.0`). Helm Job deps pinned with version bounds. |

## Uninstall Behavior

`helm uninstall rag` removes the RAG application, but does **not** delete the
ADB by default. This prevents accidental loss of enterprise data.

Use explicit cleanup tooling if the database should be deleted for a dev/test
environment.

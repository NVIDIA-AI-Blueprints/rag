# Oracle 26ai Vector Store

The Oracle branch of the NVIDIA RAG Blueprint uses Oracle Database 23ai/26ai
as the vector database backend. The adapter works with **any** Oracle Database
that has the native `VECTOR` data type — OCI Autonomous Database, Base Database
Service, Exadata, Oracle Database@Azure/@AWS, containerized Oracle on
Kubernetes, or on-premises Oracle 26ai Enterprise. It provides IVF/HNSW vector
indexes, cuVS GPU-accelerated index building (via Oracle Private AI Services),
and integrated keyword search through `CTXSYS.CONTEXT` for hybrid dense +
sparse retrieval.

## Deployment Paths

Start here:

1. **Helm + create fresh ADB 26ai**: recommended Kubernetes path. See
   `examples/oracle/helm/`.
2. **Docker Compose + create fresh ADB 26ai**: recommended local/dev path. See
   `examples/oracle/docker/`.
3. **BYO existing ADB**: advanced path for users who already have an Oracle
   Autonomous AI Database. See `examples/oracle/values.existing-adb.yaml` and
   `examples/oracle/docker/docker-compose.existing-adb.yaml`.

The Helm and Docker create-ADB flows both use the same provisioner container
from `examples/oracle/provisioner/`. The provisioner creates ADB, waits for it
to become available, creates/updates `RAG_APP`, and writes the connection
settings consumed by `rag-server` and `ingestor-server`.

## Quick Start: Helm Create-ADB

Prerequisites:

- OCI CLI configured on the workstation running Helm
- `kubectl` pointed at the target OKE cluster
- `helm` ≥ 3.14 installed
- permissions to inspect OKE/network resources and create Autonomous Database
- NGC API key for self-hosted NIM images and model downloads

Create an OCI config secret, fetch chart dependencies (`gpu-operator`,
`k8s-nim-operator`, `nvidia-blueprint-rag` — pulled from public
registries; **not** bundled in this repo), then install the Oracle
wrapper chart:

```bash
kubectl create secret generic oci-config \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem

# Fetch chart deps (do the upstream chart first because helm dep update
# is not recursive across file:// references).
helm dependency update deploy/helm/nvidia-blueprint-rag
helm dependency update examples/oracle/helm

helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --timeout 60m
```

By default the provisioner auto-discovers the current OKE cluster, VCN, and
private subnet from the active Kubernetes context. If auto-discovery fails,
provide explicit overrides:

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  --set oracle.createDatabase.compartmentId=<compartment_ocid> \
  --set oracle.createDatabase.subnetId=<private_subnet_ocid> \
  --timeout 60m
```

## Quick Start: Docker Compose Create-ADB

```bash
export OCI_COMPARTMENT_OCID=<compartment_ocid>
export OCI_SUBNET_OCID=<private_subnet_ocid>

docker compose -f examples/oracle/docker/docker-compose.create-adb.yaml up
```

The Docker provisioner writes `examples/oracle/generated/oracle.env`, which is
used by the RAG services.

## Existing ADB Mode

If you already have an ADB 23ai/26ai instance reachable from the cluster
running `rag-server` and `ingestor-server`, provision a `RAG_APP` user with the
required grants:

```sql
CREATE USER rag_app IDENTIFIED BY "<your-password>";
GRANT CONNECT, RESOURCE TO rag_app;
GRANT UNLIMITED TABLESPACE TO rag_app;
GRANT CTXAPP TO rag_app;                          -- required for hybrid (Oracle Text)
GRANT EXECUTE ON CTXSYS.CTX_QUERY TO rag_app;     -- required for hybrid
```

### 2. Connection options

The plugin supports two connection modes and works with **any** reachable
Oracle ADB — same VCN, different region, different tenancy, or on-premises:

| Mode | Customer experience | Recommended for |
|---|---|---|
| **mTLS (wallet)** | download wallet zip from OCI → mount as K8s secret → set `TNS_ADMIN` + `ORACLE_WALLET_PASSWORD` | public-endpoint ADB, cross-region, cross-tenancy |
| **TLS (walletless)** | just user / password / DSN — no wallet | private-endpoint ADB in same VCN, simplest ops |

Both go over TLS 1.3. The difference is whether the *client* presents a
certificate. For **cross-region or cross-tenancy** ADB, mTLS with wallet
is required (the ADB is reachable over the internet). cuVS GPU index
offload still works — expose the PAI LoadBalancer as public with
`--set oracle.gpuIndexOffload.service.internal=false`. See
`examples/oracle/helm/README.md` → "Cross-Region / Cross-Tenancy ADB"
for the full Helm walkthrough.

### 3. Configure the rag-server / ingestor-server

Set environment variables on both deployments:

```yaml
envVars:
  APP_VECTORSTORE_NAME: oracle           # selects this plugin
  APP_VECTORSTORE_SEARCHTYPE: hybrid     # or `dense`
  APP_VECTORSTORE_INDEXTYPE: IVF         # or `HNSW`
  ORACLE_CS: ragbp_medium                # TNS alias OR full DSN descriptor
  ORACLE_VECTOR_INDEX_TYPE: IVF
  ORACLE_DISTANCE_METRIC: COSINE         # or `EUCLIDEAN`, `DOT`
  TNS_ADMIN: /app/wallet                 # ONLY for mTLS — path to wallet dir
```

Credentials (set via Kubernetes Secret with `envFrom`):

```yaml
data:
  ORACLE_USER: <base64 of "RAG_APP">
  ORACLE_PASSWORD: <base64 of password>
  # Only for mTLS:
  ORACLE_WALLET_PASSWORD: <base64 of wallet password>
```

### 4. Mount the wallet (mTLS mode only)

```yaml
extraVolumes:
  - name: oracle-wallet
    secret:
      secretName: oracle-wallet
      defaultMode: 0400

extraVolumeMounts:
  - name: oracle-wallet
    mountPath: /app/wallet
    readOnly: true
```

For walletless TLS, omit `TNS_ADMIN`, omit `ORACLE_WALLET_PASSWORD`, omit
the volume mount, and set `ORACLE_CS` to a TLS DSN descriptor (port 1521,
not 1522), e.g.:

```
(description=
  (address=(protocol=tcps)(port=1521)(host=<your-adb-host>))
  (connect_data=(service_name=<your-service>_medium.adb.oraclecloud.com))
  (security=(ssl_server_dn_match=yes)))
```

## Supported features

| Feature | Status |
|---|---|
| Hybrid search (vector + Oracle Text BM25) | yes |
| Dense-only search | yes |
| IVF approximate index | yes |
| HNSW approximate index | yes |
| Multi-collection (multi-tenant) | yes — one Oracle table per collection |
| Multimodal (image blobs via MinIO) | yes — same as Milvus path |
| GPU-accelerated index/search | no — Oracle 26ai vector ops run on the DB |

## Index management

Vector indexes are created automatically during the first
`/v1/collections` POST.  Each collection produces:

```
ENT_<NAME>             — base table
ENT_<NAME>_VEC_IDX     — VECTOR index (IVF or HNSW per APP_VECTORSTORE_INDEXTYPE)
ENT_<NAME>_TEXT_IDX    — CTXSYS.CONTEXT domain index (for hybrid)
```

The text index is created with `SYNC (ON COMMIT)` so keyword retrieval
sees inserted documents immediately.

## Hybrid search internals

Hybrid retrieval issues a single SQL statement that combines vector
similarity and Oracle Text keyword search using weighted-score fusion:

```sql
WITH vector_results AS (
    SELECT id, text, source, content_metadata,
           VECTOR_DISTANCE(vector, :query_vector, COSINE) as vec_distance
    FROM <table>
    FETCH FIRST :top_k * 2 ROWS ONLY
),
text_results AS (
    SELECT id, SCORE(1) as text_score
    FROM <table> WHERE CONTAINS(text, :sanitized_query, 1) > 0
    FETCH FIRST :top_k * 2 ROWS ONLY
)
SELECT v.id, v.text, v.source, v.content_metadata,
       (0.7 * (1/(1+v.vec_distance)) + 0.3 * COALESCE(t.text_score/100, 0))
       as combined_score
FROM vector_results v LEFT JOIN text_results t ON v.id = t.id
ORDER BY combined_score DESC FETCH FIRST :top_k ROWS ONLY;
```

Weighted-score fusion with default weights 0.7 (vector) / 0.3 (text).
When the sanitized text query is empty (e.g. punctuation-only input),
the adapter falls back to dense-only vector search automatically.

## Production guidance

| Concern | Recommendation |
|---|---|
| Network topology | Use ADB private endpoint inside the same VCN as your OKE cluster.  Avoids public IP / NAT egress concerns and enables clean walletless TLS. |
| ADB sizing | 4 ECPU is sufficient for ~100 collections / 1M chunks.  Auto-scale recommended. |
| Concurrency tuning | Each `rag-server` worker holds one Oracle connection.  Default 8 uvicorn workers + 1 `nemotron-ranking-ms` replica saturates around par=12 search QPS — scale the reranker NIM for higher concurrency. |
| Index type | IVF is approximate but kicks in only above a few thousand rows.  At low row counts the optimizer correctly prefers full scan. |
| Hybrid lexer | Oracle Text default `BASIC_LEXER` may not retain pure single-digit tokens.  Customize with `numgroup`/`numjoin` for SKU/numeric-heavy corpora. |

## Limitations

- The plugin does not currently expose Oracle Vector Index `efSearch` /
  `efConstruction` tuning at runtime; defaults are used.
- GPU-accelerated **index building** is supported via Oracle Private AI
  Services Container (cuVS). Set `ORACLE_PAI_INDEX_URL` or deploy via the
  wrapper chart (`examples/oracle/helm/`) which wires it automatically.
  GPU-accelerated **similarity search** runs on the database (CPU); only
  index *building* is offloaded to cuVS via the `OFFLOAD_URL` parameter.
- The `ENABLE_MINIO=False` path (using Oracle BLOBs for image storage)
  is not implemented yet; MinIO remains required for multimodal citations.

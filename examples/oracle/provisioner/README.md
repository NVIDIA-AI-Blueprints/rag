# Oracle ADB Provisioner

The provisioner is the shared database-creation component used by the Oracle
Helm and Docker examples.

Responsibilities:

1. Read OCI credentials from mounted OCI CLI config or in-cluster identity.
2. Auto-discover OCI region, OKE cluster, VCN, and private subnet when possible.
3. Create Oracle Autonomous AI Database 26ai with walletless TLS.
4. Wait until the database is `AVAILABLE`.
5. Create or update the `RAG_APP` user and grants.
6. Write connection information:
   - Helm: Kubernetes Secret `oracle-creds`
   - Docker: `generated/oracle.env`

Enterprise safety rules:

- Never delete ADB on uninstall by default.
- Never drop `RAG_APP` or existing collection tables during retries.
- Do not print passwords in logs.
- Tag created ADBs with release/namespace/cluster metadata for traceability.

## Usage

Create/reuse ADB and write a Kubernetes Secret:

```bash
python provision_adb.py create \
  --namespace rag \
  --k8s-secret oracle-creds
```

Create/reuse ADB and write a Docker env file:

```bash
python provision_adb.py create \
  --output-env ../generated/oracle.env
```

Optional overrides:

```bash
python provision_adb.py create \
  --compartment-id <compartment_ocid> \
  --subnet-id <private_subnet_ocid> \
  --db-name ragbp \
  --display-name ragbp \
  --ecpus 4 \
  --storage-tb 1
```

Environment variable equivalents:

| CLI flag | Env var |
|---|---|
| `--compartment-id` | `OCI_COMPARTMENT_OCID` |
| `--subnet-id` | `OCI_SUBNET_OCID` |
| `--vcn-id` | `OCI_VCN_OCID` |
| `--region` | `OCI_REGION` |
| `--db-name` | `ORACLE_DB_NAME` |
| `--display-name` | `ORACLE_DB_DISPLAY_NAME` |
| `--admin-password` | `ORACLE_ADMIN_PASSWORD` |
| `--rag-app-password` | `ORACLE_RAG_APP_PASSWORD` |
| `--ecpus` | `ORACLE_DB_ECPUS` |
| `--storage-tb` | `ORACLE_DB_STORAGE_TB` |

Authentication:

- Default: standard OCI CLI config from `~/.oci/config`
- `OCI_CLI_PROFILE` / `OCI_PROFILE` select the profile
- `OCI_AUTH=instance_principal` enables in-cluster instance principal auth
- `OCI_AUTH=resource_principal` enables resource principal auth

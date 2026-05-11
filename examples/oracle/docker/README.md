# Docker Compose Deployment With Oracle 26ai

Docker Compose examples are intended for local development and functional
testing of the Oracle branch.

## Create ADB Automatically

```bash
cp examples/oracle/docker/.env.example examples/oracle/docker/.env
# Fill NGC_API_KEY, OCI_COMPARTMENT_OCID, OCI_SUBNET_OCID

docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

export OCI_COMPARTMENT_OCID=<compartment_ocid>
export OCI_SUBNET_OCID=<private_subnet_ocid>
docker compose -f examples/oracle/docker/docker-compose.create-adb.yaml up
```

The provisioner mounts `~/.oci`, creates ADB 26ai, bootstraps `RAG_APP`, writes
`examples/oracle/generated/oracle.env`, and then starts the RAG services.

## Use Existing ADB

```bash
cp examples/oracle/docker/.env.example examples/oracle/docker/.env
# Fill NGC_API_KEY, ORACLE_USER, ORACLE_PASSWORD, ORACLE_CS
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
docker compose -f examples/oracle/docker/docker-compose.existing-adb.yaml up
```

## Uninstall Behavior

Stopping Docker Compose does not delete the ADB. Use explicit OCI cleanup
commands for dev/test teardown.

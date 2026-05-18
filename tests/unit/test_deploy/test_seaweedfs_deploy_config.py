import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def test_compose_vectordb_uses_seaweedfs_with_persistent_data_dir():
    compose = load_yaml(REPO_ROOT / "deploy/compose/vectordb.yaml")
    services = compose["services"]

    milvus = services["milvus"]
    seaweedfs = services["seaweedfs"]

    assert milvus["environment"]["MINIO_ADDRESS"] == "seaweedfs:9010"
    assert "seaweedfs" in milvus["depends_on"]
    assert "bucketName: ${NVINGEST_OBJECTSTORE_BUCKET:-nv-ingest}" in milvus["command"]
    assert "accessKeyID: ${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}" in milvus["command"]
    assert (
        "secretAccessKey: ${OBJECTSTORE_SECRETKEY:-seaweedfsadmin}" in milvus["command"]
    )

    assert seaweedfs["container_name"] == "seaweedfs"
    assert seaweedfs["image"] == "chrislusf/seaweedfs:3.73"
    assert seaweedfs["command"] == [
        "server",
        "-dir=/data",
        "-s3",
        "-s3.port=9010",
        "-s3.config=/etc/seaweedfs/s3.json",
        "-master.volumeSizeLimitMB=1024",
    ]
    assert "rag-vol-seaweedfs:/data" in seaweedfs["volumes"]
    assert (
        "./seaweedfs-config/s3.json:/etc/seaweedfs/s3.json:ro" in seaweedfs["volumes"]
    )
    assert seaweedfs["healthcheck"]["test"] == [
        "CMD-SHELL",
        "nc -z 127.0.0.1 9010",
    ]


def test_integration_vectordb_uses_same_seaweedfs_bootstrap():
    compose = load_yaml(REPO_ROOT / "tests/integration/vectordb.yaml")
    services = compose["services"]

    milvus = services["milvus"]
    seaweedfs = services["seaweedfs"]

    assert milvus["environment"]["MINIO_ADDRESS"] == "seaweedfs:9010"
    assert "seaweedfs" in milvus["depends_on"]
    assert "bucketName: ${NVINGEST_OBJECTSTORE_BUCKET:-nv-ingest}" in milvus["command"]
    assert "accessKeyID: ${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}" in milvus["command"]
    assert (
        "secretAccessKey: ${OBJECTSTORE_SECRETKEY:-seaweedfsadmin}" in milvus["command"]
    )
    assert seaweedfs["container_name"] == "seaweedfs"
    assert "-dir=/data" in seaweedfs["command"]
    assert (
        "../../deploy/compose/seaweedfs-config/s3.json:/etc/seaweedfs/s3.json:ro"
        in seaweedfs["volumes"]
    )


def test_notebook_config_uses_host_and_container_seaweedfs_endpoints():
    notebook_config = load_yaml(REPO_ROOT / "notebooks/config.yaml")
    integration_config = load_yaml(REPO_ROOT / "tests/integration/notebook_test_config.yaml")

    for config in (notebook_config, integration_config):
        assert config["object_store"]["endpoint"] == "localhost:9010"
        assert config["object_store"]["nv_ingest_endpoint"] == "seaweedfs:9010"


def test_vdb_operator_notebook_references_valid_vectordb_profile():
    # Regression guard for NVBug 6180813: the building_rag_vdb_operator.ipynb
    # notebook previously invoked `docker compose -f .../vectordb.yaml --profile minio`,
    # but `minio` is not a defined profile in vectordb.yaml — the object store is
    # SeaweedFS. Assert any vectordb.yaml --profile reference in this notebook
    # names a profile that actually exists in vectordb.yaml.
    notebook_path = REPO_ROOT / "notebooks/building_rag_vdb_operator.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    compose = load_yaml(REPO_ROOT / "deploy/compose/vectordb.yaml")
    valid_profiles: set[str] = set()
    for service in compose["services"].values():
        for profile in service.get("profiles", []):
            if profile:
                valid_profiles.add(profile)

    pattern = re.compile(
        r"docker\s+compose\s+-f\s+\S*vectordb\.yaml\s+--profile\s+(\S+)"
    )
    found = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        for match in pattern.finditer(text):
            found.append(match.group(1))

    assert found, (
        "Expected at least one `docker compose -f .../vectordb.yaml --profile <p>` "
        "invocation in building_rag_vdb_operator.ipynb; none were found."
    )
    for profile in found:
        assert profile in valid_profiles, (
            f"Notebook references --profile {profile!r}, but vectordb.yaml only "
            f"defines profiles {sorted(valid_profiles)!r}"
        )


def test_vdb_operator_notebook_get_documents_matches_vdb_interface():
    # The library calls custom VDB operators with
    # ``get_documents(collection_name, force_get_metadata=...)``. Keep the
    # notebook implementation aligned so it works with current library mode.
    notebook_path = REPO_ROOT / "notebooks/building_rag_vdb_operator.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert re.search(
        r"def\s+get_documents\s*\([^)]*force_get_metadata\s*:\s*bool\s*=\s*False",
        code,
        flags=re.DOTALL,
    ), "Notebook OpenSearchVDB.get_documents must accept force_get_metadata."


def test_vdb_operator_notebook_cloud_mode_uses_hosted_summarizer():
    # The notebook config defaults summarizer.server_url to localhost for on-prem.
    # In cloud mode, summary generation runs through NvidiaRAGIngestor and must
    # clear that endpoint so get_llm uses the hosted NVIDIA API catalog.
    notebook_path = REPO_ROOT / "notebooks/building_rag_vdb_operator.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert re.search(
        r'if\s+DEPLOYMENT_MODE\s*==\s*"cloud"\s*:'
        r'.*config_ingestor\.summarizer\.server_url\s*=\s*""',
        code,
        flags=re.DOTALL,
    ), "Notebook cloud mode must clear config_ingestor.summarizer.server_url."


def test_compose_app_services_default_to_seaweedfs():
    ingestor = load_yaml(
        REPO_ROOT / "deploy/compose/docker-compose-ingestor-server.yaml"
    )
    rag = load_yaml(REPO_ROOT / "deploy/compose/docker-compose-rag-server.yaml")

    assert (
        ingestor["services"]["ingestor-server"]["environment"]["OBJECTSTORE_ENDPOINT"]
        == "${OBJECTSTORE_ENDPOINT:-seaweedfs:9010}"
    )
    assert rag["services"]["rag-server"]["environment"]["OBJECTSTORE_ENDPOINT"] == (
        "${OBJECTSTORE_ENDPOINT:-seaweedfs:9010}"
    )
    assert (
        ingestor["services"]["ingestor-server"]["environment"]["OBJECTSTORE_ACCESSKEY"]
        == "${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}"
    )
    assert rag["services"]["rag-server"]["environment"]["OBJECTSTORE_ACCESSKEY"] == (
        "${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}"
    )

    nv_ingest_env = ingestor["services"]["nv-ingest-ms-runtime"]["environment"]
    assert "MINIO_BUCKET=${NVINGEST_OBJECTSTORE_BUCKET:-nv-ingest}" in nv_ingest_env
    assert "MINIO_ACCESS_KEY=${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}" in nv_ingest_env
    assert "MINIO_SECRET_KEY=${OBJECTSTORE_SECRETKEY:-seaweedfsadmin}" in nv_ingest_env
    assert "YOLOX_PAGE_IMAGE_FORMAT=JPEG" in nv_ingest_env


def test_workbench_defaults_to_seaweedfs():
    compose = load_yaml(REPO_ROOT / "deploy/workbench/compose.yaml")
    services = compose["services"]

    assert (
        services["ingestor-server"]["environment"]["OBJECTSTORE_ENDPOINT"]
        == "${OBJECTSTORE_ENDPOINT:-seaweedfs:9010}"
    )
    assert services["rag-server"]["environment"]["OBJECTSTORE_ENDPOINT"] == (
        "${OBJECTSTORE_ENDPOINT:-seaweedfs:9010}"
    )
    assert services["ingestor-server"]["environment"]["OBJECTSTORE_ACCESSKEY"] == (
        "${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}"
    )
    assert services["rag-server"]["environment"]["OBJECTSTORE_ACCESSKEY"] == (
        "${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}"
    )
    assert services["milvus"]["environment"]["MINIO_ADDRESS"] == "seaweedfs:9010"
    assert (
        "bucketName: ${NVINGEST_OBJECTSTORE_BUCKET:-nv-ingest}"
        in services["milvus"]["command"]
    )
    assert (
        "accessKeyID: ${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}"
        in services["milvus"]["command"]
    )
    assert (
        "secretAccessKey: ${OBJECTSTORE_SECRETKEY:-seaweedfsadmin}"
        in services["milvus"]["command"]
    )
    assert services["seaweedfs"]["container_name"] == "seaweedfs"
    assert services["seaweedfs"]["command"][1] == "-dir=/data"
    nv_ingest_env = services["nv-ingest-ms-runtime"]["environment"]
    assert "MINIO_BUCKET=${NVINGEST_OBJECTSTORE_BUCKET:-nv-ingest}" in nv_ingest_env
    assert "MINIO_ACCESS_KEY=${OBJECTSTORE_ACCESSKEY:-seaweedfsadmin}" in nv_ingest_env
    assert "MINIO_SECRET_KEY=${OBJECTSTORE_SECRETKEY:-seaweedfsadmin}" in nv_ingest_env
    assert "YOLOX_PAGE_IMAGE_FORMAT=JPEG" in nv_ingest_env


def test_helm_values_default_to_seaweedfs_and_persistence():
    values = load_yaml(REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/values.yaml")

    assert values["envVars"]["OBJECTSTORE_ENDPOINT"] == "rag-seaweedfs-all-in-one:9010"
    assert (
        values["envVars"]["APP_VECTORSTORE_URL"]
        == "http://rag-eck-elasticsearch-es-default:9200"
    )
    assert (
        values["ingestor-server"]["envVars"]["OBJECTSTORE_ENDPOINT"]
        == "rag-seaweedfs-all-in-one:9010"
    )
    assert (
        values["ingestor-server"]["envVars"]["NVINGEST_OBJECTSTORE_BUCKET"]
        == "nv-ingest"
    )
    assert (
        values["ingestor-server"]["envVars"]["APP_VECTORSTORE_URL"]
        == "http://rag-eck-elasticsearch-es-default:9200"
    )
    assert values["seaweedfs"]["enabled"] is True
    assert values["seaweedfs"]["fullnameOverride"] == "rag-seaweedfs"
    assert values["seaweedfs"]["master"]["enabled"] is False
    assert values["seaweedfs"]["volume"]["enabled"] is False
    assert values["seaweedfs"]["filer"]["enabled"] is False
    assert values["seaweedfs"]["allInOne"]["enabled"] is True
    assert values["seaweedfs"]["allInOne"]["s3"]["enabled"] is True
    assert values["seaweedfs"]["allInOne"]["s3"]["port"] == 9010
    assert values["seaweedfs"]["allInOne"]["data"]["type"] == "persistentVolumeClaim"
    assert values["eck-elasticsearch"]["fullnameOverride"] == "rag-eck-elasticsearch"
    assert values["nv-ingest"]["fullnameOverride"] == "rag-nv-ingest"
    assert values["nv-ingest"]["envVars"]["MINIO_INTERNAL_ADDRESS"] == (
        "rag-seaweedfs-all-in-one:9010"
    )
    assert values["nv-ingest"]["envVars"]["MINIO_PUBLIC_ADDRESS"] == (
        "http://rag-seaweedfs-all-in-one:9010"
    )
    assert values["nv-ingest"]["envVars"]["MINIO_BUCKET"] == "nv-ingest"
    assert values["nv-ingest"]["redis"]["fullnameOverride"] == "rag-redis"
    assert values["nv-ingest"]["envVars"]["YOLOX_PAGE_IMAGE_FORMAT"] == "JPEG"
    assert values["nv-ingest"]["milvusDeployed"] is False


def test_helm_chart_uses_upstream_seaweedfs_dependency():
    chart = load_yaml(REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/Chart.yaml")
    dependencies = {
        dependency["name"]: dependency for dependency in chart["dependencies"]
    }

    assert "seaweedfs" in dependencies
    assert (
        dependencies["seaweedfs"]["repository"]
        == "https://seaweedfs.github.io/seaweedfs/helm"
    )
    assert dependencies["seaweedfs"]["version"] == "4.21.0"
    assert dependencies["seaweedfs"]["condition"] == "seaweedfs.enabled"


def test_helm_templates_wire_elasticsearch_secret_for_bundled_eck():
    template_dir = REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/templates"

    rag_deployment = (template_dir / "deployment.yaml").read_text(encoding="utf-8")
    ingestor_deployment = (template_dir / "ingestor-server-deployment.yaml").read_text(
        encoding="utf-8"
    )
    helpers = (template_dir / "_helpers.tpl").read_text(encoding="utf-8")

    assert "APP_VECTORSTORE_PASSWORD" in rag_deployment
    assert "APP_VECTORSTORE_PASSWORD" in ingestor_deployment
    assert "elasticsearchUserSecretName" in rag_deployment
    assert "elasticsearchUserSecretName" in ingestor_deployment
    assert "es-elastic-user" in helpers

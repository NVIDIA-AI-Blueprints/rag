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

    assert seaweedfs["image"] == "chrislusf/seaweedfs:3.73"
    assert seaweedfs["command"] == [
        "server",
        "-dir=/data",
        "-s3",
        "-s3.port=9010",
        "-s3.config=/etc/seaweedfs/s3.json",
        "-master.volumeSizeLimitMB=1024",
    ]
    assert (
        "${DOCKER_VOLUME_DIRECTORY:-.}/volumes/seaweedfs:/data"
        in seaweedfs["volumes"]
    )
    assert "./seaweedfs-config/s3.json:/etc/seaweedfs/s3.json:ro" in seaweedfs["volumes"]
    assert seaweedfs["healthcheck"]["test"] == [
        "CMD-SHELL",
        "curl -s http://localhost:9010/ >/dev/null",
    ]


def test_integration_vectordb_uses_same_seaweedfs_bootstrap():
    compose = load_yaml(REPO_ROOT / "tests/integration/vectordb.yaml")
    services = compose["services"]

    milvus = services["milvus"]
    seaweedfs = services["seaweedfs"]

    assert milvus["environment"]["MINIO_ADDRESS"] == "seaweedfs:9010"
    assert "seaweedfs" in milvus["depends_on"]
    assert "-dir=/data" in seaweedfs["command"]
    assert (
        "../../deploy/compose/seaweedfs-config/s3.json:/etc/seaweedfs/s3.json:ro"
        in seaweedfs["volumes"]
    )


def test_compose_app_services_default_to_seaweedfs():
    ingestor = load_yaml(REPO_ROOT / "deploy/compose/docker-compose-ingestor-server.yaml")
    rag = load_yaml(REPO_ROOT / "deploy/compose/docker-compose-rag-server.yaml")

    assert (
        ingestor["services"]["ingestor-server"]["environment"]["OBJECTSTORE_ENDPOINT"]
        == "seaweedfs:9010"
    )
    assert rag["services"]["rag-server"]["environment"]["OBJECTSTORE_ENDPOINT"] == (
        "seaweedfs:9010"
    )

    nv_ingest_env = ingestor["services"]["nv-ingest-ms-runtime"]["environment"]
    assert "YOLOX_PAGE_IMAGE_FORMAT=PNG" in nv_ingest_env


def test_workbench_defaults_to_seaweedfs():
    compose = load_yaml(REPO_ROOT / "deploy/workbench/compose.yaml")
    services = compose["services"]

    assert (
        services["ingestor-server"]["environment"]["OBJECTSTORE_ENDPOINT"]
        == "seaweedfs:9010"
    )
    assert services["rag-server"]["environment"]["OBJECTSTORE_ENDPOINT"] == (
        "seaweedfs:9010"
    )
    assert services["milvus"]["environment"]["MINIO_ADDRESS"] == "seaweedfs:9010"
    assert services["seaweedfs"]["command"][1] == "-dir=/data"
    assert "YOLOX_PAGE_IMAGE_FORMAT=PNG" in services["nv-ingest-ms-runtime"]["environment"]


def test_helm_values_default_to_seaweedfs_and_persistence():
    values = load_yaml(REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/values.yaml")

    assert values["envVars"]["OBJECTSTORE_ENDPOINT"] == "rag-seaweedfs:9010"
    assert (
        values["envVars"]["APP_VECTORSTORE_URL"]
        == "http://rag-eck-elasticsearch-es-default:9200"
    )
    assert (
        values["ingestor-server"]["envVars"]["OBJECTSTORE_ENDPOINT"]
        == "rag-seaweedfs:9010"
    )
    assert (
        values["ingestor-server"]["envVars"]["APP_VECTORSTORE_URL"]
        == "http://rag-eck-elasticsearch-es-default:9200"
    )
    assert values["seaweedfs"]["enabled"] is True
    assert values["seaweedfs"]["persistence"]["enabled"] is True
    assert values["eck-elasticsearch"]["fullnameOverride"] == "rag-eck-elasticsearch"
    assert values["nv-ingest"]["fullnameOverride"] == "rag-nv-ingest"
    assert values["nv-ingest"]["envVars"]["MINIO_INTERNAL_ADDRESS"] == (
        "rag-seaweedfs:9010"
    )
    assert values["nv-ingest"]["envVars"]["MINIO_PUBLIC_ADDRESS"] == (
        "http://rag-seaweedfs:9010"
    )
    assert values["nv-ingest"]["redis"]["fullnameOverride"] == "rag-redis"
    assert values["nv-ingest"]["envVars"]["YOLOX_PAGE_IMAGE_FORMAT"] == "PNG"
    assert values["nv-ingest"]["milvusDeployed"] is False
    assert values["nv-ingest"]["milvus"]["minio"]["enabled"] is False


def test_helm_seaweedfs_templates_exist_and_mount_data_dir():
    template_dir = REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/templates"

    deployment = (template_dir / "seaweedfs-deployment.yaml").read_text(
        encoding="utf-8"
    )
    service = (template_dir / "seaweedfs-service.yaml").read_text(encoding="utf-8")
    pvc = (template_dir / "seaweedfs-pvc.yaml").read_text(encoding="utf-8")
    configmap = (template_dir / "seaweedfs-configmap.yaml").read_text(
        encoding="utf-8"
    )

    assert "- -dir=/data" in deployment
    assert "mountPath: /data" in deployment
    assert "s3Port" in service
    assert "PersistentVolumeClaim" in pvc
    assert '"actions": ["Admin", "Read", "Write", "List", "Tagging"]' in configmap
    assert "tcpSocket:" in deployment
    assert 'printf "rag-seaweedfs"' in (
        template_dir / "_helpers.tpl"
    ).read_text(encoding="utf-8")


def test_helm_templates_wire_elasticsearch_secret_for_bundled_eck():
    template_dir = REPO_ROOT / "deploy/helm/nvidia-blueprint-rag/templates"

    rag_deployment = (template_dir / "deployment.yaml").read_text(encoding="utf-8")
    ingestor_deployment = (
        template_dir / "ingestor-server-deployment.yaml"
    ).read_text(encoding="utf-8")
    helpers = (template_dir / "_helpers.tpl").read_text(encoding="utf-8")

    assert "APP_VECTORSTORE_PASSWORD" in rag_deployment
    assert "APP_VECTORSTORE_PASSWORD" in ingestor_deployment
    assert "elasticsearchUserSecretName" in rag_deployment
    assert "elasticsearchUserSecretName" in ingestor_deployment
    assert "es-elastic-user" in helpers

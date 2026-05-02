#!/usr/bin/env python3
"""Provision Oracle Autonomous AI Database 26ai for NVIDIA RAG.

This helper is intentionally shared by the Oracle Helm and Docker Compose
examples.  It creates/reuses a walletless TLS ADB, bootstraps a non-destructive
RAG_APP user, and writes connection details either to a Kubernetes Secret
(`oracle-creds`) or to a Docker env file (`generated/oracle.env`).

Safety defaults:
* Do not delete databases unless the explicit `delete` command is called.
* Do not drop RAG_APP or existing collection tables on retries.
* Do not print passwords.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oci
import oracledb

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
except Exception:  # pragma: no cover - docker-only mode can run without k8s lib
    k8s_client = None
    k8s_config = None


DEFAULT_WAIT_SECONDS = 3600
DEFAULT_POLL_SECONDS = 20


@dataclass
class ProvisionerConfig:
    compartment_id: str | None
    subnet_id: str | None
    vcn_id: str | None
    region: str | None
    db_name: str
    display_name: str
    workload_type: str
    admin_password: str
    rag_app_password: str
    ecpus: int
    storage_tb: int
    namespace: str
    k8s_secret: str
    output_env: str | None
    wait_seconds: int
    poll_seconds: int
    auto_discover: bool
    reuse_existing: bool
    kubeconfig: str | None


def getenv(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in ("", None) else default


def generate_password(length: int = 24) -> str:
    """Generate an ADB-compatible password.

    ADB requires uppercase, lowercase, number, and special character.
    Avoid quotes/backslashes to keep shell/YAML/SQL handling simple.
    """
    alphabet = string.ascii_letters + string.digits + "!#$%*-_+"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(c in "!#$%*-_+" for c in pw)
        ):
            return pw


def stable_suffix(*parts: str, length: int = 6) -> str:
    """Return a stable alphanumeric suffix for idempotent resource naming."""
    raw = "|".join(p for p in parts if p)
    return hashlib.sha1(raw.encode()).hexdigest()[:length]


def normalize_db_name(name: str) -> str:
    """ADB db_name must be short and alphanumeric."""
    cleaned = "".join(ch for ch in name.lower() if ch.isalnum())
    if not cleaned:
        cleaned = "ragbp"
    if not cleaned[0].isalpha():
        cleaned = "r" + cleaned
    return cleaned[:30]


def default_names(raw_db_name: str | None, raw_display_name: str | None) -> tuple[str, str]:
    """Generate safe, stable names.

    If the caller does not explicitly provide a DB name, derive one from the
    Helm release/namespace (or Docker project) so repeated hook retries reuse
    the same ADB, while different installs in the same tenancy/region do not
    collide on plain `ragbp`.
    """
    release = getenv("HELM_RELEASE_NAME", getenv("COMPOSE_PROJECT_NAME", "rag"))
    namespace = getenv("HELM_NAMESPACE", getenv("K8S_NAMESPACE", "default"))
    # Fresh DB per install by default.  Use a random suffix unless caller
    # explicitly provides dbName/displayName.  Helm retries within the same Job
    # do not re-enter this process; separate installs should not reuse stale
    # failed-attempt databases unless reuseExisting is explicitly enabled.
    suffix = secrets.token_hex(3)
    db_name = raw_db_name or getenv("ORACLE_DB_NAME")
    display_name = raw_display_name or getenv("ORACLE_DB_DISPLAY_NAME")

    if not db_name:
        db_name = f"ragbp{suffix}"
    if not display_name:
        display_name = f"ragbp-{release}-{suffix}" if release else f"ragbp-{suffix}"
    return normalize_db_name(db_name), display_name[:80]


def load_oci_config(region_override: str | None) -> tuple[dict[str, Any], Any]:
    """Load OCI SDK config.

    Supports standard OCI CLI config first, then resource principal / instance
    principal fallback for in-cluster enterprise environments.
    """
    auth_mode = getenv("OCI_AUTH", "config")
    if auth_mode == "resource_principal":
        signer = oci.auth.signers.get_resource_principals_signer()
        cfg = {"region": region_override or getenv("OCI_REGION") or signer.region}
        return cfg, signer
    if auth_mode == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        cfg = {"region": region_override or getenv("OCI_REGION") or signer.region}
        return cfg, signer

    profile = getenv("OCI_CLI_PROFILE", getenv("OCI_PROFILE", "DEFAULT"))
    config_file = getenv("OCI_CLI_CONFIG_FILE", "~/.oci/config")
    expanded = Path(config_file).expanduser()
    config_dir = str(expanded.parent)
    pod_key = os.path.join(config_dir, "oci_api_key.pem")

    # In-pod: user's local key_file path won't exist. Rewrite the config
    # to use oci_api_key.pem from the mounted secret BEFORE the SDK validates.
    actual_config = str(expanded)
    if Path(pod_key).exists() and expanded.exists():
        content = expanded.read_text()
        if re.search(r"key_file\s*=", content):
            import tempfile
            fixed = re.sub(r"key_file\s*=\s*\S+", f"key_file={pod_key}", content)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
            tmp.write(fixed)
            tmp.close()
            actual_config = tmp.name

    cfg = oci.config.from_file(actual_config, profile)
    if region_override:
        cfg["region"] = region_override
    return cfg, None


def clients(cfg: dict[str, Any], signer: Any | None):
    kwargs = {"signer": signer} if signer else {}
    return {
        "database": oci.database.DatabaseClient(cfg, **kwargs),
        "ce": oci.container_engine.ContainerEngineClient(cfg, **kwargs),
        "vcn": oci.core.VirtualNetworkClient(cfg, **kwargs),
        "identity": oci.identity.IdentityClient(cfg, **kwargs),
    }


def load_kube_config(kubeconfig: str | None) -> None:
    if not k8s_config:
        raise RuntimeError("kubernetes package is not installed")
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config(config_file=kubeconfig)


def get_kube_nodes(kubeconfig: str | None):
    load_kube_config(kubeconfig)
    v1 = k8s_client.CoreV1Api()
    return v1.list_node().items


def get_node_private_ips(kubeconfig: str | None) -> list[str]:
    ips: list[str] = []
    for node in get_kube_nodes(kubeconfig):
        for addr in node.status.addresses or []:
            if addr.type == "InternalIP":
                ips.append(addr.address)
    return ips


def get_node_provider_ids(kubeconfig: str | None) -> list[str]:
    provider_ids: list[str] = []
    for node in get_kube_nodes(kubeconfig):
        provider_id = getattr(node.spec, "provider_id", None)
        if provider_id:
            provider_ids.append(provider_id)
    return provider_ids


def region_from_ocid(ocid: str) -> str | None:
    # Example: ocid1.instance.oc1.us-chicago-1.<unique>
    parts = ocid.split(".")
    if len(parts) >= 4 and parts[0] == "ocid1":
        return parts[3]
    return None


def all_compartments(identity, tenancy_id: str, explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    result = [tenancy_id]
    try:
        comps = oci.pagination.list_call_get_all_results(
            identity.list_compartments,
            tenancy_id,
            compartment_id_in_subtree=True,
            access_level="ACCESSIBLE",
        ).data
        result.extend(c.id for c in comps if c.lifecycle_state == "ACTIVE")
    except Exception as exc:
        print(f"Auto-discovery: unable to list compartments ({exc}); using tenancy only")
    return result


def discover_from_node_instance(
    cfg: ProvisionerConfig,
    oci_cfg: dict[str, Any],
    signer: Any | None,
) -> tuple[str, str, str] | None:
    """Discover region, compartment, and subnet from the actual OKE node OCID.

    This is the safest path because private IP ranges are not globally unique.
    Matching only `10.0.10.x` against accessible subnets can select the wrong
    region/VCN if several clusters use the same CIDR. ProviderID includes the
    compute instance OCID and therefore the region.
    """
    provider_ids = get_node_provider_ids(cfg.kubeconfig)
    if not provider_ids:
        return None
    raw = provider_ids[0]
    instance_id = raw.split("://", 1)[-1]
    region = cfg.region or region_from_ocid(instance_id) or oci_cfg.get("region")
    if not region:
        return None

    node_cfg = dict(oci_cfg)
    node_cfg["region"] = region
    compute = oci.core.ComputeClient(node_cfg, signer=signer) if signer else oci.core.ComputeClient(node_cfg)
    vcn = oci.core.VirtualNetworkClient(node_cfg, signer=signer) if signer else oci.core.VirtualNetworkClient(node_cfg)

    inst = compute.get_instance(instance_id).data
    attachments = oci.pagination.list_call_get_all_results(
        compute.list_vnic_attachments,
        inst.compartment_id,
        instance_id=instance_id,
    ).data
    if not attachments:
        return None
    # Prefer primary VNIC if present; otherwise first attachment.
    attachment = next((a for a in attachments if a.lifecycle_state == "ATTACHED"), attachments[0])
    vnic = vcn.get_vnic(attachment.vnic_id).data
    subnet = vcn.get_subnet(vnic.subnet_id).data
    if cfg.vcn_id and subnet.vcn_id != cfg.vcn_id:
        raise RuntimeError(
            f"Discovered node subnet is in VCN {subnet.vcn_id}, but --vcn-id was {cfg.vcn_id}"
        )
    print(
        "Auto-discovery: selected region/subnet from node providerID "
        f"{instance_id}: region={region}, subnet={subnet.display_name} "
        f"({subnet.cidr_block}), compartment={subnet.compartment_id}"
    )
    return region, subnet.compartment_id, subnet.id


def discover_network(
    cfg: ProvisionerConfig,
    oci_clients: dict[str, Any],
    tenancy_id: str,
    oci_cfg: dict[str, Any],
    signer: Any | None,
) -> tuple[str, str, str | None]:
    """Discover region/compartment/subnet for ADB private endpoint."""
    if cfg.compartment_id and cfg.subnet_id:
        return oci_cfg.get("region"), cfg.compartment_id, cfg.subnet_id
    if not cfg.auto_discover:
        raise ValueError("compartment-id and subnet-id are required when auto-discover is disabled")

    node_based = discover_from_node_instance(cfg, oci_cfg, signer)
    if node_based:
        return node_based

    node_ips = [ipaddress.ip_address(ip) for ip in get_node_private_ips(cfg.kubeconfig)]
    if not node_ips:
        raise RuntimeError("Could not discover Kubernetes node private IPs")

    print(f"Auto-discovery: found node IPs {[str(ip) for ip in node_ips]}")
    compartments = all_compartments(oci_clients["identity"], tenancy_id, cfg.compartment_id)
    for compartment_id in compartments:
        vcns = []
        if cfg.vcn_id:
            try:
                vcns = [oci_clients["vcn"].get_vcn(cfg.vcn_id).data]
            except Exception:
                vcns = []
        else:
            try:
                vcns = oci.pagination.list_call_get_all_results(
                    oci_clients["vcn"].list_vcns, compartment_id
                ).data
            except Exception:
                continue
        for vcn in vcns:
            try:
                subnets = oci.pagination.list_call_get_all_results(
                    oci_clients["vcn"].list_subnets,
                    compartment_id,
                    vcn_id=vcn.id,
                ).data
            except Exception:
                continue
            for subnet in subnets:
                cidr = ipaddress.ip_network(subnet.cidr_block)
                if any(ip in cidr for ip in node_ips):
                    print(
                        "Auto-discovery: selected subnet "
                        f"{subnet.display_name} ({subnet.cidr_block}) in compartment {compartment_id}"
                    )
                    print(
                        "WARNING: fallback subnet discovery used CIDR matching. "
                        "ProviderID-based discovery was unavailable; pass --subnet-id "
                        "to avoid ambiguity in multi-region environments."
                    )
                    return oci_cfg.get("region"), compartment_id, subnet.id
    raise RuntimeError(
        "Could not auto-discover an ADB subnet from current Kubernetes nodes. "
        "Pass --compartment-id and --subnet-id explicitly."
    )


def find_adb(database, compartment_id: str, display_name: str):
    dbs = oci.pagination.list_call_get_all_results(
        database.list_autonomous_databases,
        compartment_id,
        display_name=display_name,
    ).data
    for db in dbs:
        if db.lifecycle_state not in ("TERMINATED", "TERMINATING"):
            return db
    return None


def create_or_reuse_adb(cfg: ProvisionerConfig, database, compartment_id: str, subnet_id: str):
    existing = find_adb(database, compartment_id, cfg.display_name)
    if existing and cfg.reuse_existing:
        print(f"Reusing existing ADB: {existing.display_name} ({existing.id})")
        return existing
    if existing and not cfg.reuse_existing:
        raise RuntimeError(
            f"ADB named {cfg.display_name!r} already exists. This installer creates "
            "a fresh ADB by default. Pick a different dbName/displayName or set "
            "--reuse-existing to intentionally reuse it."
        )

    print(f"Creating ADB 26ai: {cfg.display_name} in subnet {subnet_id}")
    details = oci.database.models.CreateAutonomousDatabaseDetails(
        compartment_id=compartment_id,
        db_name=cfg.db_name,
        display_name=cfg.display_name,
        admin_password=cfg.admin_password,
        compute_model="ECPU",
        compute_count=cfg.ecpus,
        data_storage_size_in_tbs=cfg.storage_tb,
        db_version="26ai",
        db_workload=cfg.workload_type,
        license_model="LICENSE_INCLUDED",
        is_auto_scaling_enabled=True,
        is_mtls_connection_required=False,
        subnet_id=subnet_id,
        private_endpoint_label=cfg.db_name.lower().replace("_", "-")[:30],
        freeform_tags={
            "nvidia-rag": "oracle-26ai",
            "created-by": "nvidia-rag-oracle-provisioner",
        },
    )
    try:
        response = database.create_autonomous_database(details)
        return database.get_autonomous_database(response.data.id).data
    except oci.exceptions.ServiceError as exc:
        # If caller explicitly requested reuse, treat duplicate-name errors as
        # "created by a previous attempt" and reuse the DB.  Otherwise fail
        # clearly so a fresh install never silently attaches to stale data.
        if cfg.reuse_existing and exc.status == 400 and "already exists" in (exc.message or "").lower():
            print("ADB already exists; re-querying and reusing it")
            for _ in range(18):
                existing = find_adb(database, compartment_id, cfg.display_name)
                if existing:
                    return existing
                time.sleep(10)
        raise


def wait_for_adb(database, adb_id: str, timeout: int, poll: int):
    deadline = time.time() + timeout
    while time.time() < deadline:
        db = database.get_autonomous_database(adb_id).data
        print(f"ADB state: {db.lifecycle_state}")
        if db.lifecycle_state == "AVAILABLE":
            return db
        if db.lifecycle_state in ("FAILED", "TERMINATED", "TERMINATING"):
            raise RuntimeError(f"ADB entered terminal state: {db.lifecycle_state}")
        time.sleep(poll)
    raise TimeoutError(f"Timed out waiting for ADB {adb_id} to become AVAILABLE")


def server_tls_profile(adb) -> str:
    profiles = adb.connection_strings.profiles or []
    dsn = None
    for profile in profiles:
        if profile.display_name.endswith("_medium") and profile.tls_authentication == "SERVER":
            dsn = profile.value
            break
    if dsn is None:
        for profile in profiles:
            if profile.tls_authentication == "SERVER":
                dsn = profile.value
                break
    if dsn is None:
        raise RuntimeError("No walletless TLS connection profile found")

    # Some private endpoint hostnames are not resolvable inside customer
    # clusters until private DNS is configured.  The ADB API exposes the
    # private endpoint IP; using it directly is reliable for OKE-in-VCN
    # deployments.  When using an IP, server DN matching must be disabled.
    private_ip = getattr(adb, "private_endpoint_ip", None)
    if private_ip:
        dsn = re.sub(r"\\(host=[^)]+\\)", f"(host={private_ip})", dsn)
        dsn = re.sub(r"ssl_server_dn_match\\s*=\\s*yes", "ssl_server_dn_match=no", dsn)
    return dsn


def bootstrap_rag_app(dsn: str, admin_password: str, rag_password: str) -> None:
    conn = oracledb.connect(user="ADMIN", password=admin_password, dsn=dsn)
    cur = conn.cursor()
    statements = [
        "BEGIN "
        "FOR r IN (SELECT username FROM dba_users WHERE username='RAG_APP') LOOP "
        "EXECUTE IMMEDIATE 'ALTER USER RAG_APP IDENTIFIED BY \""
        + rag_password.replace('"', "")
        + "\"'; "
        "RETURN; END LOOP; "
        "EXECUTE IMMEDIATE 'CREATE USER RAG_APP IDENTIFIED BY \""
        + rag_password.replace('"', "")
        + "\" DEFAULT TABLESPACE DATA QUOTA UNLIMITED ON DATA'; "
        "END;",
        "GRANT CONNECT, RESOURCE TO RAG_APP",
        "GRANT CREATE SESSION TO RAG_APP",
        "GRANT CREATE TABLE TO RAG_APP",
        "GRANT CREATE VIEW TO RAG_APP",
        "GRANT CREATE SEQUENCE TO RAG_APP",
        "GRANT CREATE PROCEDURE TO RAG_APP",
        "GRANT CREATE MINING MODEL TO RAG_APP",
        "GRANT EXECUTE ON CTXSYS.CTX_DDL TO RAG_APP",
        "GRANT EXECUTE ON CTXSYS.CTX_QUERY TO RAG_APP",
        "GRANT CTXAPP TO RAG_APP",
        "GRANT CREATE ANY INDEX TO RAG_APP",
        "GRANT SELECT ANY DICTIONARY TO RAG_APP",
    ]
    for stmt in statements:
        cur.execute(stmt)
    conn.commit()
    conn.close()

    # Smoke test as app user
    app = oracledb.connect(user="RAG_APP", password=rag_password, dsn=dsn)
    app.close()


def write_env(path: str, dsn: str, rag_password: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "APP_VECTORSTORE_NAME=oracle",
                "APP_VECTORSTORE_SEARCHTYPE=hybrid",
                "APP_VECTORSTORE_INDEXTYPE=IVF",
                "ORACLE_USER=RAG_APP",
                f"ORACLE_PASSWORD={rag_password}",
                f"ORACLE_CS={dsn}",
                "ORACLE_VECTOR_INDEX_TYPE=IVF",
                "ORACLE_DISTANCE_METRIC=COSINE",
                "",
            ]
        )
    )
    print(f"Wrote Docker env file: {path}")


def write_k8s_secret(namespace: str, name: str, dsn: str, rag_password: str, kubeconfig: str | None) -> None:
    if not k8s_config:
        print("Kubernetes client not installed; skipping K8s Secret creation")
        return
    load_kube_config(kubeconfig)
    v1 = k8s_client.CoreV1Api()
    body = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        type="Opaque",
        data={
            "ORACLE_USER": base64.b64encode(b"RAG_APP").decode(),
            "ORACLE_PASSWORD": base64.b64encode(rag_password.encode()).decode(),
            "ORACLE_CS": base64.b64encode(dsn.encode()).decode(),
        },
    )
    try:
        v1.replace_namespaced_secret(name=name, namespace=namespace, body=body)
        print(f"Updated Kubernetes Secret: {namespace}/{name}")
    except Exception:
        v1.create_namespaced_secret(namespace=namespace, body=body)
        print(f"Created Kubernetes Secret: {namespace}/{name}")


def parse_create_args(args: argparse.Namespace) -> ProvisionerConfig:
    db_name, display = default_names(args.db_name, args.display_name)
    return ProvisionerConfig(
        compartment_id=args.compartment_id or getenv("OCI_COMPARTMENT_OCID"),
        subnet_id=args.subnet_id or getenv("OCI_SUBNET_OCID"),
        vcn_id=args.vcn_id or getenv("OCI_VCN_OCID"),
        region=args.region or getenv("OCI_REGION"),
        db_name=db_name,
        display_name=display,
        workload_type=(args.workload_type or getenv("ORACLE_DB_WORKLOAD_TYPE", "LH")).upper(),
        admin_password=args.admin_password or getenv("ORACLE_ADMIN_PASSWORD") or generate_password(),
        rag_app_password=args.rag_app_password or getenv("ORACLE_RAG_APP_PASSWORD") or generate_password(),
        ecpus=int(args.ecpus or getenv("ORACLE_DB_ECPUS", "4")),
        storage_tb=int(args.storage_tb or getenv("ORACLE_DB_STORAGE_TB", "1")),
        namespace=args.namespace,
        k8s_secret=args.k8s_secret,
        output_env=args.output_env,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
        auto_discover=args.auto_discover,
        reuse_existing=args.reuse_existing,
        kubeconfig=args.kubeconfig,
    )


def verify_oci_auth(oci_cfg: dict[str, Any], signer: Any | None) -> str:
    """Pre-flight check: verify OCI credentials work before attempting ADB create."""
    print("Pre-flight: verifying OCI credentials...")
    try:
        identity = oci.identity.IdentityClient(oci_cfg, signer=signer) if signer else oci.identity.IdentityClient(oci_cfg)
        tenancy = identity.get_tenancy(oci_cfg.get("tenancy", "")).data
        print(f"Pre-flight: authenticated to tenancy {tenancy.name} ({tenancy.id[:40]}...)")
        return tenancy.id
    except oci.exceptions.ServiceError as exc:
        if exc.status == 401:
            raise RuntimeError(
                "OCI authentication failed (HTTP 401). Check that:\n"
                "  1. The API key fingerprint matches the one registered in OCI Console\n"
                "  2. The private key file (oci_api_key.pem) matches the public key uploaded to OCI\n"
                "  3. The user OCID and tenancy OCID in the config are correct\n"
                "  4. The API key has not been deleted or deactivated in OCI Console\n"
                f"  OCI error: {exc.message}"
            ) from exc
        raise


def bootstrap_with_retry(dsn: str, admin_password: str, rag_password: str, retries: int = 5, delay: int = 15) -> None:
    """Bootstrap RAG_APP with retry — ADB private endpoint may not be routable immediately."""
    for attempt in range(1, retries + 1):
        try:
            bootstrap_rag_app(dsn, admin_password, rag_password)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to connect to ADB after {retries} attempts. "
                    "The database is AVAILABLE but the private endpoint may not be "
                    "routable from this pod yet. Check VCN security lists and route tables.\n"
                    f"Last error: {exc}"
                ) from exc
            print(f"Bootstrap attempt {attempt}/{retries} failed ({exc}), retrying in {delay}s...")
            time.sleep(delay)


def create_command(args: argparse.Namespace) -> int:
    cfg = parse_create_args(args)
    oci_cfg, signer = load_oci_config(cfg.region)

    tenancy_id = oci_cfg.get("tenancy") or getenv("OCI_TENANCY_OCID")
    if not tenancy_id:
        raise RuntimeError(
            "Could not determine tenancy OCID from OCI config. "
            "Check that your ~/.oci/config has a valid 'tenancy=' line "
            "for the profile you specified (--set oracle.auth.profile=...)."
        )

    tenancy_id = verify_oci_auth(oci_cfg, signer)
    cs = clients(oci_cfg, signer)
    discovered_region, compartment_id, subnet_id = discover_network(
        cfg, cs, tenancy_id, oci_cfg, signer
    )
    if discovered_region and discovered_region != oci_cfg.get("region"):
        print(f"Auto-discovery: switching OCI clients to node region {discovered_region}")
        oci_cfg = dict(oci_cfg)
        oci_cfg["region"] = discovered_region
        cs = clients(oci_cfg, signer)
    adb = create_or_reuse_adb(cfg, cs["database"], compartment_id, subnet_id)
    adb = wait_for_adb(cs["database"], adb.id, cfg.wait_seconds, cfg.poll_seconds)
    dsn = server_tls_profile(adb)
    bootstrap_with_retry(dsn, cfg.admin_password, cfg.rag_app_password)
    if cfg.output_env:
        write_env(cfg.output_env, dsn, cfg.rag_app_password)
    write_k8s_secret(cfg.namespace, cfg.k8s_secret, dsn, cfg.rag_app_password, cfg.kubeconfig)
    print(json.dumps({"adb_ocid": adb.id, "display_name": adb.display_name, "dsn": dsn}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision Oracle ADB 26ai for NVIDIA RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create", help="Create/reuse ADB and write connection outputs")
    create_parser.add_argument("--output-env", help="Write Docker env file to this path")
    create_parser.add_argument("--k8s-secret", default="oracle-creds", help="Kubernetes Secret to create/update")
    create_parser.add_argument("--namespace", default="rag", help="Kubernetes namespace for --k8s-secret")
    create_parser.add_argument("--compartment-id")
    create_parser.add_argument("--subnet-id")
    create_parser.add_argument("--vcn-id")
    create_parser.add_argument("--region")
    create_parser.add_argument("--db-name")
    create_parser.add_argument("--display-name")
    create_parser.add_argument("--workload-type", default=None, help="ADB workload type, e.g. LH (Lakehouse), OLTP, DW, AJD, APEX")
    create_parser.add_argument("--admin-password")
    create_parser.add_argument("--rag-app-password")
    create_parser.add_argument("--ecpus", type=int)
    create_parser.add_argument("--storage-tb", type=int)
    create_parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    create_parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    create_parser.add_argument("--kubeconfig")
    create_parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    create_parser.add_argument("--reuse-existing", action="store_true", help="Reuse an existing ADB with the same display name")

    sub.add_parser("delete", help="Explicit dev/test cleanup only; never called by default")

    args = parser.parse_args()
    if args.command == "create":
        return create_command(args)
    if args.command == "delete":
        raise SystemExit("Explicit delete helper intentionally not implemented in this PR.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

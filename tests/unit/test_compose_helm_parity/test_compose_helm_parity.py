import re
import fnmatch
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_image(image_str: str) -> tuple[str, str | None]:
    """Split an image string into repository and tag.
    Supports forms like 'repo:tag' and 'repo:${TAG:-default}'.
    Returns (repository, tag_or_default) where tag_or_default may be None if not parseable.
    """
    if ":" not in image_str:
        return image_str, None
    repo, tag_part = image_str.split(":", 1)
    # Handle ${VAR:-default}
    m = re.match(r"^\$\{[^:}]+:-([^}]+)\}$", tag_part)
    if m:
        return repo, m.group(1)
    # Handle ${VAR}
    if tag_part.startswith("${") and tag_part.endswith("}"):
        return repo, None
    return repo, tag_part


def get_dict_path(d: dict[str, object], path: list[str]) -> object:
    ref: object = d
    for key in path:
        if not isinstance(ref, dict) or key not in ref:
            return None
        ref = ref[key]
    return ref


def ensure_frontend_env_names(values_env_list: object) -> set[str]:
    names = set()
    if isinstance(values_env_list, list):
        for item in values_env_list:
            if isinstance(item, dict) and "name" in item:
                names.add(str(item["name"]))
    return names


def extract_compose_env_names(env_obj: object) -> set[str]:
    """Return environment variable names from docker-compose service env.
    Supports both dict mapping (key: value) and list form ["KEY=val", ...].
    """
    names: set[str] = set()
    if isinstance(env_obj, dict):
        names = {str(k) for k in env_obj.keys()}
    elif isinstance(env_obj, list):
        for entry in env_obj:
            if isinstance(entry, str) and entry:
                name = entry.split("=", 1)[0]
                if name:
                    names.add(name)
    return names


def extract_values_env_keys(values_env_obj: object) -> set[str]:
    """Return env keys defined in Helm values. For dict returns keys, for list of {name,value} returns names."""
    if isinstance(values_env_obj, dict):
        return {str(k) for k in values_env_obj.keys()}
    return ensure_frontend_env_names(values_env_obj)


def extract_compose_env_kv(env_obj: object) -> dict[str, object]:
    """Return env key->value mapping from docker-compose service env.
    Supports both dict mapping (key: value) and list form ["KEY=val", ...].
    For list entries without '=', assigns empty string as value.
    """
    result: dict[str, object] = {}
    if isinstance(env_obj, dict):
        for k, v in env_obj.items():
            result[str(k)] = v
    elif isinstance(env_obj, list):
        for entry in env_obj:
            if isinstance(entry, str) and entry:
                if "=" in entry:
                    name, val = entry.split("=", 1)
                    result[name] = val
                else:
                    result[entry] = ""
    return result


def extract_values_env_kv(values_env_obj: object) -> dict[str, object]:
    """Return env key->value mapping from Helm values envVars.
    For dict returns as-is; for list of {name, value} returns mapping.
    """
    result: dict[str, object] = {}
    if isinstance(values_env_obj, dict):
        for k, v in values_env_obj.items():
            result[str(k)] = v
    elif isinstance(values_env_obj, list):
        for item in values_env_obj:
            if isinstance(item, dict) and "name" in item:
                result[str(item["name"])] = item.get("value")
    return result


def parse_env_default(value: str) -> tuple[bool, str | None]:
    """Parse docker-compose style placeholders with nested defaults.

    Supports:
      ${VAR}
      ${VAR-default}
      ${VAR:-default}
      Nested defaults like ${VAR:-${OTHER:-grpc}}

    Returns (True, resolved_default_or_none) if a placeholder, otherwise (False, None).
    """
    if not isinstance(value, str):
        return False, None

    if not (value.startswith("${") and value.endswith("}")):
        # Not a placeholder form
        # still allow the simple ${VAR} case via regex below
        if re.match(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$", value):
            return True, None
        return False, None

    # Strip leading ${ and trailing }
    inner = value[2:-1]
    # Extract variable name
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", inner)
    if not m:
        return False, None
    varname = m.group(1)
    rest = inner[len(varname):]
    if rest == "":
        # ${VAR}
        return True, None

    # Expect either - or :- then default expression
    if rest.startswith(":-"):
        default_expr = rest[2:]
    elif rest.startswith("-"):
        default_expr = rest[1:]
    else:
        # Unexpected suffix
        return False, None

    default_expr = default_expr.strip()

    # If default is a nested placeholder, resolve recursively
    if default_expr.startswith("${") and default_expr.endswith("}"):
        _, nested = parse_env_default(default_expr)
        return True, nested

    # Strip surrounding quotes if present
    if len(default_expr) >= 2 and (
        (default_expr[0] == default_expr[-1] == '"') or (default_expr[0] == default_expr[-1] == "'")
    ):
        default_expr = default_expr[1:-1]

    return True, default_expr


def normalize_env_value(value: object) -> str:
    """Normalize env value for parity comparison.
    - Resolve nested defaults in compose placeholders to the deepest literal (e.g., grpc).
    - Strip surrounding quotes for fully quoted strings (so '""' becomes '').
    - If placeholder has no default, keep placeholder as-is.
    """
    if not isinstance(value, str):
        return str(value)

    # Treat explicit quoted-empty as empty string
    if value in ('""', "''"):
        return ""

    # Strip enclosing quotes generally to align with Helm values normalization
    def _strip_quotes(s: str) -> str:
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            return s[1:-1]
        return s

    is_placeholder, default_val = parse_env_default(value)
    if is_placeholder:
        if default_val is not None:
            return _strip_quotes(str(default_val))
        # placeholder without default: return as-is
        return value

    return _strip_quotes(value)


def test_compose_helm_image_and_env_parity():
    repo_root = Path(__file__).resolve().parents[3]

    # Paths
    values_path = repo_root / "deploy/helm/nvidia-blueprint-rag/values.yaml"
    compose_rag_path = repo_root / "deploy/compose/docker-compose-rag-server.yaml"
    compose_ingestor_path = repo_root / "deploy/compose/docker-compose-ingestor-server.yaml"
    compose_nims_path = repo_root / "deploy/compose/nims.yaml"
    compose_vectordb_path = repo_root / "deploy/compose/vectordb.yaml"
    # Config with exemptions for env parity checks, kept alongside this test
    env_exemptions_path = repo_root / "tests/unit/test_compose_helm_parity/env_parity_exemptions.yaml"

    # Load YAMLs
    values = load_yaml(values_path)
    compose_rag = load_yaml(compose_rag_path)
    compose_ingestor = load_yaml(compose_ingestor_path)
    compose_nims = load_yaml(compose_nims_path)
    compose_vectordb = load_yaml(compose_vectordb_path)
    env_exemptions = load_yaml(env_exemptions_path) if env_exemptions_path.exists() else {}

    # Mapping between docker-compose services and Helm values.yaml
    mapping = {
        str(compose_rag_path): {
            "rag-server": {
                "values_image_repo_path": ["image", "repository"],
                "values_image_tag_path": ["image", "tag"],
                "values_env_path": ["envVars"],
                # dynamic env parity from compose
                "require_all_env_from_compose": True,
                "require_values_match": True,
                # API keys are represented via Helm secrets, not envVars
                "ignore_env_keys": {"NGC_API_KEY", "NVIDIA_API_KEY"},
            },
            "rag-frontend": {
                "values_image_repo_path": ["frontend", "image", "repository"],
                "values_image_tag_path": ["frontend", "image", "tag"],
                "values_env_path": ["frontend", "envVars"],
                # enforce presence of these runtime names
                "required_env_names": [
                    "VITE_API_CHAT_URL",
                    "VITE_API_VDB_URL",
                ],
            },
        },
        str(compose_ingestor_path): {
            "ingestor-server": {
                "values_image_repo_path": ["ingestor-server", "image", "repository"],
                "values_image_tag_path": ["ingestor-server", "image", "tag"],
                "values_env_path": ["ingestor-server", "envVars"],
                # dynamic env parity from compose, with known exceptions
                "require_all_env_from_compose": True,
                "require_values_match": True,
                "ignore_env_keys": {"NGC_API_KEY", "NVIDIA_API_KEY"},
            },
            # nv-ingest runtime parity against Helm nv-ingest.envVars
            "nv-ingest-ms-runtime": {
                # image parity is not enforced here; only env parity
                "values_env_path": ["nv-ingest", "envVars"],
                "require_all_env_from_compose": True,
                "require_values_match": True,
                # API keys and deploy/runtime-only knobs may legitimately differ
                "ignore_env_keys": set(),
            },
        },
        str(compose_nims_path): {
            # Image parity checks only; env parity enforced for API key presence
            "nim-llm": {
                "values_image_repo_path": ["nim-llm", "image", "repository"],
                "values_image_tag_path": ["nim-llm", "image", "tag"],
                "requires_ngc_api_key_path": ["nim-llm", "model", "ngcAPIKey"],
            },
            "nemoretriever-embedding-ms": {
                "values_image_repo_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "image", "repository"],
                "values_image_tag_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "image", "tag"],
                "requires_ngc_api_key_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "nim", "ngcAPIKey"],
            },
            "nemoretriever-ranking-ms": {
                "values_image_repo_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "image", "repository"],
                "values_image_tag_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "image", "tag"],
                "requires_ngc_api_key_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "nim", "ngcAPIKey"],
            },
            "vlm-ms": {
                "values_image_repo_path": ["nim-vlm", "image", "repository"],
                "values_image_tag_path": ["nim-vlm", "image", "tag"],
                "requires_ngc_api_key_path": ["nim-vlm", "nim", "ngcAPIKey"],
            },
        },
        str(compose_vectordb_path): {
            # Third-party dependency: Milvus image parity (managed by nv-ingest subchart)
            "milvus": {
                "values_image_repo_path": ["nv-ingest", "milvus", "image", "all", "repository"],
                "values_image_tag_path": ["nv-ingest", "milvus", "image", "all", "tag"],
            },
            # Etcd image parity (managed by nv-ingest.milvus.etcd)
            "etcd": {
                "values_image_repo_path": ["nv-ingest", "milvus", "etcd", "image", "repository"],
                "values_image_tag_path": ["nv-ingest", "milvus", "etcd", "image", "tag"],
            },
            # MinIO image parity (managed by nv-ingest.milvus.minio)
            "minio": {
                "values_image_repo_path": ["nv-ingest", "milvus", "minio", "image", "repository"],
                "values_image_tag_path": ["nv-ingest", "milvus", "minio", "image", "tag"],
            },
        },
    }

    def assert_image_parity(compose_service: dict[str, object], values_repo_path: list[str], values_tag_path: list[str]):
        compose_image = compose_service.get("image")
        assert compose_image, "compose image must be set"
        repo, tag = parse_image(str(compose_image))
        values_repo = get_dict_path(values, values_repo_path)
        values_tag = get_dict_path(values, values_tag_path)
        assert values_repo == repo, f"Repository mismatch: {values_repo_path}='{values_repo}' != '{repo}'"
        if tag is not None:
            assert values_tag == tag, f"Tag mismatch: {values_tag_path}='{values_tag}' != '{tag}'"

    def get_value_exemptions(compose_file: str, service_name: str) -> list[str]:
        """Return glob patterns for env value mismatch exemptions for a given service.
        Config format (optional file tests/unit/test_compose_helm_parity/env_parity_exemptions.yaml):
        envValueExemptions:
          global:
            - "NIM_*"
          perService:
            docker-compose-ingestor-server.yaml:
              nv-ingest-ms-runtime:
                - "MESSAGE_CLIENT_HOST"
        """
        cfg = env_exemptions.get("envValueExemptions", {}) if isinstance(env_exemptions, dict) else {}
        patterns: list[str] = []
        global_list = cfg.get("global", []) if isinstance(cfg.get("global"), list) else []
        patterns.extend([str(p) for p in global_list if isinstance(p, (str, int, float))])
        per_service = cfg.get("perService", {}) if isinstance(cfg.get("perService"), dict) else {}
        compose_basename = Path(compose_file).name
        file_map = per_service.get(compose_basename, {}) if isinstance(per_service.get(compose_basename), dict) else {}
        service_list = file_map.get(service_name, []) if isinstance(file_map.get(service_name), list) else []
        patterns.extend([str(p) for p in service_list if isinstance(p, (str, int, float))])
        return patterns

    def get_missing_key_exemptions(compose_file: str, service_name: str) -> set[str]:
        """Return set of env keys exempted from missing-key parity for a given service.
        Config format (optional file tests/unit/test_compose_helm_parity/env_parity_exemptions.yaml):
        envMissingKeyExemptions:
          perService:
            docker-compose-ingestor-server.yaml:
              nv-ingest-ms-runtime:
                - NGC_API_KEY
        """
        cfg = env_exemptions.get("envMissingKeyExemptions", {}) if isinstance(env_exemptions, dict) else {}
        per_service = cfg.get("perService", {}) if isinstance(cfg.get("perService"), dict) else {}
        compose_basename = Path(compose_file).name
        file_map = per_service.get(compose_basename, {}) if isinstance(per_service.get(compose_basename), dict) else {}
        service_list = file_map.get(service_name, []) if isinstance(file_map.get(service_name), list) else []
        return {str(k) for k in service_list if isinstance(k, (str, int, float))}

    def is_image_parity_exempt(compose_file: str, service_name: str) -> bool:
        """Return True if image parity should be skipped for this service based on optional config.
        Config format (optional file tests/unit/test_compose_helm_parity/env_parity_exemptions.yaml):
        imageParityExemptions:
          global:
            - serviceName
          perService:
            vectordb.yaml:
              milvus: true
              etcd: true
              minio: true
        """
        if not isinstance(env_exemptions, dict):
            return False
        cfg = env_exemptions.get("imageParityExemptions", {})
        if not isinstance(cfg, dict):
            return False
        global_list = cfg.get("global", [])
        if isinstance(global_list, list):
            for name in global_list:
                if str(name) == service_name:
                    return True
        per_service = cfg.get("perService", {})
        if not isinstance(per_service, dict):
            return False
        compose_basename = Path(compose_file).name
        file_map = per_service.get(compose_basename)
        if not isinstance(file_map, dict):
            return False
        return bool(file_map.get(service_name))

    def is_value_exempt(env_key: str, exempt_exact: set[str] | None, exempt_globs: list[str] | None) -> bool:
        if exempt_exact and env_key in exempt_exact:
            return True
        if exempt_globs:
            for pat in exempt_globs:
                try:
                    if fnmatch.fnmatch(env_key, pat):
                        return True
                except Exception:
                    continue
        return False

    def assert_env_presence(
        compose_service: dict[str, object],
        values_env_path: list[str],
        required_keys: list[str] | None = None,
        required_names: list[str] | None = None,
        require_all_from_compose: bool = False,
        ignore_env_keys: set[str] | None = None,
        require_values_match: bool = False,
        value_mismatch_exempt_keys: set[str] | None = None,
        value_mismatch_exempt_globs: list[str] | None = None,
    ):
        values_env = get_dict_path(values, values_env_path)
        # values env for rag-server/ingestor-server is a dict; for frontend it is a list of {name,value}
        if required_keys:
            assert isinstance(values_env, dict), f"Expected dict at values path {values_env_path}"
            for key in required_keys:
                assert key in values_env, f"Missing env key '{key}' in values at {values_env_path}"
        if required_names:
            names = ensure_frontend_env_names(values_env)
            for name in required_names:
                assert name in names, f"Missing env name '{name}' in values at {values_env_path}"
        if require_all_from_compose:
            compose_env = compose_service.get("environment", {})
            compose_names = extract_compose_env_names(compose_env)
            if ignore_env_keys:
                compose_names -= set(ignore_env_keys)
            values_keys = extract_values_env_keys(values_env)
            missing = compose_names - values_keys
            assert not missing, (
                f"Missing env keys from compose in Helm at {values_env_path}: {sorted(missing)}"
            )
            if require_values_match:
                compose_kv = extract_compose_env_kv(compose_env)
                values_kv = extract_values_env_kv(values_env)
                diffs: list[str] = []
                for k in sorted(compose_names):
                    if is_value_exempt(k, value_mismatch_exempt_keys, value_mismatch_exempt_globs):
                        continue
                    raw_left = compose_kv.get(k)
                    raw_right = values_kv.get(k)
                    left = normalize_env_value(raw_left)
                    right = normalize_env_value(raw_right)
                    if left != right:
                        is_ph_l, def_l = parse_env_default(raw_left) if isinstance(raw_left, str) else (False, None)
                        is_ph_r, def_r = parse_env_default(raw_right) if isinstance(raw_right, str) else (False, None)
                        diffs.append(
                            (
                                f"{k}:\n"
                                f"  compose.raw={raw_left!r}\n"
                                f"  compose.normalized={left!r}\n"
                                f"  compose.placeholder={{is_placeholder={is_ph_l}, default={def_l!r}}}\n"
                                f"  helm.raw={raw_right!r}\n"
                                f"  helm.normalized={right!r}\n"
                                f"  helm.placeholder={{is_placeholder={is_ph_r}, default={def_r!r}}}"
                            )
                        )
                assert not diffs, (
                    "Env value mismatches (compose vs Helm) at "
                    f"{values_env_path} (service keys compared: {sorted(compose_names)}):\n- " + "\n- ".join(diffs)
                )

    # Execute checks
    for compose_file, services_spec in mapping.items():
        compose_data = {
            str(compose_rag_path): compose_rag,
            str(compose_ingestor_path): compose_ingestor,
            str(compose_nims_path): compose_nims,
            str(compose_vectordb_path): compose_vectordb,
        }[compose_file]
        services = compose_data.get("services", {})
        for svc_name, rules in services_spec.items():
            assert svc_name in services, f"Service '{svc_name}' not found in {compose_file}"
            svc = services[svc_name]
            if "values_image_repo_path" in rules and "values_image_tag_path" in rules:
                if not is_image_parity_exempt(compose_file, svc_name):
                    assert_image_parity(
                        svc,
                        rules["values_image_repo_path"],
                        rules["values_image_tag_path"],
                    )
            if "values_env_path" in rules:
                exempt_globs = get_value_exemptions(compose_file, svc_name)
                missing_key_exempt = get_missing_key_exemptions(compose_file, svc_name)
                ignore_keys_combined = set(rules.get("ignore_env_keys", set()) or set()) | missing_key_exempt
                assert_env_presence(
                    svc,
                    rules["values_env_path"],
                    rules.get("required_env_keys"),
                    rules.get("required_env_names"),
                    rules.get("require_all_env_from_compose", False),
                    ignore_keys_combined,
                    rules.get("require_values_match", False),
                    set(rules.get("value_mismatch_exempt_keys", set()) or []),
                    exempt_globs,
                )

            # For NIM services ensure API key is configurable in Helm when present in compose
            if "requires_ngc_api_key_path" in rules:
                compose_env = svc.get("environment")
                # Some NIM services use list form; check for presence of NGC_API_KEY in any form
                compose_env_names = extract_compose_env_names(compose_env)
                if "NGC_API_KEY" in compose_env_names:
                    ngc_key_path = rules["requires_ngc_api_key_path"]
                    ngc_key_value = get_dict_path(values, ngc_key_path)
                    assert ngc_key_value is not None, (
                        f"Expected Helm values key for NGC API under {ngc_key_path}, but it was missing"
                    )


def test_prompt_yaml_parity():
    repo_root = Path(__file__).resolve().parents[3]
    src_prompt = repo_root / "src/nvidia_rag/rag_server/prompt.yaml"
    helm_prompt = repo_root / "deploy/helm/nvidia-blueprint-rag/files/prompt.yaml"

    assert src_prompt.exists(), f"Missing source prompt file: {src_prompt}"
    assert helm_prompt.exists(), f"Missing Helm prompt file: {helm_prompt}"

    src_content = src_prompt.read_text(encoding="utf-8").strip()
    helm_content = helm_prompt.read_text(encoding="utf-8").strip()

    assert src_content == helm_content, (
        "prompt.yaml mismatch: src and Helm copies must be identical. "
        "Update deploy/helm/nvidia-blueprint-rag/files/prompt.yaml to reflect changes in src/nvidia_rag/rag_server/prompt.yaml"
    )
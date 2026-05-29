"""Checks for deployable NeMo Guardrails configurations."""

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDRAILS_CONFIG_DIR = (
    REPO_ROOT / "deploy/compose/nemoguardrails/config-store"
)


def _load_config(config_name: str) -> dict[str, Any]:
    config_path = GUARDRAILS_CONFIG_DIR / config_name / "config.yml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_guardrails_output_rails_enable_streaming() -> None:
    """RAG server streams by default, so output rails must support streaming."""
    for config_name in ("nemoguard", "nemoguard_cloud"):
        config = _load_config(config_name)

        assert (
            config["rails"]["output"]["streaming"]["enabled"] is True
        ), f"{config_name} must enable rails.output.streaming"

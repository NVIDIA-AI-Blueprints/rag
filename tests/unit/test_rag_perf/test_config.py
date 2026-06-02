# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``rag_perf.config``: loading, overrides, validation, serialisation."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_perf.config import (
    InputConfig,
    InputSource,
    LoadConfig,
    LoadMode,
    RagParams,
    RunConfig,
)


class TestConfig:
    """Tests for ``rag_perf.config`` — loading, overrides, validation, serialisation."""

    # ── Default values ───────────────────────────────────────────────────────

    def test_default_config_is_valid(self):
        cfg = RunConfig.defaults()
        assert cfg.target.url == "http://localhost:8081"
        assert cfg.load.concurrency == 8
        assert cfg.load.concurrency_list == [8]
        assert cfg.load.total_requests == 200
        assert cfg.rag.vdb_top_k == 100
        assert cfg.rag.reranker_top_k == 10
        assert cfg.rag.enable_citations is True
        assert cfg.aiperf.enabled is True
        # Default rag axes are scalars (single-point), not lists.
        assert isinstance(cfg.rag.vdb_top_k, int)
        assert isinstance(cfg.rag.reranker_top_k, int)
        assert cfg.load.iterations == 1
        assert cfg.load.sleep_between_points_s == 0

    def test_concurrency_accepts_scalar_or_list(self):
        scalar = RunConfig.defaults().with_overrides(load__concurrency=16)
        assert scalar.load.concurrency_list == [16]

        sweep = RunConfig.defaults().with_overrides(load__concurrency=[1, 4, 8])
        assert sweep.load.concurrency_list == [1, 4, 8]

    def test_aiperf_enabled_false_for_profile_only(self):
        cfg = RunConfig.defaults().with_overrides(aiperf__enabled=False)
        assert cfg.aiperf.enabled is False

    # ── YAML loading ─────────────────────────────────────────────────────────

    def test_from_yaml_loads_correctly(self, tmp_path: Path):
        yaml_content = textwrap.dedent(
            """\
            target:
              url: "http://my-server:8081"
              timeout_s: 60
            load:
              concurrency: 16
              total_requests: 100
            rag:
              collection_names: ["my-collection"]
              vdb_top_k: 50
              reranker_top_k: 5
            input:
              file: "queries.jsonl"
        """
        )
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        cfg = RunConfig.from_yaml(yaml_path)

        assert cfg.target.url == "http://my-server:8081"
        assert cfg.target.timeout_s == 60
        assert cfg.load.concurrency == 16
        assert cfg.load.total_requests == 100
        assert cfg.rag.collection_names == ["my-collection"]
        assert cfg.rag.vdb_top_k == 50
        assert cfg.rag.reranker_top_k == 5

    def test_from_yaml_empty_file_uses_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = RunConfig.from_yaml(yaml_path)
        assert cfg.load.concurrency == 8

    def test_from_yaml_partial_file_fills_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("load:\n  concurrency: 32\n")
        cfg = RunConfig.from_yaml(yaml_path)
        assert cfg.load.concurrency == 32
        assert cfg.load.total_requests == 200
        assert cfg.rag.vdb_top_k == 100

    # ── with_overrides() ─────────────────────────────────────────────────────

    def test_with_overrides_flat_key(self):
        cfg = RunConfig.defaults()
        new = cfg.with_overrides(load__concurrency=32)
        assert new.load.concurrency == 32
        assert cfg.load.concurrency == 8

    def test_with_overrides_multiple_keys(self):
        cfg = RunConfig.defaults()
        new = cfg.with_overrides(
            load__concurrency=16,
            load__total_requests=500,
            rag__vdb_top_k=50,
            rag__reranker_top_k=5,
            target__url="http://new-server:8081",
        )
        assert new.load.concurrency == 16
        assert new.load.total_requests == 500
        assert new.rag.vdb_top_k == 50
        assert new.rag.reranker_top_k == 5
        assert new.target.url == "http://new-server:8081"

    def test_with_overrides_list_value(self):
        cfg = RunConfig.defaults()
        new = cfg.with_overrides(rag__collection_names=["col1", "col2"])
        assert new.rag.collection_names == ["col1", "col2"]

    def test_with_overrides_unknown_top_key_raises(self):
        cfg = RunConfig.defaults()
        with pytest.raises(KeyError, match="nonexistent"):
            cfg.with_overrides(nonexistent__field=1)

    def test_with_overrides_unknown_leaf_key_raises(self):
        cfg = RunConfig.defaults()
        with pytest.raises(KeyError, match="typo_key"):
            cfg.with_overrides(load__typo_key=8)

    def test_with_overrides_does_not_mutate_original(self):
        cfg = RunConfig.defaults()
        _ = cfg.with_overrides(load__concurrency=99)
        assert cfg.load.concurrency == 8

    # ── Validation ───────────────────────────────────────────────────────────

    def test_vdb_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            RagParams(vdb_top_k=0)
        with pytest.raises(ValidationError):
            RagParams(vdb_top_k=401)

    def test_reranker_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            RagParams(reranker_top_k=26)

    def test_confidence_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            RagParams(confidence_threshold=1.5)

    def test_request_rate_mode_requires_rate(self):
        with pytest.raises(ValidationError, match="request_rate"):
            LoadConfig(mode=LoadMode.REQUEST_RATE, request_rate=None)

    def test_request_rate_mode_valid(self):
        cfg = LoadConfig(mode=LoadMode.REQUEST_RATE, request_rate=5.0)
        assert cfg.request_rate == 5.0

    def test_input_config_neither_set_defaults_to_synthetic(self):
        # Neither file nor synthetic → auto-default to synthetic mode so
        # RunConfig.defaults() / bare YAML still validate.
        cfg = InputConfig()
        assert cfg.file is None
        assert cfg.synthetic is not None
        assert cfg.source == InputSource.SYNTHETIC

    def test_input_config_rejects_both_file_and_synthetic(self):
        from rag_perf.config import SyntheticInputConfig

        with pytest.raises(ValidationError, match="mutually exclusive"):
            InputConfig(file="x.jsonl", synthetic=SyntheticInputConfig())

    def test_input_config_file_only_accepts_jsonl_or_csv(self):
        with pytest.raises(ValidationError, match=r"\.jsonl or \.csv"):
            InputConfig(file="queries.txt")

    def test_input_config_source_property_jsonl(self):
        cfg = InputConfig(file="examples/queries.jsonl")
        assert cfg.source == InputSource.JSONL

    def test_input_config_source_property_csv(self):
        cfg = InputConfig(file="examples/queries.csv")
        assert cfg.source == InputSource.CSV

    def test_input_config_source_property_synthetic(self):
        from rag_perf.config import SyntheticInputConfig

        cfg = InputConfig(synthetic=SyntheticInputConfig())
        assert cfg.source == InputSource.SYNTHETIC

    # ── Serialisation round-trip ──────────────────────────────────────────────

    def test_to_yaml_str_produces_valid_yaml(self):
        from ruamel.yaml import YAML

        cfg = RunConfig.defaults()
        yaml_str = cfg.to_yaml_str()

        assert "target" in yaml_str
        assert "load" in yaml_str

        yaml = YAML(typ="safe")
        data = yaml.load(io.StringIO(yaml_str))
        assert data["load"]["concurrency"] == 8

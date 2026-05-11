# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python unit tests for provision_adb.py helpers.

These tests exercise the parts of the provisioner that do not touch OCI or
Oracle: name normalization, password generation, env-file writing, and
argparse plumbing including the new ``--pai-offload-enabled`` switch.
"""
from __future__ import annotations

import argparse
import os
import re
import string
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestNormalizeDbName:
    def test_lowercases_and_strips_non_alnum(self, provisioner_module):
        assert provisioner_module.normalize_db_name("Rag-BP_DEV!") == "ragbpdev"

    def test_empty_input_falls_back_to_ragbp(self, provisioner_module):
        assert provisioner_module.normalize_db_name("") == "ragbp"

    def test_only_special_chars_falls_back_to_ragbp(self, provisioner_module):
        assert provisioner_module.normalize_db_name("@#$%") == "ragbp"

    def test_starts_with_letter_when_input_starts_with_digit(self, provisioner_module):
        out = provisioner_module.normalize_db_name("9rag")
        assert out[0].isalpha()
        # Original alphanumerics preserved after the synthetic prefix
        assert "rag" in out

    def test_capped_at_30_chars(self, provisioner_module):
        out = provisioner_module.normalize_db_name("a" * 100)
        assert len(out) == 30
        assert all(c == "a" for c in out)


class TestStableSuffix:
    def test_deterministic_for_same_inputs(self, provisioner_module):
        a = provisioner_module.stable_suffix("rag", "default")
        b = provisioner_module.stable_suffix("rag", "default")
        assert a == b
        assert len(a) == 6
        assert re.fullmatch(r"[0-9a-f]+", a)

    def test_changes_with_inputs(self, provisioner_module):
        a = provisioner_module.stable_suffix("rag", "default")
        b = provisioner_module.stable_suffix("rag", "staging")
        assert a != b

    def test_skips_empty_parts(self, provisioner_module):
        # `|` join skips empty parts; ("a", "") and ("a",) hash identically
        a = provisioner_module.stable_suffix("a", "")
        b = provisioner_module.stable_suffix("a")
        assert a == b


class TestGeneratePassword:
    @pytest.mark.parametrize("length", [16, 24, 30])
    def test_meets_adb_complexity_rules(self, provisioner_module, length):
        for _ in range(20):
            pw = provisioner_module.generate_password(length=length)
            assert len(pw) == length
            assert any(c.islower() for c in pw)
            assert any(c.isupper() for c in pw)
            assert any(c.isdigit() for c in pw)
            assert any(c in "!#$%*-_+" for c in pw)

    def test_no_quotes_or_backslash(self, provisioner_module):
        """ADB passwords are embedded in ALTER USER … IDENTIFIED BY "<pw>";
        quote/backslash chars would break that quoting."""
        for _ in range(20):
            pw = provisioner_module.generate_password()
            assert '"' not in pw
            assert "'" not in pw
            assert "\\" not in pw
            assert "`" not in pw


class TestRegionFromOcid:
    def test_extracts_region_from_instance_ocid(self, provisioner_module):
        ocid = "ocid1.instance.oc1.us-chicago-1.abc123"
        assert provisioner_module.region_from_ocid(ocid) == "us-chicago-1"

    def test_returns_none_for_short_ocid(self, provisioner_module):
        assert provisioner_module.region_from_ocid("ocid1.x") is None

    def test_returns_none_for_non_ocid(self, provisioner_module):
        assert provisioner_module.region_from_ocid("not-an-ocid") is None


class TestDefaultNames:
    def test_explicit_names_pass_through_unchanged(self, provisioner_module, monkeypatch):
        monkeypatch.setenv("HELM_RELEASE_NAME", "rag")
        db, disp = provisioner_module.default_names("explicit_db_name", "Explicit Display")
        assert db == "explicitdbname"  # normalized
        assert disp == "Explicit Display"

    def test_uses_helm_release_in_display_when_unset(self, provisioner_module, monkeypatch):
        monkeypatch.setenv("HELM_RELEASE_NAME", "myrelease")
        monkeypatch.delenv("ORACLE_DB_NAME", raising=False)
        monkeypatch.delenv("ORACLE_DB_DISPLAY_NAME", raising=False)
        db, disp = provisioner_module.default_names(None, None)
        assert db.startswith("ragbp")
        assert "myrelease" in disp

    def test_db_name_capped_at_30(self, provisioner_module):
        very_long = "x" * 200
        db, _ = provisioner_module.default_names(very_long, None)
        assert len(db) <= 30

    def test_display_name_capped_at_80(self, provisioner_module):
        db, disp = provisioner_module.default_names(None, "y" * 200)
        assert len(disp) == 80


# ---------------------------------------------------------------------------
# write_env
# ---------------------------------------------------------------------------
class TestWriteEnv:
    def test_writes_required_keys(self, provisioner_module, tmp_path):
        path = tmp_path / "oracle.env"
        provisioner_module.write_env(
            str(path), dsn="my_medium", rag_password="Sup3rSecret!",
        )
        body = path.read_text()
        assert "APP_VECTORSTORE_NAME=oracle" in body
        assert "APP_VECTORSTORE_SEARCHTYPE=hybrid" in body
        # HNSW so cuVS OFFLOAD_URL actually engages — IVF would silently NOT use cuVS
        assert "APP_VECTORSTORE_INDEXTYPE=HNSW" in body
        assert "ORACLE_VECTOR_INDEX_TYPE=HNSW" in body
        assert "ORACLE_USER=RAG_APP" in body
        assert "ORACLE_PASSWORD=Sup3rSecret!" in body
        assert "ORACLE_CS=my_medium" in body

    def test_writes_pai_index_url_when_provided(self, provisioner_module, tmp_path):
        path = tmp_path / "oracle.env"
        provisioner_module.write_env(
            str(path),
            dsn="x_medium",
            rag_password="pw",
            pai_index_url="http://10.0.50.42:8080/v1/index",
        )
        body = path.read_text()
        assert "ORACLE_PAI_INDEX_URL=http://10.0.50.42:8080/v1/index" in body

    def test_omits_pai_index_url_when_empty(self, provisioner_module, tmp_path):
        path = tmp_path / "oracle.env"
        provisioner_module.write_env(str(path), dsn="x", rag_password="pw")
        assert "ORACLE_PAI_INDEX_URL" not in path.read_text()

    def test_creates_parent_directory(self, provisioner_module, tmp_path):
        path = tmp_path / "nested" / "deep" / "oracle.env"
        provisioner_module.write_env(str(path), dsn="x", rag_password="pw")
        assert path.exists()


# ---------------------------------------------------------------------------
# write_k8s_secret (mocked kubernetes client)
# ---------------------------------------------------------------------------
class TestWriteK8sSecret:
    def _mk_args(self):
        """Return a SimpleNamespace shaped like argparse output for default flow."""
        from types import SimpleNamespace
        return SimpleNamespace(
            namespace="rag",
            k8s_secret="oracle-creds",
            kubeconfig=None,
        )

    def test_replace_path_when_secret_exists(
        self, provisioner_module, monkeypatch
    ):
        import base64
        recorded = {}

        class FakeV1Secret:
            def __init__(self, metadata, type, data):
                self.metadata = metadata
                self.type = type
                self.data = data

        class FakeMeta:
            def __init__(self, name, namespace):
                self.name = name
                self.namespace = namespace

        class FakeApi:
            def replace_namespaced_secret(self, name, namespace, body):
                recorded["op"] = "replace"
                recorded["name"] = name
                recorded["namespace"] = namespace
                recorded["data"] = body.data
            def create_namespaced_secret(self, namespace, body):  # pragma: no cover
                raise AssertionError("should not create when replace succeeds")

        class FakeK8sClient:
            CoreV1Api = lambda self=None: FakeApi()  # noqa: E731
            V1Secret = staticmethod(FakeV1Secret)
            V1ObjectMeta = staticmethod(FakeMeta)

        monkeypatch.setattr(provisioner_module, "k8s_client", FakeK8sClient())
        # k8s_config truthy so the early-return is skipped
        monkeypatch.setattr(provisioner_module, "k8s_config", object())
        monkeypatch.setattr(provisioner_module, "load_kube_config", lambda kc: None)

        provisioner_module.write_k8s_secret(
            namespace="rag",
            name="oracle-creds",
            dsn="my_medium",
            rag_password="pw",
            kubeconfig=None,
            pai_index_url="http://10.0.50.42:8080/v1/index",
        )

        assert recorded["op"] == "replace"
        assert recorded["namespace"] == "rag"
        assert recorded["name"] == "oracle-creds"
        # All three creds plus PAI URL stamped
        assert "ORACLE_USER" in recorded["data"]
        assert "ORACLE_PASSWORD" in recorded["data"]
        assert "ORACLE_CS" in recorded["data"]
        assert "ORACLE_PAI_INDEX_URL" in recorded["data"]
        assert (
            base64.b64decode(recorded["data"]["ORACLE_PAI_INDEX_URL"]).decode()
            == "http://10.0.50.42:8080/v1/index"
        )

    def test_create_path_when_replace_404s(self, provisioner_module, monkeypatch):
        seq = []

        class FakeApi:
            def replace_namespaced_secret(self, name, namespace, body):
                seq.append("replace")
                raise RuntimeError("404")
            def create_namespaced_secret(self, namespace, body):
                seq.append("create")

        class FakeK8sClient:
            CoreV1Api = lambda self=None: FakeApi()  # noqa: E731
            V1Secret = staticmethod(lambda metadata, type, data: object())
            V1ObjectMeta = staticmethod(lambda name, namespace: object())

        monkeypatch.setattr(provisioner_module, "k8s_client", FakeK8sClient())
        monkeypatch.setattr(provisioner_module, "k8s_config", object())
        monkeypatch.setattr(provisioner_module, "load_kube_config", lambda kc: None)

        provisioner_module.write_k8s_secret(
            namespace="rag", name="oracle-creds",
            dsn="x", rag_password="pw", kubeconfig=None,
        )
        assert seq == ["replace", "create"]

    def test_no_op_without_kubernetes_lib(self, provisioner_module, monkeypatch, capsys):
        monkeypatch.setattr(provisioner_module, "k8s_config", None)
        provisioner_module.write_k8s_secret(
            namespace="rag", name="oracle-creds",
            dsn="x", rag_password="pw", kubeconfig=None,
        )
        out = capsys.readouterr().out
        assert "skipping" in out.lower()

    def test_omits_pai_url_key_when_empty(self, provisioner_module, monkeypatch):
        recorded = {}

        class FakeApi:
            def replace_namespaced_secret(self, name, namespace, body):
                recorded["data"] = body.data
            def create_namespaced_secret(self, namespace, body):
                recorded["data"] = body.data

        class FakeV1Secret:
            def __init__(self, metadata, type, data):
                self.data = data

        class FakeK8sClient:
            CoreV1Api = lambda self=None: FakeApi()  # noqa: E731
            V1Secret = staticmethod(FakeV1Secret)
            V1ObjectMeta = staticmethod(lambda name, namespace: object())

        monkeypatch.setattr(provisioner_module, "k8s_client", FakeK8sClient())
        monkeypatch.setattr(provisioner_module, "k8s_config", object())
        monkeypatch.setattr(provisioner_module, "load_kube_config", lambda kc: None)

        provisioner_module.write_k8s_secret(
            namespace="rag", name="oracle-creds",
            dsn="x", rag_password="pw", kubeconfig=None, pai_index_url="",
        )
        assert "ORACLE_PAI_INDEX_URL" not in recorded["data"]


# ---------------------------------------------------------------------------
# parse_create_args
# ---------------------------------------------------------------------------
class TestParseCreateArgs:
    def _ns(self, **overrides):
        """Build the argparse Namespace shape parse_create_args expects."""
        defaults = dict(
            compartment_id=None, subnet_id=None, vcn_id=None, region=None,
            db_name=None, display_name=None, workload_type=None,
            admin_password=None, rag_app_password=None,
            ecpus=None, storage_tb=None,
            namespace="rag", k8s_secret="oracle-creds", output_env=None,
            wait_seconds=10, poll_seconds=1,
            auto_discover=True, reuse_existing=False, kubeconfig=None,
            pai_index_url="", pai_offload_enabled=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_workload_is_LH(self, provisioner_module, monkeypatch):
        for k in ("ORACLE_DB_WORKLOAD_TYPE", "ORACLE_PAI_INDEX_URL", "ORACLE_DB_NAME"):
            monkeypatch.delenv(k, raising=False)
        cfg = provisioner_module.parse_create_args(self._ns())
        assert cfg.workload_type == "LH"

    def test_workload_type_uppercased(self, provisioner_module):
        cfg = provisioner_module.parse_create_args(self._ns(workload_type="oltp"))
        assert cfg.workload_type == "OLTP"

    def test_pai_offload_enabled_flag_propagates(self, provisioner_module):
        cfg = provisioner_module.parse_create_args(self._ns(pai_offload_enabled=True))
        assert cfg.pai_offload_enabled is True

    def test_pai_index_url_implies_offload_enabled(self, provisioner_module):
        """Backwards compat: passing --pai-index-url alone (legacy v1 chart)
        should enable the security-path-opening best-effort, even without the
        explicit --pai-offload-enabled flag."""
        cfg = provisioner_module.parse_create_args(
            self._ns(pai_index_url="http://x:8080/v1/index"),
        )
        assert cfg.pai_offload_enabled is True
        assert cfg.pai_index_url == "http://x:8080/v1/index"

    def test_no_pai_means_offload_disabled(self, provisioner_module, monkeypatch):
        monkeypatch.delenv("ORACLE_PAI_INDEX_URL", raising=False)
        cfg = provisioner_module.parse_create_args(self._ns())
        assert cfg.pai_offload_enabled is False
        assert cfg.pai_index_url == ""

    def test_pai_index_url_falls_back_to_env(self, provisioner_module, monkeypatch):
        monkeypatch.setenv("ORACLE_PAI_INDEX_URL", "http://env-url:8080/v1/index")
        cfg = provisioner_module.parse_create_args(self._ns())
        assert cfg.pai_index_url == "http://env-url:8080/v1/index"
        assert cfg.pai_offload_enabled is True

    def test_explicit_arg_wins_over_env(self, provisioner_module, monkeypatch):
        monkeypatch.setenv("ORACLE_PAI_INDEX_URL", "http://from-env:8080/v1/index")
        cfg = provisioner_module.parse_create_args(
            self._ns(pai_index_url="http://from-flag:8080/v1/index"),
        )
        assert cfg.pai_index_url == "http://from-flag:8080/v1/index"

    def test_passwords_default_to_generated(self, provisioner_module, monkeypatch):
        for k in ("ORACLE_ADMIN_PASSWORD", "ORACLE_RAG_APP_PASSWORD"):
            monkeypatch.delenv(k, raising=False)
        cfg = provisioner_module.parse_create_args(self._ns())
        # Generated passwords are at least 24 chars (default length)
        assert len(cfg.admin_password) >= 24
        assert len(cfg.rag_app_password) >= 24
        assert cfg.admin_password != cfg.rag_app_password

    def test_ecpus_storage_default_from_env(self, provisioner_module, monkeypatch):
        monkeypatch.setenv("ORACLE_DB_ECPUS", "8")
        monkeypatch.setenv("ORACLE_DB_STORAGE_TB", "2")
        cfg = provisioner_module.parse_create_args(self._ns())
        assert cfg.ecpus == 8
        assert cfg.storage_tb == 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance tests: OracleVDB implements every public method that
the dispatcher consumes from VDBRag (and from the parallel Milvus /
Elasticsearch implementations).

This catches the class of bug where:

  * RAG calls ``vdb.health_check()`` but Oracle only defines
    ``check_health()`` — silent AttributeError at runtime.
  * Milvus returns ``list[dict]`` from ``list_collections`` but Oracle
    returns ``list[str]`` — frontend breaks.
  * A new method gets added to VDBRag base; Oracle silently lags.

By inspecting the actual class hierarchy at import time, these tests
prove every consumer-visible method is implemented BEFORE the customer
hits it at runtime.
"""
from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest


# These tests exercise live Python imports, so guard on oracledb being
# available (it's a runtime dep declared via the [oracle] extra).
oracledb = pytest.importorskip("oracledb")


# ---------------------------------------------------------------------------
# 1. OracleVDB inherits from VDBRag.
# ---------------------------------------------------------------------------
def test_oracle_vdb_inherits_from_vdb_rag_base():
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
    from nvidia_rag.utils.vdb.vdb_base import VDBRag

    assert issubclass(OracleVDB, VDBRag), (
        "OracleVDB must inherit from VDBRag — that's how the dispatcher "
        "treats it as polymorphic with Milvus / Elasticsearch."
    )


# ---------------------------------------------------------------------------
# 2. Every abstract method on VDBRag is concretely implemented.
# ---------------------------------------------------------------------------
def test_oracle_vdb_implements_every_abstract_method():
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
    from nvidia_rag.utils.vdb.vdb_base import VDBRag

    abstract_methods = getattr(VDBRag, "__abstractmethods__", set())
    if not abstract_methods:
        pytest.skip("VDBRag has no abstract methods")
    not_implemented = set(getattr(OracleVDB, "__abstractmethods__", set()))
    assert not not_implemented, (
        f"OracleVDB still has abstract methods unimplemented: "
        f"{sorted(not_implemented)}. The dispatcher will raise "
        "TypeError at construction time."
    )


# ---------------------------------------------------------------------------
# 3. OracleVDB has every method that MilvusVDB has (signature parity).
# ---------------------------------------------------------------------------
def _public_methods(cls) -> dict:
    """Return public method names → (sig, is_async) for instance methods."""
    out = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        member = getattr(cls, name)
        if not callable(member):
            continue
        try:
            sig = inspect.signature(member)
        except (ValueError, TypeError):
            continue
        out[name] = sig
    return out


def test_oracle_vdb_has_every_public_method_milvus_does():
    """Mirror MilvusVDB. If we forget a method, the dispatcher will
    crash at runtime when something tries to call it."""
    try:
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB
    except ImportError as e:
        pytest.skip(f"MilvusVDB not importable: {e}")
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

    try:
        from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB
        elastic_methods = _public_methods(ElasticVDB)
    except ImportError:
        elastic_methods = {}

    milvus_methods = _public_methods(MilvusVDB)
    oracle_methods = _public_methods(OracleVDB)

    # Methods that are MILVUS-ONLY extensions (not in the VDBRag ABC,
    # not implemented by ElasticsearchVDB either). Verified by inspecting
    # vdb_base.py and elasticsearch/elastic_vdb.py — same omissions.
    MILVUS_ONLY_EXTENSIONS = (
        set(milvus_methods) - set(elastic_methods) - {"reindex"}
        if elastic_methods else set()
    )

    INTENTIONALLY_NOT_IMPLEMENTED = MILVUS_ONLY_EXTENSIONS | {
        # GPU-search: Oracle 26ai's vector engine runs on the DB CPU; no
        # equivalent Milvus-style GPU index runtime.
        "build_gpu_index",
        # Per-collection Milvus partition keys; Oracle uses one table per
        # collection so no partition concept.
        "create_partition", "drop_partition", "list_partitions",
    }
    missing = (set(milvus_methods) - set(oracle_methods)
               - INTENTIONALLY_NOT_IMPLEMENTED)
    assert not missing, (
        f"OracleVDB is missing public methods that MilvusVDB AND "
        f"ElasticsearchVDB both have: {sorted(missing)}. These appear "
        "to be part of the shared contract. Either implement them or "
        "document the omission in INTENTIONALLY_NOT_IMPLEMENTED above."
    )


# ---------------------------------------------------------------------------
# 4. Each shared method has a parameter-compatible signature.
# ---------------------------------------------------------------------------
def test_shared_methods_have_compatible_signatures():
    """For every method present in BOTH MilvusVDB and OracleVDB AND
    ElasticsearchVDB, the *required* parameter names must match (extra
    optional kwargs OK)."""
    try:
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB
        from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB
    except ImportError as e:
        pytest.skip(f"VDBs not importable: {e}")
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

    def _required_params(sig: inspect.Signature) -> list[str]:
        return [
            p.name for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and p.name != "self"
        ]

    milvus_methods = _public_methods(MilvusVDB)
    elastic_methods = _public_methods(ElasticVDB)
    oracle_methods = _public_methods(OracleVDB)

    leaks = []
    for name, milvus_sig in milvus_methods.items():
        if name not in oracle_methods:
            continue
        oracle_sig = oracle_methods[name]
        m_req = _required_params(milvus_sig)
        o_req = _required_params(oracle_sig)

        # Compare against Elasticsearch too — if Oracle's signature
        # matches Elasticsearch (which is the closest "thin" backend),
        # the dispatcher already handles that shape, so it's fine.
        if name in elastic_methods:
            e_req = _required_params(elastic_methods[name])
            if o_req == e_req:
                continue

        oracle_extra_required = set(o_req) - set(m_req)
        if oracle_extra_required:
            leaks.append(
                f"{name}: Oracle has extra REQUIRED params Milvus doesn't: "
                f"{sorted(oracle_extra_required)}"
            )
    assert not leaks, (
        "Signature drift between Milvus, Elasticsearch, and Oracle:\n  "
        + "\n  ".join(leaks)
    )


# ---------------------------------------------------------------------------
# 5. Smoke: every consumer-visible method on OracleVDB is callable.
# ---------------------------------------------------------------------------
def test_every_public_method_is_callable_attribute():
    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

    for name in dir(OracleVDB):
        if name.startswith("_"):
            continue
        member = getattr(OracleVDB, name, None)
        if member is None:
            continue
        # Either a method/function or a property/cached_property
        assert callable(member) or isinstance(member, (property,)), (
            f"OracleVDB.{name} is neither callable nor a property"
        )


# ---------------------------------------------------------------------------
# 6. Health check returns a dict with the keys the frontend expects.
# ---------------------------------------------------------------------------
def test_check_health_returns_dict_with_status_field():
    """Inspect the source: ``check_health`` must return a dict with
    a ``status`` key. The frontend's ``/v1/health`` endpoint relies on
    that. (We don't actually call it because it requires a live DB.)"""
    from pathlib import Path
    src_file = Path(__file__).resolve().parents[3] / "src" / "nvidia_rag" / "utils" / "vdb" / "oracle" / "oracle_vdb.py"
    src = src_file.read_text()
    assert 'def check_health' in src, "OracleVDB must define check_health"
    # Find the check_health method body and verify it sets a "status" key
    idx = src.index('def check_health')
    method_body = src[idx:idx+500]
    assert '"status"' in method_body or "'status'" in method_body, (
        "check_health must return a dict containing a 'status' key — "
        "the frontend reads health.status."
    )


# ---------------------------------------------------------------------------
# 7. Constructor accepts the same keyword args the dispatcher passes.
# ---------------------------------------------------------------------------
def test_constructor_accepts_dispatcher_kwargs():
    """Read the dispatcher's call site and verify OracleVDB.__init__
    accepts every kwarg the dispatcher passes. If we drift, the
    dispatcher hits a TypeError on the first ``vector_store=oracle`` call."""
    import re
    from pathlib import Path
    src = Path(
        inspect.getsourcefile(__import__("nvidia_rag.utils.vdb", fromlist=["__init__"]))
    ).read_text()
    # Find the OracleVDB(...) construction call
    m = re.search(
        r"OracleVDB\s*\(\s*([^)]+)\)",
        src, re.DOTALL,
    )
    assert m, "Dispatcher does not appear to call OracleVDB(...)"
    call_kwargs = re.findall(r"^\s*(\w+)\s*=", m.group(1), re.MULTILINE)

    from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB
    init_sig = inspect.signature(OracleVDB.__init__)
    init_param_names = set(init_sig.parameters) | {"self"}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in init_sig.parameters.values()
    )
    if has_var_keyword:
        return  # any kwargs accepted
    missing = [k for k in call_kwargs if k not in init_param_names]
    assert not missing, (
        f"Dispatcher passes kwargs {missing} that OracleVDB.__init__ "
        f"does not accept. Either add them or use **kwargs to absorb."
    )

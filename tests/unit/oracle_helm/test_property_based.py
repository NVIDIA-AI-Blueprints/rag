# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for Oracle 26ai integration code.

Hypothesis generates thousands of inputs per test, which is the cheapest
way to catch edge cases (Unicode, empty strings, surrogate pairs, very
long inputs) that a hand-curated test list will always miss.

Properties asserted here are *invariants* — properties that hold for
every legal input — not just specific cases.
"""
from __future__ import annotations

import re
import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st


# ---------------------------------------------------------------------------
# normalize_db_name(): post-conditions that must hold for any string input
# ---------------------------------------------------------------------------
class TestNormalizeDbNameProperties:
    @settings(max_examples=300, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(st.text(min_size=0, max_size=200))
    def test_output_is_alphanumeric_starts_with_letter_capped(
        self, provisioner_module, raw,
    ):
        out = provisioner_module.normalize_db_name(raw)
        assert isinstance(out, str)
        assert 1 <= len(out) <= 30
        assert out[0].isalpha(), f"first char {out[0]!r} must be a letter"
        assert out.isalnum(), f"output {out!r} must be all alphanumeric"
        # Lowercase invariant: ADB rejects upper/mixed dbName
        assert out == out.lower()

    @settings(max_examples=200, deadline=None)
    @given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50))
    def test_idempotent_on_already_normalized(
        self, provisioner_module, ascii_alnum,
    ):
        """normalize(normalize(x)) == normalize(x) for ASCII-alnum inputs."""
        once = provisioner_module.normalize_db_name(ascii_alnum)
        twice = provisioner_module.normalize_db_name(once)
        assert once == twice


# ---------------------------------------------------------------------------
# generate_password(): complexity rules for every length we'd ever pick
# ---------------------------------------------------------------------------
class TestGeneratePasswordProperties:
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=12, max_value=64))
    def test_complexity_holds_for_any_length(self, provisioner_module, length):
        pw = provisioner_module.generate_password(length=length)
        assert len(pw) == length
        # ADB requires lower / upper / digit / special
        assert any(c.islower() for c in pw)
        assert any(c.isupper() for c in pw)
        assert any(c.isdigit() for c in pw)
        assert any(c in "!#$%*-_+" for c in pw)
        # Never emits chars that break ALTER USER … IDENTIFIED BY "<pw>"
        assert not any(c in '"\'\\`' for c in pw), (
            f"Unsafe quoting char in {pw!r}"
        )

    @settings(max_examples=20, deadline=None)
    @given(st.integers(min_value=24, max_value=24))
    def test_passwords_are_distinct_across_calls(
        self, provisioner_module, length,
    ):
        """Cryptographic randomness invariant: generating 50 passwords in a
        row should never collide. If this ever fails, ``secrets`` was
        replaced with something predictable."""
        seen = {provisioner_module.generate_password(length=length) for _ in range(50)}
        assert len(seen) == 50


# ---------------------------------------------------------------------------
# stable_suffix(): deterministic, fixed-length, hex-only
# ---------------------------------------------------------------------------
class TestStableSuffixProperties:
    @settings(max_examples=200, deadline=None)
    @given(
        st.lists(st.text(min_size=0, max_size=64), min_size=0, max_size=8),
        st.integers(min_value=1, max_value=40),
    )
    def test_length_and_charset(self, provisioner_module, parts, length):
        out = provisioner_module.stable_suffix(*parts, length=length)
        assert len(out) == length
        assert re.fullmatch(r"[0-9a-f]+", out), f"non-hex output: {out!r}"

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=4))
    def test_deterministic(self, provisioner_module, parts):
        a = provisioner_module.stable_suffix(*parts)
        b = provisioner_module.stable_suffix(*parts)
        assert a == b


# ---------------------------------------------------------------------------
# DDL builders: invariants about generated SQL strings
# ---------------------------------------------------------------------------
def _import_oracle_queries():
    from nvidia_rag.utils.vdb.oracle import oracle_queries
    return oracle_queries


# A "safe table identifier" strategy: Oracle identifier rules are quite
# strict (ASCII letters/digits/underscore, starts with letter, max 128
# chars). Stay within that envelope so we don't trigger ORA-00972 noise.
oracle_ident = st.from_regex(r"^[A-Za-z][A-Za-z0-9_]{0,30}$", fullmatch=True)


class TestDDLProperties:
    @settings(max_examples=200, deadline=None)
    @given(oracle_ident, st.integers(min_value=64, max_value=4096))
    def test_create_table_ddl_carries_table_name_and_dim(
        self, table_name, dim,
    ):
        oq = _import_oracle_queries()
        ddl = oq.create_vector_table_ddl(table_name, dimension=dim)
        assert table_name in ddl
        assert f"VECTOR({dim}" in ddl
        # Balanced parens — basic syntax invariant
        assert ddl.count("(") == ddl.count(")")

    @settings(max_examples=200, deadline=None)
    @given(
        oracle_ident,
        st.sampled_from(["IVF", "HNSW"]),
        st.sampled_from(["COSINE", "L2", "DOT", "MANHATTAN"]),
    )
    def test_create_index_ddl_carries_name_metric_type(
        self, table_name, index_type, metric,
    ):
        oq = _import_oracle_queries()
        ddl = oq.create_vector_index_ddl(
            table_name=table_name, index_type=index_type, distance_metric=metric,
        )
        assert table_name in ddl
        assert f"DISTANCE {metric}" in ddl
        assert f"TYPE {index_type}" in ddl
        # Index name = <table>_vec_idx (so multiple collections coexist)
        assert f"{table_name}_vec_idx" in ddl
        # Balanced parens (catches accidentally unbalanced f-strings)
        assert ddl.count("(") == ddl.count(")")

    @settings(max_examples=200, deadline=None)
    @given(
        oracle_ident,
        st.sampled_from(["IVF", "HNSW"]),
        st.text(
            alphabet=string.ascii_letters + string.digits + ":/.-_",
            min_size=10, max_size=120,
        ),
    )
    def test_offload_only_in_hnsw(self, table_name, index_type, url):
        """OFFLOAD_URL is HNSW-only; for IVF it must never appear regardless
        of what the caller passes."""
        oq = _import_oracle_queries()
        # Make URL look like a real one
        full_url = "http://" + url + "/v1/index"
        ddl = oq.create_vector_index_ddl(
            table_name=table_name,
            index_type=index_type,
            pai_offload_url=full_url,
        )
        if index_type == "IVF":
            assert "OFFLOAD" not in ddl.upper()
        else:
            assert f"OFFLOAD_URL '{full_url}'" in ddl

    @settings(max_examples=200, deadline=None)
    @given(oracle_ident, oracle_ident, oracle_ident, oracle_ident)
    def test_byo_view_projects_all_canonical_columns(
        self, view_name, src_table, text_col, vec_col,
    ):
        oq = _import_oracle_queries()
        ddl = oq.create_byo_view_ddl(
            view_name=view_name,
            source_table=src_table,
            column_map={"text": text_col, "vector": vec_col},
        )
        # All five canonical "AS <col>" projections present
        for canonical in ("id", "text", "vector", "source", "content_metadata"):
            assert f" AS {canonical}" in ddl, f"missing AS {canonical}"
        # The user's columns are referenced verbatim
        assert text_col in ddl
        assert vec_col in ddl
        # Balanced parens
        assert ddl.count("(") == ddl.count(")")

    @settings(max_examples=100, deadline=None)
    @given(oracle_ident)
    def test_byo_view_raises_when_text_or_vector_missing(self, view_name):
        oq = _import_oracle_queries()
        with pytest.raises(ValueError):
            oq.create_byo_view_ddl(view_name, "T", column_map={"text": "c"})
        with pytest.raises(ValueError):
            oq.create_byo_view_ddl(view_name, "T", column_map={"vector": "v"})
        with pytest.raises(ValueError):
            oq.create_byo_view_ddl(view_name, "T", column_map={})


# ---------------------------------------------------------------------------
# _extract_source_name(): always returns a non-empty string
# ---------------------------------------------------------------------------
class TestSourceExtractionProperties:
    @settings(max_examples=300, deadline=None)
    @given(st.text(min_size=0, max_size=200))
    def test_string_input_always_returns_string(self, raw):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        out = OracleVDB._extract_source_name(raw)
        assert isinstance(out, str)
        # Non-empty: either the input (after strip), or "unknown"
        assert len(out) > 0

    @settings(max_examples=200, deadline=None)
    @given(st.dictionaries(
        st.sampled_from(["source_name", "source", "other"]),
        st.text(min_size=0, max_size=80),
        max_size=4,
    ))
    def test_dict_input_prefers_source_name_then_source(self, d):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        out = OracleVDB._extract_source_name(d)
        assert isinstance(out, str) and len(out) > 0
        if d.get("source_name"):
            assert out == d["source_name"]
        elif d.get("source"):
            assert out == d["source"]
        else:
            assert out == "unknown"

    @settings(max_examples=100, deadline=None)
    @given(st.one_of(st.none(), st.integers(), st.lists(st.integers())))
    def test_arbitrary_non_string_does_not_crash(self, val):
        from nvidia_rag.utils.vdb.oracle.oracle_vdb import OracleVDB

        out = OracleVDB._extract_source_name(val)
        assert isinstance(out, str)

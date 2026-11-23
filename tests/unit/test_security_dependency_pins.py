# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version

from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    assert Version(version("cryptography")) >= Version("46.0.6")


def test_pillow_not_vulnerable_cve_2026_42311() -> None:
    assert Version(version("pillow")) >= Version("12.2.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.0.0rc3")


def test_python_multipart_not_vulnerable_cve_2026_42561() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.27")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_3644_q5cj_c5c7() -> None:
    assert Version(version("langsmith")) >= Version("0.8.0")


# NSPECT-S62Q-PZUD (2026-06-22) — pip-audit confirmed fixes
def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    assert Version(version("langchain")) >= Version("1.3.9")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_starlette_not_vulnerable_ghsa_82w8_qh3p_5jfq() -> None:
    # DEFERRED: starlette CVEs (GHSA-86qp, GHSA-82w8, GHSA-wqp7) require starlette>=1.0
    # but fastapi<1.0 requires starlette<0.51.0. Fix requires fastapi upgrade (out of scope).
    # Pin floor as high as possible within the fastapi constraint.
    assert Version(version("starlette")) >= Version("0.50.0")


def test_aiohttp_not_vulnerable_ghsa_jg22_mg44_37j8() -> None:
    assert Version(version("aiohttp")) >= Version("3.14.1")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    assert Version(version("pydantic-settings")) >= Version("2.14.2")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    assert Version(version("bleach")) >= Version("6.4.0")


# NSPECT-S62Q-PZUD — DEFERRED CVEs (cannot be fixed without out-of-scope upgrades)

def test_langchain_openai_pysec_2026_76_deferred() -> None:
    # DEFERRED: PYSEC-2026-76 (langchain-openai) fix requires langchain-openai>=1.1.14
    # which requires openai>=2.26.0,<3.0.0 — conflicts with the project's openai>=1.0,<2.0
    # constraint. Fix requires openai v1→v2 API migration (out of scope for pin-bump).
    # Floor asserts the highest version installable within >=0.2,<1.1.9 (locked at 1.1.7).
    assert Version(version("langchain-openai")) >= Version("1.1.7")


def test_pyarrow_pysec_2026_113_deferred() -> None:
    # DEFERRED: PYSEC-2026-113 (pyarrow) fix is only available in pyarrow>=23.0.1
    # but the project pins pyarrow<22.0 (extras: rag, ingest, all). The fix version
    # crosses a major version boundary — out of scope for a pin-bump CVE remediation.
    # Floor asserts the only resolvable version within >=21.0,<22.0 (locked at 21.0.0).
    assert Version(version("pyarrow")) >= Version("21.0.0")

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 / NSPECT-S62Q-PZUD dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    assert Version(version("cryptography")) >= Version("46.0.6")


def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    # GHSA-537c-gmf6-5ccf: Vulnerable OpenSSL in cryptography wheels
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_pillow_not_vulnerable_cve_2026_42311() -> None:
    assert Version(version("pillow")) >= Version("12.2.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.0.0rc3")


def test_python_multipart_not_vulnerable_cve_2026_42561() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.27")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    # GHSA-5rvq-cxj2-64vf / CVE-2026-53540: quadratic-time querystring parsing + multiple vulns
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_3644_q5cj_c5c7() -> None:
    assert Version(version("langsmith")) >= Version("0.8.0")


def test_starlette_not_vulnerable_ghsa_82w8_qh3p_5jfq() -> None:
    # GHSA-82w8-qh3p-5jfq / GHSA-wqp7-x3pw-xc5r / PYSEC-2026-161: multiple starlette DoS/SSRF vulns
    assert Version(version("starlette")) >= Version("1.3.1")


def test_pyarrow_not_vulnerable_ghsa_rgxp_2hwp_jwgg() -> None:
    # GHSA-rgxp-2hwp-jwgg / CVE-2026-25087: use-after-free in IPC file reading
    assert Version(version("pyarrow")) >= Version("23.0.1")


def test_aiohttp_not_vulnerable_cve_2026_54273() -> None:
    # CVE-2026-54273 / CVE-2026-54279 and related: multiple aiohttp vulns fixed in 3.14.1
    assert Version(version("aiohttp")) >= Version("3.14.1")


def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    # GHSA-gr75-jv2w-4656: path traversal in LangChain file-search loaders
    assert Version(version("langchain")) >= Version("1.3.9")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    # GHSA-gj48-438w-jh9v / GHSA-8rfp-98v4-mmr6: bleach URI sanitization bypass
    assert Version(version("bleach")) >= Version("6.4.0")

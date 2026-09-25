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


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    """GHSA-f4xh-w4cj-qxq8: LangSmith TracingMiddleware arbitrary file read."""
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_pyarrow_not_vulnerable_ghsa_rgxp_2hwp_jwgg() -> None:
    """GHSA-rgxp-2hwp-jwgg: Apache Arrow potential use-after-free reading IPC file."""
    assert Version(version("pyarrow")) >= Version("23.0.1")


def test_aiohttp_not_vulnerable_cve_2026_50269() -> None:
    """CVE-2026-50269: aiohttp CRLF injection in payload.py."""
    assert Version(version("aiohttp")) >= Version("3.14.0")


def test_pillow_not_vulnerable_cve_2026_54058() -> None:
    """CVE-2026-54058: Pillow OOB read in PyImaging_MapBuffer (McIdas AREA)."""
    assert Version(version("pillow")) >= Version("12.3.0")


def test_pillow_not_vulnerable_cve_2026_59197() -> None:
    """CVE-2026-59197: Pillow heap OOB write in ImageFilter.RankFilter."""
    assert Version(version("pillow")) >= Version("12.3.0")


def test_langchain_not_vulnerable_cve_2026_44843() -> None:
    """CVE-2026-44843: LangChain sensitive data exposure via deserialization."""
    assert Version(version("langchain")) >= Version("1.3.3")


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    """CVE-2026-48776: LangGraph SDK unsafe URL path construction."""
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_starlette_not_vulnerable_ghsa_82w8_qh3p_5jfq() -> None:
    """GHSA-82w8-qh3p-5jfq: Starlette form() DoS via application/x-www-form-urlencoded."""
    assert Version(version("starlette")) >= Version("1.3.1")


def test_python_multipart_not_vulnerable_cve_2026_53539() -> None:
    """CVE-2026-53539: python-multipart quadratic querystring parsing CPU DoS."""
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_cryptography_not_vulnerable_cve_2026_69249() -> None:
    """CVE-2026-69249: cryptography network-reachable DoS."""
    assert Version(version("cryptography")) >= Version("49.0.0")


def test_cryptography_not_vulnerable_cve_2026_69247() -> None:
    """CVE-2026-69247: cryptography High severity information disclosure."""
    assert Version(version("cryptography")) >= Version("50.0.0")

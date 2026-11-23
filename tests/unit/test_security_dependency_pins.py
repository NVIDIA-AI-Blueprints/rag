# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 / NSPECT-S62Q-PZUD dependency remediation."""

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


# NSPECT-S62Q-PZUD — 2026-07-15 batch fix
def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    # GHSA-537c-gmf6-5ccf: OpenSSL bundled in cryptography wheels (DoS)
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    # GHSA-5rvq-cxj2-64vf: quadratic-time querystring parsing (DoS)
    assert Version(version("python-multipart")) >= Version("0.0.30")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    # GHSA-f4xh-w4cj-qxq8: TracingMiddleware arbitrary file read
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_aiohttp_not_vulnerable_ghsa_m6qw_4cw2_hm4m() -> None:
    # GHSA-m6qw-4cw2-hm4m (CVE-2026-50269): CRLF injection in multipart headers
    assert Version(version("aiohttp")) >= Version("3.14.0")


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    # CVE-2026-48776: unsafe URL path construction / path traversal
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_bleach_not_vulnerable_bdsa_2026_15483() -> None:
    # BDSA-2026-15483/15484/15486: ReDoS, formaction URI injection, Unicode URI bypass
    assert Version(version("bleach")) >= Version("6.4.0")


def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    # GHSA-gr75-jv2w-4656: path traversal and sandbox escape in file-search middleware
    assert Version(version("langchain")) >= Version("1.3.9")


def test_pillow_not_vulnerable_cve_2026_54059() -> None:
    # CVE-2026-54059/54060/55379/55380: decompression bomb bypasses; CVE-2026-55798: command injection
    assert Version(version("pillow")) >= Version("12.3.0")

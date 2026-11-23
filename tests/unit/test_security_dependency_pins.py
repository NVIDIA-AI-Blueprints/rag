# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version

from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_69247() -> None:
    # CVE-2026-69247/CVE-2026-69248/CVE-2026-69249: pkcs7_decrypt + chain validation issues
    assert Version(version("cryptography")) >= Version("50.0.0")


def test_pillow_not_vulnerable_cve_2026_59197() -> None:
    # CVE-2026-54058..59205: heap OOB write, OOB read, corrupt heap in various codec paths
    assert Version(version("pillow")) >= Version("12.3.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.1.0")


def test_python_multipart_not_vulnerable_cve_2026_53539() -> None:
    # GHSA-5rvq-cxj2-64vf / CVE-2026-53539/53540: quadratic querystring parsing DoS
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    # GHSA-f4xh-w4cj-qxq8: TracingMiddleware arbitrary server-side file read
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_aiohttp_not_vulnerable_cve_2026_54274() -> None:
    # CVE-2026-54274..54280/69243/69244: OOB read, request smuggling, cookie bypass batch
    assert Version(version("aiohttp")) >= Version("3.14.3")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    # GHSA-gj48-438w-jh9v / GHSA-8rfp-98v4-mmr6: formaction javascript: URI bypass + XSS
    assert Version(version("bleach")) >= Version("6.4.0")


def test_starlette_not_vulnerable_cve_2026_54283() -> None:
    # CVE-2026-54283 (GHSA-82w8-qh3p-5jfq): form() max_fields silently ignored DoS
    assert Version(version("starlette")) >= Version("1.3.1")


def test_langchain_not_vulnerable_cve_2026_55443() -> None:
    # CVE-2026-55443 (GHSA-gr75-jv2w-4656): LangChain components SSRF via callback URL
    assert Version(version("langchain")) >= Version("1.3.9")


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    # CVE-2026-48776 (GHSA-w39p-vh2g-g8g5): unsafe URL path construction — path traversal
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_click_not_vulnerable_cve_2026_7246() -> None:
    # CVE-2026-7246 (GHSA-47fr-3ffg-hgmw): click.edit() command injection
    assert Version(version("click")) >= Version("8.3.3")


def test_setuptools_not_vulnerable_cve_2026_59890() -> None:
    # CVE-2026-59890 (GHSA-h35f-9h28-mq5c): FileList arbitrary file inclusion in sdist
    assert Version(version("setuptools")) >= Version("83.0.0")


def test_langchain_classic_min_version() -> None:
    # Transitive guardrail: langchain ecosystem churn; override >=1.0.7
    assert Version(version("langchain-classic")) >= Version("1.0.7")


def test_langchain_text_splitters_min_version() -> None:
    # Transitive guardrail: langchain ecosystem churn; override >=1.1.2
    assert Version(version("langchain-text-splitters")) >= Version("1.1.2")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    # GHSA-4xgf-cpjx-pc3j / CVE-2026-58203: env-file parsing vulnerability
    assert Version(version("pydantic-settings")) >= Version("2.14.2")

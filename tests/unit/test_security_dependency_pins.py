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
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    """GHSA-f4xh-w4cj-qxq8: LangSmith TracingMiddleware arbitrary server-side file read."""
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_aiohttp_not_vulnerable_cve_2026_batch() -> None:
    """CVE-2026-34993,CVE-2026-47265,CVE-2026-54273-54280,CVE-2026-50269: aiohttp vulnerabilities."""
    assert Version(version("aiohttp")) >= Version("3.14.1")


def test_starlette_not_vulnerable_cve_2026_batch() -> None:
    """GHSA-86qp-5c8j-p5mr,GHSA-wqp7-x3pw-xc5r,GHSA-82w8-qh3p-5jfq and others: starlette CVEs."""
    assert Version(version("starlette")) >= Version("1.3.1")


def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    """GHSA-gr75-jv2w-4656: langchain vulnerability."""
    assert Version(version("langchain")) >= Version("1.3.9")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    """GHSA-gj48-438w-jh9v,GHSA-8rfp-98v4-mmr6: bleach XSS vulnerabilities."""
    assert Version(version("bleach")) >= Version("6.4.0")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    """GHSA-4xgf-cpjx-pc3j: pydantic-settings vulnerability."""
    assert Version(version("pydantic-settings")) >= Version("2.14.2")

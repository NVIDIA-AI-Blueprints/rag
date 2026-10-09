# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-S62Q-PZUD dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    assert Version(version("cryptography")) >= Version("48.0.1")  # GHSA-537c-gmf6-5ccf


def test_pillow_not_vulnerable_cve_2026_42311() -> None:
    assert Version(version("pillow")) >= Version("12.2.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.0.0rc3")


def test_python_multipart_not_vulnerable_cve_2026_42561() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.31")  # CVE-2026-53538/39/40


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_3644_q5cj_c5c7() -> None:
    assert Version(version("langsmith")) >= Version("0.8.18")  # GHSA-f4xh-w4cj-qxq8 / CVE-2026-45134


def test_aiohttp_not_vulnerable_cve_2026_50269() -> None:
    assert Version(version("aiohttp")) >= Version("3.14.1")  # CVE-2026-50269/54274/54277/54279/54280


def test_bleach_not_vulnerable_ghsa_8rfp_98v4_mmr6() -> None:
    assert Version(version("bleach")) >= Version("6.4.0")  # GHSA-8rfp-98v4-mmr6


def test_langchain_not_vulnerable_cve_2026_55443() -> None:
    assert Version(version("langchain")) >= Version("1.3.9")  # CVE-2026-55443 / GHSA-gr75-jv2w-4656


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")  # CVE-2026-48776 / GHSA-w39p-vh2g-g8g5


def test_starlette_not_vulnerable_cve_2026_48818() -> None:
    assert Version(version("starlette")) >= Version("1.3.1")  # CVE-2026-48817/48818 + GHSA-82w8-qh3p-5jfq


def test_pyarrow_not_vulnerable_ghsa_rgxp_2hwp_jwgg() -> None:
    assert Version(version("pyarrow")) >= Version("23.0.1")  # GHSA-rgxp-2hwp-jwgg / CVE-2026-25087


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    assert Version(version("pydantic-settings")) >= Version("2.14.2")  # GHSA-4xgf-cpjx-pc3j

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    assert Version(version("cryptography")) >= Version("48.0.1")


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


def test_langsmith_not_vulnerable_ghsa_3644_q5cj_c5c7() -> None:
    assert Version(version("langsmith")) >= Version("0.8.18")


# NSPECT-S62Q-PZUD / NSPECT-UV6I-R3V9 batch (2026-07-04)
def test_aiohttp_not_vulnerable_cve_2026_50269_et_al() -> None:
    # CVE-2026-50269/54274/54277/54279/54280 and others — fixed in 3.14.1
    assert Version(version("aiohttp")) >= Version("3.14.1")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    # GHSA-gj48-438w-jh9v, GHSA-8rfp-98v4-mmr6 — fixed in 6.4.0
    assert Version(version("bleach")) >= Version("6.4.0")


def test_langchain_not_vulnerable_cve_2026_44843() -> None:
    # GHSA-gr75-jv2w-4656 / CVE-2026-44843 — fixed in 1.3.9
    assert Version(version("langchain")) >= Version("1.3.9")


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    # CVE-2026-48776 — fixed in 0.3.15
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_langsmith_not_vulnerable_cve_2026_45134() -> None:
    # GHSA-f4xh-w4cj-qxq8 / CVE-2026-45134 — fixed in 0.8.18
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_starlette_not_vulnerable_cve_2026_48710_et_al() -> None:
    # CVE-2026-48710/48817/48818/54282/54283 — fixed in 1.3.1
    assert Version(version("starlette")) >= Version("1.3.1")


def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    # GHSA-537c-gmf6-5ccf — fixed in 48.0.1
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    # GHSA-4xgf-cpjx-pc3j — fixed in 2.14.2
    assert Version(version("pydantic-settings")) >= Version("2.14.2")


def test_python_multipart_not_vulnerable_cve_2026_53538_et_al() -> None:
    # CVE-2026-53538/53539/53540 — fixed in 0.0.31
    assert Version(version("python-multipart")) >= Version("0.0.31")

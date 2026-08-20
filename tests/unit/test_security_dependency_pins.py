# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    # CVE-2026-69247/69248/69249 — raise floor from 46.0.6 to 50.0.0
    assert Version(version("cryptography")) >= Version("50.0.0")


def test_pillow_not_vulnerable_cve_2026_54059_et_al() -> None:
    # GHSA-45hq-cxwh-f6vc, GHSA-5x94-69rx-g8h2, GHSA-8v84-f9pq-wr9x, GHSA-phj9-mv4w-65pm + 8 more — fixed 12.3.0
    assert Version(version("pillow")) >= Version("12.3.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    # CVE-2026-1839 — fixed 5.0.0rc3; override floor 5.1.0 for additional sweep findings
    assert Version(version("transformers")) >= Version("5.1.0")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    # GHSA-5rvq-cxj2-64vf (CVE-2026-53539), GHSA-6jv3-5f52-599m, GHSA-v9pg-7xvm-68hf — fixed 0.0.31
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    # GHSA-f4xh-w4cj-qxq8 (CVE-2026-59152) — fixed 0.8.18
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_starlette_not_vulnerable_ghsa_82w8_qh3p_5jfq() -> None:
    # GHSA-82w8-qh3p-5jfq (CVE-2026-54283), GHSA-wqp7-x3pw-xc5r, GHSA-86qp-5c8j-p5mr,
    # GHSA-jp82-jpqv-5vv3, GHSA-x746-7m8f-x49c — fixed 1.3.1
    assert Version(version("starlette")) >= Version("1.3.1")


def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    # GHSA-gr75-jv2w-4656 (CVE-2026-55443) — fixed 1.3.9
    assert Version(version("langchain")) >= Version("1.3.9")


def test_aiohttp_not_vulnerable_ghsa_4m7w_qmgq_4wj5() -> None:
    # GHSA-4m7w-qmgq-4wj5 + 12 more aiohttp CVEs — fixed 3.14.3
    assert Version(version("aiohttp")) >= Version("3.14.3")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    # GHSA-gj48-438w-jh9v, GHSA-8rfp-98v4-mmr6 — fixed 6.4.0
    assert Version(version("bleach")) >= Version("6.4.0")


def test_langgraph_sdk_not_vulnerable_ghsa_w39p_vh2g_g8g5() -> None:
    # GHSA-w39p-vh2g-g8g5 (CVE-2026-48776) — fixed 0.3.15
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    # GHSA-4xgf-cpjx-pc3j (CVE-2026-58203) — fixed 2.14.2
    assert Version(version("pydantic-settings")) >= Version("2.14.2")


def test_click_not_vulnerable_ghsa_47fr_3ffg_hgmw() -> None:
    # GHSA-47fr-3ffg-hgmw (CVE-2026-7246) — fixed 8.3.3
    assert Version(version("click")) >= Version("8.3.3")

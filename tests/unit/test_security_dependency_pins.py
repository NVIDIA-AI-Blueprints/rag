# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9 dependency remediation (pip-audit verified pins)."""

from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_pillow_not_vulnerable_cve_2026_42311() -> None:
    assert Version(version("pillow")) >= Version("12.2.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.0.0rc3")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.30")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_starlette_not_vulnerable_ghsa_82w8_qhp_5jfq() -> None:
    assert Version(version("starlette")) >= Version("1.3.1")

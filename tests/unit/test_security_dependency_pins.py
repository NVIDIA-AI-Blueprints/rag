# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for security dependency remediation.

NSPECT-UV6I-R3V9: pip-audit verified pins (Track A).
NSPECT-S62Q-PZUD: UNVERIFIED recommended pins — scanner not run in venv (Track B).
"""

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


# UNVERIFIED: scanner not run in venv; recommended fixes from NSPECT-S62Q-PZUD nSpect source/container surface
def test_cryptography_not_vulnerable_ghsa_537c_gmf6_5ccf() -> None:
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_python_multipart_not_vulnerable_ghsa_5rvq_cxj2_64vf() -> None:
    assert Version(version("python-multipart")) >= Version("0.0.30")


def test_starlette_not_vulnerable_ghsa_82w8_qh3p_5jfq() -> None:
    assert Version(version("starlette")) >= Version("1.3.1")


def test_starlette_not_vulnerable_ghsa_wqp7_x3pw_xc5r() -> None:
    assert Version(version("starlette")) >= Version("1.1.0")


def test_pyarrow_not_vulnerable_ghsa_rgxp_2hwp_jwgg() -> None:
    assert Version(version("pyarrow")) >= Version("23.0.1")

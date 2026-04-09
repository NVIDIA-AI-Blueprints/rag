# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PoC / regression tests for CWE-22 path traversal in MCP server file upload tools.
#
# The tool_upload_documents and tool_update_documents functions accept arbitrary
# file_paths from MCP clients and read them without any path validation.
# An attacker can supply paths like "/etc/passwd" or "../../sensitive.txt" to
# read arbitrary files from the server's filesystem and exfiltrate them via
# the ingestor upload.

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import examples.nvidia_rag_mcp.mcp_server as mcp_server

try:
    from fastmcp.tools import FunctionTool
except Exception:
    FunctionTool = None


def _tool_fn(tool: Any):
    if FunctionTool is not None and isinstance(tool, FunctionTool):
        inner = getattr(tool, "func", None) or getattr(tool, "__wrapped__", None)
        if inner is not None:
            return inner
    return tool


def _make_fake_aiohttp(captured_files: list):
    """Build a fake aiohttp that records which file contents are uploaded."""

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"ok": True}

        async def text(self):
            return "ok"

    class FakeFormData:
        def __init__(self):
            self.fields: list[tuple] = []

        def add_field(self, name, value, filename=None, content_type=None):
            self.fields.append((name, value, filename, content_type))
            if name == "documents":
                captured_files.append({"filename": filename, "content": value})

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, data=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()
                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False
            return Ctx()

        def patch(self, url, data=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()
                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False
            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    return SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
        FormData=FakeFormData,
    )


@pytest.mark.anyio
async def test_upload_rejects_absolute_path_outside_allowed_dir(monkeypatch, tmp_path):
    """
    Path traversal PoC: tool_upload_documents must reject absolute paths
    outside the allowed upload directory.

    An attacker-controlled MCP client could pass "/etc/passwd" to read
    arbitrary files. After the fix, a ValueError should be raised.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("super-secret-data")

    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir()
    monkeypatch.setenv("MCP_UPLOAD_DIR", str(allowed_dir))

    captured_files: list = []
    fake = _make_fake_aiohttp(captured_files)
    monkeypatch.setattr(mcp_server, "aiohttp", fake, raising=True)

    tool = _tool_fn(mcp_server.tool_upload_documents)

    with pytest.raises(ValueError, match="not within the allowed upload directory"):
        await tool(
            collection_name="test",
            file_paths=[str(secret)],
        )

    assert len(captured_files) == 0, "Sensitive file was read despite being outside allowed dir"


@pytest.mark.anyio
async def test_update_rejects_absolute_path_outside_allowed_dir(monkeypatch, tmp_path):
    """
    Same traversal via tool_update_documents (PATCH variant).
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("super-secret-data")

    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir()
    monkeypatch.setenv("MCP_UPLOAD_DIR", str(allowed_dir))

    captured_files: list = []
    fake = _make_fake_aiohttp(captured_files)
    monkeypatch.setattr(mcp_server, "aiohttp", fake, raising=True)

    tool = _tool_fn(mcp_server.tool_update_documents)

    with pytest.raises(ValueError, match="not within the allowed upload directory"):
        await tool(
            collection_name="test",
            file_paths=[str(secret)],
        )
    assert len(captured_files) == 0


@pytest.mark.anyio
async def test_upload_rejects_dot_dot_traversal(monkeypatch, tmp_path):
    """
    Relative path traversal via '../' must be rejected.
    """
    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir()

    secret = tmp_path / "secret.txt"
    secret.write_text("traversal-secret")

    monkeypatch.setenv("MCP_UPLOAD_DIR", str(allowed_dir))

    captured_files: list = []
    fake = _make_fake_aiohttp(captured_files)
    monkeypatch.setattr(mcp_server, "aiohttp", fake, raising=True)

    tool = _tool_fn(mcp_server.tool_upload_documents)

    traversal_path = str(allowed_dir / ".." / "secret.txt")
    with pytest.raises(ValueError, match="not within the allowed upload directory"):
        await tool(
            collection_name="test",
            file_paths=[traversal_path],
        )
    assert len(captured_files) == 0


@pytest.mark.anyio
async def test_upload_allows_file_inside_allowed_dir(monkeypatch, tmp_path):
    """
    Files within the allowed upload directory should be accepted.
    """
    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir()

    legit_file = allowed_dir / "doc.pdf"
    legit_file.write_bytes(b"%PDF-1.4 legit content")

    monkeypatch.setenv("MCP_UPLOAD_DIR", str(allowed_dir))

    captured_files: list = []
    fake = _make_fake_aiohttp(captured_files)
    monkeypatch.setattr(mcp_server, "aiohttp", fake, raising=True)

    tool = _tool_fn(mcp_server.tool_upload_documents)
    result = await tool(
        collection_name="test",
        file_paths=[str(legit_file)],
    )

    assert result.get("ok") is True
    assert len(captured_files) == 1
    assert captured_files[0]["filename"] == "doc.pdf"
    assert captured_files[0]["content"] == b"%PDF-1.4 legit content"


@pytest.mark.anyio
async def test_upload_rejects_symlink_escape(monkeypatch, tmp_path):
    """
    Symlink escape: a symlink inside the allowed dir pointing outside must be rejected.
    """
    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir()

    secret = tmp_path / "secret.txt"
    secret.write_text("symlink-secret")

    link = allowed_dir / "evil_link.txt"
    link.symlink_to(secret)

    monkeypatch.setenv("MCP_UPLOAD_DIR", str(allowed_dir))

    captured_files: list = []
    fake = _make_fake_aiohttp(captured_files)
    monkeypatch.setattr(mcp_server, "aiohttp", fake, raising=True)

    tool = _tool_fn(mcp_server.tool_upload_documents)

    with pytest.raises(ValueError, match="not within the allowed upload directory"):
        await tool(
            collection_name="test",
            file_paths=[str(link)],
        )
    assert len(captured_files) == 0

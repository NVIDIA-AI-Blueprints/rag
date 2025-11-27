# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import asyncio
import os
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

import nvidia_rag.utils.mcp.mcp_server as mcp_server


def test_ensure_rag_initialized_creates_instance(monkeypatch):
    created: dict[str, Any] = {}

    class DummyConfig:
        pass

    class DummyRAG:
        def __init__(self, config=None):
            created["called"] = True
            created["config_type"] = type(config).__name__

    monkeypatch.setattr(mcp_server, "NvidiaRAGConfig", DummyConfig, raising=True)
    monkeypatch.setattr(mcp_server, "NvidiaRAG", DummyRAG, raising=True)
    # Reset global state
    monkeypatch.setattr(mcp_server, "RAG", None, raising=True)

    mcp_server._ensure_rag_initialized()
    assert created.get("called") is True
    assert created.get("config_type") == "DummyConfig"
    # Second call should not recreate
    created.clear()
    mcp_server._ensure_rag_initialized()
    assert created == {}


@pytest.mark.anyio
async def test_tool_generate_concatenates_generator(monkeypatch):
    captured: dict[str, Any] = {}

    class DummyResponse:
        def __init__(self, chunks: Iterable[str]):
            self.generator = iter(chunks)

    class DummyRAG:
        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return DummyResponse(["Hello", " ", "world"])

    monkeypatch.setattr(mcp_server, "RAG", DummyRAG(), raising=True)
    out = await mcp_server.tool_generate(messages=[{"role": "user", "content": "hi"}])
    assert out == "Hello world"
    # Ensure defaults are applied/sanitized
    assert "stop" in captured["kwargs"]
    assert captured["kwargs"]["stop"] == []
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.anyio
async def test_tool_search_returns_model_dump(monkeypatch):
    class DummyCitations:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    captured: dict[str, Any] = {}

    class DummyRAG:
        def search(self, **kwargs):
            captured["kwargs"] = kwargs
            return DummyCitations({"ok": True, "total": 0})

    monkeypatch.setattr(mcp_server, "RAG", DummyRAG(), raising=True)
    out = await mcp_server.tool_search(query="q")
    assert out == {"ok": True, "total": 0}
    assert captured["kwargs"]["query"] == "q"


@pytest.mark.anyio
async def test_tool_get_summary_calls_classmethod(monkeypatch):
    recorded: dict[str, Any] = {}

    async def fake_get_summary(**kwargs):
        recorded["kwargs"] = kwargs
        await asyncio.sleep(0)
        return {"summary": "done"}

    class DummyRAGClass:
        # tool_get_summary calls NvidiaRAG.get_summary as a class/static method
        get_summary = staticmethod(fake_get_summary)

        # _ensure_rag_initialized may still try to instantiate NvidiaRAG; make it harmless
        def __init__(self, *args, **kwargs):
            pass

    # Avoid side effects from _ensure_rag_initialized during this test
    monkeypatch.setattr(mcp_server, "_ensure_rag_initialized", lambda: None, raising=True)
    # Ensure RAG is non-None so the internal assert RAG is not None passes
    monkeypatch.setattr(mcp_server, "RAG", object(), raising=True)
    monkeypatch.setattr(mcp_server, "NvidiaRAG", DummyRAGClass, raising=True)
    out = await mcp_server.tool_get_summary(
        collection_name="c", file_name="f", blocking=True, timeout=5
    )
    assert out == {"summary": "done"}
    assert recorded["kwargs"] == {"collection_name": "c", "file_name": "f", "blocking": True, "timeout": 5}


@pytest.mark.anyio
async def test_sse_auth_env_middleware_sets_env_from_headers(monkeypatch):
    # Ensure clean environment
    for k in ("NVIDIA_API_KEY",):
        os.environ.pop(k, None)

    calls: dict[str, Any] = {}

    async def dummy_app(scope, receive, send):
        calls["called"] = True

    mw = mcp_server.SSEAuthEnvMiddleware(dummy_app)
    scope = {
        "type": "http",
        "headers": [
            (b"authorization", b"Bearer mytoken"),
        ],
    }
    await mw(scope, None, None)
    assert os.environ.get("NVIDIA_API_KEY") == "mytoken"
    assert calls.get("called") is True

    # Reset and test X-API-Key
    os.environ.pop("NVIDIA_API_KEY", None)
    calls.clear()
    scope = {
        "type": "http",
        "headers": [
            (b"x-api-key", b"xyz123"),
        ],
    }
    await mw(scope, None, None)
    assert os.environ.get("NVIDIA_API_KEY") == "xyz123"
    assert calls.get("called") is True


@pytest.mark.anyio
async def test__amain_stdio_sets_env_and_runs_stdio(monkeypatch):
    # Prepare namespace for stdio transport
    ns = SimpleNamespace(transport="stdio", host="127.0.0.1", port=9999, api_key="abc123")

    # Patch server.run_stdio_async
    called = {"run": False}

    async def fake_run_stdio_async():
        called["run"] = True

    monkeypatch.setattr(mcp_server.server, "run_stdio_async", fake_run_stdio_async, raising=True)

    # Run _amain directly with our namespace
    await mcp_server._amain(ns)
    assert called["run"] is True
    assert os.environ.get("NVIDIA_API_KEY") == "abc123"


def test_main_streamable_http_uses_server_run(monkeypatch):
    # Namespace for streamable_http transport
    ns = SimpleNamespace(transport="streamable_http", host="0.0.0.0", port=9901, api_key="xyz123")

    class DummyParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return ns

    # Patch ArgumentParser to return our dummy parser
    monkeypatch.setattr(
        mcp_server.argparse,
        "ArgumentParser",
        lambda *a, **k: DummyParser(),
        raising=True,
    )

    # Patch server.run and anyio.run to observe calls
    called = {"server_run": False, "anyio_run": False}

    def fake_server_run(*args, **kwargs):
        called["server_run"] = True
        # Ensure correct transport and mount_path
        assert kwargs.get("transport") == "streamable-http"
        assert kwargs.get("mount_path") == "/mcp"

    def fake_anyio_run(*args, **kwargs):
        called["anyio_run"] = True

    monkeypatch.setattr(mcp_server.server, "run", fake_server_run, raising=True)
    monkeypatch.setattr(mcp_server.anyio, "run", fake_anyio_run, raising=True)

    # Run main; it should invoke server.run and NOT anyio.run
    mcp_server.main()
    assert called["server_run"] is True
    assert called["anyio_run"] is False
    # API key should be propagated to environment
    assert os.environ.get("NVIDIA_API_KEY") == "xyz123"

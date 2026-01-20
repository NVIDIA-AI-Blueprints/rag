# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for `nvidia_rag_mcp.mcp_server`.
#
# These tests avoid real HTTP calls by monkeypatching `aiohttp` and, when
# needed, unwrapping FastMCP `FunctionTool` wrappers to call the underlying
# async functions directly.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

try:
    import examples.nvidia_rag_mcp.mcp_server as mcp_server

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MCP_AVAILABLE = False
    mcp_server = None  # type: ignore[assignment]

try:  # FastMCP exports FunctionTool in recent versions
    from fastmcp.tools import FunctionTool  # type: ignore
except Exception:  # pragma: no cover - older FastMCP
    FunctionTool = None  # type: ignore


pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="MCP server dependencies not installed (optional dependency)",
)


def _tool_fn(tool: Any):
    """
    Helper to obtain the underlying coroutine function for FastMCP tools.

    Newer FastMCP versions wrap tool functions in a `FunctionTool` object.
    When that is the case, the original coroutine is exposed via `.func`
    (or sometimes `__wrapped__`). These tests call the unwrapped function
    directly, since they are exercising the HTTP adapter logic rather than
    the FastMCP runtime itself.
    """
    if FunctionTool is not None and isinstance(tool, FunctionTool):  # type: ignore[arg-type]
        inner = getattr(tool, "func", None) or getattr(tool, "__wrapped__", None)
        if inner is not None:
            return inner
    return tool


@pytest.mark.anyio
async def test_tool_generate_concatenates_stream(monkeypatch):
    class FakeContent:
        def __init__(self, payloads: list[str]):
            self._payloads = payloads

        async def iter_chunked(self, n: int):
            for p in self._payloads:
                yield p.encode("utf-8")

    class FakeResp:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "text/event-stream"}
            data1 = 'data: {"choices":[{"message":{"content":"Hello"}}]}\n'
            data2 = 'data: {"choices":[{"message":{"content":" world"},"finish_reason":"stop"}]}\n'
            self.content = FakeContent([data1, data2])

        async def json(self):
            return {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_generate)
    out = await tool(messages=[{"role": "user", "content": "hi"}])
    assert out == "Hello world"


@pytest.mark.anyio
async def test_tool_search_returns_json(monkeypatch):
    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"ok": True, "total": 1}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_search)
    out = await tool(query="q")
    assert out == {"ok": True, "total": 1}


@pytest.mark.anyio
async def test_tool_get_summary_returns_json(monkeypatch):
    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"summary": "done"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_get_summary)
    out = await tool(collection_name="c", file_name="f", blocking=True, timeout=5)
    assert out == {"summary": "done"}


@pytest.mark.anyio
async def test_tool_get_documents_calls_ingestor(monkeypatch):
    """tool_get_documents should GET /v1/documents with collection_name and optional vdb_endpoint."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"documents": [], "ok": True}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_get_documents)
    out = await tool(collection_name="c", vdb_endpoint="http://milvus:19530")
    assert out == {"documents": [], "ok": True}
    assert "/v1/documents" in captured["url"]
    assert captured["params"]["collection_name"] == "c"
    assert captured["params"]["vdb_endpoint"] == "http://milvus:19530"


@pytest.mark.anyio
async def test_tool_delete_documents_calls_ingestor(monkeypatch):
    """tool_delete_documents should DELETE /v1/documents with collection_name query param and JSON body of document_names."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"deleted": ["a.pdf", "b.pdf"], "ok": True}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def delete(self, url, params=None, json=None):
            captured["url"] = url
            captured["params"] = params
            captured["json"] = json

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_delete_documents)
    out = await tool(collection_name="c", document_names=["a.pdf", "b.pdf"])

    assert out["ok"] is True
    assert "/v1/documents" in captured["url"]
    assert captured["params"]["collection_name"] == "c"
    assert captured["json"] == ["a.pdf", "b.pdf"]


@pytest.mark.anyio
async def test_tool_update_documents_uses_patch_and_form(monkeypatch, tmp_path):
    """tool_update_documents should PATCH /v1/documents with form-data including files and JSON payload."""

    p1 = tmp_path / "a.pdf"
    p1.write_bytes(b"%PDF-1.4 a")
    p2 = tmp_path / "b.pdf"
    p2.write_bytes(b"%PDF-1.4 b")

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"ok": True, "updated": ["a.pdf", "b.pdf"]}

        async def text(self):
            return "ok"

    class FakeFormData:
        def __init__(self):
            self.fields: list[tuple[str, str]] = []

        def add_field(self, name, value, filename=None, content_type=None):
            self.fields.append((name, filename or name))

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def patch(self, url, data=None):
            captured["url"] = url
            captured["data"] = data

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
        FormData=FakeFormData,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_update_documents)
    out = await tool(
        collection_name="c",
        file_paths=[str(p1), str(p2)],
        blocking=True,
        generate_summary=False,
        custom_metadata=None,
        split_options={"chunk_size": 512, "chunk_overlap": 150},
    )
    assert out.get("ok") is True
    assert "/v1/documents" in captured["url"]
    doc_fields = [f for f in captured["data"].fields if f[0] == "documents"]
    assert ("documents", "a.pdf") in doc_fields
    assert ("documents", "b.pdf") in doc_fields


@pytest.mark.anyio
async def test_tool_list_collections_calls_ingestor(monkeypatch):
    """tool_list_collections should GET /v1/collections with optional vdb_endpoint."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"collections": ["c1", "c2"]}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_list_collections)
    out = await tool(vdb_endpoint="http://milvus:19530")
    assert out == {"collections": ["c1", "c2"]}
    assert "/v1/collections" in captured["url"]
    assert captured["params"]["vdb_endpoint"] == "http://milvus:19530"


@pytest.mark.anyio
async def test_tool_update_collection_metadata_calls_ingestor(monkeypatch):
    """tool_update_collection_metadata should PATCH /v1/collections/{name}/metadata with JSON body."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"message": "ok", "collection_name": "c"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def patch(self, url, json=None):
            captured["url"] = url
            captured["json"] = json

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_update_collection_metadata)
    out = await tool(
        collection_name="c",
        description="d",
        tags=["t1"],
        owner="o",
        business_domain="b",
        status="Active",
    )
    assert out["collection_name"] == "c"
    assert "/v1/collections/c/metadata" in captured["url"]
    assert captured["json"]["description"] == "d"
    assert captured["json"]["tags"] == ["t1"]
    assert captured["json"]["owner"] == "o"
    assert captured["json"]["business_domain"] == "b"
    assert captured["json"]["status"] == "Active"


@pytest.mark.anyio
async def test_tool_update_document_metadata_calls_ingestor(monkeypatch):
    """tool_update_document_metadata should PATCH /v1/collections/{c}/documents/{d}/metadata with JSON body."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"message": "ok", "collection_name": "c"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def patch(self, url, json=None):
            captured["url"] = url
            captured["json"] = json

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_update_document_metadata)
    out = await tool(
        collection_name="c",
        document_name="doc.pdf",
        description="d",
        tags=["t1", "t2"],
    )
    assert out["collection_name"] == "c"
    assert "/v1/collections/c/documents/doc.pdf/metadata" in captured["url"]
    assert captured["json"]["description"] == "d"
    assert captured["json"]["tags"] == ["t1", "t2"]


@pytest.mark.anyio
async def test_tool_create_collections_calls_ingestor(monkeypatch):
    """tool_create_collections should POST /v1/collections with JSON array of names."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"collections": ["c1", "c2"], "ok": True}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_create_collections)
    out = await tool(collection_names=["c1", "c2"])
    assert out["ok"] is True
    assert "/v1/collections" in captured["url"]
    assert captured["json"] == ["c1", "c2"]


@pytest.mark.anyio
async def test_tool_delete_collections_calls_ingestor(monkeypatch):
    """tool_delete_collections should DELETE /v1/collections with JSON array of names."""

    captured: dict[str, Any] = {}

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"deleted": ["c1"], "ok": True}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def delete(self, url, json=None):
            captured["url"] = url
            captured["json"] = json

            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_delete_collections)
    out = await tool(collection_names=["c1"])
    assert out["ok"] is True
    assert "/v1/collections" in captured["url"]
    assert captured["json"] == ["c1"]


@pytest.mark.anyio
async def test_tool_upload_documents(monkeypatch, tmp_path):
    """tool_upload_documents should POST /v1/documents with form-data including files and JSON payload."""

    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4...")

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"ok": True, "uploaded": ["doc.pdf"]}

        async def text(self):
            return "ok"

    class FakeFormData:
        def __init__(self):
            self.fields: list[tuple[str, str]] = []

        def add_field(self, name, value, filename=None, content_type=None):
            self.fields.append((name, filename or name))

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

    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
        FormData=FakeFormData,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)

    tool = _tool_fn(mcp_server.tool_upload_documents)
    out = await tool(
        collection_name="c",
        file_paths=[str(p)],
        blocking=True,
        generate_summary=True,
        custom_metadata=None,
        split_options={"chunk_size": 512, "chunk_overlap": 150},
    )
    assert out.get("ok") is True


def test_main_streamable_http_uses_server_run(monkeypatch):
    ns = SimpleNamespace(transport="streamable_http", host="0.0.0.0", port=9901)

    class DummyParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return ns

    monkeypatch.setattr(
        mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True
    )

    called = {"server_run": False}

    def fake_server_run(*args, **kwargs):
        called["server_run"] = True
        assert kwargs.get("transport") == "streamable-http"

    monkeypatch.setattr(mcp_server.server, "run", fake_server_run, raising=True)
    mcp_server.main()
    assert called["server_run"] is True


def test_main_sse_uses_server_run(monkeypatch):
    ns = SimpleNamespace(transport="sse", host="127.0.0.1", port=8000)

    class DummyParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return ns

    monkeypatch.setattr(
        mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True
    )

    called = {"server_run": False}

    def fake_server_run(*args, **kwargs):
        called["server_run"] = True
        assert kwargs.get("transport") == "sse"

    monkeypatch.setattr(mcp_server.server, "run", fake_server_run, raising=True)
    mcp_server.main()
    assert called["server_run"] is True


def test_main_stdio_uses_server_run(monkeypatch):
    ns = SimpleNamespace(transport="stdio", host="127.0.0.1", port=8000)

    class DummyParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return ns

    monkeypatch.setattr(
        mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True
    )

    called = {"server_run": False}

    def fake_server_run(*args, **kwargs):
        called["server_run"] = True
        assert kwargs.get("transport") == "stdio"

    monkeypatch.setattr(mcp_server.server, "run", fake_server_run, raising=True)
    mcp_server.main()
    assert called["server_run"] is True


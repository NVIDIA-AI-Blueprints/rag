# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Any

import pytest

# Try to import mcp_server, skip tests if not available (optional dependency)
try:
    import nvidia_rag_mcp.mcp_server as mcp_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp_server = None

# Skip all tests in this module if MCP is not available
pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="MCP dependencies not installed (optional dependency)"
)


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

    # Provide a fake aiohttp with ClientSession and ClientTimeout used by server
    FakeClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, total=None: None})
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout=None: FakeSession(),
        ClientTimeout=FakeClientTimeout,
        ContentTypeError=Exception,
    )
    monkeypatch.setattr(mcp_server, "aiohttp", fake_aiohttp, raising=True)
    out = await mcp_server.tool_generate(messages=[{"role": "user", "content": "hi"}])
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
    out = await mcp_server.tool_search(query="q")
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
    out = await mcp_server.tool_get_summary(collection_name="c", file_name="f", blocking=True, timeout=5)
    assert out == {"summary": "done"}


@pytest.mark.anyio
async def test_tool_get_documents_calls_ingestor(monkeypatch):
    """tool_get_documents should GET /v1/documents with collection_name (and optional vdb_endpoint)."""

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

    out = await mcp_server.tool_get_documents(collection_name="c", vdb_endpoint="http://milvus:19530")
    assert out == {"documents": [], "ok": True}
    assert "/v1/documents" in captured["url"]
    assert captured["params"]["collection_name"] == "c"
    assert captured["params"]["vdb_endpoint"] == "http://milvus:19530"


@pytest.mark.anyio
async def test_tool_delete_documents_calls_ingestor(monkeypatch):
    """tool_delete_documents should DELETE /v1/documents with collection_name and repeated document_names."""

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

        def delete(self, url, params=None):
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

    out = await mcp_server.tool_delete_documents(
        collection_name="c", document_names=["a.pdf", "b.pdf"]
    )
    assert out["ok"] is True
    assert "/v1/documents" in captured["url"]
    # Params should contain collection_name and two document_names entries
    assert ("collection_name", "c") in captured["params"]
    assert ("document_names", "a.pdf") in captured["params"]
    assert ("document_names", "b.pdf") in captured["params"]


@pytest.mark.anyio
async def test_tool_update_documents_uses_patch_and_form(monkeypatch, tmp_path):
    """tool_update_documents should PATCH /v1/documents with form-data including files and JSON payload."""

    # Prepare fake files
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
            # Record field names and filenames for assertions
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

    out = await mcp_server.tool_update_documents(
        collection_name="c",
        file_paths=[str(p1), str(p2)],
        blocking=True,
        generate_summary=False,
        custom_metadata=None,
        split_options={"chunk_size": 512, "chunk_overlap": 150},
    )
    assert out.get("ok") is True
    assert "/v1/documents" in captured["url"]
    # Ensure form-data was constructed with both documents and a data field
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

    out = await mcp_server.tool_list_collections(vdb_endpoint="http://milvus:19530")
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

    out = await mcp_server.tool_update_collection_metadata(
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

    out = await mcp_server.tool_update_document_metadata(
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

    out = await mcp_server.tool_create_collections(collection_names=["c1", "c2"])
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

    out = await mcp_server.tool_delete_collections(collection_names=["c1"])
    assert out["ok"] is True
    assert "/v1/collections" in captured["url"]
    assert captured["json"] == ["c1"]


@pytest.mark.anyio
async def test_tool_upload_documents(monkeypatch, tmp_path):
    # Prepare a fake file to upload
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4...")  # minimal bytes

    class FakeResp:
        def __init__(self):
            self.status = 200

        async def json(self):
            return {"ok": True, "uploaded": ["doc.pdf"]}

        async def text(self):
            return "ok"

    class FakeFormData:
        def __init__(self):
            self.fields = []

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

    out = await mcp_server.tool_upload_documents(
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

    monkeypatch.setattr(mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True)

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

    monkeypatch.setattr(mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True)

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

    monkeypatch.setattr(mcp_server.argparse, "ArgumentParser", lambda *a, **k: DummyParser(), raising=True)

    called = {"server_run": False}

    def fake_server_run(*args, **kwargs):
        called["server_run"] = True
        assert kwargs.get("transport") == "stdio"

    monkeypatch.setattr(mcp_server.server, "run", fake_server_run, raising=True)
    mcp_server.main()
    assert called["server_run"] is True


@pytest.mark.anyio
async def test_tool_create_and_delete_collections(monkeypatch):
    # Mock aiohttp for POST and DELETE flows
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status = 200

        async def json(self):
            return self._data

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp({"ok": True, "collections": json})

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

        def delete(self, url, json=None, params=None):
            class Ctx:
                async def __aenter__(self_inner):
                    # Accept either json list or params for flexibility
                    payload = json if json is not None else params
                    return FakeResp({"ok": True, "deleted": payload})

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

    out_create = await mcp_server.tool_create_collections(["c1", "c2"])
    assert out_create.get("ok") is True
    assert out_create.get("collections") == ["c1", "c2"]

    out_delete = await mcp_server.tool_delete_collections(["c1", "c2"])
    assert out_delete.get("ok") is True
    assert out_delete.get("deleted") == ["c1", "c2"]

@pytest.mark.anyio
async def test_tool_create_and_delete_collections(monkeypatch):
    # Mock aiohttp for POST and DELETE flows
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status = 200

        async def json(self):
            return self._data

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            class Ctx:
                async def __aenter__(self_inner):
                    return FakeResp({"ok": True, "collections": json})

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

        def delete(self, url, json=None, params=None):
            class Ctx:
                async def __aenter__(self_inner):
                    # Accept either json list or params for flexibility
                    payload = json if json is not None else params
                    return FakeResp({"ok": True, "deleted": payload})

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

    out_create = await mcp_server.tool_create_collections(["c1", "c2"])
    assert out_create.get("ok") is True
    assert out_create.get("collections") == ["c1", "c2"]

    out_delete = await mcp_server.tool_delete_collections(["c1", "c2"])
    assert out_delete.get("ok") is True
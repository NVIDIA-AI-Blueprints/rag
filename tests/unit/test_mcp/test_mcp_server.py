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

import nvidia_rag_mcp.mcp_server as mcp_server


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
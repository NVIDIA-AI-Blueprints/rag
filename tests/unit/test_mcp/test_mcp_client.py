# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for nvidia_rag_mcp.mcp_client
#
# These tests are written to be independent of the real `mcp` SDK by
# installing a small fake `mcp.client.session` module into `sys.modules`
# before importing `nvidia_rag_mcp.mcp_client`.

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _install_fake_mcp_session():
    """Install a minimal fake `mcp.client.session.ClientSession` into sys.modules."""
    # Create package module objects for mcp, mcp.client and mcp.client.session
    mcp_pkg = ModuleType("mcp")
    mcp_pkg.__path__ = []  # mark as package

    client_pkg = ModuleType("mcp.client")
    client_pkg.__path__ = []

    session_mod = ModuleType("mcp.client.session")

    class FakeClientInfo:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version

    class FakeClientSession:
        """
        Minimal stub that mimics the `ClientSession` __init__ signature we care about.

        The real implementation accepts read/write stream parameters plus a client
        identity (client_info / client_name / name). We only need the __init__
        signature so that `_build_session_kwargs` can introspect it and decide
        which keyword arguments to emit.
        """

        def __init__(
            self,
            *,
            read_stream=None,
            write_stream=None,
            client_info: FakeClientInfo | None = None,
        ):
            self.read_stream = read_stream
            self.write_stream = write_stream
            self.client_info = client_info

    # Attach FakeClientSession as ClientSession in the fake module hierarchy
    session_mod.ClientSession = FakeClientSession  # type: ignore[attr-defined]

    # Also provide a minimal `mcp.types.ClientInfo` in case the client imports it.
    types_mod = ModuleType("mcp.types")
    types_mod.ClientInfo = FakeClientInfo  # type: ignore[attr-defined]

    sys.modules["mcp"] = mcp_pkg
    sys.modules["mcp.client"] = client_pkg
    sys.modules["mcp.client.session"] = session_mod
    sys.modules["mcp.types"] = types_mod


_install_fake_mcp_session()
mcp_client = importlib.import_module("nvidia_rag_mcp.mcp_client")


def test_to_jsonable_with_model_dump():
    class WithModelDump:
        def model_dump(self):
            return {"a": 1, "b": [2, 3]}

    value = WithModelDump()
    out = mcp_client._to_jsonable(value)
    assert out == {"a": 1, "b": [2, 3]}


def test_to_jsonable_with_dict_and_nested():
    class WithDict:
        def dict(self):
            return {"x": 1, "y": {"z": 2}}

    value = {"obj": WithDict(), "flag": True, "nums": [1, 2, 3]}
    out = mcp_client._to_jsonable(value)
    assert out["flag"] is True
    assert out["nums"] == [1, 2, 3]
    assert out["obj"] == {"x": 1, "y": {"z": 2}}


def test_to_jsonable_falls_back_to_str():
    class Weird:
        __slots__ = ()

        def __repr__(self):
            return "Weird()"

    out = mcp_client._to_jsonable(Weird())
    assert out == "Weird()"


def test_build_session_kwargs_uses_read_write_and_client_info():
    """
    Verify that `_build_session_kwargs` inspects the ClientSession __init__
    signature and emits the correct keyword arguments for streams and client
    identity when our fake mcp client is installed.
    """
    read = SimpleNamespace()
    write = SimpleNamespace()

    kwargs = mcp_client._build_session_kwargs(read, write)

    # From our FakeClientSession signature we expect:
    # - read_stream and write_stream to be set
    # - client_info to be present and have `name` / `version` attrs
    assert kwargs["read_stream"] is read
    assert kwargs["write_stream"] is write

    client_info = kwargs.get("client_info")
    assert client_info is not None
    assert getattr(client_info, "name", "") == "nvidia-rag-mcp-client"
    assert getattr(client_info, "version", "") == "0.0.0"


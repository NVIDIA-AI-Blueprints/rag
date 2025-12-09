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

import argparse
from typing import Any
import sys
import types
from types import SimpleNamespace

from nvidia_rag_mcp.mcp_client import (
    _build_arg_parser,
    _build_session_kwargs,
    _to_jsonable,
)


def test_build_arg_parser_has_commands_and_options():
    parser = _build_arg_parser()
    # Ensure subcommands exist
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert actions, "No subparsers found"
    subparsers = actions[0]
    subcommands = subparsers.choices
    assert "list" in subcommands
    assert "call" in subcommands

    # Ensure transport option includes supported modes (sse, streamable_http)
    list_parser = subparsers.choices["list"]
    transport_actions = [a for a in list_parser._actions if getattr(a, "dest", "") == "transport"]
    assert transport_actions, "No --transport option found on list subcommand"
    transport_action = transport_actions[0]
    assert set(transport_action.choices) >= {"sse", "streamable_http"}


def test_to_jsonable_handles_common_types_and_objects():
    class WithModelDump:
        def model_dump(self) -> dict[str, Any]:
            return {"a": 1, "b": [1, 2]}

    class WithDict:
        def dict(self) -> dict[str, Any]:
            return {"x": {"y": 2}}

    class WithToDict:
        def to_dict(self) -> dict[str, Any]:
            return {"k": "v"}

    class WithAttrs:
        def __init__(self):
            self.public = 3
            self._private = 9

    assert _to_jsonable(5) == 5
    assert _to_jsonable([1, 2, {"a": 3}]) == [1, 2, {"a": 3}]
    assert _to_jsonable({"n": 1, "m": [1, 2]}) == {"n": 1, "m": [1, 2]}
    assert _to_jsonable(WithModelDump()) == {"a": 1, "b": [1, 2]}
    assert _to_jsonable(WithDict()) == {"x": {"y": 2}}
    assert _to_jsonable(WithToDict()) == {"k": "v"}
    assert _to_jsonable(WithAttrs()) == {"public": 3}


def _install_fake_mcp_client_session(monkeypatch, params: list[str]):
    # Create fake module hierarchy: mcp.client.session
    mcp_mod = types.ModuleType("mcp")
    client_mod = types.ModuleType("mcp.client")
    session_mod = types.ModuleType("mcp.client.session")

    # Build a fake ClientSession with a dynamic __init__ signature.
    # Signature parameters are determined by 'params' list so that
    # _build_session_kwargs can use inspect.signature to discover them.
    # Example: params = ["client_info", "read_stream", "write_stream"]
    param_names = list(params)
    args_def = ", ".join([f"{name}=None" for name in param_names])
    src = (
        "def __init__(self"
        + (", " + args_def if args_def else "")
        + ", **kwargs):\n"
        + "    # Accept known kwargs; anything else should raise to surface mismatch in tests\n"
        + "    for name in "
        + repr(param_names)
        + ":\n"
        + "        kwargs.pop(name, None)\n"
        + "    if kwargs:\n"
        + "        raise TypeError(f'Unexpected kwargs: {sorted(kwargs.keys())}')\n"
    )
    namespace: dict[str, Any] = {}
    exec(src, namespace)
    __init__ = namespace["__init__"]

    ClientSession = type("ClientSession", (), {"__init__": __init__})
    session_mod.ClientSession = ClientSession  # type: ignore[attr-defined]

    # Register modules
    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.session", session_mod)


def test_build_session_kwargs_prefers_read_write_stream(monkeypatch):
    _install_fake_mcp_client_session(
        monkeypatch, ["client_info", "read_stream", "write_stream"]
    )
    read = object()
    write = object()
    kwargs = _build_session_kwargs(read, write)
    assert "client_info" in kwargs or "client_name" in kwargs or "name" in kwargs
    # When signature includes read_stream/write_stream, those should be used
    assert kwargs.get("read_stream", None) is read
    assert kwargs.get("write_stream", None) is write


def test_build_session_kwargs_supports_writer_reader(monkeypatch):
    _install_fake_mcp_client_session(monkeypatch, ["client_name", "reader", "writer"])
    read = object()
    write = object()
    kwargs = _build_session_kwargs(read, write)
    assert kwargs.get("reader", None) is read
    assert kwargs.get("writer", None) is write
    assert kwargs.get("client_name") == "nvidia-rag-mcp-client"


def test_build_session_kwargs_supports_name_only(monkeypatch):
    _install_fake_mcp_client_session(monkeypatch, ["name"])
    read = object()
    write = object()
    kwargs = _build_session_kwargs(read, write)
    assert kwargs.get("name") == "nvidia-rag-mcp-client"

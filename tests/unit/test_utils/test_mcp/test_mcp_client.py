# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from nvidia_rag.utils.mcp.mcp_client import (
    _build_arg_parser,
    _build_session_kwargs,
    _canonicalize_sse_url,
    _parse_headers,
    _resolve_stdio_command,
    _streamable_base_url,
    _to_jsonable,
)


def test_parse_headers_supports_colon_and_equals():
    headers = _parse_headers(
        [
            "Authorization: Bearer abc",
            "X-API-Key=xyz",
            "invalidheader",
        ]
    )
    assert headers == {"Authorization": "Bearer abc", "X-API-Key": "xyz"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://0.0.0.0:8000", "http://127.0.0.1:8000"),
        ("http://0.0.0.0", "http://127.0.0.1"),
        ("http://0.0.0.0:9901/sse", "http://127.0.0.1:9901/sse"),
        ("http://127.0.0.1:9901/sse", "http://127.0.0.1:9901/sse"),
        ("", ""),
        (None, None),
    ],
)
def test_canonicalize_sse_url_rewrites_host(raw: str | None, expected: str | None):
    assert _canonicalize_sse_url(raw) == expected


def test_resolve_stdio_command_requires_command_exits():
    ns = SimpleNamespace(command=None, args_list=None)
    with pytest.raises(SystemExit) as ei:
        _resolve_stdio_command(ns)  # type: ignore[arg-type]
    assert ei.value.code == 2


def test_resolve_stdio_command_parses_args_list():
    ns = SimpleNamespace(command="python", args_list="-m foo.bar --opt=1")
    cmd, args = _resolve_stdio_command(ns)  # type: ignore[arg-type]
    assert cmd == "python"
    assert args == ["-m", "foo.bar", "--opt=1"]


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


def test_build_arg_parser_has_commands_and_options():
    parser = _build_arg_parser()
    # Ensure subcommands exist
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert actions, "No subparsers found"
    subparsers = actions[0]
    subcommands = subparsers.choices
    assert "list" in subcommands
    assert "call" in subcommands

    # Ensure transport option includes all supported modes, including streamable_http
    list_parser = subparsers.choices["list"]
    transport_actions = [a for a in list_parser._actions if getattr(a, "dest", "") == "transport"]
    assert transport_actions, "No --transport option found on list subcommand"
    transport_action = transport_actions[0]
    # Choices should contain stdio, sse, and streamable_http
    assert set(transport_action.choices) >= {"stdio", "sse", "streamable_http"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Empty/none path -> mount at /mcp, and canonicalize 0.0.0.0 -> 127.0.0.1
        ("http://0.0.0.0:8000", "http://127.0.0.1:8000/mcp"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/mcp"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000/mcp"),
        # Existing mount path is preserved (trailing slash trimmed)
        ("http://127.0.0.1:8000/mcp", "http://127.0.0.1:8000/mcp"),
        ("http://127.0.0.1:8000/mcp/", "http://127.0.0.1:8000/mcp"),
        # Arbitrary subpath keeps its structure
        (
            "http://0.0.0.0:8000/custom/path/",
            "http://127.0.0.1:8000/custom/path",
        ),
    ],
)
def test_streamable_base_url_normalizes_and_mounts(raw: str, expected: str) -> None:
    assert _streamable_base_url(raw) == expected


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

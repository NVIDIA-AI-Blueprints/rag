# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
MCP client for interacting with the NVIDIA RAG MCP server.

Uses the official Model Context Protocol Python SDK to:
- List available tools and their schemas
- Call tools with JSON arguments

Supported Transports:
- stdio: Launch local MCP server as a subprocess
- sse: Connect to remote/local MCP server over Server-Sent Events (HTTP)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Iterable, List, Tuple
import inspect
from urllib.parse import urlparse, urlunparse


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    # Try common object-to-dict patterns
    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(value, attr):
            try:
                data = getattr(value, attr)()
                return _to_jsonable(data)
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            data = {k: v for k, v in vars(value).items() if not k.startswith("_")}
            return _to_jsonable(data)
        except Exception:
            pass
    # Fallback to string
    return str(value)


@contextmanager
def _temporary_env(extra_env: Iterable[str]) -> Iterable[str]:
    """Temporarily apply KEY=VALUE entries to os.environ."""
    old_vals: Dict[str, Tuple[bool, str]] = {}
    try:
        for kv in extra_env or []:
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            if key not in old_vals:
                old_vals[key] = (key in os.environ, os.environ.get(key, ""))
            os.environ[key] = val
        yield extra_env
    finally:
        for key, (existed, old_val) in old_vals.items():
            if existed:
                os.environ[key] = old_val
            else:
                os.environ.pop(key, None)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MCP client (Python SDK) for NVIDIA RAG MCP server")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--transport", choices=["stdio", "sse", "streamable_http"], default="stdio")
    common.add_argument("--command", help="Command to run for stdio transport (e.g., python)")
    common.add_argument(
        "--args",
        dest="args_list",
        help="Arguments for stdio command as a single string, e.g. '-m nvidia_rag.utils.mcp.mcp_server'",
    )
    common.add_argument("--env", action="append", help="Env var to pass to child MCP server: KEY=VALUE", default=[])
    common.add_argument("--url", help="URL for SSE/streamable_http transport, e.g., http://127.0.0.1:8000")
    common.add_argument(
        "--header",
        action="append",
        default=[],
        help="HTTP header for SSE/streamable_http transport as 'Key: Value' or 'Key=Value' (may repeat)",
    )

    p_list = sub.add_parser("list", parents=[common], help="List tools or show details for a specific tool")
    p_list.add_argument("--tool", help="Optional tool name to show details")

    p_call = sub.add_parser("call", parents=[common], help="Call a tool with JSON args")
    p_call.add_argument("--tool", required=True, help="Tool name to call")
    p_call.add_argument(
        "--json-args",
        default="{}",
        help='JSON string for tool arguments, e.g., \'{"messages":[...]}\'',
    )

    return p


def _resolve_stdio_command(ns: argparse.Namespace) -> Tuple[str, List[str]]:
    """Parse and validate stdio command and arguments."""
    if not ns.command:
        print("Error: --command is required for stdio transport", file=sys.stderr)
        sys.exit(2)
    cmd = ns.command
    args: List[str] = shlex.split(ns.args_list) if ns.args_list else []
    return cmd, args


def _parse_headers(header_list: Iterable[str]) -> Dict[str, str]:
    """Parse HTTP headers from 'Key: Value' or 'Key=Value' format."""
    headers: Dict[str, str] = {}
    for h in header_list or []:
        if ":" in h:
            k, v = h.split(":", 1)
        elif "=" in h:
            k, v = h.split("=", 1)
        else:
            continue
        headers[k.strip()] = v.strip()
    return headers


def _canonicalize_sse_url(raw_url: str) -> str:
    """
    Ensure SSE URL is routable by replacing 0.0.0.0 with 127.0.0.1.
    Preserves port and path unchanged.
    """
    if not raw_url:
        return raw_url
    parsed = urlparse(raw_url)
    netloc = parsed.netloc
    if netloc.startswith("0.0.0.0"):
        host_port = netloc.split(":")
        port = host_port[1] if len(host_port) > 1 else ""
        netloc = "127.0.0.1" + (f":{port}" if port else "")
    fixed = parsed._replace(netloc=netloc)
    return urlunparse(fixed)


def _streamable_base_url(raw_url: str) -> str:
    """
    Build the correct base URL for streamable_http.

    FastMCP.run(transport="streamable-http", mount_path="/mcp") exposes the
    streamable HTTP endpoint at the mount path itself (e.g. /mcp), not /messages.

    Rules:
    - If the path is empty or '/', assume mount_path='/mcp' → use /mcp.
    - Otherwise, strip any trailing slash and use that path.
    """
    base = _canonicalize_sse_url(raw_url)
    parsed = urlparse(base)
    path = parsed.path or ""

    if path in ("", "/"):
        new_path = "/mcp"
    else:
        new_path = path[:-1] if path.endswith("/") else path

    return urlunparse(parsed._replace(path=new_path))


@asynccontextmanager
async def _open_connection(ns: argparse.Namespace):
    """
    Establish MCP connection and yield (read, write) stream pair.
    
    For stdio: Launches local server subprocess with environment variables.
    For SSE: Connects to remote/local HTTP server with automatic endpoint probing.
    For streamable_http: Connects to FastMCP streamable-http endpoint.
    """
    if ns.transport == "stdio":
        from mcp.client.stdio import stdio_client
        try:
            from mcp.client.stdio import StdioServerParameters  # type: ignore
        except Exception:
            StdioServerParameters = None  # type: ignore

        cmd, args = _resolve_stdio_command(ns)
        with _temporary_env(ns.env):
            # Try new-style API with StdioServerParameters first
            if StdioServerParameters is not None:  # type: ignore[truthy-bool]
                try:
                    params = StdioServerParameters(command=cmd, args=args)  # type: ignore[call-arg]
                    async with stdio_client(params) as pair:  # type: ignore[misc]
                        yield pair
                        return
                except TypeError:
                    pass
            # Fallback to older API signatures for compatibility
            for attempt in (
                lambda: stdio_client(command=cmd, args=args),  # type: ignore[misc]
                lambda: stdio_client(cmd, args),  # type: ignore[misc]
                lambda: stdio_client([cmd] + args),  # type: ignore[misc]
                lambda: stdio_client(cmd),  # type: ignore[misc]
            ):
                try:
                    async with attempt() as pair:  # type: ignore[misc]
                        yield pair
                        return
                except TypeError:
                    continue
        return

    # HTTP transports (SSE, streamable_http)
    if not ns.url:
        print("Error: --url is required for non-stdio transports (sse, streamable_http)", file=sys.stderr)
        sys.exit(2)
    headers = _parse_headers(ns.header)

    if ns.transport == "sse":
        from mcp.client.sse import sse_client

        # Probe common SSE endpoints: /sse, /messages, then base URL
        base = _canonicalize_sse_url(ns.url)
        base_no_slash = base[:-1] if base.endswith("/") else base
        candidates = [
            base_no_slash + "/sse",
            base_no_slash + "/sse/",
            base_no_slash + "/messages",
            base_no_slash + "/messages/",
            base,
        ]
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                async with sse_client(url=candidate, headers=headers) as pair:  # type: ignore[misc]
                    yield pair
                    return
            except TypeError:
                # Try alternative signature for older SDK versions
                try:
                    async with sse_client(candidate, headers) as pair:  # type: ignore[misc]
                        yield pair
                        return
                except Exception as e:
                    last_exc = e
                    continue
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        return

    if ns.transport == "streamable_http":
        try:
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        except Exception as e:
            print(f"Error: streamable_http transport requires mcp package with streamable_http support: {e}", file=sys.stderr)
            sys.exit(2)

        # Normalize to the FastMCP mount path (e.g. http://host:port/mcp)
        url = _streamable_base_url(ns.url)
        last_exc: Exception | None = None
        try:
            # streamablehttp_client yields (read, write, close_handle) or similar
            async with streamablehttp_client(url=url, headers=headers) as streams:  # type: ignore[misc]
                if isinstance(streams, (tuple, list)) and len(streams) >= 2:
                    read, write = streams[0], streams[1]
                else:
                    # Fallback: assume it's already a (read, write) pair
                    read, write = streams  # type: ignore[misc]
                yield (read, write)
                return
        except Exception as e:
            last_exc = e
        if last_exc:
            raise last_exc
        return

    print(f"Unsupported transport: {ns.transport}", file=sys.stderr)
    sys.exit(2)


def _build_session_kwargs(read: Any, write: Any) -> Dict[str, Any]:
    """Build ClientSession kwargs compatible with various MCP SDK versions."""
    try:
        from mcp.client.session import ClientSession  # type: ignore
    except Exception:
        from mcp import ClientSession  # type: ignore
    
    sig = None
    try:
        sig = inspect.signature(ClientSession.__init__)  # type: ignore[attr-defined]
    except Exception:
        pass
    params = set(sig.parameters.keys()) if sig else set()
    kwargs: Dict[str, Any] = {}
    
    # Client identity (try different parameter names for version compatibility)
    if "client_info" in params:
        try:
            from mcp.types import ClientInfo  # type: ignore
            kwargs["client_info"] = ClientInfo(name="nvidia-rag-mcp-client", version="0.0.0")  # type: ignore[call-arg]
        except Exception:
            kwargs["client_info"] = {"name": "nvidia-rag-mcp-client", "version": "0.0.0"}  # type: ignore[assignment]
    elif "client_name" in params:
        kwargs["client_name"] = "nvidia-rag-mcp-client"
    elif "name" in params:
        kwargs["name"] = "nvidia-rag-mcp-client"
    
    # Streams (try different parameter names for version compatibility)
    if "read_stream" in params:
        kwargs["read_stream"] = read
    if "write_stream" in params:
        kwargs["write_stream"] = write
    if "reader" in params and "read_stream" not in kwargs:
        kwargs["reader"] = read
    if "writer" in params and "write_stream" not in kwargs:
        kwargs["writer"] = write
    
    return kwargs


async def _list_tools_async(ns: argparse.Namespace) -> int:
    """List available MCP tools or show details for a specific tool."""
    try:
        from mcp.client.session import ClientSession  # type: ignore
    except Exception:
        from mcp import ClientSession  # type: ignore
    async with _open_connection(ns) as (read, write):
        kwargs = _build_session_kwargs(read, write)
        async with ClientSession(**kwargs) as session:  # type: ignore[misc]
            await session.initialize()
            resp = await session.list_tools()
        
        tools = getattr(resp, "tools", resp)
        
        # Show details for specific tool if requested
        if ns.tool:
            selected = None
            for t in tools or []:
                name = getattr(t, "name", None)
                if not name:
                    try:
                        name = t.name  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if name == ns.tool:
                    selected = t
                    break
            if not selected:
                print(f"Tool '{ns.tool}' not found", file=sys.stderr)
                return 1
            print(json.dumps(_to_jsonable(selected), indent=2))
            return 0
        
        # List all tools with descriptions
        for t in tools or []:
            name = getattr(t, "name", None) or ""
            desc = getattr(t, "description", None) or ""
            print(f"{name}: {desc}".rstrip(": "))
        return 0


async def _call_tool_async(ns: argparse.Namespace) -> int:
    """Call an MCP tool with JSON arguments and print the result."""
    try:
        from mcp.client.session import ClientSession  # type: ignore
    except Exception:
        from mcp import ClientSession  # type: ignore
    
    if not ns.tool:
        print("--tool is required for call", file=sys.stderr)
        return 2
    
    try:
        arguments = json.loads(ns.json_args or "{}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --json-args: {e}", file=sys.stderr)
        return 2

    # streamable_http uses a high-level client, separate from stdio/SSE
    if ns.transport == "streamable_http":
        if not ns.url:
            print("Error: --url is required for streamable_http transport", file=sys.stderr)
            return 2
        headers = _parse_headers(ns.header)
        try:
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        except Exception as e:
            print(f"Error: streamable_http transport requires mcp package with streamable_http support: {e}", file=sys.stderr)
            return 1

        # Target the streamable-http endpoint at the mount path (e.g. /mcp),
        # accepting either bare host or mount path as input.
        url = _streamable_base_url(ns.url)
        try:
            from mcp.client.session import ClientSession  # type: ignore
        except Exception:
            from mcp import ClientSession  # type: ignore

        try:
            async with streamablehttp_client(url=url, headers=headers) as streams:  # type: ignore[misc]
                if isinstance(streams, (tuple, list)) and len(streams) >= 2:
                    read, write = streams[0], streams[1]
                else:
                    read, write = streams  # type: ignore[misc]

                async with ClientSession(read, write) as session:  # type: ignore[misc]
                    await session.initialize()
                    result = await session.call_tool(ns.tool, arguments=arguments)
        except Exception as e:
            print(f"Error connecting via streamable_http: {e}", file=sys.stderr)
            return 1
        print(json.dumps(_to_jsonable(result), indent=2))
        return 0

    async with _open_connection(ns) as (read, write):
        kwargs = _build_session_kwargs(read, write)
        async with ClientSession(**kwargs) as session:  # type: ignore[misc]
            await session.initialize()
            result = await session.call_tool(ns.tool, arguments=arguments)
        print(json.dumps(_to_jsonable(result), indent=2))
        return 0


def main() -> None:
    """Main entry point for the MCP client CLI."""
    parser = _build_arg_parser()
    ns = parser.parse_args()
    
    import anyio
    
    if ns.cmd == "list":
        code = anyio.run(_list_tools_async, ns)
        raise SystemExit(code)
    elif ns.cmd == "call":
        code = anyio.run(_call_tool_async, ns)
        raise SystemExit(code)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

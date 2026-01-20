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

from __future__ import annotations
"""
MCP Client for NVIDIA RAG
-------------------------

This CLI connects to an MCP server over one of three transports:
- sse                → HTTP SSE endpoint (e.g., http://127.0.0.1:8000/sse)
- streamable_http    → FastMCP streamable-http endpoint (e.g., http://127.0.0.1:8000/mcp)
- stdio              → Spawns/attaches to a server over stdio (requires --command/--args)

Capabilities:
- list tools exposed by the MCP server
- call a specific tool with JSON arguments

Compatibility:
- Adapts to multiple `mcp` SDK versions by introspecting ClientSession signatures
- Converts results to JSON via _to_jsonable for stable CLI output

StdIO notes:
- Use --command (e.g., python) and --args to point to the MCP server startup, such as:
  --command=python --args="-m nvidia_rag_mcp.mcp_server --transport stdio"
"""

import argparse
import json
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict
import inspect
import shlex


def _unpack_streams(streams):
    """Extract read/write streams from transport context."""
    if isinstance(streams, (tuple, list)) and len(streams) >= 2:
        return streams[0], streams[1]
    return streams


def _to_jsonable(value: Any) -> Any:
    """
    Best-effort conversion of SDK objects to JSON-serializable structures.
    Tries common adapters then falls back to public attrs or str(value).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
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
    return str(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI with `list` and `call` subcommands."""
    p = argparse.ArgumentParser(description="MCP client (Python SDK) for NVIDIA RAG MCP server")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--transport", choices=["sse", "streamable_http", "stdio"])
    common.add_argument("--command", help="Command to run for transport (e.g., python)")
    common.add_argument(
        "--args",
        dest="args_list",
        help="Arguments for transport command as a single string, e.g. '-m nvidia_rag.utils.mcp.mcp_server'",
    )
    common.add_argument("--env", action="append", help="Env var to pass to child MCP server: KEY=VALUE", default=[])
    common.add_argument("--url", help="URL for SSE/streamable_http transport, e.g., http://127.0.0.1:8000")
    common.add_argument(
        "--header",
        action="append",
        default=[],
        help="HTTP header for SSE/streamable_http transport as 'Key: Value' or 'Key=Value' (may repeat)",
    )

    _ = sub.add_parser("list", parents=[common], help="List tools or show details for a specific tool")

    p_call = sub.add_parser("call", parents=[common], help="Call a tool with JSON args")
    p_call.add_argument("--tool", required=True, help="Tool name to call")
    p_call.add_argument(
        "--json-args",
        default="{}",
        help='JSON string for tool arguments, e.g., \'{"messages":[...]}\'',
    )

    return p


@asynccontextmanager
async def _open_connection(ns: argparse.Namespace):
    """
    Establish MCP connection and yield (read, write) stream pair.
    
    For SSE: Connects to remote/local HTTP server with automatic endpoint probing.
    For streamable_http: Connects to FastMCP streamable-http endpoint.
    """
    if ns.transport == "stdio":
        try:
            from mcp.client.stdio import stdio_client
            try:
                from mcp.client.stdio import StdioServerParameters  # type: ignore
            except Exception:
                StdioServerParameters = None  # type: ignore
        except Exception as e:
            print(f"Error: stdio transport requires mcp package with stdio support: {e}", file=sys.stderr)
            return

        if not ns.command:
            print("Error: --command is required for transport stdio", file=sys.stderr)
            sys.exit(2)
        args = shlex.split(ns.args_list or "")
        try:
            if StdioServerParameters is not None:
                params = StdioServerParameters(command=ns.command, args=args)  # type: ignore
                async with stdio_client(params) as streams:
                    read, write = _unpack_streams(streams)
                    yield (read, write)
                    return
            else:
                async with stdio_client(command=ns.command, args=args) as streams:
                    read, write = _unpack_streams(streams)
                    yield (read, write)
                    return
        except TypeError:
            try:
                async with stdio_client(ns.command, args) as streams:
                    read, write = _unpack_streams(streams)
                    yield (read, write)
                    return
            except Exception as e:
                print(f"Error starting stdio client: {e}", file=sys.stderr)
                return
        except (ImportError, TypeError, AttributeError) as e:
            print(f"Error: stdio transport failed: {type(e).__name__}: {e}", file=sys.stderr)
            return
        except Exception as e:
            print(f"Error: unexpected stdio transport failure: {type(e).__name__}: {e}", file=sys.stderr)
            return

    if not ns.url:
        print("Error: --url is required for transports sse, streamable_http", file=sys.stderr)
        sys.exit(2)

    if ns.transport == "sse":
        from mcp.client.sse import sse_client

        try:
            async with sse_client(url=ns.url) as streams:
                read, write = _unpack_streams(streams)
                yield (read, write)
                return
        except Exception as e:
            print(f"Error: sse transport requires mcp package with sse support: {e}", file=sys.stderr)
            return

    elif ns.transport == "streamable_http":
        from mcp.client.streamable_http import streamablehttp_client

        try:
            async with streamablehttp_client(url=ns.url) as streams:
                read, write = _unpack_streams(streams)
                yield (read, write)
                return
        except Exception as e:
            print(f"Error: streamable_http transport requires mcp package with streamable_http support: {e}", file=sys.stderr)
            return

    else:
        print(f"Unsupported transport: {ns.transport}", file=sys.stderr)
        return


def _build_session_kwargs(read: Any, write: Any) -> Dict[str, Any]:
    """
    Build ClientSession kwargs compatible with multiple MCP SDK versions.
    Prefers read_stream/write_stream, falls back to reader/writer, and sets a client identity.
    """
    try:
        from mcp.client.session import ClientSession
    except Exception:
        from mcp import ClientSession
    
    sig = None
    try:
        sig = inspect.signature(ClientSession.__init__)
    except Exception:
        pass
    params = set(sig.parameters.keys()) if sig else set()
    kwargs: Dict[str, Any] = {}
    
    # Client identity (try different parameter names for version compatibility)
    if "client_info" in params:
        try:
            from mcp.types import ClientInfo
            kwargs["client_info"] = ClientInfo(name="nvidia-rag-mcp-client", version="0.0.0")
        except Exception:
            kwargs["client_info"] = {"name": "nvidia-rag-mcp-client", "version": "0.0.0"}
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
    """List available MCP tools and print 'name: description' per line."""
    try:
        from mcp.client.session import ClientSession
    except Exception:
        from mcp import ClientSession
        
    async with _open_connection(ns) as (read, write):
        kwargs = _build_session_kwargs(read, write)
        async with ClientSession(**kwargs) as session:
            await session.initialize()
            resp = await session.list_tools()
        
        tools = getattr(resp, "tools", resp)
        
        # List all tools with descriptions
        for t in tools or []:
            name = getattr(t, "name", None) or ""
            desc = getattr(t, "description", None) or ""
            print(f"{name}: {desc}".rstrip(": "))
        return 0


async def _call_tool_async(ns: argparse.Namespace) -> int:
    """Call an MCP tool with JSON arguments and print the JSON-serialized result."""
    try:
        from mcp.client.session import ClientSession
    except Exception:
        from mcp import ClientSession
    
    if not ns.tool:
        print("--tool is required for call", file=sys.stderr)
        return 2
    
    try:
        arguments = json.loads(ns.json_args or "{}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --json-args: {e}", file=sys.stderr)
        return 2

    try:
        async with _open_connection(ns) as (read, write):
            kwargs = _build_session_kwargs(read, write)
            async with ClientSession(**kwargs) as session:
                await session.initialize()
                result = await session.call_tool(ns.tool, arguments=arguments)
                print(json.dumps(_to_jsonable(result), indent=2))
    except Exception as e:
        print(f"Error calling tool: {e}", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    """
    Main entry point for the MCP client CLI.
    Examples:
      List tools (SSE):
        python examples/nvidia_rag_mcp/mcp_client.py list --transport=sse --url=http://127.0.0.1:8000/sse
      List tools (stdio):
        python examples/nvidia_rag_mcp/mcp_client.py list --transport=stdio --command=python \
          --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio"
      Call generate (streamable_http):
        python examples/nvidia_rag_mcp/mcp_client.py call --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
          --tool=generate --json-args='{"messages":[{"role":"user","content":"Hi"}]}'
      Call upload_documents (stdio):
        python examples/nvidia_rag_mcp/mcp_client.py call --transport=stdio --command=python \
          --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
          --tool=upload_documents \
          --json-args='{"collection_name":"my_collection","file_paths":["/abs/path/file.pdf"]}'
    """
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

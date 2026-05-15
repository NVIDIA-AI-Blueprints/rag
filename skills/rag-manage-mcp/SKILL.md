---
name: rag-manage-mcp
description: Set up, use, validate, or troubleshoot the NVIDIA RAG Blueprint MCP server, MCP client, and NeMo Agent Toolkit integration. Use when the user asks for RAG MCP tools, agent integration, NAT MCP integration, MCP server examples, or MCP debugging.
author: nvidia-rag-team
version: "0.1.0"
license: Apache-2.0
data_classification: internal
---

# RAG Manage MCP

## Overview

Use and troubleshoot the RAG MCP server, MCP client, and NeMo Agent Toolkit
integration examples.

## Prerequisites

- Confirm whether the user needs MCP server, MCP client, or NAT integration.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Read `references/mcp.md` before setup.
- Confirm a RAG deployment or library config is available.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate target MCP workflow and RAG endpoint.
2. Prepare environment, dependencies, config, and tool registration.
3. Execute the relevant example or integration.
4. Verify tool discovery and a sample RAG query through MCP.
5. Report tool names, endpoint used, and any failed calls.

## Reference

- `references/mcp.md`
- `../../docs/mcp.md`
- `../../examples/nvidia_rag_mcp/mcp_server.py`
- `../../examples/nvidia_rag_mcp/mcp_client.py`
- `../../notebooks/mcp_server_usage.ipynb`
- `../../notebooks/nat_mcp_integration.ipynb`
- `../../examples/rag_react_agent/README.md`

## Error Handling

If MCP tools are not visible, inspect server startup, client config, transport,
and endpoint reachability. Do not expose API keys or raw tool payloads that
contain confidential user data.

## Examples

- "Run the RAG MCP server."
- "Connect a client to RAG MCP tools."
- "Use RAG with NeMo Agent Toolkit."
- "Debug MCP tool discovery."

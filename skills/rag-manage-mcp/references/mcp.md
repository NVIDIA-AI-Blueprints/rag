# MCP Reference

## Source Files

- `docs/mcp.md`
- `examples/nvidia_rag_mcp/mcp_server.py`
- `examples/nvidia_rag_mcp/mcp_client.py`
- `notebooks/mcp_server_usage.ipynb`
- `notebooks/nat_mcp_integration.ipynb`
- `examples/rag_react_agent/README.md`

## Decision Table

| User goal | Action |
|---|---|
| Start RAG MCP server | Read `docs/mcp.md`, confirm RAG endpoint health, then run the documented server example. |
| Connect an MCP client | Use `examples/nvidia_rag_mcp/mcp_client.py` or `notebooks/mcp_server_usage.ipynb` and verify tool discovery. |
| Use NeMo Agent Toolkit | Read `notebooks/nat_mcp_integration.ipynb` and `examples/rag_react_agent/README.md`. |
| Debug missing tools | Check MCP server startup, client config, transport, endpoint reachability, and tool registration. |

## Config and Runtime Inputs

- RAG endpoint URL.
- MCP transport/client configuration.
- Any API keys required by the underlying RAG deployment, checked without
  printing values.
- Collection/query inputs, treated as user data and sanitized before reporting.

## Verification

Verify three layers:

1. RAG endpoint health.
2. MCP server startup and tool registration.
3. MCP client can discover tools and run a sample query.

Report the tool names and sample result, but sanitize request/response content
if it includes confidential data.

## Known Failure Modes

- RAG endpoint is unhealthy before MCP starts.
- MCP server starts but registers no tools.
- Client points to the wrong transport or port.
- Tool call reaches RAG but query fails because collections are empty or filters
  are invalid.

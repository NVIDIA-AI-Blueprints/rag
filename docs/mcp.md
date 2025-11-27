### NVIDIA RAG MCP Server

This guide explains how to deploy the RAG MCP server and how to use it from an MCP client (CLI and Python).

#### Prerequisites
- Python 3.10+ environment with:
  - `mcp`, `anyio`, `httpx`, `httpx-sse`, `uvicorn`
- Valid API key for your model backend:
  - If using NVCF: `NVCF_API_KEY` (and commonly `NVCF_ORG_ID`, `NVCF_TEAM_ID`, `NVCF_REGION`, `NVCF_BASE_URL`)
  - If using a direct NVIDIA key: `NVIDIA_API_KEY`

---

## Deploying the MCP Server

You can run the server in three supported transports:
- stdio (for local process-to-process communication)
- sse (Server-Sent Events) over HTTP for remote/local networking
- streamable_http (Model Context Protocol streamable HTTP transport over HTTP)

### SSE Mode

Start the server on `http://127.0.0.1:8000`:

```bash
# Option A: provide key via flag
python -m nvidia_rag.utils.mcp.mcp_server \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$YOUR_API_KEY"

# Option B: provide key via env (NVCF preferred if your key is nvapi-*)
export NVCF_API_KEY="$YOUR_API_KEY"
# Optional and commonly needed for NVCF:
# export NVCF_ORG_ID=...
# export NVCF_TEAM_ID=...
# export NVCF_REGION=us-east-1
# export NVCF_BASE_URL=https://api.nvcf.nvidia.com
python -m nvidia_rag.utils.mcp.mcp_server \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000
```

Authentication via headers is also supported in SSE mode:
- `Authorization: Bearer <KEY>` or `x-api-key: <KEY>`
The server will read the header and set `NVIDIA_API_KEY` internally for downstream use.

To suppress server-side INFO logs, set the `LOGLEVEL` environment variable:
```bash
LOGLEVEL=ERROR python -m nvidia_rag.utils.mcp.mcp_server \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$YOUR_API_KEY"
```

### stdio Mode

Run the server via stdio:

```bash
# Option A: run directly (useful for debugging)
python -m nvidia_rag.utils.mcp.mcp_server --transport stdio --api-key "$YOUR_API_KEY"

# Option B: launch via the client (see client section below)
```

### Streamable HTTP Mode

`streamable_http` exposes the MCP server over HTTP using the Model Context Protocol
streamable HTTP transport. This is useful when you want HTTP-based connectivity but
with proper MCP semantics (messages, tools, etc.).

Start the streamable_http server (by default it binds to `http://127.0.0.1:8000`
and serves MCP under the `/mcp` mount path):

```bash
# Option A: provide key via flag
python -m nvidia_rag.utils.mcp.mcp_server \
  --transport streamable_http \
  --api-key "$YOUR_API_KEY"

# Option B: provide key via env (NVCF preferred if your key is nvapi-*)
export NVCF_API_KEY="$YOUR_API_KEY"
# Optional and commonly needed for NVCF:
# export NVCF_ORG_ID=...
# export NVCF_TEAM_ID=...
# export NVCF_REGION=us-east-1
# export NVCF_BASE_URL=https://api.nvcf.nvidia.com
python -m nvidia_rag.utils.mcp.mcp_server \
  --transport streamable_http
```

The MCP client will typically connect to the base URL (e.g. `http://127.0.0.1:8000`)
and normalize it to the FastMCP mount path (e.g. `http://127.0.0.1:8000/mcp`) internally.

---

## Using the MCP Client (CLI)

The repository includes a simple client at `nvidia_rag.utils.mcp.mcp_client`. It supports
two commands:
- `list`: list tools
- `call`: call a specific tool

It supports three transports:
- `stdio`
- `sse`
- `streamable_http`

### SSE Transport Examples

First, ensure the SSE server is running (see SSE Mode section above). To suppress server-side logs, set `LOGLEVEL=ERROR`:

```bash
LOGLEVEL=ERROR python -m nvidia_rag.utils.mcp.mcp_server \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$YOUR_API_KEY"
```

#### List tools (SSE)
```bash
python -m nvidia_rag.utils.mcp.mcp_client list \
  --transport sse \
  --url http://127.0.0.1:8000
```

#### Call `generate` (SSE)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport sse \
  --url http://127.0.0.1:8000 \
  --tool generate \
  --json-args '{"messages":[{"role":"user","content":"Hello from SSE demo"}]}'
```

#### Call `search` (SSE)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport sse \
  --url http://127.0.0.1:8000 \
  --tool search \
  --json-args '{"query":"Tell me about Robert Frost poems","collection_name":"test","reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

#### Call `get_summary` (SSE)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport sse \
  --url http://127.0.0.1:8000 \
  --tool get_summary \
  --json-args '{"collection_name":"test","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

### Streamable HTTP Transport Examples

First, ensure the streamable_http server is running (see Streamable HTTP Mode section above).

> Note: Connect to the **base URL** (e.g. `http://127.0.0.1:8000`); the client normalizes to
> the MCP mount path (e.g. `/mcp`) internally.

#### List tools (streamable_http)
```bash
python -m nvidia_rag.utils.mcp.mcp_client list \
  --transport streamable_http \
  --url http://127.0.0.1:8000
```

#### Call `generate` (streamable_http)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport streamable_http \
  --url http://127.0.0.1:8000 \
  --tool generate \
  --json-args '{"messages":[{"role":"user","content":"Hello from streamable_http demo"}],
                "collection_name":"test"}'
```

#### Call `search` (streamable_http)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport streamable_http \
  --url http://127.0.0.1:8000 \
  --tool search \
  --json-args '{"query":"Tell me about Robert Frost poems","collection_name":"test",
                "reranker_top_k":2,"vdb_top_k":5,
                "enable_query_rewriting":false,"enable_reranker":true}'
```

#### Call `get_summary` (streamable_http)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport streamable_http \
  --url http://127.0.0.1:8000 \
  --tool get_summary \
  --json-args '{"collection_name":"test","file_name":"woods_frost.pdf",
                "blocking":false,"timeout":60}'
```

### stdio Transport Examples

The stdio transport launches the server as a subprocess. Use `--env=LOGLEVEL=ERROR` to suppress server-side logs.

#### List tools (stdio)
```bash
python -m nvidia_rag.utils.mcp.mcp_client list \
  --transport stdio \
  --command python \
  --args "-m nvidia_rag.utils.mcp.mcp_server --api-key $YOUR_API_KEY" \
  --env LOGLEVEL=ERROR
```

#### Call `generate` (stdio)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport stdio \
  --command python \
  --args "-m nvidia_rag.utils.mcp.mcp_server --api-key $YOUR_API_KEY" \
  --env LOGLEVEL=ERROR \
  --tool generate \
  --json-args '{"messages":[{"role":"user","content":"Hello from stdio demo"}]}'
```

#### Call `search` (stdio)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport stdio \
  --command python \
  --args "-m nvidia_rag.utils.mcp.mcp_server --api-key $YOUR_API_KEY" \
  --env LOGLEVEL=ERROR \
  --tool search \
  --json-args '{"query":"Tell me about Robert Frost poems","collection_name":"test","reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

#### Call `get_summary` (stdio)
```bash
python -m nvidia_rag.utils.mcp.mcp_client call \
  --transport stdio \
  --command python \
  --args "-m nvidia_rag.utils.mcp.mcp_server --api-key $YOUR_API_KEY" \
  --env LOGLEVEL=ERROR \
  --tool get_summary \
  --json-args '{"collection_name":"test","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

---

## Available MCP Tools

The server exposes three main tools:

1. **`generate`**: Generate an answer using NVIDIA RAG (optionally with knowledge base). Provide chat messages and optional generation parameters.
   - Required: `messages` (array of chat messages)
   - Optional: `collection_name`, `temperature`, `top_p`, `max_tokens`, etc.

2. **`search`**: Search the vector database and return citations for a given query.
   - Required: `query` (search query string), `collection_name`
   - Optional: `reranker_top_k`, `vdb_top_k`, `enable_query_rewriting`, `enable_reranker`

3. **`get_summary`**: Retrieve the pre-generated summary for a document from a collection.
   - Required: `collection_name`, `file_name`
   - Optional: `blocking` (wait for summary generation), `timeout`

---

## Troubleshooting

- **401 Unauthorized**:
  - Ensure the correct key is available to the server:
    - SSE: pass `--api-key` when starting the server, or set `NVCF_API_KEY`/`NVIDIA_API_KEY` environment variables.
    - stdio: pass `--api-key` in the server args, or use `--env` to forward environment variables.
  - For NVCF deployments, you may also need `NVCF_ORG_ID`, `NVCF_TEAM_ID`, `NVCF_REGION`, `NVCF_BASE_URL`.

- **404 Not Found (SSE)**:
  - Use the base URL (e.g., `http://127.0.0.1:8000`). The client probes standard SSE endpoints (`/sse`, `/messages`) automatically.
  - Ensure the server was started with `--transport sse` and is still running.

- **404 Not Found (streamable_http)**:
  - Ensure the server was started with `--transport streamable_http` and is still running.
  - Use the base URL (e.g., `http://127.0.0.1:8000`); the client will append the MCP mount path (e.g. `/mcp`) internally.

- **Server-side logs cluttering output**:
  - Set `LOGLEVEL=ERROR` environment variable when starting the server (SSE) or use `--env=LOGLEVEL=ERROR` with the client (stdio).

- **Version mismatches**:
  - Ensure `mcp`, `anyio`, and `uvicorn` versions are compatible with your environment.

---

## Related Notebook
- See `notebooks/mcp_server_usage.ipynb` for an executable end-to-end walkthrough (SSE, streamable_http, and stdio).

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
## MCP Server and Client Usage

This guide shows how to run the NVIDIA RAG MCP server and use the included Python MCP client to:
- List available tools
- Generate answers
- Search the vector database
- Retrieve document summaries


### Prerequisites

- NVIDIA RAG HTTP services are reachable:
  - RAG API base URL: `VITE_API_CHAT_URL` (defaults to `http://localhost:8081`)
  - Ingestor API base URL: `INGESTOR_URL` (defaults to `http://127.0.0.1:8082`)
- Python 3.11+ and dependencies:

```bash
pip install -r nvidia_rag_mcp/requirements.txt
```

### 1) Seed the Knowledge Base (optional but recommended)

Create a collection and upload a sample document so search/generate/summary tools have content.

```bash
export INGESTOR_URL="${INGESTOR_URL:-http://127.0.0.1:8082}"
COLLECTION="my_collection"
PDF_PATH="data/multimodal/woods_frost.pdf"

# Create collection
curl -sS -X POST "$INGESTOR_URL/v1/collections" \
  -H "Content-Type: application/json" \
  -d "[\"$COLLECTION\"]"

# Upload document with summary generation enabled
curl -sS -X POST "$INGESTOR_URL/v1/documents" \
  -F "documents=@${PDF_PATH}" \
  -F "data={\"collection_name\":\"$COLLECTION\",\"blocking\":true,\"custom_metadata\":[],\"generate_summary\":true}"
```

### 2) Start the MCP Server

From the repo root, start either transport:

#### SSE
```bash
# Free port 8000 if needed (Linux)
fuser -k 8000/tcp || true

python nvidia_rag_mcp/mcp_server.py --transport sse --host 127.0.0.1 --port 8000
```

#### streamable_http
```bash
# Free port 8000 if needed (Linux)
fuser -k 8000/tcp || true

python nvidia_rag_mcp/mcp_server.py --transport streamable_http
```

Notes:
- The server uses `VITE_API_CHAT_URL` (default `http://localhost:8081`) to call the RAG HTTP endpoints.
- For streamable_http, probing `http://127.0.0.1:8000/mcp` may return HTTP 406 for GET; that can still indicate readiness.

### 3) Use the MCP Client (List tools, Generate, Search, Get Summary)

The CLI lives at `nvidia_rag_mcp/mcp_client.py`. The examples below show both transports.

#### List Tools

SSE:
```bash
python nvidia_rag_mcp/mcp_client.py list \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse
```

streamable_http:
```bash
python nvidia_rag_mcp/mcp_client.py list \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp
```

Expected tools: `generate`, `search`, `get_summary`.

#### Generate

SSE:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=generate \
  --json-args='{"messages":[{"role":"user","content":"Say \"ok\""}],"collection_name":"my_collection"}'
```

streamable_http:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=generate \
  --json-args='{"messages":[{"role":"user","content":"Say \"ok\""}],"collection_name":"my_collection"}'
```

#### Search

SSE:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=search \
  --json-args='{"query":"Tell me about Robert Frost''s poems","collection_name":"my_collection","reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

streamable_http:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=search \
  --json-args='{"query":"Tell me about Robert Frost''s poems","collection_name":"my_collection","reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

#### Get Summary

SSE:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

streamable_http:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

### Troubleshooting

- 404 on `/sse`: Ensure you are pointing the client at the base SSE URL (`http://127.0.0.1:8000/sse`).
- 406 on `/mcp`: For streamable_http, a 406 on GET can still indicate the server is up; use the client list/call commands above.
- Port conflicts: Free port 8000 before launching (e.g., `fuser -k 8000/tcp` on Linux).
- Ensure the RAG and Ingestor services are running and reachable at the configured URLs.

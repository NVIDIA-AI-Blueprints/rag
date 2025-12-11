<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
## MCP Server and Client Usage

This guide shows how to run the NVIDIA RAG MCP server and use the included Python MCP client to interact with the RAG system.

### Available Tools

The MCP server exposes two categories of tools:

#### Retrieval Tools

These tools interact with the RAG server to query and generate responses from your knowledge base:

| Tool | Description |
|------|-------------|
| `generate` | Generate answers using the RAG pipeline with context from the knowledge base |
| `search` | Search the vector database for relevant documents |
| `get_summary` | Retrieve document summaries from the knowledge base |

#### Ingestor Tools

These tools manage collections and documents in the vector database:

| Tool | Description |
|------|-------------|
| `create_collections` | Create one or more collections in the vector database |
| `upload_documents` | Upload documents to a collection with optional summary generation |
| `delete_collections` | Delete one or more collections from the vector database |

### Supported Transport Modes

The MCP server supports three transport modes:

| Transport | Description | Server Required |
|-----------|-------------|-----------------|
| `sse` | Server-Sent Events over HTTP | Yes |
| `streamable_http` | HTTP-based streaming | Yes |
| `stdio` | Standard input/output | No |

**Note:** The `stdio` transport can be run without starting the MCP server separately. The client spawns the server process directly, making it ideal for local development and testing.


### Prerequisites

- NVIDIA RAG HTTP services are reachable:
  - RAG API base URL: `VITE_API_CHAT_URL` (defaults to `http://localhost:8081`)
  - Ingestor API base URL: `INGESTOR_URL` (defaults to `http://127.0.0.1:8082`)
- Python 3.11+ and dependencies:

```bash
pip install -r nvidia_rag_mcp/requirements.txt
```

### 1) Start the MCP Server

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

### 2) Seed the Knowledge Base

Before using retrieval tools, create a collection and upload documents using the ingestor tools:

```bash
# Create collection
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'

# Upload document with summary generation enabled
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true}'
```

### 3) Use the MCP Client (List tools, Generate, Search, Get Summary)

The CLI lives at `nvidia_rag_mcp/mcp_client.py`. The examples below show all transports (SSE, streamable_http, stdio).

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

stdio:
```bash
python nvidia_rag_mcp/mcp_client.py list \
  --transport=stdio \
  --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio"
```

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

stdio:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
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

stdio:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
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

stdio:
```bash
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

### 4) Ingestor Tools (Create, Upload, Delete)

Use these before/after the RAG tools to manage collections and documents.

SSE:
```bash
# Create collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'

# Upload document(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'

# Delete collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

streamable_http:
```bash
# Create collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'

# Upload document(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'

# Delete collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

stdio:
```bash
# Create collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'

# Upload document(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'

# Delete collection(s)
python nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

### Troubleshooting

- 404 on `/sse`: Ensure you are pointing the client at the base SSE URL (`http://127.0.0.1:8000/sse`).
- 406 on `/mcp`: For streamable_http, a 406 on GET can still indicate the server is up; use the client list/call commands above.
- Port conflicts: Free port 8000 before launching (e.g., `fuser -k 8000/tcp` on Linux).
- Ensure the RAG and Ingestor services are running and reachable at the configured URLs.

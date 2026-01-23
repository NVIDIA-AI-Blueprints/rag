<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# MCP Server and Client Usage

This guide shows how to run the NVIDIA RAG MCP server and use the included Python MCP client to interact with the RAG system.

## Prerequisites

Confirm you have the following prerequisites before you run the NVIDIA RAG MCP server:

- NVIDIA RAG HTTP services are reachable. Follow the [quickstart guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md) to start the end-to-end RAG workflow.
- Python 3.11+ and dependencies:

```bash
pip install -r examples/nvidia_rag_mcp/requirements.txt
```

## RAG Tools

These tools interact with the RAG server to query and generate responses from your knowledge base:

| Tool | Description |
|------|-------------|
| `generate` | Generate answers using the RAG pipeline with context from the knowledge base |
| `search` | Search the vector database for relevant documents |
| `get_summary` | Retrieve document summaries from the knowledge base |

## Ingestor Tools

These tools manage collections and documents in the vector database:

| Tool | Description |
|------|-------------|
| `create_collections` | Create one or more collections in the vector database |
| `list_collections` | List collections from the vector database via the ingestor service |
| `upload_documents` | Upload documents to a collection with optional summary generation |
| `get_documents` | Retrieve documents that have been ingested into a collection |
| `update_documents` | Update (re-upload) existing documents in a collection |
| `delete_documents` | Delete one or more documents from a collection |
| `update_collection_metadata` | Update catalog metadata for an existing collection |
| `update_document_metadata` | Update catalog metadata for a specific document in a collection |
| `delete_collections` | Delete one or more collections from the vector database |

## Supported Transport Modes

The MCP server supports three transport modes:

| Transport | Description | Server Required |
|-----------|-------------|-----------------|
| `sse` | Server-Sent Events over HTTP | Yes |
| `streamable_http` | HTTP-based streaming | Yes |
| `stdio` | Standard input/output | No |

**Note:** The `stdio` transport can be run without starting the MCP server separately. The client spawns the server process directly, making it ideal for local development and testing.

## Use the MCP Server

1. Start the MCP Server

From the repo root, start either transport:


 * For SSE
```bash
# Free port 8000 if needed (Linux)
fuser -k 8000/tcp || true

python examples/nvidia_rag_mcp/mcp_server.py --transport sse --host 127.0.0.1 --port 8000
```

 * For streamable_http
```bash
# Free port 8000 if needed (Linux)
fuser -k 8000/tcp || true

python examples/nvidia_rag_mcp/mcp_server.py --transport streamable_http --host 127.0.0.1 --port 8000
```

2. Use the `list` subcommand to see all exposed tools (RAG + Ingestor).

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py list \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py list \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py list \
  --transport=stdio \
  --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio"
```

You should see output similar to the following:  
`generate`, `search`, `get_summary`, `get_documents`, `delete_documents`, `update_documents`, `list_collections`, `update_collection_metadata`, `update_document_metadata`, `create_collections`, `delete_collections`, `upload_documents`.

3. The examples below show all transports (SSE, streamable_http, stdio) for the RAG tools. The CLI lives at `examples/nvidia_rag_mcp/mcp_client.py`. 

Here is the `generate` example:

* For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=generate \
  --json-args='{"messages":[{"role":"user","content":"Say \"ok\""}],"collection_names":["my_collection"]}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=generate \
  --json-args='{"messages":[{"role":"user","content":"Say \"ok\""}],"collection_names":["my_collection"]}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=generate \
  --json-args='{"messages":[{"role":"user","content":"Say \"ok\""}],"collection_names":["my_collection"]}'
```

Here are the `search` examples:

* For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=search \
  --json-args='{"query":"Tell me about Robert Frost''s poems","collection_names":["my_collection"],"reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=search \
  --json-args='{"query":"Tell me about Robert Frost''s poems","collection_names":["my_collection"],"reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=search \
  --json-args='{"query":"Tell me about Robert Frost''s poems","collection_names":["my_collection"],"reranker_top_k":2,"vdb_top_k":5,"enable_query_rewriting":false,"enable_reranker":true}'
```

Here are the `get_summary` examples"

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse \
  --url=http://127.0.0.1:8000/sse \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

* For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http \
  --url=http://127.0.0.1:8000/mcp \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio \
  --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=get_summary \
  --json-args='{"collection_name":"my_collection","file_name":"woods_frost.pdf","blocking":false,"timeout":60}'
```

4. Use these before/after the RAG tools to manage collections and documents in the vector database.

Here are the `create_collections` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=create_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

Here are the `list_collections` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=list_collections \
  --json-args='{}'
```

* For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=list_collections \
  --json-args='{}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=list_collections \
  --json-args='{}'
```

Here are the `upload_documents` examples:

* For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=upload_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":true,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

Here are the `get_documents` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=get_documents \
  --json-args='{"collection_name":"my_collection"}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=get_documents \
  --json-args='{"collection_name":"my_collection"}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=get_documents \
  --json-args='{"collection_name":"my_collection"}'
```

Here are the `update_documents` examples: 

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=update_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":false,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=update_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":false,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=update_documents \
  --json-args='{"collection_name":"my_collection","file_paths":["data/multimodal/woods_frost.pdf"],"blocking":true,"generate_summary":false,"split_options":{"chunk_size":512,"chunk_overlap":150}}'
```

Here are the `delete_document` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=delete_documents \
  --json-args='{"collection_name":"my_collection","document_names":["woods_frost.pdf"]}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=delete_documents \
  --json-args='{"collection_name":"my_collection","document_names":["woods_frost.pdf"]}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=delete_documents \
  --json-args='{"collection_name":"my_collection","document_names":["woods_frost.pdf"]}'
```

Here are the `update_collection_metadata` examples:

* For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=update_collection_metadata \
  --json-args='{"collection_name":"my_collection","description":"Updated description","tags":["tag1","tag2"],"owner":"owner@example.com","business_domain":"demo","status":"Active"}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=update_collection_metadata \
  --json-args='{"collection_name":"my_collection","description":"Updated description","tags":["tag1","tag2"],"owner":"owner@example.com","business_domain":"demo","status":"Active"}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=update_collection_metadata \
  --json-args='{"collection_name":"my_collection","description":"Updated description","tags":["tag1","tag2"],"owner":"owner@example.com","business_domain":"demo","status":"Active"}'
```

Here are `update_document_metadata` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=update_document_metadata \
  --json-args='{"collection_name":"my_collection","document_name":"woods_frost.pdf","description":"Updated description","tags":["tag1","tag2"]}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=update_document_metadata \
  --json-args='{"collection_name":"my_collection","document_name":"woods_frost.pdf","description":"Updated description","tags":["tag1","tag2"]}'
```

 * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=update_document_metadata \
  --json-args='{"collection_name":"my_collection","document_name":"woods_frost.pdf","description":"Updated description","tags":["tag1","tag2"]}'
```

Here are the `delete_collections` examples:

 * For SSE:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=sse --url=http://127.0.0.1:8000/sse \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

 * For streamable_http:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=streamable_http --url=http://127.0.0.1:8000/mcp \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

  * For stdio:
```bash
python examples/nvidia_rag_mcp/mcp_client.py call \
  --transport=stdio --command=python \
  --args="examples/nvidia_rag_mcp/mcp_server.py --transport stdio" \
  --tool=delete_collections \
  --json-args='{"collection_names":["my_collection"]}'
```

## Troubleshoot MCP Server and Client Usage

- 404 on `/sse`: Ensure you are pointing the client at the base SSE URL (`http://127.0.0.1:8000/sse`).
- 406 on `/mcp`: For streamable_http, a 406 on GET can still indicate the server is up; use the client list/call commands above.
- Port conflicts: Free port 8000 before launching (e.g., `fuser -k 8000/tcp` on Linux).
- Ensure the RAG and Ingestor services are running and reachable at the configured URLs.

For more information, refer to [Troubleshoot RAG Blueprint](troubleshooting.md). 

## Related Topics

- [MCP server usage notebook](../notebooks/mcp_server_usage.ipynb)

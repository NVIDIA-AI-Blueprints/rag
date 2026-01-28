<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# MCP Server and Client Usage

This guide shows how to run the NVIDIA RAG MCP server and use the included Python MCP client to interact with the RAG system.

The MCP server acts as a thin adapter layer that exposes NVIDIA RAG and Ingestor HTTP APIs as standard MCP tools (e.g., `generate`, `search`, `upload_documents`). This makes it easy to plug RAG into MCP-aware clients such as IDEs, agents, and orchestration frameworks without writing custom integration code for each one—any MCP-compatible client can discover and call these tools in a uniform way. You can learn more about MCP from [here](https://modelcontextprotocol.io/docs/getting-started/intro).

## Overview

**[MCP Server](../examples/nvidia_rag_mcp/mcp_server.py)**  
The server exposes NVIDIA RAG and Ingestor HTTP APIs as MCP tools using FastMCP. It acts as a thin adapter that forwards requests to the backend services (RAG at port 8081, Ingestor at port 8082). Supports three transport modes: `sse`, `streamable_http`, and `stdio`.

**[MCP Client](../examples/nvidia_rag_mcp/mcp_client.py)**  
A command-line client for interacting with the MCP server. It can list available tools and call them with JSON arguments. The client adapts to multiple MCP SDK versions and supports the same three transport modes as the server.

### Available Tools

The MCP server exposes two categories of tools:

#### RAG Tools

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
| `create_collection` | Create a collection in the vector database (supports metadata schema and catalog metadata) |
| `list_collections` | List collections from the vector database via the ingestor service |
| `upload_documents` | Upload documents to a collection with optional summary generation |
| `get_documents` | Retrieve documents that have been ingested into a collection |
| `update_documents` | Update (re-upload) existing documents in a collection |
| `delete_documents` | Delete one or more documents from a collection |
| `update_collection_metadata` | Update catalog metadata for an existing collection |
| `update_document_metadata` | Update catalog metadata for a specific document in a collection |
| `delete_collections` | Delete one or more collections from the vector database |

### Supported Transport Modes

The MCP server supports three transport modes:

| Transport | Description | Server Required |
|-----------|-------------|-----------------|
| `sse` | Server-Sent Events over HTTP | Yes |
| `streamable_http` | HTTP-based streaming | Yes |
| `stdio` | Standard input/output | No |

**Note:** The `stdio` transport can be run without starting the MCP server separately. The client spawns the server process directly, making it ideal for local development and testing.

## **End-to-End Usage Example**  
For a complete workflow demonstration including collection creation, document upload, and RAG queries, see the [MCP server usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/mcp_server_usage.ipynb).


## Related Topics

- [NeMo Agent Toolkit integration with NVIDIA RAG MCP server](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/nat_mcp_integration.ipynb)

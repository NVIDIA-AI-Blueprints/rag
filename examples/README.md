# NVIDIA RAG Examples

This directory contains example integrations and extensions for NVIDIA RAG.

## Examples

| Example | Description | Documentation |
|---------|-------------|---------------|
| [nvidia_nat_rag](./nvidia_nat_rag/) | Integration with NeMo Agent Toolkit (NAT) providing RAG query and search capabilities for agent workflows | [README](./nvidia_nat_rag/README.md) |
| [nvidia_rag_mcp](./nvidia_rag_mcp/) | MCP (Model Context Protocol) server and client for exposing NVIDIA RAG capabilities to MCP-compatible applications | [Documentation](../docs/nvidia-rag-mcp.md) |

## nvidia_nat_rag

This plugin integrates NVIDIA RAG with NeMo Agent Toolkit, enabling intelligent agents to use RAG tools for document retrieval and question answering. It demonstrates:

- Creating custom NAT tools that wrap NVIDIA RAG functionality
- Using the React Agent workflow for intelligent tool selection
- Integrating Milvus Lite as an embedded vector database

See the [nvidia_nat_rag README](./nvidia_nat_rag/README.md) for setup and usage instructions.

## nvidia_rag_mcp

This example provides an MCP server and client that exposes NVIDIA RAG and Ingestor capabilities as MCP tools. It supports multiple transport modes (SSE, streamable HTTP, stdio) and enables MCP-compatible applications to:

- Generate answers using the RAG pipeline
- Search the vector database for relevant documents
- Manage collections and documents in the vector database

See the [MCP documentation](../docs/nvidia-rag-mcp.md) for detailed setup and usage instructions.

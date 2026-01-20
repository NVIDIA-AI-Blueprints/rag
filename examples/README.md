# NVIDIA RAG Examples

This directory contains example integrations and extensions for NVIDIA RAG.

## Examples

| Example | Description | Documentation |
|---------|-------------|---------------|
| [rag_react_agent](./rag_react_agent/) | Integration with [NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit) providing RAG query and search capabilities for agent workflows | [README](./rag_react_agent/README.md) |
| [nvidia_rag_mcp](./nvidia_rag_mcp/) | MCP (Model Context Protocol) server and client for exposing NVIDIA RAG capabilities to MCP-compatible applications | [Documentation](../docs/nvidia-rag-mcp.md) |

## rag_react_agent

This plugin integrates NVIDIA RAG with [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit), enabling intelligent agents to use RAG tools for document retrieval and question answering. It demonstrates:

- Creating custom NAT tools that wrap NVIDIA RAG functionality
- Using the React Agent workflow for intelligent tool selection

See the [rag_react_agent README](./rag_react_agent/README.md) for setup and usage instructions.

## nvidia_rag_mcp

This example provides an MCP server and client that exposes NVIDIA RAG and Ingestor capabilities as MCP tools. It supports multiple transport modes (SSE, streamable HTTP, stdio) and enables MCP-compatible applications to:

- Generate answers using the RAG pipeline
- Search the vector database for relevant documents
- Manage collections and documents in the vector database

See the [MCP documentation](../docs/nvidia-rag-mcp.md) for detailed setup and usage instructions.

# NVIDIA RAG Examples

This directory contains example integrations and extensions for NVIDIA RAG.

## Examples

| Example | Description | Documentation |
|---------|-------------|---------------|
| [rag_react_agent](./rag_react_agent/) | Integration with [NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit) providing RAG query and search capabilities for agent workflows | [README](./rag_react_agent/README.md) |
| [nvidia_rag_mcp](./nvidia_rag_mcp/) | MCP (Model Context Protocol) server and client for exposing NVIDIA RAG capabilities to MCP-compatible applications | [Documentation](../docs/mcp.md) |
| [rag_event_ingest](./rag_event_ingest/) | Automated document & video ingestion from object storage (MinIO) via Kafka, with VSS video analysis | [Notebook](../notebooks/rag_event_ingest.ipynb) |

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

See the [MCP documentation](../docs/mcp.md) for detailed setup and usage instructions.

## rag_event_ingest

This example deploys an event-driven ingestion pipeline that monitors MinIO object storage for new file uploads via Kafka events. Documents are indexed through the RAG Ingestor, while videos are analyzed by NVIDIA VSS (Video Search & Summarization). All content becomes queryable through the RAG Agent.

Components:
- **kafka_consumer/** - Event-driven consumer that routes files to RAG or VSS based on file type
- **deploy/** - Docker Compose for Kafka, MinIO, and the consumer
- **data/** - Sample documents and videos for testing

See the [notebook](../notebooks/rag_event_ingest.ipynb) for step-by-step deployment and testing.

# Building Agentic RAG with NeMo Agent Toolkit

This example demonstrates how to build intelligent agents that leverage **NVIDIA RAG** capabilities using [NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit). The agent can autonomously decide when and how to query your document knowledge base.

## Overview

This example shows how to:

1. **Expose RAG as agent tools** - Wrap NVIDIA RAG query and search capabilities as tools that agents can use
2. **Build a ReAct agent** - Use NAT's ReAct workflow to create an agent that reasons about when to use RAG

The ReAct (Reason + Act) agent pattern enables the LLM to iteratively reason about which tools to use based on the user's query, making it ideal for building conversational AI applications with document retrieval capabilities.

## Prerequisites

- Python 3.11+
- Access to NVIDIA AI endpoints (API key required)
- **Data ingested into Milvus** - Complete the [rag_library_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb) notebook to set up Milvus and ingest documents before running this example

## Quick Start

All commands should be run from the `examples/rag_react_agent/` directory.

### 1. Set Environment Variables

```bash
# Required: NVIDIA API key for embeddings, reranking, and LLM
export NVIDIA_API_KEY="your-nvidia-api-key"

# Optional: If using custom endpoints
# export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

### 2. Configure Vector Database Endpoint

By default, the example connects to Milvus at `http://localhost:19530`. You can configure this in two ways:

**Option A: Environment Variable (takes precedence)**

```bash
# For standard Milvus server
# export APP_VECTORSTORE_URL="http://localhost:19530"

# Or for a remote Milvus instance
# export APP_VECTORSTORE_URL="http://milvus-host:19530"
```

**Option B: Update config.yml**

Edit `src/rag_react_agent/configs/config.yml` and update the `vdb_endpoint` field:

```yaml
functions:
  rag_query:
    vdb_endpoint: "http://localhost:19530"  # Your Milvus endpoint
  rag_search:
    vdb_endpoint: "http://localhost:19530"  # Your Milvus endpoint
```

> **Note**: If you have followed rag_library_lite_usage.ipynb notebook and have a setup using milvus-lite, provide an absolute path to the `.db` file (e.g., `/home/user/data/milvus.db`).

### 3. Install Dependencies and Run the Agent

```bash
# From examples/rag_react_agent/ directory
# Install all dependencies including nvidia-rag and NeMo Agent Toolkit
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Usage

### Running the RAG Agent

The example uses NAT's **ReAct Agent** workflow, which enables the LLM to reason about which RAG tools to use based on the user's query.

```bash
# From examples/rag_react_agent/ directory with .venv activated
nat run --config_file src/rag_react_agent/configs/config.yml --input "what is giraffe doing?"
```

### Example Queries

Try different queries to see how the agent decides which tool to use:

```bash
# Query that triggers rag_query (generates a response using retrieved documents)
nat run --config_file src/rag_react_agent/configs/config.yml --input "Summarize the main themes of the documents"

# Query that triggers rag_search (returns relevant document chunks)
nat run --config_file src/rag_react_agent/configs/config.yml --input "Find all animals mentioned in documents"
```

### Expected Output

When running successfully, you'll see the agent's reasoning process:

```
Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 3
Number of LLMs: 1

------------------------------
[AGENT]
Agent input: what is giraffe doing?
Agent's thoughts: 
Thought: I don't have any information about what giraffe is doing. 

Action: rag_query 
Action Input: {'query': 'giraffe current activity'}
------------------------------

------------------------------
[AGENT]
Calling tools: rag_query
Tool's input: {'query': 'giraffe current activity'}
Tool's response: 
Driving a car at the beach
------------------------------

------------------------------
[AGENT]
Agent input: what is giraffe doing?
Agent's thoughts: 
Thought: I now know the final answer 
Final Answer: Giraffe is driving a car at the beach.
------------------------------

--------------------------------------------------
Workflow Result:
['Giraffe is driving a car at the beach.']
--------------------------------------------------
```

## Configuration

The configuration file at `src/rag_react_agent/configs/config.yml` defines the RAG tools and agent workflow:

```yaml
functions:
  # RAG Query Tool - Queries documents and returns LLM or VLM Generated response
  rag_query:
    _type: nvidia_rag_query
    # Ensure collection_name matches with the collection name used in the rag library notebook.
    collection_names: ["test_library"]    # Milvus collection names
    vdb_endpoint: "http://localhost:19530" # Milvus endpoint URL

  # RAG Search Tool - Searches for relevant document chunks
  rag_search:
    _type: nvidia_rag_search
    collection_names: ["test_library"]
    vdb_endpoint: "http://localhost:19530"
    reranker_top_k: 3                     # Number of results after reranking
    vdb_top_k: 20                         # Number of results from vector search

  # Utility tool for date/time queries
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

# ReAct Agent workflow - enables the LLM to reason about tool usage
workflow:
  _type: react_agent
  tool_names:
    - rag_query
    - rag_search
    - current_datetime
  llm_name: nim_llm
  verbose: true                           # Shows agent reasoning process
```

### RAG Tools

| Tool | Type | Description |
|------|------|-------------|
| `rag_query` | `nvidia_rag_query` | Queries documents and returns an AI-generated response based on retrieved context |
| `rag_search` | `nvidia_rag_search` | Searches for relevant document chunks without generating a response |

### Tool Configuration Options

#### `nvidia_rag_query`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `collection_names` | List of Milvus collection names to query | `[]` |
| `vdb_endpoint` | Vector database endpoint URL or absolute path to Milvus Lite `.db` file | `"http://localhost:19530"` |

#### `nvidia_rag_search`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `collection_names` | List of Milvus collection names to search | `[]` |
| `vdb_endpoint` | Vector database endpoint URL or absolute path to Milvus Lite `.db` file | `"http://localhost:19530"` |
| `reranker_top_k` | Number of results to return after reranking | `10` |
| `vdb_top_k` | Number of results to retrieve before reranking | `100` |

## Troubleshooting

### Error: Function type `nvidia_rag_query` not found

The tools are not registered. Ensure you've installed the package:

```bash
# From examples/rag_react_agent/ directory
uv sync
source .venv/bin/activate
```

### Error: Token limit exceeded

If you encounter token limit errors, reduce the number of results:

```yaml
rag_search:
  _type: nvidia_rag_search
  reranker_top_k: 1    # Reduce from 3
  vdb_top_k: 10        # Reduce from 20
```

This commonly occurs when documents contain large base64-encoded images.

### Error: NVIDIA API key not set

```bash
export NVIDIA_API_KEY="your-api-key"
```

### Error: Connection to Milvus failed

Ensure Milvus is running and accessible at the configured endpoint. If you followed the [rag_library_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb) notebook, Milvus should be running at `http://localhost:19530`.

```bash
# Check if Milvus is running
docker ps | grep milvus
```

## Learn More

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)

## License

Apache-2.0

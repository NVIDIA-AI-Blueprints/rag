# RAG React Agent Example

This plugin integrates [NVIDIA RAG](https://github.com/NVIDIA-AI-Blueprints/rag) with NeMo Agent Toolkit, providing RAG query and search capabilities for your agent workflows.

## Overview

This example demonstrates how to:

1. **Create custom NAT tools** that wrap NVIDIA RAG functionality
2. **Use the React Agent workflow** to enable intelligent tool selection
3. **Integrate Milvus Lite** as an embedded vector database for document retrieval

The React Agent autonomously decides when to use RAG tools based on user queries, making it easy to build conversational AI applications with document retrieval capabilities.

## Prerequisites

- Python 3.11+
- NeMo Agent Toolkit installed
- Access to NVIDIA AI endpoints (API key required)

## Installation

All commands should be run from the `examples/rag_react_agent/` directory.

### 1. Set Environment Variables

```bash
# Required: NVIDIA API key for embeddings, reranking, and LLM
export NVIDIA_API_KEY="your-nvidia-api-key"

# Optional: If using custom endpoints
# export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

### 2. Setting Up the Vector Store

The plugin uses Milvus Lite as an embedded vector database. You need to ingest documents to create the vector store before running RAG queries.

> **Note**: The ingestion script requires a separate virtual environment due to dependency conflicts with the main toolkit.

#### Create a Separate Virtual Environment for Ingestion

```bash
# From examples/rag_react_agent/ directory
cd examples/rag_react_agent

# Create and activate a new virtual environment for ingestion
uv venv .venv-ingest
source .venv-ingest/bin/activate

# Install ingestion dependencies
uv pip install -r requirements-ingest.txt
```

#### Ingest Sample Data

Sample documents are available in the repository's `data/multimodal/` directory:

```bash
# Ingest the sample documents
python ingest_data.py \
  --collection test_library \
  --files ../../data/multimodal/multimodal_test.pdf ../../data/multimodal/woods_frost.docx \
  --db-path ./milvus.db
```

This creates a `milvus.db` file containing the vector embeddings in the current directory.

#### Deactivate Ingestion Environment

After ingestion, deactivate the ingestion environment:

```bash
deactivate
```

### 3. Install the Plugin and Run Agent

```bash
# From examples/rag_react_agent/ directory
# Create virtual environment and install all dependencies (including nvidia-rag from local repo)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Usage

### Running a RAG Workflow

The example uses the **React Agent** workflow from NeMo Agent Toolkit, which enables the LLM to reason about which RAG tools to use based on the user's query.

```bash
# From examples/rag_react_agent/ directory with .venv activated
nat run --config_file src/rag_react_agent/configs/config.yml --input "what is giraffee doing?"
```

### Example Queries

Try different queries to see how the React Agent decides which tool to use:

```bash
# Query that triggers rag_query (generates a response using retrieved documents)
nat run --config_file src/rag_react_agent/configs/config.yml --input "Summarize the main themes of the documents"

# Query that triggers rag_search (returns relevant document chunks)
nat run --config_file src/rag_react_agent/configs/config.yml --input "Find all animals mentioned in documents"
```

### Expected Output

When running successfully, you'll see the React Agent's reasoning process:

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

The plugin includes a sample configuration at `src/rag_react_agent/configs/config.yml`:

```yaml
functions:
  # nvidia_rag_query: Queries documents and returns AI-generated response
  rag_query:
    _type: nvidia_rag_query
    collection_names: ["test_library"]    # Milvus collection names
    vdb_endpoint: "./milvus.db"           # Path to Milvus Lite database
    use_knowledge_base: true
    # embedding_endpoint: "localhost:9080"  # Optional: for on-prem embeddings

  # nvidia_rag_search: Searches for relevant document chunks
  rag_search:
    _type: nvidia_rag_search
    collection_names: ["test_library"]
    vdb_endpoint: "./milvus.db"
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

# React Agent workflow - enables the LLM to reason about tool usage
workflow:
  _type: react_agent
  tool_names:
    - rag_query
    - rag_search
    - current_datetime
  llm_name: nim_llm
  verbose: true                           # Shows agent reasoning process
```

### Key Configuration Details

- **`workflow._type: react_agent`**: Uses the React (Reason + Act) agent pattern, which allows the LLM to iteratively reason about which tools to use
- **`verbose: true`**: Displays the agent's thought process, useful for debugging and understanding tool selection
- **`vdb_endpoint`**: Points to the local Milvus Lite database created during ingestion

### Available Functions

| Function | Description |
|----------|-------------|
| `nvidia_rag_query` | Queries documents using NVIDIA RAG and returns an AI-generated response |
| `nvidia_rag_search` | Searches for relevant document chunks without generating a response |

## Troubleshooting

### Error: Function type `nvidia_rag_query` not found

The plugin is not installed. Run:
```bash
# From examples/rag_react_agent/ directory
uv sync
source .venv/bin/activate
```

### Error: Token limit exceeded

If you get a token limit error, reduce the number of results returned:
```yaml
rag_search:
  _type: nvidia_rag_search
  reranker_top_k: 1    # Reduce from 3
  vdb_top_k: 10        # Reduce from 20
```

This often happens when documents contain large base64-encoded images (charts, figures).

### Error: NVIDIA API key not set

```bash
export NVIDIA_API_KEY="your-api-key"
```

## Directory Structure

```
examples/rag_react_agent/
├── README.md                 # This file
├── pyproject.toml           # Package configuration
├── ingest_data.py           # Document ingestion script
├── requirements-ingest.txt  # Dependencies for ingestion
├── milvus.db                # Vector database (created after ingestion)
└── src/
    └── rag_react_agent/
        ├── __init__.py
        ├── configs/
        │   └── config.yml    # Sample config
        └── register.py       # RAG function implementations & plugin registration
```

> **Note**: This plugin uses `nvidia-rag` from the parent repository (`../..`) in editable mode. 
> Changes to the nvidia_rag source code will be immediately available without reinstallation.

## License

Apache-2.0

# NVIDIA NAT RAG Plugin

This plugin integrates [NVIDIA RAG](https://github.com/NVIDIA-AI-Blueprints/rag) with NeMo Agent Toolkit, providing RAG query and search capabilities for your agent workflows.

## Prerequisites

- Python 3.11+
- NeMo Agent Toolkit installed
- Access to NVIDIA AI endpoints (API key required)

## Installation

All commands should be run from the `examples/nvidia_nat_rag/` directory.

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
# From examples/nvidia_nat_rag/ directory
cd examples/nvidia_nat_rag

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
# From examples/nvidia_nat_rag/ directory
# Create virtual environment and install all dependencies (including nvidia-rag from local repo)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Configuration

### Sample Config File

The plugin includes a sample configuration at:
```
src/nat/plugins/rag/configs/config.yml
```

### Available Functions

#### `nvidia_rag_query`
Queries documents using NVIDIA RAG and returns an AI-generated response.

```yaml
functions:
  rag_query:
    _type: nvidia_rag_query
    config_file: config.yaml          # Path to nvidia_rag config
    collection_names: ["test_library"]     # Milvus collection names
    vdb_endpoint: "./milvus.db"
    use_knowledge_base: true
    # embedding_endpoint: "localhost:9080"  # Optional: for on-prem embeddings
```

#### `nvidia_rag_search`
Searches for relevant document chunks without generating a response.

```yaml
functions:
  rag_search:
    _type: nvidia_rag_search
    config_file: config.yaml
    collection_names: ["test_library"]
    vdb_endpoint: "./milvus.db"
    reranker_top_k: 3                 # Number of results after reranking
    vdb_top_k: 20                     # Number of results from vector search
```

## Usage

### Running a RAG Workflow

```bash
# From examples/nvidia_nat_rag/ directory with .venv activated
nat run --config_file src/nat/plugins/rag/configs/config.yml --input "what is giraffee doing?"
```

### Example Workflow Config

```yaml
functions:
  rag_query:
    _type: nvidia_rag_query
    config_file: config.yaml
    collection_names: ["test_library"]
    vdb_endpoint: "./milvus.db"
    use_knowledge_base: true

  rag_search:
    _type: nvidia_rag_search
    config_file: config.yaml
    collection_names: ["test_library"]
    vdb_endpoint: "./milvus.db"
    reranker_top_k: 3
    vdb_top_k: 20

  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

workflow:
  _type: react_agent
  tool_names:
    - rag_query
    - rag_search
    - current_datetime
  llm_name: nim_llm
  verbose: true
```

## Troubleshooting

### Error: Function type `nvidia_rag_query` not found

The plugin is not installed. Run:
```bash
# From examples/nvidia_nat_rag/ directory
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
examples/nvidia_nat_rag/
├── LICENSE.md
├── README.md                 # This file
├── pyproject.toml           # Package configuration
├── ingest_data.py           # Document ingestion script
├── requirements-ingest.txt  # Dependencies for ingestion
├── milvus.db                # Vector database (created after ingestion)
└── src/
    └── nat/
        ├── meta/
        │   └── pypi.md
        └── plugins/
            └── rag/
                ├── __init__.py
                ├── configs/
                │   └── config.yml    # Sample config
                ├── rag_functions.py  # RAG function implementations
                └── register.py       # Plugin registration
```

> **Note**: This plugin uses `nvidia-rag` from the parent repository (`../..`) in editable mode. 
> Changes to the nvidia_rag source code will be immediately available without reinstallation.

## License

Apache-2.0

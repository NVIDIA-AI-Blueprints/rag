<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Use the NVIDIA RAG Blueprint Python Package

Use this documentation to learn about the [NVIDIA RAG Blueprint](readme.md) Python Package. 
For a notebook that walks you through these code examples, see [NVIDIA RAG Python Package](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb).

:::{tip}
Looking for a containerless deployment without Docker? See the [NVIDIA RAG Python Package - Lite Mode](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_lite_usage.ipynb) notebook for a simplified setup using Milvus Lite and NVIDIA cloud APIs.
:::

## Prerequisites

Before running this notebook, ensure you have:
1. **Python 3.11+** installed on your system
2. **[uv](https://docs.astral.sh/uv/)** - A fast Python package manager (installation instructions below)

### **Development Mode Note:**

- Installing with `uv pip install -e ..[all]` allows you to make live edits to the `nvidia_rag` source code and have those changes reflected without reinstalling the package.
- After making changes to the source code, you need to:
  - Restart the kernel of your notebook server
  - Re-execute the cells under `Setting up the dependencies` and `Import the packages` sections

## Environment Setup

### Step 1: Install uv (if not already installed)

Run the cell below to check if `uv` is installed and install it if needed.

```
import subprocess
import shutil
```

# Check if uv is installed
if shutil.which("uv"):
    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    print(f"✅ uv is already installed: {result.stdout.strip()}")
else:
    print("⚠️ uv is not installed. Installing now...")
    # Install uv using the official installer
    !curl -LsSf https://astral.sh/uv/install.sh | sh
    print("\n✅ uv installed! Please restart your terminal/kernel and re-run this notebook.")


### Step 2: Install the NVIDIA RAG Package

Choose one of the installation options below:
- **Option A**: Install from PyPI (recommended for most users)
- **Option B**: Install from source in development mode (for contributors)
- **Option C**: Build and install from source wheel

# Option A: Install from PyPI (recommended)
# Uncomment the line below to install from PyPI
# !uv pip install nvidia-rag[all]

# Option B: Install from source in development mode (for contributors)
# Note: ".." refers to the parent directory where pyproject.toml is located
!uv pip install -e "..[all]"

# Option C: Build and install from source wheel
# Uncomment the lines below to build and install from source
# !cd .. && uv build
# !uv pip install ../dist/nvidia_rag-*-py3-none-any.whl[all]


### Step 3: Verify the installation

The location of the package shown in the output should be inside your Python environment.

Expected location: `<workspace_path>/rag/.venv/lib/python3.12/site-packages`

!uv pip show nvidia_rag | grep Location

## Setting up the dependencies

After the environment for the python package is setup we now launch all the dependent services and NIMs the pipeline depends on.
Fulfill the [prerequisites here](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md) to setup docker on your system.

### 1. Setup the default configurations

!uv pip install python-dotenv
import os
from getpass import getpass

Provide your NGC_API_KEY after executing the cell below. You can obtain a key by following steps [here](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/api-key.md).


# del os.environ['NVIDIA_API_KEY']  ## delete key and reset if needed
if os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    print("Valid NGC_API_KEY already in environment. Delete to reset")
else:
    candidate_api_key = getpass("NVAPI Key (starts with nvapi-): ")
    assert candidate_api_key.startswith("nvapi-"), (
        f"{candidate_api_key[:5]}... is not a valid key"
    )
    os.environ["NGC_API_KEY"] = candidate_api_key


Login to nvcr.io which is needed for pulling the containers of dependencies


!echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

### 2. Setup the Milvus vector DB services
By default milvus uses GPU Indexing. Ensure you have provided correct GPU ID.
Note: If you don't have a GPU available, you can switch to CPU-only Milvus by following the instructions in [milvus-configuration.md](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/milvus-configuration.md).

os.environ["VECTORSTORE_GPU_DEVICE_ID"] = "0"

!docker compose -f ../deploy/compose/vectordb.yaml up -d

### 3. Setup the NIMs

#### Option 1: Deploy on-prem models

Move to Option 2 if you are interested in using cloud models.

Ensure you meet [the hardware requirements](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/support-matrix.md). By default the NIMs are configured to use 2xH100.

# Create the model cache directory
!mkdir -p ~/.cache/model-cache

# Set the MODEL_DIRECTORY environment variable in the Python kernel
import os

os.environ["MODEL_DIRECTORY"] = os.path.expanduser("~/.cache/model-cache")
print("MODEL_DIRECTORY set to:", os.environ["MODEL_DIRECTORY"])

# Set deployment mode for on-prem NIMs
DEPLOYMENT_MODE = "on_prem"

# Configure GPU IDs for the various microservices if needed
os.environ["EMBEDDING_MS_GPU_ID"] = "0"
os.environ["RANKING_MS_GPU_ID"] = "0"
os.environ["YOLOX_MS_GPU_ID"] = "0"
os.environ["YOLOX_GRAPHICS_MS_GPU_ID"] = "0"
os.environ["YOLOX_TABLE_MS_GPU_ID"] = "0"
os.environ["OCR_MS_GPU_ID"] = "0"
os.environ["LLM_MS_GPU_ID"] = "1"

# ⚠️ Deploying NIMs - This may take a while as models download. If kernel times out, just rerun this cell.
!USERID=$(id -u) docker compose -f ../deploy/compose/nims.yaml up -d

# Watch the status of running containers (run this cell repeatedly or in a terminal)
!docker ps

Ensure all the below are running and healthy before proceeding further
```output
NAMES                           STATUS
nemoretriever-ranking-ms        Up ... (healthy)
compose-page-elements-1         Up ...
compose-nemoretriever-ocr-1     Up ...
compose-graphic-elements-1      Up ...
compose-table-structure-1       Up ...
nemoretriever-embedding-ms      Up ... (healthy)
nim-llm-ms                      Up ... (healthy)
```

#### Option 2: Using Nvidia Hosted models

# Set deployment mode for NVIDIA hosted cloud APIs
DEPLOYMENT_MODE = "cloud"

# Configure NV-Ingest to use NVIDIA hosted cloud APIs
os.environ["OCR_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"
os.environ["OCR_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"
)
os.environ["YOLOX_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
)
os.environ["YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
)
os.environ["YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"] = "http"


### 4. Setup the Nvidia Ingest runtime and redis service

!docker compose -f ../deploy/compose/docker-compose-ingestor-server.yaml up nv-ingest-ms-runtime redis -d


---
# API usage example

After setting up the python package and starting all dependent services, finally we can execute some snippets showcasing all different functionalities offered by the `nvidia_rag` package.










## Set logging level

First let's set the required logging level. Set to INFO for displaying basic important logs. Set to DEBUG for full verbosity. 

```python
import logging
LOGLEVEL = logging.WARNING # Set to INFO, DEBUG, WARNING or ERROR
logging.basicConfig(level=LOGLEVEL)

for name in logging.root.manager.loggerDict:
    if name == "nvidia_rag" or name.startswith("nvidia_rag."):
        logging.getLogger(name).setLevel(LOGLEVEL)
    if name == "nv_ingest_client" or name.startswith("nv_ingest_client."):
        logging.getLogger(name).setLevel(LOGLEVEL) 
```

## Import the NvidiaRAGIngestor packages

`NvidiaRAG` exposes APIs to interact with the uploaded documents 
and `NvidiaRAGIngestor` exposes APIs for document upload and management. 
You can import both or either one based on your requirements. 

You can create a config object from a YAML file or a dictionary. A sample configuration file is available at [`notebooks/config.yaml`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/config.yaml).

```python
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor
from nvidia_rag.utils.configuration import NvidiaRAGConfig

# Option 1: Create config from YAML file
config = NvidiaRAGConfig.from_yaml("config.yaml")

# Option 2: Create config from dictionary
# config = NvidiaRAGConfig.from_dict({
#     "llm": {
#         "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
#         "server_url": "",  # Empty uses NVIDIA API catalog
#     },
#     "embeddings": {
#         "model_name": "nvidia/llama-3.2-nv-embedqa-1b-v2",
#         "server_url": "https://integrate.api.nvidia.com/v1",
#     },
#     "ranking": {
#         "model_name": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
#         "server_url": "",  # Empty uses NVIDIA API catalog
#     },
# })

# Initialize with config
rag = NvidiaRAG(config=config)
ingestor = NvidiaRAGIngestor(config=config)
```

:::{tip}
For cloud deployments using NVIDIA hosted APIs, set `server_url` to empty string `""` for LLM and ranking services, and use `https://integrate.api.nvidia.com/v1` for embeddings. For on-prem deployments, use your local NIM endpoints (e.g., `http://localhost:8999` for LLM).
:::

## 1. Create a new collection

Creates a new collection in the vector database. 

```python
response = ingestor.create_collection(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530"
)
print(response)
```


## 2. List all collections

Retrieves all available collections from the vector database. 

```python
response = ingestor.get_collections(vdb_endpoint="http://localhost:19530")
print(response)  
```


## 3. Add a document

Uploads new documents to the specified collection in the vector database. 
In case you have a requirement of updating existing documents in the specified collection, 
you can call `update_documents` instead of `upload_documents`. 

```python
response = await ingestor.upload_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    split_options={"chunk_size": 512, "chunk_overlap": 150},
    filepaths=["../data/multimodal/woods_frost.docx", "../data/multimodal/multimodal_test.pdf"],
    generate_summary=False
)
task_id = response.get("task_id")
print(response)  
```


## 4. Check document upload status

Checks the status of a document upload or update task. 
Before you use this code, replace `task_id` with your actual task ID. 

```python
response = await ingestor.status(task_id="task_id")
print(response)  
```


## [Optional] Update a document in a collection

In you need to update an existing document in the specified collection, use the following code.

```python
response = await ingestor.update_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    filepaths=["../data/multimodal/woods_frost.docx"],
    generate_summary=False
)
print(response)  
```


## 5. Get documents in a collection

Retrieves the list of documents uploaded to a collection. 

```python
response = ingestor.get_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
)
print(response)  
```


## Import the NvidiaRAG packages
You can import `NvidiaRAG()` which exposes APIs to interact with the uploaded documents.

## Query a document using RAG

Sends a chat-style query to the RAG system using the specified models and endpoints. 


### Check health of all dependent services.

```python
health_status_with_deps = await rag.health()
print(health_status_with_deps.message)  
``` 


### Prepare output parser

```python
import json
import base64
from IPython.display import display, Image, Markdown

async def print_streaming_response_and_citations(rag_response):
    """
    Print the streaming response and citations from the RAG response.
    """
    # Check for API errors before processing
    if rag_response.status_code != 200:
        print("Error: ", rag_response.status_code)
        return

    # Extract the streaming generator from the response
    response_generator = rag_response.generator
    first_chunk_data = None
    async for chunk in response_generator:
        if chunk.startswith("data: "):
            chunk = chunk[len("data: "):].strip()
        if not chunk:
            continue
        try:
            data = json.loads(chunk)
        except Exception as e:
            print(f"JSON decode error: {e}")
            continue
        choices = data.get("choices", [])
        if not choices:
            continue
        # Save the first chunk with citations
        if first_chunk_data is None and data.get("citations"):
            first_chunk_data = data
        # Print streaming text
        delta = choices[0].get("delta", {})
        text = delta.get("content")
        if not text:
            message = choices[0].get("message", {})
            text = message.get("content", "")
        print(text, end='', flush=True)
    print()  # Newline after streaming

    # Display citations after streaming is done
    if first_chunk_data and first_chunk_data.get("citations"):
        citations = first_chunk_data["citations"]
        for idx, citation in enumerate(citations.get("results", [])):
            doc_type = citation.get("document_type", "text")
            content = citation.get("content", "")
            doc_name = citation.get("document_name", f"Citation {idx+1}")
            display(Markdown(f"**Citation {idx+1}: {doc_name}**"))
            try:
                image_bytes = base64.b64decode(content)
                display(Image(data=image_bytes))
            except Exception:
                display(Markdown(f"```\n{content}\n```"))  
```


### Call the API

```python
await print_streaming_response_and_citations(
    await rag.generate(
        messages=[{"role": "user", "content": "What is the price of a hammer?"}],
        use_knowledge_base=True,
        collection_names=["test_library"],
    )
)  
```


## Search for documents

Performs a search in the vector database for relevant documents. 

### Define output parser

```python
import base64
from IPython.display import display, Image, Markdown

def print_search_citations(citations):
    """
    Display all citations from the Citations object returned by search().
    Handles base64-encoded images and text.
    """
    if not citations or not hasattr(citations, "results") or not citations.results:
        print("No citations found.")
        return

    for idx, citation in enumerate(citations.results):
        # If using pydantic models, citation fields may be attributes, not dict keys
        doc_type = getattr(citation, "document_type", "text")
        content = getattr(citation, "content", "")
        doc_name = getattr(citation, "document_name", f"Citation {idx + 1}")

        display(Markdown(f"**Citation {idx + 1}: {doc_name}**"))
        try:
            image_bytes = base64.b64decode(content)
            display(Image(data=image_bytes))
        except Exception:
            display(Markdown(f"```\n{content}\n```"))  
```


### Call the API

```python
print_search_citations(
    await rag.search(
        query="What is the price of a hammer?",
        collection_names=["test_library"],
        reranker_top_k=10,
        vdb_top_k=100,
    )
)  
```


## Retrieve documents summary

If you enabled summary generation during document upload by using `generate_summary: bool`, 
use the following code to get the summary.

```python
response = await rag.get_summary(
        collection_name="test_library",
        file_name="woods_frost.docx",
        blocking=False,
        timeout=20
)
print(response)  
```


## Delete documents from a collection

Deletes documents from the specified collection.

```python
response = ingestor.delete_documents(
    collection_name="test_library",
    document_names=["../data/multimodal/multimodal_test.pdf"],
    vdb_endpoint="http://localhost:19530"
)
print(response)  
```

## Delete collections

Deletes the specified collection and all its documents from the vector database. 

```python
response = ingestor.delete_collections(vdb_endpoint="http://localhost:19530", collection_names=["test_library"])
print(response)  
```


## Customize prompts

You can customize prompts by passing them to the `NvidiaRAG` constructor:

```python
from nvidia_rag import NvidiaRAG
from nvidia_rag.utils.configuration import NvidiaRAGConfig

custom_prompts = {
    "rag_template": {
        "system": "/no_think",
        "human": """You are a helpful AI assistant named Envie.
You will reply to questions only based on the context that you are provided.

Context: {context}"""
    }
}

config = NvidiaRAGConfig.from_yaml("config.yaml")
rag = NvidiaRAG(config=config, prompts=custom_prompts)
```

For more details on available prompts and customization options, see [Prompt Customization](prompt-customization.md#prompt-customization-in-python-library-mode).

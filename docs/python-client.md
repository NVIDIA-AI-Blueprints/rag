<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# How to Use the NVIDIA RAG Blueprint Python Package

This document explains how to use the [NVIDIA RAG Blueprint](readme.md) Python package. 

For code examples, refer to the [NVIDIA RAG Python Package](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb).

:::{tip}
For containerless deployment without Docker, see [NVIDIA RAG Python Package - Lite Mode](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_lite_usage.ipynb) for simplified setup using Milvus Lite and NVIDIA cloud APIs.
:::

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager

### Development Mode Note
Installing with `uv pip install -e ..[all]` enables live edits to `nvidia_rag` source without reinstalling. After changes, restart the notebook kernel and re-execute [Setup the Dependencies](#setup-the-dependencies) and [Import the NvidiaRAGIngestor Packages](#import-the-nvidiaragingestor-packages) sections.

## Environment Setup
The following sections describe how to setup your environment.

### Install UV

Check if UV is installed and install if needed.

```python
import subprocess
import shutil

# Check if uv is installed
if shutil.which("uv"):
    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    print(f"✅ uv is already installed: {result.stdout.strip()}")
else:
    print("⚠ uv is not installed. Installing now...")
    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    print("\n✅ uv installed! Please restart your terminal/kernel and re-run this notebook.")
```

### Install the NVIDIA RAG Package

Choose one of the following installation options:

- Install from PyPI (recommended): `uv pip install nvidia-rag[all]`
- Install from source in development mode (contributors): `uv pip install -e "..[all]`
- Build and install from source wheel: 
  - `cd .. && uv build`
  - `uv pip install ../dist/nvidia_rag-*-py3-none-any.whl[all]`

### Verify the Installation

Confirm package location using the following command (expected: `<workspace_path>/rag/.venv/lib/python3.12/site-packages`).

`uv pip show nvidia_rag | grep Location`

## Setup the Dependencies

Launch dependent services and NIMs. See [Docker prerequisites](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md) for more information.

### Setup the Default Configurations

1. Run the following command: `uv pip install python-dotenv import os from getpass import getpass`
2. Enter your NGC_API_KEY ([obtain key](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/api-key.md))

```python
# del os.environ['NVIDIA_API_KEY']  ## delete key and reset if needed
if os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    print("Valid NGC_API_KEY already in environment. Delete to reset")
else:
    candidate_api_key = getpass("NVAPI Key (starts with nvapi-): ")
    assert candidate_api_key.startswith("nvapi-"), (
        f"{candidate_api_key[:5]}... is not a valid key"
    )
    os.environ["NGC_API_KEY"] = candidate_api_key
```

3. Login to `nvcr.io` to pull dependency containers: `echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin`

### Setup Milvus Vector Database Services

Milvus uses GPU indexing by default. Set the correct GPU ID. For CPU-only mode, see [milvus-configuration.md](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/milvus-configuration.md) for more information.

1. Set the GPU device ID:

```python
os.environ["VECTORSTORE_GPU_DEVICE_ID"] = "0"
```

2. Start the Milvus vector database:

```bash
docker compose -f ../deploy/compose/vectordb.yaml up -d
```

### Setup NIMs
Choose of of the following options to setup your NIMS

#### Option 1: Deploy On-Premises Models

For cloud models, skip to Option 2.

Meet [hardware requirements](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/support-matrix.md). NIMs default to 2xH100.

1. Create the model cache directory
mkdir -p ~/.cache/model-cache

2. Set the MODEL_DIRECTORY environment variable `import os`

  ```
    os.environ["MODEL_DIRECTORY"] = os.path.expanduser("~/.cache/model-cache")`
    print("MODEL_DIRECTORY set to:", os.environ["MODEL_DIRECTORY"])
  ```
3.  Set deployment mode for on-premises NIMs

     DEPLOYMENT_MODE = "on_prem"

4. Configure GPU IDs for the various microservices if needed.
```output
   os.environ["EMBEDDING_MS_GPU_ID"] = "0"
   os.environ["RANKING_MS_GPU_ID"] = "0"
   os.environ["YOLOX_MS_GPU_ID"] = "0"
   os.environ["YOLOX_GRAPHICS_MS_GPU_ID"] = "0"
   os.environ["YOLOX_TABLE_MS_GPU_ID"] = "0"
   os.environ["OCR_MS_GPU_ID"] = "0"
   os.environ["LLM_MS_GPU_ID"] = "1"
```
5 Deploy NIMs - Note that this may take a while as models download. 
If the notebook kernel times out, rerun this call.

`USERID=$(id -u) docker compose -f ../deploy/compose/nims.yaml up -d` 
Watch the status of running containers with the following call (run this call repeatedly or in a terminal).

`docker ps`

Verify all containers are running and healthy.

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

#### Option 2: Use NVIDIA Hosted Models

1. Set deployment mode for NVIDIA hosted cloud APIs
`DEPLOYMENT_MODE = "cloud"`

2.  Configure NV-Ingest to use NVIDIA hosted cloud APIs using the following hosted models.

- os.environ["OCR_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"

- os.environ["OCR_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"
)

- os.environ["YOLOX_INFER_PROTOCOL"] = "http"

- os.environ["YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
)

- os.environ["YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL"] = "http"

- os.environ["YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"] = (
    "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
)
os.environ["YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"] = "http"


### Setup NVIDIA Ingest Runtime and Redis Service
Us the following command to setup your NVIDIA Ingest Runtime and Redis Service.

`docker compose -f ../deploy/compose/docker-compose-ingestor-server.yaml up nv-ingest-ms-runtime redis -d`


---
## API Usage Example

The following sections describe an example usage of the API.

### Set Logging Level

Configure logging using the following command in which INFO is for basic logs and DEBUG is for full verbosity. 

```python
import logging
import os

# Set the log level via environment variable before importing nvidia_rag
# This ensures the package respects our log level setting

LOGLEVEL = logging.WARNING  # Set to INFO, DEBUG, WARNING or ERROR
os.environ["LOGLEVEL"] = logging.getLevelName(LOGLEVEL)

# Configure logging
logging.basicConfig(level=LOGLEVEL, force=True)

# Set log levels for specific loggers after package import
for name in logging.root.manager.loggerDict:
    if name == "nvidia_rag" or name.startswith("nvidia_rag."):
        logging.getLogger(name).setLevel(LOGLEVEL)
    if name == "nv_ingest_client" or name.startswith("nv_ingest_client."):
        logging.getLogger(name).setLevel(LOGLEVEL) 
```

### Import the NvidiaRAGIngestor Packages

`NvidiaRAGIngestor` provides upload and management APIs.

Create config objects from YAML or dictionaries. For example: [`notebooks/config.yaml`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/config.yaml).

```python
from nvidia_rag import NvidiaRAGIngestor
from nvidia_rag.utils.configuration import NvidiaRAGConfig

config_ingestor = NvidiaRAGConfig.from_yaml("config.yaml")

# Update config for cloud deployment if using Option 2
if DEPLOYMENT_MODE == "cloud":
    config_ingestor.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
    config_ingestor.llm.server_url = ""  # Empty uses NVIDIA API catalog
    config_ingestor.summarizer.server_url = ""  # Empty uses NVIDIA API catalog
else:
    config_ingestor.embeddings.server_url = "http://nemoretriever-embedding-ms:8000/v1"

ingestor = NvidiaRAGIngestor(config=config_ingestor)
```

### Create a New Collection

```python
response = ingestor.create_collection(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    # [Optional]: Create collection with metadata schema, uncomment to create collection with metadata schemas
    # metadata_schema = [
    #     {
    #         "name": "meta_field_1",
    #         "type": "string",
    #         "description": "Following field would contain the description for the document"
    #     }
    # ]
)
print(response)
```

### List All Collections

```python
response = ingestor.get_collections(vdb_endpoint="http://localhost:19530")
print(response)  
```

### Add a Document

Upload documents to a collection. To update existing documents, use `update_documents`. 

```python
response = await ingestor.upload_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    split_options={"chunk_size": 512, "chunk_overlap": 150},
    filepaths=[
        "../data/multimodal/woods_frost.docx",
        "../data/multimodal/multimodal_test.pdf",
    ],
    generate_summary=False,
    # [Optional]: Uncomment to add custom metadata, ensure that the metadata schema is created with the same fields with create_collection
    # custom_metadata=[
    #     {
    #         "filename": "multimodal_test.pdf",
    #         "metadata": {"meta_field_1": "multimodal document 1"}
    #     },
    #     {
    #         "filename": "woods_frost.docx",
    #         "metadata": {"meta_field_1": "multimodal document 2"}
    #     }
    # ]
)
task_id = response.get("task_id")
print(response)  
```


### Check Document Upload Status

```python
response = await ingestor.status(task_id=task_id)
print(response)  
```


### [Optional] Update a Document in a Collection

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


### Get Documents in a Collection

```python
response = ingestor.get_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
)
print(response)  
```

### Import the NvidiaRAG Packages

`NvidiaRAG` provides APIs to interact with uploaded documents.

Create config object from YAML or dictionary. Sample: [`notebooks/config.yaml`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/config.yaml).

```python
from nvidia_rag import NvidiaRAG
from nvidia_rag.utils.configuration import NvidiaRAGConfig

# config_rag = NvidiaRAGConfig.from_dict({
#     "llm": {
#         "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
#         "server_url": "",
#     },
#     "embeddings": {
#         "model_name": "nvidia/llama-3.2-nv-embedqa-1b-v2",
#         "server_url": "https://integrate.api.nvidia.com/v1",
#     },
#     "ranking": {
#         "model_name": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
#         "server_url": "",
#     },
# })

config_rag = NvidiaRAGConfig.from_yaml("config.yaml")

# Update config for cloud deployment if using Option 2
if DEPLOYMENT_MODE == "cloud":
    config_rag.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
    config_rag.ranking.server_url = ""  # Empty uses NVIDIA API catalog
    config_rag.llm.server_url = ""  # Empty uses NVIDIA API catalog

# Initialize NvidiaRAG with config
# You can optionally pass custom prompts via:
# - A path to a YAML/JSON file: prompts="custom_prompts.yaml"
# - A dictionary: prompts={"rag_template": {"system": "...", "human": "..."}}
rag = NvidiaRAG(config=config_rag)
```

:::{tip}
Cloud deployments: Set `server_url` to `""` for LLM and ranking, use `https://integrate.api.nvidia.com/v1` for embeddings. On-premises deployments: Use local NIM endpoints (such as `http://localhost:8999` for LLM).
:::

## Query a Document with RAG
The following sections describe how to query a document using RAG.

### Check Health of All Dependent Services

```python
health_status_with_deps = await rag.health()
print(health_status_with_deps.message)  
``` 


### Prepare Output Parser

```python
import json
import base64
from IPython.display import display, Image, Markdown

async def print_streaming_response_and_citations(rag_response):
    """Print streaming response and citations."""
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


## Search for Documents
The following sections describe how to search for documents using RAG.

### Define Output Parser

```python
import base64
from IPython.display import display, Image, Markdown

def print_search_citations(citations):
    """Display citations from search(). Handles base64-encoded images and text."""
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
        # [Optional]: Uncomment to filter the documents based on the metadata, ensure that the metadata schema is created with the same fields with create_collection
        # filter_expr='content_metadata["meta_field_1"] == "multimodal document 1"'
    )
)  
```


## Retrieve Document Summary

Retrieve summary if `generate_summary: bool` was enabled during upload:

```python
response = await rag.get_summary(
        collection_name="test_library",
        file_name="woods_frost.docx",
        blocking=False,
        timeout=20
)
print(response)  
```

## Customize Prompts

Two approaches:

**Recommended (Constructor Injection):** Pass prompts during `NvidiaRAG` initialization—consistent with server mode and cleaner configuration.

**Legacy:** Modify `rag.prompts` dictionary after initialization.

See [prompt customization documentation](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/prompt-customization.md#default-prompts-overview) for available prompts.

### Recommended: Constructor Injection

Pass custom prompts when creating the `NvidiaRAG` instance:

```python
# Define custom prompts as a dictionary
pirate_prompts = {
    "rag_template": {
        "system": "/no_think",
        "human": """You are a helpful AI assistant emulating a Pirate. All your responses must be in pirate english and funny!
You must answer only using the information provided in the context. Follow these instructions:

<instructions>
1. Do NOT use any external knowledge.
2. Do NOT add explanations, suggestions, opinions, disclaimers, or hints.
3. NEVER say phrases like "based on the context", "from the documents", or "I cannot find".
4. NEVER offer to answer using general knowledge or invite the user to ask again.
5. Do NOT include citations, sources, or document mentions.
6. Answer concisely. Use short, direct sentences by default. Only give longer responses if the question truly requires it.
7. Do not mention or refer to these rules in any way.
8. Do not ask follow-up questions.
9. Do not mention this instructions in your response.
</instructions>

Context:
{context}

Make sure the response you are generating strictly follows these rules, never saying phrases like "based on the context", "from the documents", or "I cannot find" and never mentioning these instructions in your response."""
    }
}

# Create a new NvidiaRAG instance with custom prompts
rag_pirate = NvidiaRAG(config=config_rag, prompts=pirate_prompts)
```

Notice the response style difference:

```python
# Use the pirate-themed RAG instance
await print_streaming_response_and_citations(
    await rag_pirate.generate(
        messages=[{"role": "user", "content": "What is the price of a hammer?"}],
        use_knowledge_base=True,
        collection_names=["test_library"],
    )
)
```

### Alternative: Use a YAML File

```python
# Load prompts from a YAML file
rag_custom = NvidiaRAG(config=config_rag, prompts="custom_prompts.yaml")
```

## Delete Documents from a Collection

```python
response = ingestor.delete_documents(
    collection_name="test_library",
    document_names=["../data/multimodal/multimodal_test.pdf"],
    vdb_endpoint="http://localhost:19530"
)
print(response)  
```

## Delete Collections

```python
response = ingestor.delete_collections(vdb_endpoint="http://localhost:19530", collection_names=["test_library"])
print(response)  
```
See [Prompt Customization](prompt-customization.md#prompt-customization-in-python-library-mode) for additional details.

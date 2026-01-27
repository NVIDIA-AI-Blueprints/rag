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

## Import the packages

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

## Create a new collection

Creates a new collection in the vector database. 

```python
response = ingestor.create_collection(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530"
)
print(response)
```


## List all collections

Retrieves all available collections from the vector database. 

```python
response = ingestor.get_collections(vdb_endpoint="http://localhost:19530")
print(response)  
```


## Add a document

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
print(response)  
```


## Check document upload status

Checks the status of a document upload or update task. 
Before you use this code, replace `task_id` with your actual task ID. 

```python
response = await ingestor.status(
    task_id="*********************************"
)
print(response)  
```


## Update a document in a collection

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


## Get documents in a collection

Retrieves the list of documents uploaded to a collection. 

```python
response = ingestor.get_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
)
print(response)  
```


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

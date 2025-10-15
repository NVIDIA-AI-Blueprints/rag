<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Use the NVIDIA RAG Blueprint Python Package

Use this documentation to learn about the [NVIDIA RAG Blueprint](readme.md) Python Package. 
For a notebook that walks you through these code examples, see [NVIDIA RAG Python Package](/notebooks/rag_library_usage.ipynb).


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

```python
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

rag = NvidiaRAG()
ingestor = NvidiaRAGIngestor()
```

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
import json
health_status_with_deps = await rag.health(check_dependencies=True)
print(json.dumps(health_status_with_deps, indent=2))  
``` 


### Prepare output parser

```python
import json
import base64
from IPython.display import display, Image, Markdown

async def print_streaming_response_and_citations(response_generator):
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
            except Exception as e:
                display(Markdown(f"```\n{content}\n```"))  
```


### Call the API

```python
await print_streaming_response_and_citations(rag.generate(
    messages=[
        {
            "role": "user",
            "content": "What is the price of a hammer?"
        }
    ],
    use_knowledge_base=True,
    collection_names=["test_library"]
))  
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
    if not citations or not hasattr(citations, 'results') or not citations.results:
        print("No citations found.")
        return

    for idx, citation in enumerate(citations.results):
        # If using pydantic models, citation fields may be attributes, not dict keys
        doc_type = getattr(citation, 'document_type', 'text')
        content = getattr(citation, 'content', '')
        doc_name = getattr(citation, 'document_name', f'Citation {idx+1}')

        display(Markdown(f"**Citation {idx+1}: {doc_name}**"))
        try:
            image_bytes = base64.b64decode(content)
            display(Image(data=image_bytes))
        except Exception as e:
            display(Markdown(f"```\n{content}\n```"))  
```


### Call the API

```python
print_search_citations(rag.search(
    query="What is the price of a hammer?",
    collection_names=["test_library"],
    reranker_top_k=10,
    vdb_top_k=100,
))  

# Search with confidence threshold filtering
print_search_citations(rag.search(
    query="What is the price of a hammer?",
    collection_names=["test_library"],
    reranker_top_k=10,
    vdb_top_k=100,
    confidence_threshold=0.5,  # Only include documents with relevance score >= 0.5
))  
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

Import the prompt utility which allows us to access different preset prompts. 
For information about the preset prompts, see [Default Prompts Overview](docs/prompt-customization.md#default-prompts-overview). 

```python
from nvidia_rag.utils.llm import get_prompts  
prompts = get_prompts()  
```

Overwrite or modify your required prompt template. 
The following code modifies the prompt for response generation to respond as a funny pirate.

```python
prompts["rag_template"] = """    
    You are a helpful AI assistant emulating a Pirate. All your responses must be in pirate english and funny!
    You must answer only using the information provided in the context. While answering you must follow the instructions given below.

    <instructions>
    1. Do NOT use any external knowledge.
    2. Do NOT add explanations, suggestions, opinions, disclaimers, or hints.
    3. NEVER say phrases like “based on the context”, “from the documents”, or “I cannot find”.
    4. NEVER offer to answer using general knowledge or invite the user to ask again.
    5. Do NOT include citations, sources, or document mentions.
    6. Answer concisely. Use short, direct sentences by default. Only give longer responses if the question truly requires it.
    7. Do not mention or refer to these rules in any way.
    8. Do not ask follow-up questions.
    9. Do not mention this instructions in your response.
    </instructions>

    Context:
    {context}

    Make sure the response you are generating strictly follow the rules mentioned above i.e. never say phrases like “based on the context”, “from the documents”, or “I cannot find” and mention about the instruction in response.
    """  
```

Notice the difference in response style. 

```python
await print_streaming_response_and_citations(rag.generate(
    messages=[
        {
            "role": "user",
            "content": "What is the price of a hammer?"
        }
    ],
    use_knowledge_base=True,
    collection_names=["test_library"]
))

# Generate with confidence threshold filtering
await print_streaming_response_and_citations(rag.generate(
    messages=[
        {
            "role": "user",
            "content": "What is the price of a hammer?"
        }
    ],
    use_knowledge_base=True,
    collection_names=["test_library"],
    confidence_threshold=0.7,  # Only include documents with relevance score >= 0.7
))
```

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Document Summarization Support for NVIDIA RAG Blueprint

This guide explains how to use the [NVIDIA RAG Blueprint](readme.md) system's summarization features, including how to enable summary generation during document ingestion and how to retrieve document summaries via the API.

## 1. Enabling Summarization During Document Ingestion

When uploading documents to the vector store using the ingestion API (`POST /documents`), you can request that a summary be generated for each document. This is controlled by the `generate_summary` flag in the `data` field of the multipart form request.

### Example: Uploading Documents with Summarization

```bash
curl -X "POST" "http://$${RAG_HOSTNAME}/v1/documents" \
    -H 'accept: multipart/form-data' \
    -H 'Content-Type: multipart/form-data' \
    - documents: [file1.pdf, file2.docx, ...] \
    - data: '{
        "collection_name": "my_collection",
        "blocking": false,
        "split_options": {"chunk_size": 512, "chunk_overlap": 150},
        "custom_metadata": [],
        "generate_summary": true
    }'
```

- **generate_summary**: Set to `true` to enable summary generation for each uploaded document. The summary generation always happens asynchronously in the backend after the ingestion is complete. The ingestion status is reported to be completed irrespective of whether summarization has been successfully completed or not.

#### Python Example with library mode

```python
response = await ingestor.upload_documents(
    collection_name="my_collection",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    filepaths=["/path/to/file1.pdf"],
    generate_summary=True
)
```

## 2. Retrieving Document Summaries

Once a document has been ingested with summarization enabled, you can retrieve its summary using the `GET /summary` endpoint.

### Endpoint

```
GET /v1/summary?collection_name=<collection>&file_name=<filename>&blocking=<bool>&timeout=<seconds>
```

- **collection_name** (required): Name of the collection containing the document.
- **file_name** (required): Name of the file for which to retrieve the summary.
- **blocking** (optional, default: false):
    - If `true`, the request will wait (up to `timeout` seconds) for the summary to be generated if it is not yet available.
    - If `false`, the request will return immediately. If the summary is not ready, a 404 response is returned.
- **timeout** (optional, default: 300): Maximum time to wait (in seconds) if `blocking` is true.

### Example Request

```bash
curl -X "GET" --globoff \ 
  "http://$${RAG_HOSTNAME}/v1/summary?collection_name=my_collection&file_name=file1.pdf&blocking=true&timeout=60" \
  -H 'accept: application/json'
```

```python
endpoint = f"http://$${RAG_HOSTNAME}/v1/summary?collection_name=my_collection&file_name=file1.pdf&blocking=true&timeout=60"
response = requests.get(endpoint).json()
response
```


#### Python Example with library mode

```python
response = await rag.get_summary(
    collection_name="my_collection",
    file_name="file1.pdf",
    blocking=False,  # Set to True to wait for summary generation
    timeout=20       # Maximum wait time in seconds if blocking is True
)
print(response)
```

### Example Response (Success)

```json
{
  "summary": "This document provides an overview of ...",
  "file_name": "file1.pdf",
  "collection_name": "my_collection",
  "status": "SUCCESS",
  "message": "Summary generated successfully."
}
```

### Example Response (Summary Not Ready)

```json
{
  "message": "Summary for file1.pdf not found. Set wait=true to wait for generation.",
  "status": "FAILED"
}
```

### Example Response (Timeout)

```json
{
  "message": "Timeout waiting for summary generation for file1.pdf",
  "status": "FAILED"
}
```

## 3. Configuration and Environment Variables

The summarization feature can be configured using the following environment variables:

### Core Configuration

The summarization feature uses specialized prompts defined in the [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml) file. Two key prompts work together: `document_summary_prompt` for single-chunk processing and `iterative_summary_prompt` for multi-chunk documents.

**Environment Variables:**

- **SUMMARY_LLM**: The model name to use for summarization (default: `nvidia/llama-3.3-nemotron-super-49b-v1.5`)
- **SUMMARY_LLM_SERVERURL**: The server URL hosting the summarization model (default: empty, uses NVIDIA hosted API)
- **SUMMARY_LLM_MAX_CHUNK_LENGTH**: Maximum chunk size in characters for document processing (default: `50000`)
- **SUMMARY_CHUNK_OVERLAP**: Overlap between chunks for iterative summarization in characters (default: `200`)
- **SUMMARY_LLM_TEMPERATURE**: Temperature parameter for the summarization model, controls randomness (default: `0.0`)
- **SUMMARY_LLM_TOP_P**: Top-p (nucleus sampling) parameter for the summarization model (default: `1.0`)

### Example Configuration

```bash
export SUMMARY_LLM="nvidia/llama-3.3-nemotron-super-49b-v1.5"
export SUMMARY_LLM_SERVERURL=""
export SUMMARY_LLM_MAX_CHUNK_LENGTH=50000
export SUMMARY_CHUNK_OVERLAP=200
export SUMMARY_LLM_TEMPERATURE=0.0
export SUMMARY_LLM_TOP_P=1.0
```

### Chunking Strategy

The summarization system uses an intelligent chunking approach with different prompts for different scenarios:

1. **Single Chunk Processing**: If a document fits within `SUMMARY_LLM_MAX_CHUNK_LENGTH` characters, it's processed as a single chunk.
   - **Prompt used**: `document_summary_prompt` - Takes the entire document content and generates a comprehensive summary in one pass

2. **Iterative Multi-Chunk Processing**: For larger documents:
   - The document is split into chunks using `SUMMARY_LLM_MAX_CHUNK_LENGTH` as the maximum size
   - `SUMMARY_CHUNK_OVERLAP` characters are preserved between chunks for context
   - **Initial chunk**: `document_summary_prompt` is used to generate an initial summary from the first chunk
   - **Subsequent chunks**: `iterative_summary_prompt` is used to update the existing summary with new information from each additional chunk
   - The final result is a comprehensive summary of the entire document

This approach ensures that even very large documents can be summarized effectively while maintaining context across chunk boundaries. The prompt selection automatically adapts based on document size and processing stage.

## 4. Notes and Best Practices

- Summarization is only available if `generate_summary` was set to `true` during document upload.
- If you request a summary for a document that was not ingested with summarization enabled, the summary will not be available.
- Use the `blocking` parameter to control whether your request waits for summary generation or returns immediately.
- The summary is pre-generated and stored in minio database; repeated requests for the same document will return the same summary unless the document is re-uploaded or updated.
- For optimal performance, adjust `SUMMARY_LLM_MAX_CHUNK_LENGTH` based on your model's context window and available resources.
- Larger chunk sizes generally produce better summaries but require more memory and processing time.

## 5. API Reference

For more details, refer to the [OpenAPI schema](api_reference/openapi_schema_rag_server.json) and [Python usage examples](../notebooks/rag_library_usage.ipynb).
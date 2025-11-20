<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Document Summarization Support for NVIDIA RAG Blueprint

This guide explains how to use the [NVIDIA RAG Blueprint](readme.md) system's summarization features, including how to enable summary generation during document ingestion and how to retrieve document summaries via the API.

## Architecture Overview

The diagram below illustrates the complete RAG pipeline with the summarization workflow:

![Summarization Pipeline Architecture](assets/summarization_flow_diagram.png)

## Overview

The NVIDIA RAG Blueprint features **intelligent document summarization** with real-time progress tracking, enabling you to:

- **Generate summaries with multiple strategies** – Choose between single-pass (fastest), hierarchical (balanced), or iterative (best quality)
- **Fast shallow extraction** – 10x faster text-only summaries skipping OCR/tables/images
- **Flexible page filtering** – Summarize specific pages using ranges, negative indexing, or even/odd patterns
- **Real-time status tracking** – Monitor summary generation progress with chunk-level updates
- **Global rate limiting** – Prevent GPU/API overload with Redis-based semaphores
- **Token-based chunking** – Aligned with nv-ingest using the same tokenizer for consistency

## 1. Enabling Summarization During Document Ingestion

When uploading documents to the vector store using the ingestion API (`POST /documents`), you can request that a summary be generated for each document. This is controlled by the `generate_summary` flag and optional `summary_options` in the `data` field of the multipart form request.

### Example: Basic Summarization

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

- **generate_summary**: Set to `true` to enable summary generation for each uploaded document. The summary generation always happens asynchronously in the backend after the ingestion is complete. The ingestion status is reported to be completed irrespective of whether summarization has been successfully completed or not. You can track the summary generation status independently using the `GET /summary` endpoint.

### Example: Advanced Summarization Options

You can customize summarization behavior using the `summary_options` parameter:

```bash
curl -X "POST" "http://$${RAG_HOSTNAME}/v1/documents" \
    -H 'accept: multipart/form-data' \
    -H 'Content-Type: multipart/form-data' \
    - documents: [file1.pdf] \
    - data: '{
        "collection_name": "my_collection",
        "blocking": false,
        "generate_summary": true,
        "summary_options": {
            "page_filter": [[1, 10], [-5, -1]],
            "shallow_summary": true,
            "summarization_strategy": "single"
        }
    }'
```

#### Summary Options Explained

- **page_filter** (optional): Select specific pages to summarize
  - **Ranges**: `[[1, 10], [20, 30]]` - Pages 1-10 and 20-30
  - **Negative indexing**: `[[-5, -1]]` - Last 5 pages (Pythonic indexing where -1 is last page)
  - **String filters**: `"even"` or `"odd"` - All even or odd pages
  - **Examples**:
    - `[[1, 10]]` - First 10 pages
    - `[[-10, -1]]` - Last 10 pages
    - `[[1, 10], [-5, -1]]` - First 10 and last 5 pages
    - `"even"` - All even-numbered pages

- **shallow_summary** (optional, default: `false`): Enable fast text-only extraction
  - **`true`**: Text-only extraction, skips OCR/table detection/image processing (~10x faster)
  - **`false`**: Full multimodal extraction with OCR, tables, charts, and images
  - Use `true` for quick summaries of text-heavy documents
  - Use `false` for comprehensive summaries of documents with complex layouts

- **summarization_strategy** (optional, default: `null`): Choose summarization approach
  - **`"single"`**: ⚡ Fastest - Merge content, chunk by configured size, and summarize only the first chunk
    - Best for: Quick overviews, short documents
    - Speed: Fastest (one LLM call)
  - **`"hierarchical"`**: 🔀 Balanced - Tree-based summarization: summarize all chunks, merge summaries until they fit chunk size, repeat recursively until reaching one final summary
    - Best for: Balance between speed and quality
    - Speed: Fast (parallel processing with tree reduction)
  - **`null` (or omit)**: 📚 Best Quality - Sequential refinement chunk by chunk (iterative)
    - Best for: Long documents requiring best quality
    - Speed: Slower but highest quality

#### Python Example with library mode

```python
# Basic summarization
response = await ingestor.upload_documents(
    collection_name="my_collection",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    filepaths=["/path/to/file1.pdf"],
    generate_summary=True
)

# Advanced summarization with options
response = await ingestor.upload_documents(
    collection_name="my_collection",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    filepaths=["/path/to/file1.pdf"],
    generate_summary=True,
    summary_options={
        "page_filter": [[1, 10], [-5, -1]],  # First 10 and last 5 pages
        "shallow_summary": True,  # Fast text-only extraction
        "summarization_strategy": "hierarchical"  # Balanced approach
    }
)
```

## 2. Retrieving Document Summaries

Once a document has been ingested with summarization enabled, you can retrieve its summary using the `GET /summary` endpoint. The endpoint provides real-time status tracking with chunk-level progress information.

### Endpoint

```
GET /v1/summary?collection_name=<collection>&file_name=<filename>&blocking=<bool>&timeout=<seconds>
```

- **collection_name** (required): Name of the collection containing the document.
- **file_name** (required): Name of the file for which to retrieve the summary.
- **blocking** (optional, default: false):
    - If `true`, the request will wait (up to `timeout` seconds) for the summary to be generated if it is not yet available. During this time, the endpoint polls Redis for status updates.
    - If `false`, the request will return immediately with the current status. If the summary is not ready, you'll receive the current generation status (PENDING, IN_PROGRESS, or NOT_FOUND).
- **timeout** (optional, default: 300): Maximum time to wait (in seconds) if `blocking` is true.

### Status Values

The endpoint returns one of the following status values:

- **SUCCESS**: Summary generation completed successfully. The `summary` field contains the generated text.
- **PENDING**: Summary generation is queued but not yet started.
- **IN_PROGRESS**: Summary is currently being generated. The response includes `progress` information showing current chunk processing (e.g., "Processing chunk 3/5").
- **FAILED**: Summary generation failed. The `error` field contains the failure reason.
- **NOT_FOUND**: No summary was requested for this document (it was not uploaded with `generate_summary=true`).

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
  "message": "Summary retrieved successfully.",
  "summary": "This document provides an overview of ...",
  "file_name": "file1.pdf",
  "collection_name": "my_collection",
  "status": "SUCCESS"
}
```

### Example Response (In Progress)

When summary generation is in progress, you'll receive real-time progress information:

```json
{
  "message": "Summary generation is in progress. Set blocking=true to wait for completion.",
  "status": "IN_PROGRESS",
  "file_name": "file1.pdf",
  "collection_name": "my_collection",
  "started_at": "2025-01-24T10:30:00.000Z",
  "updated_at": "2025-01-24T10:30:15.000Z",
  "progress": {
    "current": 3,
    "total": 5,
    "message": "Processing chunk 3/5"
  }
}
```

**HTTP Status Code**: 202 (Accepted)

### Example Response (Pending)

```json
{
  "message": "Summary generation is pending. Set blocking=true to wait for completion.",
  "status": "PENDING",
  "file_name": "file1.pdf",
  "collection_name": "my_collection",
  "queued_at": "2025-01-24T10:30:00.000Z"
}
```

**HTTP Status Code**: 202 (Accepted)

**Note**: For PENDING status, only `queued_at` is set. Fields like `started_at`, `updated_at`, and `progress` will be `null` since processing hasn't started yet.

### Example Response (Not Found)

```json
{
  "message": "Summary for file1.pdf not found. To generate a summary, upload the document with generate_summary=true.",
  "status": "NOT_FOUND",
  "file_name": "file1.pdf",
  "collection_name": "my_collection"
}
```

**HTTP Status Code**: 404 (Not Found)

### Example Response (Failed)

```json
{
  "message": "Summary generation failed for file1.pdf",
  "status": "FAILED",
  "error": "Error details here",
  "file_name": "file1.pdf",
  "collection_name": "my_collection",
  "started_at": "2025-01-24T10:30:00.000Z",
  "completed_at": "2025-01-24T10:30:30.000Z"
}
```

**HTTP Status Code**: 500 (Internal Server Error)

### Example Response (Timeout)

When using blocking mode, if the summary is not generated within the specified timeout:

```json
{
  "message": "Timeout waiting for summary generation for file1.pdf after 300 seconds",
  "status": "FAILED",
  "error": "Timeout after 300 seconds",
  "file_name": "file1.pdf",
  "collection_name": "my_collection"
}
```

**HTTP Status Code**: 408 (Request Timeout)

## 3. Configuration and Environment Variables

The summarization feature can be configured using the following environment variables:

### Redis Configuration (Status Tracking)

Summary generation status is tracked using Redis to enable cross-service visibility (between ingestor and RAG servers). Configure the following environment variables for both services:

- **REDIS_HOST**: Redis server hostname (default: `localhost`)
- **REDIS_PORT**: Redis server port (default: `6379`)
- **REDIS_DB**: Redis database number (default: `0`)

**Status Tracking Behavior:**
- Status information is stored in Redis with a 24-hour TTL (automatically cleaned up)
- If Redis is unavailable, the system gracefully degrades: summaries will still be generated and stored in MinIO, but real-time status tracking will not be available
- Status values include: `PENDING`, `IN_PROGRESS` (with chunk progress), `SUCCESS`, `FAILED`, and `NOT_FOUND`
- Redis semaphore counter is automatically reset when ingestor server starts, preventing stale values from crashed processes

### Core Configuration

The summarization feature uses specialized prompts defined in the [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml) file. The system automatically selects the appropriate prompts based on extraction mode and document size:

- **`shallow_summary_prompt`**: Used for text-only (shallow) extraction workflows when `shallow_summary: true`
- **`document_summary_prompt`**: Used for full multimodal extraction (default) for single-chunk processing
- **`iterative_summary_prompt`**: Used for multi-chunk documents in iterative strategy

For more details on customizing these prompts, see [Prompt Customization Guide](./prompt-customization.md).

**Environment Variables:**

- **SUMMARY_LLM**: The model name to use for summarization (default: `nvidia/llama-3.3-nemotron-super-49b-v1.5`)
- **SUMMARY_LLM_SERVERURL**: The server URL hosting the summarization model (default: empty, uses NVIDIA hosted API)
- **SUMMARY_LLM_MAX_CHUNK_LENGTH**: Maximum chunk size in **tokens** for document processing (default: `9000`)
- **SUMMARY_CHUNK_OVERLAP**: Overlap between chunks for summarization in **tokens** (default: `400`)
- **SUMMARY_LLM_TEMPERATURE**: Temperature parameter for the summarization model, controls randomness (default: `0.0`)
- **SUMMARY_LLM_TOP_P**: Top-p (nucleus sampling) parameter for the summarization model (default: `1.0`)
- **SUMMARY_MAX_PARALLELIZATION**: Global rate limit for concurrent summary tasks across all workers (default: `20`)

### Example Configuration

```bash
export SUMMARY_LLM="nvidia/llama-3.3-nemotron-super-49b-v1.5"
export SUMMARY_LLM_SERVERURL=""
export SUMMARY_LLM_MAX_CHUNK_LENGTH=9000
export SUMMARY_CHUNK_OVERLAP=400
export SUMMARY_LLM_TEMPERATURE=0.0
export SUMMARY_LLM_TOP_P=1.0
export SUMMARY_MAX_PARALLELIZATION=20
```

### Token-Based Chunking

The summarization system uses **token-based chunking** aligned with nv-ingest for consistency:

- **Tokenizer**: Uses `e5-large-unsupervised` (same as nv-ingest)
- **Max Chunk Length**: 9000 tokens
- **Chunk Overlap**: 400 tokens
- **Benefits**: More accurate chunking, better context preservation, aligned with ingestion pipeline

### Global Rate Limiting

To prevent GPU/API overload, the system uses **Redis-based global rate limiting**:

- **Default Limit**: Maximum 20 concurrent summary tasks across all workers
- **Redis Semaphore**: Coordinates access across multiple ingestor server instances
- **Connection Pooling**: Up to 50 Redis connections for efficient coordination
- **Behavior**: Tasks wait in queue until a slot becomes available
- **Reset on Startup**: Counter automatically reset when ingestor server starts

### Chunking Strategy

The summarization system uses an intelligent chunking approach with different prompts for different scenarios:

**Prompt Selection Based on Extraction Mode:**
- **Shallow extraction** (`shallow_summary: true`): Uses `shallow_summary_prompt` - optimized for fast text-only processing
- **Full extraction** (`shallow_summary: false`, default): Uses `document_summary_prompt` and `iterative_summary_prompt` - comprehensive multimodal processing

**Processing by Document Size:**

1. **Single Chunk Processing**: If a document fits within `SUMMARY_LLM_MAX_CHUNK_LENGTH` tokens, it's processed as a single chunk.
   - **Prompt used**: `shallow_summary_prompt` (for shallow) or `document_summary_prompt` (for full extraction)
   - **Strategy**: All strategies use the appropriate prompt for single-chunk documents

2. **Multi-Chunk Processing**: For larger documents:
   - The document is split into chunks using `SUMMARY_LLM_MAX_CHUNK_LENGTH` as the maximum size (in tokens)
   - `SUMMARY_CHUNK_OVERLAP` tokens are preserved between chunks for context
   - **Strategy behavior**:
     - **`single`**: Concatenates all chunks into one, truncates if exceeds max length (fastest)
     - **`hierarchical`**: Processes chunks in parallel using extraction-appropriate prompt, then combines summaries (balanced)
     - **`null/iterative`**: Initial chunk uses extraction-appropriate prompt, subsequent chunks use `iterative_summary_prompt` to update existing summary (best quality)

This approach ensures that even very large documents can be summarized effectively while maintaining context across chunk boundaries. The prompt selection automatically adapts based on extraction mode, document size, processing stage, and selected strategy.

## 4. Notes and Best Practices

- Summarization is only available if `generate_summary` was set to `true` during document upload.
- If you request a summary for a document that was not ingested with summarization enabled, you'll receive a `NOT_FOUND` status.
- Use the `blocking` parameter to control whether your request waits for summary generation or returns immediately with the current status.
- The summary is pre-generated and stored in MinIO; repeated requests for the same document will return the same summary unless the document is re-uploaded or updated.
- **Status Tracking**: Monitor summary generation progress in real-time using the `GET /summary` endpoint. The `IN_PROGRESS` status includes chunk-level progress (e.g., "Processing chunk 3/5").
- **Redis Requirement**: For status tracking and global rate limiting to work across services, ensure Redis is configured and accessible to both ingestor and RAG servers. Without Redis, summaries will still be generated but status tracking and rate limiting will be unavailable.
- **Timeout Handling**: When using `blocking=true`, set an appropriate timeout based on your document size. Large documents may take several minutes to summarize.

### Performance Recommendations

- **For fastest summaries**: Use `shallow_summary=true` + `summarization_strategy="single"` + `page_filter` to select key pages
- **For best quality**: Use `shallow_summary=false` + `summarization_strategy=null` (iterative) + process all pages
- **For balanced approach**: Use `shallow_summary=true` + `summarization_strategy="hierarchical"` + selective pages

### Token-Based Chunking Best Practices

- Adjust `SUMMARY_LLM_MAX_CHUNK_LENGTH` based on your model's context window and available resources
- Larger chunk sizes generally produce better summaries but require more memory and processing time
- The default 9000 tokens works well for most models and document types
- Keep `SUMMARY_CHUNK_OVERLAP` at 400-500 tokens to maintain context continuity between chunks

### Rate Limiting Considerations

- Increase `SUMMARY_MAX_PARALLELIZATION` if you have more GPU resources available
- Decrease it if experiencing GPU memory issues or API rate limits
- Monitor Redis semaphore usage to optimize for your workload

## 5. API Reference

For more details, refer to the [OpenAPI schema](api_reference/openapi_schema_rag_server.json) and [Python usage examples](../notebooks/rag_library_usage.ipynb).
<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# API - Ingestor Server Schema


This documentation contains the OpenAPI reference for the ingestor server.

:::{tip}
To view this documentation on docs.nvidia.com, browse to [https://docs.nvidia.com/rag/latest/api-ingestor](https://docs.nvidia.com/rag/latest/api-ingestor).
:::


:::{swagger-plugin} ../docs/api_reference/openapi_schema_ingestor_server.json
:::


## Task Status Tracking

The `/status` endpoint returns detailed information about ingestion task progress. Use the `task_id` returned from `POST /documents` to monitor task completion.

### Task States

| State | Description |
|-------|-------------|
| `PENDING` | Task is queued and processing has not started |
| `FINISHED` | Task completed successfully |
| `FAILED` | Task failed due to an error |
| `UNKNOWN` | Task not found or state cannot be determined |

### Progress Tracking

The status response includes progress metrics updated after each batch completes:

- **`total_documents`**: Total number of documents in the ingestion job
- **`documents_completed`**: Number of documents that have completed processing
- **`batches_completed`**: Number of processing batches completed

:::{note}
For more granular progress updates during batch processing, use the `nv_ingest_status` object described below, which tracks individual document extraction progress and updates more frequently than the batch-level metrics.
:::

### NV-Ingest Extraction Status

The `/status` endpoint response includes an `nv_ingest_status` object that provides real-time document extraction progress, updating more frequently than batch-level metrics. This is useful for monitoring individual document processing when polling the status endpoint:

- **`extraction_completed`**: Count of documents with completed extraction
- **`document_wise_status`**: Dictionary mapping each filename to its current extraction status

#### Document Extraction States

| Status | Description |
|--------|-------------|
| `not_started` | Document queued, extraction not yet initiated |
| `submitted` | Document submitted to NV-Ingest for processing |
| `processing` | Document extraction is in progress |
| `completed` | Document extraction completed successfully |
| `failed` | Document extraction failed |

### Example Response

:::{note}
The example below shows key fields relevant to progress tracking. For the complete and current response schema, refer to the OpenAPI specification at the top of this page.
:::

```json
{
  "state": "FINISHED",
  "result": {
    "message": "Document upload job successfully completed.",
    "total_documents": 3,
    "documents_completed": 3,
    "batches_completed": 2,
    "documents": [
      {
        "document_name": "document1.pdf",
        "metadata": {},
        "document_info": {}
      },
      {
        "document_name": "document2.pdf",
        "metadata": {},
        "document_info": {}
      },
      {
        "document_name": "document3.pdf",
        "metadata": {},
        "document_info": {}
      }
    ],
    "failed_documents": [],
    "validation_errors": []
  },
  "nv_ingest_status": {
    "extraction_completed": 3,
    "document_wise_status": {
      "document1.pdf": "completed",
      "document2.pdf": "completed",
      "document3.pdf": "completed"
    }
  }
}
```

## Related Topics

- [API - RAG Server Schema](api-rag.md)
- [NVIDIA RAG Blueprint Documentation](index.md)

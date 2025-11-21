<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Milvus Collection Schema Requirements for NVIDIA RAG Blueprint

When you create a collection in Milvus to use with the [NVIDIA RAG Blueprint](readme.md) server, there are specific schema requirements that must be followed to ensure compatibility with the search and generate APIs. This document outlines the required fields and their configurations.

> **Note**: If you are using either LangChain's Milvus integration or NVIDIA's nv-ingest tool for data ingestion, these schema requirements are automatically handled for you. Both tools will create and configure the collection with the correct schema fields. You only need to ensure these requirements when manually creating collections or using custom ingestion methods.



## Required Schema Fields

The following fields are required in your Milvus collection schema:

1. **Vector Field**
   - Name: `vector`
   - Description: Stores the document embeddings

3. **Text Field**
   - Name: `text`
   - Description: Stores the document content

4. **Source Field**
   - Name: `source`
   - Can be configured in two ways:
     1. Simple string format: Directly store the filename
     2. JSON format: Store a JSON object with a `source_id` field containing the filename
     ```json
     {
       "source_id": "document.pdf"
     }
     ```

5. **Content Metadata Field** (Optional)
   - Name: `content_metadata`
   - Type: `JSON` (DataType.JSON)
   - Description: Stores additional metadata about the document content
   - Can be used for filtering during search and retrieval

## Example Schema Definition

Here's an example of a complete schema definition that meets all requirements:

```python
{
    'auto_id': True,
    'description': '',
    'fields': [
        {
            'name': 'pk',
            'description': '',
            'type': DataType.INT64,
            'is_primary': True,
            'auto_id': True
        },
        {
            'name': 'vector',
            'description': '',
            'type': DataType.FLOAT_VECTOR,
            'params': {'dim': 2048}
        },
        {
            'name': 'source',
            'description': '',
            'type': DataType.JSON
        },
        {
            'name': 'content_metadata',
            'description': '',
            'type': DataType.JSON
        },
        {
            'name': 'text',
            'description': '',
            'type': DataType.VARCHAR,
            'params': {'max_length': 65535}
        }
    ],
    'enable_dynamic_field': True
}
```

## Usage with RAG Server

When using this schema with the RAG server:

1. The search API will use the `vector` field for similarity search
2. The `text` field will be used to return the actual content
3. The `source` field will be used to track document sources
4. The `content_metadata` field can be used for filtering using the `filter_expr` parameter in search and generate APIs

For more information about using metadata for filtering, refer to the [Custom Metadata Documentation](./custom-metadata.md). 
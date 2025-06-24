<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable Filtering Search Results by Uploading Custom Metadata

## Overview

You can upload custom metadata for documents during ingestion and retrieval. 
By uploading custom metadata you can attach additional information to documents, 
and use it for filtering results during retrieval operations. 
Custom metadata supports the following:

- Document categorization
- Temporal filtering
- Custom document properties
- Filtering search results

For basic usage examples and implementation details, refer to the following:

- [Ingestion API Usage Notebook](../notebooks/ingestion_api_usage.ipynb) - Demonstrates how to add custom metadata during document ingestion.
- [Retriever API Usage Notebook](../notebooks/retriever_api_usage.ipynb) - Demonstrates how to use metadata filtering during document retrieval.

## Limitations

The following are limitation when you use custom metadata:

- Metadata fields must be consistent across documents in the same collection.
- Currently supports string and datetime data types for metadata fields.
- Complex filter expressions may impact retrieval performance.
- Timestamp filtering requires strict ISO 8601 format compliance.
- Metadata updates require re-ingestion of documents.

## Define Metadata Schema for Collection

When creating a collection using the `/v1/collection` endpoint, you can define a metadata schema to enforce data validation during document ingestion. The schema helps ensure that metadata fields are properly typed and consistent across all documents in the collection.

### Schema Definition

The metadata schema is defined as a list of field definitions when creating a collection. Each field definition must specify:
- `name`: The name of the metadata field
- `type`: The data type of the field (currently supported: "string" or "datetime")
- `description`: A description of the field's purpose

Example schema definition:
```json
[
    {
        "name": "timestamp",
        "type": "datetime",
        "description": "Timestamp of when the document was created"
    },
    {
        "name": "category",
        "type": "string",
        "description": "Document category for classification"
    }
]
```

### Usage in Collection Creation

When creating a collection using the `/v1/collection` endpoint, include the metadata schema in the request:

```python
data = {
    "collection_name": "my_collection",
    "embedding_dimension": 2048,
    "metadata_schema": [
        {
            "name": "timestamp",
            "type": "datetime",
            "description": "Timestamp of when the document was created"
        },
        {
            "name": "category",
            "type": "string",
            "description": "Document category for classification"
        }
    ]
}
```

### Schema Validation

The system validates metadata during ingestion:
- Datetime fields use ISO 8601 format
- String fields accept any text
- Invalid metadata causes document rejection

## Add Custom Metadata During Ingestion

You can add custom metadata during the document ingestion process by using the `/v1/documents` endpoint. 
You cand specify metadata for each file, 
and you can specify different metadata for different documents in the same ingestion batch.


### Metadata Structure

You specify custom metadata as a list of objects, where each object contains the following:

- `filename`: The name of the document.
- `metadata`: A dictionary that contains key-value pairs of metadata.

The following example contains metadata fields `timestamp`, `category`, and `department`. 
You can create whatever metadata is helpful for your scenario.
```json
[
    {
        "filename": "document1.pdf",
        "metadata": {
            "timestamp": "2024-03-15T10:23:00",
            "category": "technical",
            "department": "engineering"
        }
    },
    {
        "filename": "document2.pdf",
        "metadata": {
            "timestamp": "2024-03-16T14:30:00",
            "category": "marketing",
            "department": "sales"
        }
    }
]
```

### Example: Add Custom Metadata During Ingestion

The following example adds custom metadata during ingestion.

```python
CUSTOM_METADATA = [
    {
        "filename": "technical_doc.pdf",
        "metadata": {
            "timestamp": "2024-03-15T10:23:00",  # ISO 8601 format as string
            "category": "technical",              # string
            "department": "engineering",          # string
            "priority": "1",                      # numeric as string
            "is_active": "true"                   # boolean as string
        }
    },
    {
        "filename": "marketing_doc.pdf",
        "metadata": {
            "timestamp": "2024-03-16T14:30:00",
            "category": "marketing",
            "department": "sales",
            "priority": "2",
            "is_active": "false"
        }
    }
]

# Include in upload request
data = {
    "collection_name": "my_collection",
    "blocking": False,
    "split_options": {
        "chunk_size": 512,
        "chunk_overlap": 150
    },
    "custom_metadata": CUSTOM_METADATA
}
```

## Considerations for Custom Metadata

Consider the following before you create your custom metadata.

- **Metadata types** — You can specify strings, numeric values, boolean values, and timestamps, but you must specify all values as strings. For example, specify `"priority": "1"` and `"is_active": "true"`.
- **Timestamp format** — Specify timestamps in ISO 8601 format. For example, `"timestamp": "2024-03-15T10:23:00"`.


## Use Custom Metadata to Filter Results During Retrieval

You can use custom metadata to filter documents during retrieval operations 
by using the `filter_expr` parameter in both the `/v1/search` and `/v1/generate` endpoints.


### Filter Expression Syntax

Use filter expressions that follow the Milvus boolean expression syntax. 
For more information, refer to [Filtering Explained](https://milvus.io/docs/boolean.md).

Use the following information to write filter expressions:

- Access metadata fields by using `content_metadata["field_name"]`.
- You can use the following operators:
  - Comparison: ==, !=, >, >=, <, <=
  - Logical: AND, OR, NOT
  - Range: LIKE, IN
- Since all metadata values are strings, comparisons are done with string values. For example, `content_metadata["priority"] == "1"`.

### Example Filter Expressions

The following example filters results by category.

```python
filter_expr = 'content_metadata["category"] == "technical"'
```

The following example filters results by time range.

```python
filter_expr = 'content_metadata["timestamp"] >= "2024-03-01T00:00:00" and content_metadata["timestamp"] <= "2024-03-31T23:59:59"'
```

The following example filters by category and uses multiple logical operators.

```python
filter_expr = '(content_metadata["department"] == "engineering" and content_metadata["priority"] == "high") or content_metadata["category"] == "critical"'
```

### Example: Use a Filter Expression in Search

The following example uses a filter expression to narrow results.

```python
payload = {
    "query": "What are the technical specifications?",
    "reranker_top_k": 10,
    "vdb_top_k": 100,
    "collection_names": ["my_collection"],
    "enable_query_rewriting": True,
    "enable_reranker": True,
    "filter_expr": 'content_metadata["category"] == "technical" and content_metadata["priority"] == "high"'
}
```

### Example: Using Filter Expressions in Generate

The following example uses a filter expression to narrow results.

```python
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest engineering updates?"
        }
    ],
    "use_knowledge_base": True,
    "collection_names": ["my_collection"],
    "filter_expr": 'content_metadata["department"] == "engineering" and content_metadata["timestamp"] >= "2024-03-01T00:00:00"'
}
```

## Best Practices

The following are the best practices when you work with custom metadata:

- Metadata Design
  - Plan metadata structure before ingestion.
  - Use consistent naming conventions.
  - Include essential filtering fields.
  - Keep metadata values consistent across documents.
  - Document the expected string format for each metadata field.

- Timestamp Usage
  - Consider time zone implications.
  - Use consistent timestamp precision.

- Filter Expressions
  - Test filter expressions with small datasets first.
  - Use parentheses to clarify complex expressions.
  - Consider performance implications of complex filters.

- Error Handling
  - Validate metadata during ingestion.
  - Handle missing metadata fields gracefully.
  - Log invalid filter expressions.

### 💡 Pro Tip: Implementing Tag-like Metadata

While the current implementation doesn't support array-type metadata fields (like tags), you can implement tag-like functionality using boolean metadata fields. This is particularly useful when you need to categorize documents with multiple attributes.

For example, instead of using an array of tags like:
```json
{
    "category": ["finance", "earnings"]
}
```

You can define boolean metadata fields during collection creation:
```json
{
    "name": "is_finance",
    "type": "string",
    "description": "Indicates if document is related to finance"
},
{
    "name": "is_earnings",
    "type": "string",
    "description": "Indicates if document is related to earnings"
}
```

Then during ingestion, set these fields to "yes" or "no":
```json
{
    "filename": "financial_report.pdf",
    "metadata": {
        "is_finance": "yes",
        "is_earnings": "yes"
    }
}
```

During retrieval, you can filter using boolean logic:
```python
filter_expr = 'content_metadata["is_finance"] == "yes" and content_metadata["is_earnings"] == "yes"'
```

Note: This approach requires defining all possible tags at collection creation time, as the metadata schema cannot be modified after collection creation.

## Troubleshooting

The following are some issues that might arise when you work with custom metadata:

- Filter Expression Errors
  - Verify that the metadate field names are correct.
  - Verify that all values are correctly enclosed in quotes.
  - Verify all metadata values are strings in filter expressions.
  - Verify the operator syntax. For valid expression syntax, refer to [Milvus Filtering Documentation](https://milvus.io/docs/boolean.md).

- Timestamp Filtering Issues
  - Verify that the metadata uses the ISO 8601 format.
  - Verify that the time zones are consistent.
  - Validate the date range logic.

- Missing Metadata
  - Verify that the metadata was added during ingestion.
  - Verify that you specified the correct document filename.
  - Validate the metadata structure.

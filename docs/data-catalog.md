<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Data Catalog for NVIDIA RAG Blueprint

The Data Catalog feature enables comprehensive metadata management for collections and documents in the [NVIDIA RAG Blueprint](readme.md). This feature provides organization, governance, and discovery capabilities for your knowledge base through collection-level and document-level catalog metadata.

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), the Data Catalog endpoints are automatically available. No additional configuration is required.

## Overview

The Data Catalog provides two types of metadata:

1. **Collection Catalog Metadata**: Organizational metadata for entire collections (description, tags, owner, business domain, status)
2. **Document Catalog Metadata**: Metadata for individual documents within collections (description, tags)

Additionally, the system automatically populates content metrics such as `number_of_files`, `has_tables`, `has_charts`, and `has_images` to help you understand what content each collection contains.

## Collection Catalog Metadata

### Supported Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `description` | string | Human-readable description of the collection | "Q4 2024 Financial Reports" |
| `tags` | array[string] | Tags for categorization and discovery | `["finance", "q4-2024"]` |
| `owner` | string | Team or person responsible | "Finance Team" |
| `created_by` | string | User who created the collection | "john.doe@company.com" |
| `business_domain` | string | Business domain or department | "Finance", "Legal", "Engineering" |
| `status` | string | Collection lifecycle status | "Active", "Archived", "Deprecated" |
| `date_created` | timestamp | Automatically set on creation | "2024-11-18T10:30:00+00:00" |
| `last_updated` | timestamp | Automatically updated on changes | "2024-11-18T15:45:00+00:00" |

### Auto-Populated Content Metrics

The system automatically analyzes ingested content and provides these metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `number_of_files` | integer | Total number of documents in the collection |
| `last_indexed` | timestamp | Last time documents were ingested |
| `ingestion_status` | string | Current ingestion status |
| `has_tables` | boolean | Whether collection contains table content |
| `has_charts` | boolean | Whether collection contains charts/diagrams |
| `has_images` | boolean | Whether collection contains images |

## Creating Collections with Catalog Metadata

### Using the API

```python
import requests

url = "http://localhost:8081/v1/collection"

data = {
    "collection_name": "financial_reports_2024",
    "embedding_dimension": 2048,
    "description": "Q4 2024 Financial Reports and Analysis",
    "tags": ["finance", "reports", "q4-2024"],
    "owner": "Finance Team",
    "created_by": "john.doe@company.com",
    "business_domain": "Finance",
    "status": "Active",
    "metadata_schema": []  # Add custom metadata schema if needed
}

response = requests.post(url, json=data)
print(response.json())
```

### Using the Python Client

```python
from nvidia_rag import NvidiaRAGIngestor

ingestor = NvidiaRAGIngestor(mode="client")

result = ingestor.create_collection(
    collection_name="financial_reports_2024",
    vdb_endpoint="http://milvus:19530",
    embedding_dimension=2048,
    description="Q4 2024 Financial Reports and Analysis",
    tags=["finance", "reports", "q4-2024"],
    owner="Finance Team",
    created_by="john.doe@company.com",
    business_domain="Finance",
    status="Active"
)
```

:::{note}
All catalog metadata fields are optional. If not provided, they will be empty strings or empty arrays by default.
:::

## Updating Collection Metadata

You can update collection catalog metadata at any time without re-ingesting documents:

```python
import requests

url = "http://localhost:8081/v1/collections/financial_reports_2024/metadata"

updates = {
    "description": "Q4 2024 Financial Reports - Final Version",
    "tags": ["finance", "reports", "q4-2024", "final", "approved"],
    "status": "Archived",
    "business_domain": "Finance"
}

response = requests.patch(url, json=updates)
print(response.json())
```

:::{note}
The PATCH endpoint performs a merge update. Only provided fields are updated; omitted fields retain their current values.
:::

## Document Catalog Metadata

### Updating Document Metadata

After ingesting documents, you can add descriptive metadata to individual documents:

```python
import requests

url = "http://localhost:8081/v1/collections/financial_reports_2024/documents/annual_report.pdf/metadata"

updates = {
    "description": "Annual Financial Report 2024 - Comprehensive Overview",
    "tags": ["annual", "comprehensive", "board-approved"]
}

response = requests.patch(url, json=updates)
print(response.json())
```

## Retrieving Collections with Catalog Data

### Using the API

```python
import requests

url = "http://localhost:8081/v1/collections"
response = requests.get(url)
result = response.json()

for collection in result.get("collections", []):
    info = collection.get('collection_info', {})
    print(f"Collection: {collection['collection_name']}")
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Tags: {info.get('tags', [])}")
    print(f"  Owner: {info.get('owner', 'N/A')}")
    print(f"  Status: {info.get('status', 'N/A')}")
    print(f"  Files: {info.get('number_of_files', 0)}")
    print(f"  Has Tables: {info.get('has_tables', False)}")
    print(f"  Has Charts: {info.get('has_charts', False)}")
    print(f"  Has Images: {info.get('has_images', False)}")
    print()
```

### Example Response

```json
{
  "collections": [
    {
      "collection_name": "financial_reports_2024",
      "num_entities": 1250,
      "metadata_schema": [...],
      "collection_info": {
        "description": "Q4 2024 Financial Reports - Final Version",
        "tags": ["finance", "reports", "q4-2024", "final"],
        "owner": "Finance Team",
        "created_by": "john.doe@company.com",
        "business_domain": "Finance",
        "status": "Archived",
        "date_created": "2024-11-18T10:30:00+00:00",
        "last_updated": "2024-11-18T15:45:00+00:00",
        "number_of_files": 15,
        "last_indexed": "2024-11-18T14:20:00+00:00",
        "ingestion_status": "completed",
        "has_tables": true,
        "has_charts": true,
        "has_images": false
      }
    }
  ],
  "total_collections": 1,
  "message": "Collections listed successfully."
}
```

## Use Cases

### Data Governance and Compliance

Track ownership, business domain, and lifecycle status of collections for compliance and auditing requirements:

```python
# Mark collections for different governance stages
ingestor.update_collection_metadata(
    collection_name="legal_contracts",
    status="Active",
    owner="Legal Team",
    business_domain="Legal"
)
```

### Knowledge Base Organization

Use tags and descriptions to organize and discover collections:

```python
# Tag collections by project, team, or topic
ingestor.create_collection(
    collection_name="project_apollo_docs",
    description="Project Apollo Technical Documentation",
    tags=["apollo", "engineering", "technical", "2024"],
    business_domain="Engineering"
)
```

### Lifecycle Management

Manage collection lifecycles by updating status as collections evolve:

```python
# Archive completed project documentation
ingestor.update_collection_metadata(
    collection_name="project_apollo_docs",
    status="Archived",
    tags=["apollo", "engineering", "technical", "2024", "completed"]
)
```

### Content Analysis

Use auto-populated metrics to understand collection content types:

```python
# Query collections to find those with tables for structured data extraction
collections = ingestor.get_collections()
table_collections = [
    c for c in collections 
    if c['collection_info'].get('has_tables', False)
]
```

## Data Catalog vs Custom Metadata

The RAG Blueprint provides two complementary metadata systems:

| Feature | Data Catalog (This Document) | [Custom Metadata](custom-metadata.md) |
|---------|------------------------------|---------------------------------------|
| **Purpose** | Collection/document management and governance | Document content filtering for retrieval |
| **Scope** | Entire collections and documents | Individual document chunks |
| **Schema** | Fixed catalog fields (description, tags, owner, etc.) | User-defined per collection (flexible) |
| **Updates** | Update anytime via PATCH endpoints | Set during ingestion only |
| **Use Case** | "Which collections does Finance own?" | "Show documents with priority > 5" |
| **Filtering** | Organization and discovery | Semantic search and retrieval |

**When to Use:**
- Use **Data Catalog** for collection organization, governance, and discovery
- Use **Custom Metadata** for filtering document chunks during retrieval
- Use **Both** together for comprehensive data management

## Vector Database Support

Data Catalog is supported on both Milvus and Elasticsearch with full feature parity:

| Feature | Milvus | Elasticsearch |
|---------|--------|---------------|
| Collection Catalog Metadata | ✅ | ✅ |
| Document Catalog Metadata | ✅ | ✅ |
| Auto-Populated Metrics | ✅ | ✅ |
| Runtime Metadata Updates | ✅ | ✅ |

## API Reference

For complete API specifications including request/response schemas and error codes, see the [API - Ingestor Server Schema](api-ingestor.md).

### Endpoints

- **POST /v1/collection**: Create collection with catalog metadata
- **PATCH /v1/collections/{collection_name}/metadata**: Update collection metadata
- **PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata**: Update document metadata
- **GET /v1/collections**: Get all collections with catalog data

## Related Documentation

- [Custom Metadata Support](custom-metadata.md) - For document content metadata and filtering
- [API - Ingestor Server Schema](api-ingestor.md) - Complete API reference
- [Use the Python Package](python-client.md) - Python client documentation
- [Multi-Collection Retrieval](multi-collection-retrieval.md) - Using multiple collections in queries


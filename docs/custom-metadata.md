<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Advanced Metadata Filtering with Natural Language Generation

The [NVIDIA RAG Blueprint](readme.md) features **advanced metadata filtering with natural language generation**, enabling you to:

- **Generate filter expressions from natural language** using LLMs
- **Define comprehensive metadata schemas** with type validation
- **Filter documents using complex expressions** with full operator support
- **Work with multiple collections** having different schemas
- **Leverage AI-powered filtering** for intuitive document retrieval
- **Validate and process filters** with robust error handling
- **Optimize performance** with caching and parallel processing

## Quick Start

### 1. Enable Natural Language Filter Generation

```python
config = {
    "filter_expression_generator": {
        "enable_filter_generator": True,
        "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "temperature": 0.1,
        "max_tokens": 1024
    }
}
```

### 2. Define Metadata Schema

```python
collection_data = {
    "collection_name": "technical_docs",
    "embedding_dimension": 2048,
    "metadata_schema": [
        {"name": "category", "type": "string", "required": True, "description": "Document category (e.g., 'AI', 'engineering', 'marketing')"},
        {"name": "priority", "type": "integer", "required": False, "description": "Priority level (1-10)"},
        {"name": "rating", "type": "float", "required": False, "description": "Document quality rating (0.0-5.0)"},
        {"name": "tags", "type": "array", "array_type": "string", "required": False, "description": "Document tags for categorization"},
        {"name": "created_date", "type": "datetime", "required": False, "description": "Document creation timestamp"},
        {"name": "is_public", "type": "boolean", "required": False, "description": "Whether document is publicly accessible"}
    ]
}
```

### 3. Add Metadata During Ingestion

```python
custom_metadata = [
    {
        "filename": "ai_guide.pdf",
        "metadata": {
            "category": "AI",
            "priority": 8,
            "rating": 4.5,
            "tags": ["machine-learning", "neural-networks"],
            "created_date": "2024-01-15T10:30:00",
            "is_public": True
        }
    }
]

data = {
    "collection_name": "technical_docs",
    "custom_metadata": custom_metadata,
    "split_options": {"chunk_size": 512, "chunk_overlap": 150}
}
```

### 4. Use Natural Language Filtering

```python
payload = {
    "query": "What are the latest AI developments?",
    "collection_names": ["technical_docs"],
    "enable_filter_generator": True,
    "filter_expr": "",
    "reranker_top_k": 10,
    "vdb_top_k": 100
}
```

## 📓 Interactive Notebook

For a comprehensive, interactive demonstration of metadata functionality, check out our dedicated notebook:

**[📖 nb_metadata.ipynb](../notebooks/nb_metadata.ipynb)**

This notebook demonstrates:
- **Real metadata ingestion** with Ford vehicle manuals (2015 Edge, 2023 Edge, 2024 Escape) including manufacturer, model, year, rating, tags, features, and document properties
- **Q&A without filtering** - shows how queries return results from all vehicle models
- **Q&A with metadata filtering** - demonstrates filtering by specific model (`content_metadata["model"] == "edge"`) to get targeted results
- **Complex filter expressions** - combines multiple criteria like manufacturer, rating, date ranges, and boolean conditions
- **Error handling examples** - shows validation failures for missing required fields, wrong data types, and invalid filter syntax
- **Metadata extraction from queries** - demonstrates how to extract metadata from user questions for enhanced RAG responses

## Important Notes

### 🎯 **Vector Database Support**
- **Milvus**: Full support for natural language filter generation and complex expressions
- **Elasticsearch**: Limited to basic filter validation only (no natural language generation)
- **Natural Language Generation**: Only works with Milvus vector database
- **Filter Expression Types**: Milvus uses string expressions, Elasticsearch uses list of dictionaries

### 🚨 **Key Limitations**
- **IS NULL/IS NOT NULL operations**: Not supported
- **Empty string/array comparisons**: Not supported
- **Direct array indexing**: Not supported (e.g., `content_metadata["tags"][0]`)
- **NULL values**: Not supported in filter expressions
- **Schema evolution**: Removing fields may break existing filters

## Vector Database Support

| Feature | Milvus | Elasticsearch |
|---------|--------|---------------|
| **Natural Language Filter Generation** | ✅ Fully automated with LLM integration | 🔧 Advanced users can leverage native Elasticsearch Query DSL for sophisticated queries |
| **Filter Expression Complexity** | ✅ String-based syntax with validation | 🚀 Full Elasticsearch Query DSL support - Boolean, range, nested, geo, and aggregation queries |
| **Schema Validation** | ✅ Comprehensive metadata schema validation | 🔧 Flexible schema-less design with dynamic mapping capabilities |
| **Array Operations** | ✅ Built-in functions: `array_contains`, `array_length`, etc. | 🚀 Native nested object support with powerful array querying capabilities |
| **Query Performance** | ⚡ Optimized for vector similarity with metadata filtering | ⚡ Industry-leading full-text search with advanced scoring algorithms |
| **Advanced Features** | 🎯 Simple, intuitive filter syntax | 🚀 Multi-field search, fuzzy matching, proximity queries, aggregations, and analytics |
| **UI Support** | ✅ **Primary support** - Full filtering interface in UI | ❌ **No UI support** - Requires direct API integration |

### Key Differences

- **🎯 Milvus**: Designed for simplicity with automated natural language filter generation, perfect for users who want straightforward metadata filtering
- **🚀 Elasticsearch**: Provides full access to enterprise-grade search capabilities, ideal for advanced users who need complex querying, analytics, and fine-grained control

**📝 Note**: The UI supports basic arithmetic filter operators to showcase functionality, while the RAG-Server API provides full support for all mentioned operators and advanced features.

## Natural Language Filter Generation

### What It Does

The natural language filter generation automatically converts your queries into precise metadata filters, helping you get more accurate and relevant results by filtering documents based on specific criteria mentioned in your question.

### How to Use It

Simply enable the feature and ask questions naturally:

```python
# Enable filter generation in your request
payload = {
    "query": "Show me AI documents with rating above 4.0",
    "collection_names": ["technical_docs"],
    "enable_filter_generator": True,  # 🎯 Enable this
    "reranker_top_k": 10,
    "vdb_top_k": 100
}
```

### How It Helps You

**Without Filter Generation:**
- Query: "Show me AI documents with rating above 4.0"
- Result: All documents, regardless of category or rating

**With Filter Generation:**
- Query: "Show me AI documents with rating above 4.0"
- Generated Filter: `content_metadata["category"] == "AI" and content_metadata["rating"] > 4.0`
- Result: Only AI documents with rating > 4.0

### Example Queries and Generated Filters

| Your Question | Generated Filter | What It Does |
|---------------|------------------|--------------|
| "Show me AI documents with rating above 4.0" | `content_metadata["category"] == "AI" and content_metadata["rating"] > 4.0` | Filters to AI category + high ratings |
| "Public documents with engineering tags" | `content_metadata["is_public"] == true and array_contains(content_metadata["tags"], "engineering")` | Filters to public docs with engineering tags |
| "High priority tech documents from 2024" | `content_metadata["priority"] > 7 and content_metadata["category"] == "tech" and content_metadata["created_date"] >= "2024-01-01"` | Filters to urgent tech docs from 2024 |

### Improving Existing Filters

You can also improve existing filters by providing them with your query:

```python
# Existing filter
existing_filter = 'content_metadata["category"] == "tech"'

# User request to improve it
payload = {
    "query": "Make it more specific for urgent tech documents",
    "enable_filter_generator": True,
    "filter_expr": existing_filter  # Will be improved
}

# Generated improved filter:
# content_metadata["category"] == "tech" and content_metadata["priority"] == "urgent"
```

### Error Handling

The system gracefully handles filter generation failures:

- **LLM Unavailable**: Falls back to empty filter (no filtering)
- **Invalid Generation**: Returns None, continues without filtering
- **Schema Mismatch**: Logs warning, skips incompatible collections
- **Processing Errors**: Returns original query, maintains functionality

## Metadata Schema Definition

### Supported Data Types

#### Basic Types
- **`string`**: Text data with configurable length limits
- **`integer`**: Whole numbers (e.g., priority levels, counts)
- **`float`**: Decimal numbers (e.g., ratings, scores)
- **`number`**: Generic numeric type (accepts both integer and float)
- **`boolean`**: True/false values
- **`datetime`**: Date and time values (ISO 8601 format)

#### Complex Types
- **`array`**: Lists of values with typed elements
  - **Valid array types**: `string`, `number`, `integer`, `float`, `boolean`
  - **Example**: `{"type": "array", "array_type": "string"}`

### Schema Validation Rules

#### Field Name Validation
- **Non-empty**: Field names cannot be empty or whitespace-only
- **Unique**: Each field name must be unique within the schema
- **Case-sensitive**: Field names are case-sensitive

**Note**: The `filename` field is automatically added to all collections if you don't define it in your schema. You can also define your own `filename` field in your schema, and the system will use your definition instead of the automatic one.

#### Field Properties
- **`name`**: Field identifier (required)
- **`type`**: Data type (required)
- **`required`**: Whether field is mandatory (default: `false`)
- **`array_type`**: Type of array elements (required only for `array` type)
- **`max_length`**: Maximum length for string/array fields (optional)
- **`description`**: Optional field description for documentation (optional)

#### Type-Specific Validation
- **String fields**: Configurable max length, accepts any text
- **Numeric fields**: Supports arithmetic operations and comparisons
- **Datetime fields**: Flexible parsing with ISO 8601 normalization
- **Boolean fields**: Accepts various truth values ("true", "false", "1", "0", etc.)
- **Array fields**: Requires `array_type`, validates element types

### Example Schemas

#### Technical Documentation Schema

```json
[
    {
        "name": "category",
        "type": "string",
        "required": true,
        "description": "Document category (e.g., 'AI', 'engineering', 'marketing')"
    },
    {
        "name": "priority",
        "type": "integer",
        "required": false,
        "description": "Priority level (1-10)"
    },
    {
        "name": "rating",
        "type": "float",
        "required": false,
        "description": "Document quality rating (0.0-5.0)"
    },
    {
        "name": "tags",
        "type": "array",
        "array_type": "string",
        "required": false,
        "max_length": 50,
        "description": "Document tags for categorization"
    },
    {
        "name": "created_date",
        "type": "datetime",
        "required": false,
        "description": "Document creation timestamp"
    },
    {
        "name": "is_public",
        "type": "boolean",
        "required": false,
        "description": "Whether document is publicly accessible"
    }
]
```

## Adding Metadata During Ingestion

### Metadata Structure

Metadata is specified as a list of objects during document ingestion:

```python
custom_metadata = [
    {
        "filename": "document_name.pdf",
        "metadata": {
            "field1": "value1",
            "field2": "value2",
            # ... more fields
        }
    }
]
```

### Validation During Ingestion

The system validates metadata during ingestion:

- **Required fields**: All required fields must be present
- **Type validation**: Values are validated against schema types
- **Array validation**: Array elements must match specified `array_type`
- **Length validation**: String and array fields respect `max_length` limits
- **Unknown fields**: Files with metadata fields not defined in the schema will fail validation
- **Error handling**: Invalid metadata causes document rejection with detailed errors

**Note**: The system uses strict validation. Any metadata fields not defined in the schema will cause the entire file to fail ingestion.

## Filter Expression Syntax

### Basic Syntax

Filter expressions use the format: `content_metadata["field_name"] operator value`

**Milvus Filter Syntax Documentation:**
See the [Milvus Filtering Explained](https://milvus.io/docs/boolean.md#Filtering-Explained) guide for full details.

**💡 Note:** This document contains extensive examples throughout - from quick start examples, natural language filter generation, to complex expressions and API usage examples.


### Supported Operators by Type

#### String Operations
- **Equality**: `==`, `=`, `!=`
- **Pattern matching**: `like`, `LIKE` (supports wildcards)
- **Membership**: `in`, `IN`, `not in`, `NOT IN`

#### Numeric Operations (integer, float, number)
- **Comparison**: `==`, `=`, `!=`, `>`, `>=`, `<`, `<=`
- **Range**: `between`, `BETWEEN`
- **Membership**: `in`, `IN`, `not in`, `NOT IN`

#### Datetime Operations
- **Comparison**: `==`, `=`, `!=`, `>`, `>=`, `<`, `<=`
- **Range**: `between`, `BETWEEN`
- **Relative**: `before`, `BEFORE`, `after`, `AFTER`

#### Boolean Operations
- **Equality**: `==`, `=`, `!=`
#### Array Operations
- **Equality**: `==`, `=`, `!=`
- **Membership**: `in`, `IN`, `not in`, `NOT IN`
- **Includes**: `includes`, `INCLUDES`, `does not include`, `DOES NOT INCLUDE`
- **Functions**: `array_contains`, `array_contains_all`, `array_contains_any`, `array_length`

#### Logical Operations
- **Logical**: `AND`, `OR`, `NOT`
- **Grouping**: `(condition1) AND (condition2)`

### Filter Expression Examples

```python
# String filtering
'content_metadata["category"] == "technical"'
'content_metadata["title"] like "%policy%"'

# Numeric filtering
'content_metadata["priority"] > 5'
'content_metadata["rating"] between 3.5 and 5.0'

# Array filtering
'array_contains(content_metadata["tags"], "engineering")'
'content_metadata["tags"] includes ["tech"]'
'content_metadata["tags"] does not include ["deprecated"]'

# Complex expressions
'(content_metadata["category"] == "technical") AND (content_metadata["priority"] > 5)'
```

### Using Filters in API Calls

#### Search Endpoint
```python
payload = {
    "query": "What are the technical specifications?",
    "collection_names": ["technical_docs"],
    "filter_expr": '(content_metadata["category"] == "technical") AND (content_metadata["priority"] > 5)',
    "reranker_top_k": 10,
    "vdb_top_k": 100,
    "enable_filter_generator": True  # Enable natural language generation
}
```

#### Generate Endpoint
```python
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest engineering updates?"
        }
    ],
    "use_knowledge_base": True,
    "collection_names": ["technical_docs"],
    "enable_filter_generator": True
}
```

### Elasticsearch Filter Example

For Elasticsearch, filters must be provided as a list of dictionaries using Elasticsearch query syntax:

```python
# Elasticsearch filter example
filter_expr = [
    {"term": {"metadata.content_metadata.category": "AI"}},
    {"range": {"metadata.content_metadata.priority": {"gt": 5}}}
]
```

**Note**: Elasticsearch filters use the `metadata.content_metadata.field_name` format and support standard Elasticsearch query types like `term`, `range`, `wildcard`, `terms`, etc.

**Advanced Elasticsearch Support**: All ES queries are supported. Advanced developers who are familiar with Elasticsearch can refer to the [official Elasticsearch query and filter documentation](https://www.elastic.co/docs/explore-analyze/query-filter) and write any query or filter anything they need. This advanced functionality is intended for experienced Elasticsearch users.

## Advanced Filtering Features

### Array Functions

| Function | Description | Example |
|----------|-------------|---------|
| `array_contains(field, value)` | Check if array contains a specific value | `array_contains(content_metadata["tags"], "tech")` |
| `array_contains_all(field, array)` | Check if array contains all values from another array | `array_contains_all(content_metadata["tags"], ["tech", "ai"])` |
| `array_contains_any(field, array)` | Check if array contains any value from another array | `array_contains_any(content_metadata["tags"], ["tech", "ai"])` |
| `array_length(field)` | Get the length of an array | `array_length(content_metadata["tags"]) > 3` |

## Configuration and Setup

### Filter Expression Generator Configuration

```python
# Configuration file (config.yaml)
filter_expression_generator:
  enable_filter_generator: true  # Set to true to enable filter generation (default is false)
  model_name: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
  server_url: ""  # Leave empty for default endpoint
  temperature: 0.1  # Low temperature for consistent results
  top_p: 0.9
  max_tokens: 1024
```

### Metadata Configuration

```python
# Metadata configuration
metadata:
  max_array_length: 1000             # Maximum length for array metadata fields
  max_string_length: 65535           # Maximum length for string metadata fields
  allow_partial_filtering: false     # Allow filter expressions to work with collections that support them
```

### Environment Variables

```bash
# Enable filter generation
export ENABLE_FILTER_GENERATOR=true

# LLM configuration
export APP_FILTEREXPRESSIONGENERATOR_MODELNAME="nvidia/llama-3.3-nemotron-super-49b-v1.5"
export APP_FILTEREXPRESSIONGENERATOR_SERVERURL=""

# Note: Metadata configuration is not currently exposed via environment variables
# Default behavior is controlled by the configuration.py file at the code level
```

### Partial Filtering Modes

#### Flexible Mode (`allow_partial_filtering: true`)
- **Operation succeeds** if at least one collection supports the filter expression
- **Collections that support the filter** are processed normally
- **Collections that don't support the filter** are skipped

#### Strict Mode (`allow_partial_filtering: false`)
- **Operation fails** if any collection doesn't support the filter expression
- **All collections must support** the filter expression for the request to succeed
- **No partial results** are returned - it's all or nothing

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **No filter generated** | LLM unavailable or query too vague | Check LLM service, make query more specific |
| **Field not found** | Field doesn't exist in collection schema | Check available fields in error message |
| **Operator not supported** | Operator incompatible with field type | Use appropriate operators for the field type |
| **Syntax error** | Invalid filter expression syntax | Review syntax and use provided examples |
| **Unknown field error** | Metadata contains fields not in schema | Remove unknown fields or add them to schema |
| **Missing required field** | Required field not provided in metadata | Add the required field to your metadata |

## API Reference

### API Endpoints

#### Search with Filter Generation

```http
POST /v1/search
Content-Type: application/json

{
    "query": "Show me AI documents with rating above 4.0",
    "collection_names": ["research_papers"],
    "enable_filter_generator": true,
    "reranker_top_k": 10,
    "vdb_top_k": 100
}
```

#### Generate with Filter Generation

```http
POST /v1/generate
Content-Type: application/json

{
    "messages": [{"role": "user", "content": "What are the latest engineering updates?"}],
    "use_knowledge_base": true,
    "collection_names": ["research_papers"],
    "enable_filter_generator": true
}
```

## Summary

This comprehensive documentation covers the advanced metadata filtering system with natural language generation capabilities. The system provides:

### 🚀 **Key Capabilities**
- **Natural Language Filter Generation**: Convert user queries to structured filters using LLMs
- **Comprehensive Metadata Support**: Full type system with validation and processing
- **Multi-Collection Support**: Flexible filtering across heterogeneous collections
- **Production-Ready Features**: Error handling, caching, and performance optimization

### 🛠️ **Implementation Features**
- **Type-Safe Metadata**: String, datetime, number, boolean, and array types
- **Advanced Filtering**: Complex expressions with logical operators and functions
- **AI-Powered Generation**: LLM-based filter creation from natural language
- **Robust Validation**: Comprehensive error handling and detailed feedback

### 🎯 **Production Readiness**
- **198+ Integration Tests**: Comprehensive test coverage without external dependencies
- **Performance Optimization**: Caching, parallel processing, and schema optimization
- **Error Recovery**: Graceful degradation and detailed error messages
- **Configuration Management**: Flexible setup via environment variables

This documentation provides everything needed to implement and use the advanced metadata filtering system with natural language generation capabilities in production environments.


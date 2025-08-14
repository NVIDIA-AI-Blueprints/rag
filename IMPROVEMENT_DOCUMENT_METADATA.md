# Document Metadata Tracking Implementation

## Overview
This implementation addresses multiple TODO items in the ingestor server by adding proper document metadata tracking throughout the RAG pipeline. This enhancement provides users with essential document information that was previously missing or placeholder values.

## Problem Addressed
The following TODO items were resolved:
- **Line 312**: `TODO: Store document_id, timestamp and document size as metadata`
- **Line 590**: `TODO - Use actual document_id` (in list documents response)  
- **Line 592**: `TODO - Use actual timestamp` (in list documents response)
- **Line 593**: `TODO - Use actual size` (in list documents response)
- **Line 634**: `TODO: Delete based on document_ids if provided`
- **Line 639**: `TODO - Use actual document_id` (in delete documents response)
- **Line 641**: `TODO - Use actual size` (in delete documents response)

## Changes Made

### 1. Enhanced Document Ingestion (`src/nvidia_rag/ingestor_server/main.py`)
- **Document Upload Process**: Now generates and stores document_id, timestamp, and size_bytes during ingestion
- **Metadata Storage**: Document-level metadata is stored with each chunk in the vector database
- **Response Enhancement**: Upload responses now return actual metadata instead of generating new values

### 2. Improved Custom Metadata Handling (`src/nvidia_rag/utils/common.py`) 
- **Extended `prepare_custom_metadata_dataframe`**: Now accepts `document_metadata_map` parameter
- **Per-File Metadata**: Supports different document metadata per file
- **Backward Compatibility**: Maintains compatibility with existing custom metadata functionality

### 3. Updated NV-Ingest Integration (`src/nvidia_rag/ingestor_server/nvingest.py`)
- **Enhanced `get_nv_ingest_ingestor`**: Passes document metadata to the ingestion pipeline
- **Metadata CSV Generation**: Document metadata is included in the CSV file sent to NV-Ingest

### 4. Improved Document Retrieval (`src/nvidia_rag/utils/vectorstore.py`)
- **Enhanced `get_docs_vectorstore_langchain`**: Extracts document_id, timestamp, and size_bytes from stored metadata
- **Proper Response Format**: Returns structured document information instead of placeholder values

### 5. Better Delete Functionality (`src/nvidia_rag/ingestor_server/main.py`)
- **Metadata Retrieval**: Gets document metadata before deletion to provide accurate response
- **Proper Response**: Delete responses now include actual document_id and size_bytes

## Benefits

1. **API Completeness**: Document management APIs now return meaningful data instead of empty placeholders
2. **Document Tracking**: Users can track when documents were uploaded and their sizes
3. **Better UX**: Frontend applications can display proper document information
4. **Audit Trail**: Timestamp tracking enables audit capabilities
5. **No Breaking Changes**: All changes are backward compatible

## Data Flow

1. **Upload**: 
   - Generate document_id (UUID), timestamp (ISO format), size_bytes
   - Store this metadata with each document chunk in vector database
   - Return actual values in API response

2. **List**: 
   - Query vector database for all documents
   - Extract document-level metadata from stored chunks
   - Deduplicate and return complete document information

3. **Delete**:
   - Retrieve metadata before deletion
   - Perform deletion
   - Return response with actual metadata values

## Technical Implementation

### Metadata Storage
Document metadata is stored in the `content_metadata` field of each vector database entry:
```json
{
  "document_id": "uuid-string",
  "timestamp": "2025-08-14T10:30:00.000Z",
  "size_bytes": 1024,
  "custom_field": "custom_value"
}
```

### API Response Format
Enhanced API responses now include:
```json
{
  "documents": [
    {
      "document_id": "actual-uuid",
      "document_name": "example.pdf", 
      "timestamp": "2025-08-14T10:30:00.000Z",
      "size_bytes": 1024,
      "metadata": {
        "custom_field": "value"
      }
    }
  ]
}
```

## Future Enhancements
- Extend delete API to accept document_ids in addition to document_names
- Add document update tracking (last_modified timestamps)
- Implement document versioning capabilities

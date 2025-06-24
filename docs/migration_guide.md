<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Migration Guide: RAG v2.1.0 to RAG v2.2.0

## Overview

This guide summarizes the key API changes and new features introduced in RAG v2.2.0. Update your integrations to take advantage of new summarization, metadata, and multi-collection capabilities, and to prepare for upcoming deprecations.

## API changes

- Summarization support
  - A `generate_summary: bool` field has been added to the `POST /documents` and `PATCH /documents` endpoints.
  - A new `GET /summary` endpoint has been added to the `rag-server`, allowing users to retrieve summaries of uploaded files.

- Custom metadata support
  - `POST /collections` will be deprecated in favor of `POST /collection` for the ingestor-server.
    - `POST /collection` allows only a single collection to be created at a time.
    - Developers can now define a custom metadata schema for all files uploaded to a collection.
    - `POST /collections` will be deprecated in a future release; developers are encouraged to migrate to `POST /collection`.
  - Metadata information is now available in the responses of the `GET /collections` and `GET /documents APIs`.

- Multi-collection support
  - The `collection_names: List[str]` field has been added to the request schema of the `POST /generate` and `POST /search` endpoints, replacing `collection_name: str`. The old `collection_name: str` field will be deprecated in a future release.

# Migration Guide: RAG v2.0.0 to RAG v2.1.0

## Overview
In RAG 2.1.0, the behaviour of POST /documents API which can be used to upload documents has changed. Developers can now upload documents in a non-blocking manner.

## API Changes

Updated openapi schemas are available [here](./api_reference/).

### 2.1 Changed Endpoints and Features

1. **Documents management**:
   - **Upload documents**:
     - *New field*: `blocking: bool` is added in the request schema. By default it is set to `True`. Developers are expected to call this API and then monitor the status of doc upload using `/status` API.

# Migration Guide: RAG v1.0.0 to RAG v2.0.0

## Overview

In **RAG v1.0.0**, a single server managed both **ingestion** and **retrieval/generation** APIs.

In **RAG v2.0.0**, the architecture has evolved to utilize **two separate servers**:

1. **RAG Server** - Manages retrieval and generation APIs.
2. **Ingestion Server** - Manages ingestion APIs.

Also the pipeline by default using on-prem models as default. Earlier it used to use NVIDIA cloud hosted models as default. The minimum hardware requirements for deploying the blueprint in its default settings is specified [here](../README.md#minimum-system-requirements).
This guide outlines the key changes and steps required for migration.

---

## 1. Server Architecture Changes

| Feature                 | RAG v1.0.0 (Single Server) | RAG v2.0.0 (Separate Servers)             |
|-------------------------|----------------------------|-------------------------------------------|
| API Hosting             | Single server for all APIs | Two servers: **RAG Server** and **Ingestion Server** |
| Retrieval & Generation  | Same server as ingestion   | Hosted separately in RAG Server           |
| Document Ingestion      | Same server as retrieval   | Hosted separately in Ingestion Server     |

---

## 2. API Changes

Updated openapi schemas are available [here](./api_reference/).

### 2.1 New Endpoints and Features

1. **Collection Management**:
   - **Create Collection**:
     - *New Endpoint*: `POST /collections`
     - *Description*: Allows the creation of document collections. Previously, collections were implicitly created during document uploads.
   - **Delete Collection**:
     - *New Endpoint*: `DELETE /collections/{collection_name}`
     - *Description*: Enables deletion of entire collections.

2. **Multi-file Document Upload**:
   - *Enhanced Endpoint*: `POST /documents`
   - *Description*: Supports uploading multiple files in a single request. Previously, only single-file uploads were supported.

### 2.2 Endpoints Moved to Separate Servers

| API Endpoint | RAG v1.0.0 | RAG v2.0.0 |
|--------------|------------|------------|
| `/documents` (POST) - Upload Document | Unified Server | Now in Ingestion Server |
| `/documents` (GET) - List Documents   | Unified Server | Now in Ingestion Server |
| `/documents` (DELETE) - Delete Document | Unified Server | Now in Ingestion Server |
| `/generate` (POST) - Generate Answer  | Unified Server | Now in RAG Server       |
| `/search` (POST) - Document Search    | Unified Server | Now in RAG Server       |

---

### 2.3 Breaking Endpoint Changes

1. **Ingestion API Enhancements**:
   - `PATCH /documents` introduced in v2.0.0 for **deleting & uploading documents in a single request**. `POST /documents` will throw error if a document exists in the collection
   - `POST /documents`  now accepts multiple files as a list instead a single file. The payload schema in v2.0.0 is non-backward compatible with v1.0.0.
   - A seperate `POST /collections` API is now needed to be called to create a new collection. In v1.0.0, a new collection was automatically created when `POST /documents` was called.
   - New optional parameters introduced for all APIs to improve the runtime configurability of the pipeline.
   - `DELETE /documents` API now accepts multiple files (List[str]) in the payload instead of a single string. This is again non-backward compatible with v1.0.0.

2. **Document Search and Generate Enhancements**:
   - `search` and `generate` API now includes additional options added to refine retrieval results.
   - Both of these APIs remain backward compatible with v1.0.0.

1. **Health API remains unchanged**:
   - `/health` endpoint still exists in both servers and is backward compatible.

---

## 3. Migration Steps

### Step 1: Deploy Two Separate Containers

Ensure that you run two separate containers for **RAG Server** and **Ingestion Server** by following the [quickstart guide](quickstart.md)

### Step 2: Update API Calls

Modify API calls in your client applications:

- **For Retrieval & Generation**, update requests to point to the RAG Server (e.g., `http://rag-server:8081`).
- **For Document Ingestion**, update requests to point to the Ingestion Server (e.g., `http://ingestion-server:8082`).

### Step 3: Adjust API Payloads

You can understand the updated schemas for APIs in v2.0.0 by following the [notebooks](../notebooks/).


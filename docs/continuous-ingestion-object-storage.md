<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Continuous Ingestion from Object Storage RAG Blueprint

Continuous ingestion from object storage connects the [RAG blueprint](readme.md) to continuous integration. This enables an event-driven pipeline that automatically indexes documents. Continuous integration means that when you add documents to a storage bucket, the system detects new uploads, routes them for processing, and indexes their content—making all data immediately searchable and available for analysis through the [RAG Frontend](user-interface.md).

## Hardware Requirements

| Requirement | Details |
|-------------|---------|
| **GPU** | 2x RTX PRO 6000 Blackwell or 2x H100 |
| **OS** | Ubuntu 22.04 or later |
| **Docker** | Docker 24.0+ with Docker Compose v2 |
| **NVIDIA Driver** | 570+ |
| **NVIDIA Container Toolkit** | Required |


## Overview

You can create an event-driven continuous ingestion pipeline that works as follows:

1. Upload documents to object storage.

2. The system detects new uploads via storage events and routes them for processing.

3. Content is automatically indexed into the RAG vector store.

4. You can then query the ingested content through the RAG UI or API.

Continuous ingestion supports documents such as PDF, DOCX, and other formats supported by the [ingestor](api-ingestor.md).

## Architecture

The continuous ingestion architecture features the following high-level flow:

1. Object storage: Files are written to storage using a protocol that emits events (for example, MinIO configured with Kafka notifications).

2. Event trigger: Upload events are published to a Kafka topic.

3. Consumer: A Kafka consumer subscribes to the topic, retrieves the events, downloads the corresponding files from object storage, and routes them for processing.

4. Document path: Files are passed to a file-based processing pipeline (such as the NeMo Retriever Library or ingestor-server) and then indexed in the vector database.

The continuous ingestion architecture follows the end-to-end sequence described above and can be summarized as:

- Document ingestion flow: (1) → (2) → (3) → file-based processing → VectorDB → RAG Agent.

## Implementation Components

The reference implementation includes the following components:

- Object storage (MinIO): A bucket configured with Kafka notifications on put (and optionally delete) events.

- Kafka: A broker and topic (for example, aidp-topic) used to publish storage event notifications.

- Kafka consumer: A service that:

-- Subscribes to the Kafka topic and consumes storage events.

-- Downloads new objects from MinIO.

-- Sends files to the RAG ingestor for indexing.

The deployment is defined in `examples/rag_event_ingest/deploy/docker-compose.yaml`, which runs MinIO, Kafka, and the Kafka consumer on the same Docker network as the RAG stack (`nvidia-rag`).

### Prerequisites

- [Deploy the NVIDIA RAG Blueprint](deploy-docker-self-hosted.md) (NIMs, Milvus, ingestor-server, RAG server) so the consumer can reach the ingestor and the rest of the stack.
- Ensure the `nvidia-rag` Docker network exists (created by the RAG deployment).
- For the notebook, clone the repo, set `NGC_API_KEY`, and have the required hardware (see notebook for GPU and software requirements).

### Option 1: Use the Notebook

The notebook provides a guided walkthrough of the following steps:

- Environment setup
- NVIDIA RAG deployment
- Continuous ingestion pipeline deployment (Kafka, MinIO, and consumer)
- Testing document uploads with RAG queries
- Cleanup

To follow along, open and run: [rag_event_ingest.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_event_ingest.ipynb).

### Option 2: Deploy the Example with Docker Compose

From the repository root, after the RAG stack is up:

```bash
docker compose -f examples/rag_event_ingest/deploy/docker-compose.yaml up -d
```

This command launches the following components:

- Kafka (with an optional Kafka UI available on port 8080)
- MinIO (object storage and console using ports 9201 and 9211 in the example)
- Kafka consumer — connects to the ingestor at `INGESTOR_SERVER_URL` (default: `http://ingestor-server:8082`) and uses `COLLECTION_NAME` (default: `aidp_bucket`)

After deployment, upload documents and query ingested content as follows:

1. Open the MinIO Console UI at `http://<host-ip>:9211/login`.
2. Log in with the default credentials (`minioadmin` / `minioadmin`).
3. Navigate to the `aidp-bucket` bucket and upload your documents (PDF, DOCX, etc.).
4. The system automatically publishes upload events to Kafka, the consumer retrieves the files, and documents are sent to the ingestor for indexing into the `aidp_bucket` collection.
5. Query the ingested content through the RAG Frontend UI at `http://<host-ip>:8090` (select the `aidp_bucket` collection) or via the RAG API at `http://<host-ip>:8081/generate`.

### Key Environment Variables

The following environment variables configure the Kafka consumer. For details, refer to `examples/rag_event_ingest/deploy/docker-compose.yaml`.

Consumer environment variables

| Variable | Description | Default Value|
|----------|---------|--------|
| `KAFKA_BOOTSTRAP_SERVERS` | Address of the Kafka broker(s). | `kafka:9092` |
| `KAFKA_TOPIC` |Kafka topic used for object storage events. | `aidp-topic` |
| `MINIO_ENDPOINT` | MinIO endpoint in <host>:<port> format. | `minio-source-1:9000` |
| `INGESTOR_SERVER_URL` | Base URL for the RAG ingestor service. | `http://ingestor-server:8082` |
| `COLLECTION_NAME` | Target RAG collection for content indexing. | `aidp_bucket` |

## Reference

- [RAG Blueprint deployment (Docker self-hosted)](deploy-docker-self-hosted.md)
- [Ingestor API](api-ingestor.md)
- [Notebook: Document continuous ingestion from object storage](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_event_ingest.ipynb)
- [Example: `examples/rag_event_ingest/`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/examples/rag_event_ingest/) — Kafka consumer and `deploy/docker-compose.yaml`

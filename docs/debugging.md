<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Pipeline Debugging Guide for NVIDIA RAG Blueprint

This guide provides comprehensive debugging guidance for the [NVIDIA RAG Blueprint](readme.md), helping you verify that your deployment is working correctly and troubleshoot issues when they arise.

:::{note}
This guide is mostly intended for developers who are working with docker setup.
:::

## How to Verify Your RAG System is Running Correctly

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), you need to verify that all components are healthy and functioning properly.

### Get detailed health status of all dependencies

```bash
curl -X GET "http://localhost:8081/v1/health?check_dependencies=true"  # RAG Server
curl -X GET "http://localhost:8082/v1/health?check_dependencies=true"  # Ingestor Server
```

- Ensure `"status":"healthy"` is shown for all dependent services in the response
- Ensure correct depedencies are listed. For example if you are using a different model for LLM ensure the model name is correct and the correct endpoint for cloud (NVIDIA-hosted) or local (self-hosted) appears in the response schema.
- If the service names do not match your expectations, ensure the correct configurations are loaded in `deploy/compose/.env` file and you have sourced it properly before starting the containers.
- To understand the set envs inside the container, [enable debug mode](#how-to-enable-advanced-debugging) and inspect the logs.

## How to Verify Your Ingestion Pipeline is Working

The ingestion pipeline processes documents and stores them in the vector database. Follow these steps to ensure it's functioning correctly.

### 1. Check if All Required Ingestion Services are Running

After starting your ingestion containers, verify these core services are healthy:

**Required Core Services:**
- `ingestor-server` (Port 8082) - Main ingestion API
- `nv-ingest-ms-runtime` (Port 7670) - Document processing engine
- `milvus` (Port 19530) - Vector database
- `redis` (Port 6379) - Task queue

### 2. Verify Container Status After Deployment

```bash
# List all running containers to see their status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check specifically for ingestion-related containers
docker ps | grep -E "(ingestor-server|nv-ingest|nemoretriever-embedding|milvus|redis)"
```

   *Example Output*

   ```output
   NAMES                                   STATUS
   compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
   ingestor-server                         Up 5 minutes
   compose-redis-1                         Up 5 minutes
   rag-frontend                            Up 9 minutes
   rag-server                              Up 9 minutes
   milvus-standalone                       Up 36 minutes
   milvus-minio                            Up 35 minutes (healthy)
   milvus-etcd                             Up 35 minutes (healthy)
   nemoretriever-ranking-ms                Up 38 minutes (healthy)
   compose-page-elements-1                 Up 38 minutes
   compose-paddle-1                        Up 38 minutes
   compose-graphic-elements-1              Up 38 minutes
   compose-table-structure-1               Up 38 minutes
   nemoretriever-embedding-ms              Up 38 minutes (healthy)
   nim-llm-ms                              Up 38 minutes (healthy)
   ```



### 3. Test Ingestion Service Health

```bash
# Check ingestor server health with all dependencies
curl -X GET "http://localhost:8082/v1/health?check_dependencies=true" | jq

# Verify NV-Ingest runtime is ready for processing
curl -X GET "http://localhost:7670/v1/health/ready"

# Check embedding service is responding
curl -X GET "http://localhost:9080/v1/health/ready"
```

### 4. Test Document Upload Functionality

```bash
# Test document upload to verify end-to-end ingestion
curl -X POST "http://localhost:8082/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test_document.pdf" \
  -F "data={\"collection_name\":\"test_collection\"}"
```

If ingestion is not working refer common debugging steps [here](#what-to-do-when-ingestion-fails).

## How to Verify Your Retrieval Pipeline is Working

The retrieval pipeline processes user queries and generates responses. Follow these steps to ensure it's functioning correctly.

### 1. Check if All Required Retrieval Services are Running

After starting your RAG containers, verify these core services are healthy:

**Required Core Services:**
- `rag-server` (Port 8081) - Main RAG API orchestrator
- `milvus` (Port 19530) - Vector database

### 2. Test Retrieval Service Health

```bash
# Check RAG server health with all dependencies
curl -X GET "http://localhost:8081/v1/health?check_dependencies=true" | jq

# Verify LLM service is ready for inference if deployed locally (self-hosted)
curl -X GET "http://localhost:8999/v1/health/ready"

# Check ranking service is responding if deployed locally (self-hosted)
curl -X GET "http://localhost:1976/v1/health/ready"
```

### 3. Test Query Processing Functionality

```bash
# Test query endpoint to verify end-to-end retrieval
curl -X POST "http://localhost:8081/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the main topic of the document?"
      }
    ],
    "use_knowledge_base": true,
    "collection_names": ["test_data"] # Ensure this collection exists
  }'
```

If retrieval is not working refer common debugging steps [here](#what-to-do-when-retrieval-fails).

## What to Do When Ingestion Fails

If your document ingestion is not working properly, follow these steps to identify and resolve the issue.

### 1. Check Container Logs for Error Messages

Start by examining the logs of key ingestion services to identify the specific error:

```bash
# Check ingestor server logs for API errors
docker logs ingestor-server --tail 100

# Check NV-Ingest runtime logs for processing errors
docker logs nv-ingest-ms-runtime --tail 100

# Check embedding service logs for model issues
docker logs nemoretriever-embedding-ms --tail 100
```

### 2. Common Ingestion Problems and Solutions

**File Upload Failures:**
- Check file size limits (400MB max per file)
- Verify file format is supported
- Check disk space availability

**Vector Database Connection Issues:**
```bash
# Check Milvus connectivity
curl -X GET "http://localhost:9091/healthz"

# Check Milvus logs for database errors
docker logs milvus-standalone --tail 50
```

**Embedding Service Issues:**
```bash
# Check embedding service logs
docker logs nemoretriever-embedding-ms --tail 100

# Verify GPU availability and memory
nvidia-smi
```

**NV-Ingest Processing Errors:**
```bash
# Check NV-Ingest logs for processing errors
docker logs nv-ingest-ms-runtime --tail 200 | grep -i error

# Check Redis connectivity for task queue
docker logs redis --tail 50
```

### 3. How to Verify Your Documents Were Successfully Ingested

**Check Collection Status:**
```bash
# List collections in Milvus to verify data was stored
curl -X GET "http://localhost:8082/collections"

# Check ingestor-server logs for successful ingestion
docker logs ingestor-server
```


## What to Do When Retrieval Fails

If your query processing is not working properly, follow these steps to identify and resolve the issue.

### 1. Check Container Logs for Error Messages

Start by examining the logs of key retrieval services:

```bash
# Check RAG server logs for API errors
docker logs rag-server --tail 100

# Check LLM service logs for model issues
docker logs nim-llm-ms --tail 100

# Check ranking service logs for reranking errors
docker logs nemoretriever-ranking-ms --tail 100
```

### 2. Common Retrieval Problems and Solutions

**Query Processing Failures:**
- Check query format and parameters
- Verify collection name exists
- Check vector database connectivity
- LLM service crashed during startup

**LLM Service Issues:**
```bash
# Check LLM service health
curl -X GET "http://localhost:8999/v1/health/ready"

# Check GPU memory usage
nvidia-smi


# Check NIM LLM container logs
docker logs -f nim-llm-ms
```

**Vector Search Issues:**

Delete the existing volumes directory and retry.

```bash
sudo rm -rf deploy/compose/volumes
```

## How to Enable Advanced Debugging

When basic troubleshooting doesn't resolve your issue, enable detailed logging and monitoring.

### 1. Enable Detailed Logging

**Set Debug Log Level:**
```bash
# For ingestor server
export LOGLEVEL=DEBUG
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --no-deps ingestor-server

# For RAG server
export LOGLEVEL=DEBUG
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --no-deps rag-server
```

### 2. Monitor System Resources

```bash
# Check GPU usage and memory
nvidia-smi

# Check container memory and CPU usage
docker stats

# Check disk space availability
df -h
```

## Step-by-Step Troubleshooting Checklist

### For Ingestion Issues:
- [ ] All required containers are running
- [ ] Vector database is accessible
- [ ] Embedding service is healthy
- [ ] File format is supported
- [ ] Sufficient disk space available
- [ ] GPU resources are available
- [ ] Check container logs for specific errors

### For Retrieval Issues:
- [ ] RAG server is running and healthy
- [ ] LLM service is accessible
- [ ] Vector database contains data
- [ ] Collection name is correct
- [ ] Query format is valid
- [ ] Check service logs for errors
- [ ] Verify GPU memory availability

### For Quality Issues:
- [ ] Reranker is enabled and working
- [ ] Top-K values are appropriate
- [ ] Collection has sufficient data
- [ ] Query rewriting is configured correctly



## Related Topics

- [Troubleshooting](troubleshooting.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Changelog](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CHANGELOG.md)

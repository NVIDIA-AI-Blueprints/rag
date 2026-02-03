<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Migration Guide: Milvus to Oracle AI Database 26ai

This guide covers migrating the NVIDIA RAG Blueprint from Milvus to Oracle AI Database 26ai as the vector database. Oracle 26ai is now the **default** vector store for this blueprint.

## Overview

Oracle AI Database 26ai is Oracle's next-generation AI-native database that integrates vector search capabilities directly into the database engine. Key benefits include:

- **Native VECTOR data type** - Store and query vectors alongside relational data
- **IVF and HNSW indexes** - CPU-optimized vector indexes for fast similarity search
- **Hybrid search** - Combine vector similarity with Oracle Text full-text search
- **No separate vector DB** - Unified database for all data types
- **Enterprise features** - Built-in security, scalability, and high availability

## Prerequisites

Before migrating, ensure you have:

1. **Oracle 26ai database instance** - Either:
   - Oracle Cloud Autonomous Database with AI Vector Search
   - On-premises Oracle 26ai installation
   - Docker container for development (Oracle Free tier)

2. **Database user with privileges**:
   - CREATE TABLE
   - CREATE INDEX
   - UNLIMITED TABLESPACE (or appropriate quota)

3. **Network connectivity** from RAG servers to Oracle database

4. **Python environment** with Oracle dependencies

## Step 1: Install Oracle Dependencies

Install the Oracle optional dependencies:

```bash
# Using pip
pip install nvidia_rag[oracle]

# Using uv
uv sync --extra oracle
```

This installs:
- `oracledb>=2.0.0` - Oracle Database Python driver (thin client, no Oracle Client needed)
- `langchain-community>=0.4` - LangChain integration with OracleVS

## Step 2: Configure Environment Variables

Set the following environment variables for Oracle connection:

```bash
# Vector store selection (oracle is now default)
export APP_VECTORSTORE_NAME=oracle

# Oracle connection credentials
export ORACLE_USER=rag_user
export ORACLE_PASSWORD=your_secure_password
export ORACLE_DSN=hostname:1521/service_name

# Optional: Vector index configuration
export ORACLE_VECTOR_INDEX_TYPE=IVF    # IVF (default) or HNSW
export ORACLE_DISTANCE_METRIC=COSINE   # COSINE (default), L2, DOT, MANHATTAN

# Optional: Enable hybrid search (vector + text)
export APP_VECTORSTORE_SEARCH_TYPE=hybrid
```

### Connection String Formats

Oracle DSN can be specified in multiple formats:

```bash
# Easy Connect format
export ORACLE_DSN=hostname:1521/service_name

# Easy Connect Plus with options
export ORACLE_DSN=hostname:1521/service_name?connect_timeout=30

# TNS alias (requires tnsnames.ora)
export ORACLE_DSN=mydb_alias
```

## Step 3: Prepare Oracle Database

Connect to your Oracle database as a DBA and create the RAG user:

```sql
-- Create user
CREATE USER rag_user IDENTIFIED BY your_secure_password;

-- Grant privileges
GRANT CONNECT, RESOURCE TO rag_user;
GRANT CREATE TABLE TO rag_user;
GRANT CREATE INDEX TO rag_user;
GRANT UNLIMITED TABLESPACE TO rag_user;

-- For hybrid search (Oracle Text)
GRANT CTXAPP TO rag_user;
```

### Verify Vector Search Support

Ensure your Oracle 26ai instance has vector search enabled:

```sql
-- Check database version (should be 26.x)
SELECT * FROM V$VERSION;

-- Verify VECTOR data type is available
SELECT VECTOR('[1.0, 2.0, 3.0]', 3, FLOAT32) FROM DUAL;
```

## Step 4: Deploy Services

### Option A: Using Docker Compose (Development)

Start the Oracle container for development:

```bash
# Set Oracle password
export ORACLE_PASSWORD=oracle123

# Start Oracle 26ai container
docker compose -f deploy/compose/vectordb.yaml --profile oracle up -d

# Wait for Oracle to be ready (first startup takes ~5 minutes)
docker logs -f oracle-26ai
```

Then start the RAG services:

```bash
# Source environment
source deploy/compose/.env

# Start RAG and Ingestor servers
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

### Option B: External Oracle Database (Production)

For production, connect to your existing Oracle 26ai instance:

```bash
# Set connection to your Oracle instance
export ORACLE_USER=rag_user
export ORACLE_PASSWORD=your_secure_password
export ORACLE_DSN=your-oracle-host:1521/your_service

# Source other environment variables
source deploy/compose/.env

# Start RAG services
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

## Step 5: Re-ingest Documents

**Important**: Documents must be re-ingested when switching vector databases. Data does not automatically migrate from Milvus to Oracle.

1. Access the RAG UI at `http://localhost:8090`
2. Create a new collection
3. Upload your documents
4. Verify ingestion via health check:

```bash
curl -X GET 'http://localhost:8082/v1/health?check_dependencies=true' -H 'accept: application/json'
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_VECTORSTORE_NAME` | Vector store type | `oracle` |
| `ORACLE_USER` | Database username | - |
| `ORACLE_PASSWORD` | Database password | - |
| `ORACLE_DSN` | Connection DSN | - |
| `ORACLE_VECTOR_INDEX_TYPE` | Index type: `IVF` or `HNSW` | `IVF` |
| `ORACLE_DISTANCE_METRIC` | Distance: `COSINE`, `L2`, `DOT`, `MANHATTAN` | `COSINE` |
| `APP_VECTORSTORE_SEARCH_TYPE` | Search type: `dense` or `hybrid` | `dense` |

### Vector Index Types

**IVF (Inverted File Index)** - Default, recommended for CPU deployment:
- Good balance of speed and accuracy
- Lower memory usage
- Best for large-scale deployments

**HNSW (Hierarchical Navigable Small World)**:
- Higher accuracy
- Faster query time
- Higher memory usage
- Best for smaller datasets with high accuracy requirements

### Distance Metrics

| Metric | Use Case |
|--------|----------|
| `COSINE` | Default, best for normalized embeddings (most NLP models) |
| `L2` | Euclidean distance, good for image embeddings |
| `DOT` | Dot product, fast but requires normalized vectors |
| `MANHATTAN` | L1 distance, robust to outliers |

## Hybrid Search Configuration

Hybrid search combines vector similarity search with Oracle Text full-text search for improved relevance:

```bash
export APP_VECTORSTORE_SEARCH_TYPE=hybrid
```

This creates both:
- Vector index (IVF/HNSW) on the `vector` column
- Oracle Text index (CONTEXT) on the `text` column

Queries will combine semantic similarity with keyword matching.

## Rollback to Milvus

To switch back to Milvus:

```bash
# Change vector store
export APP_VECTORSTORE_NAME=milvus

# Start Milvus containers
docker compose -f deploy/compose/vectordb.yaml up -d

# Restart RAG services
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

Note: You will need to re-ingest documents into Milvus.

## Troubleshooting

### Connection Errors

**ORA-01017: invalid username/password; logon denied**
- Verify `ORACLE_USER` and `ORACLE_PASSWORD` are correct
- Check if the user exists and is not locked

**ORA-12541: TNS:no listener**
- Verify `ORACLE_DSN` format and hostname
- Check network connectivity: `telnet hostname 1521`
- Ensure Oracle listener is running

**ORA-12514: TNS:listener does not currently know of service requested**
- Verify the service name in `ORACLE_DSN`
- List available services: `lsnrctl services`

### Vector Operations

**ORA-51801: VECTOR data type not supported**
- Ensure you're using Oracle 26ai (not an older version)
- Check `COMPATIBLE` parameter is set to 23.4.0 or higher

**Vector dimension mismatch**
- Ensure your embedding model dimension matches the collection
- Default dimension is 2048 (for NeMo Retriever embeddings)
- Check with: `SELECT VECTOR_DIMS(vector) FROM your_table WHERE ROWNUM = 1`

### Performance Issues

**Slow ingestion**
- Increase batch size in `oracle_vdb.py`
- Consider disabling indexes during bulk load, then rebuild
- Use parallel DML: `ALTER SESSION ENABLE PARALLEL DML`

**Slow queries**
- Ensure vector index exists: `SELECT * FROM USER_INDEXES WHERE INDEX_TYPE = 'VECTOR'`
- Rebuild index if needed: `ALTER INDEX idx_name REBUILD`
- Check index statistics: `DBMS_STATS.GATHER_INDEX_STATS`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   NVIDIA RAG Blueprint                   │
├─────────────────────────────────────────────────────────┤
│  RAG Server / Ingestor Server                           │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────┐                                    │
│  │   OracleVDB     │ ◄── VDBRagIngest base class        │
│  │   (oracle_vdb)  │                                    │
│  └────────┬────────┘                                    │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │   oracledb      │    │  LangChain      │            │
│  │   (thin client) │    │  OracleVS       │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                   │
│           ┌─────────────────────┐                       │
│           │  Oracle 26ai DB     │                       │
│           │  - VECTOR columns   │                       │
│           │  - IVF indexes      │                       │
│           │  - Oracle Text      │                       │
│           └─────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Change the Vector Database](change-vectordb.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [Troubleshoot](troubleshooting.md)

## External Resources

- [Oracle AI Vector Search Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)
- [Oracle Database 26ai Release Notes](https://docs.oracle.com/en/database/oracle/oracle-database/26/)
- [LangChain OracleVS Integration](https://python.langchain.com/docs/integrations/vectorstores/oracle)
- [oracledb Python Driver](https://python-oracledb.readthedocs.io/)

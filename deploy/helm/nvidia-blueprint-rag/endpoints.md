# Endpoints

This document describes the configurable endpoints used by the RAG server and its components. These endpoints can be configured to use NVIDIA-hosted models, self-hosted models, OSS components, or any other reachable endpoint.

## Core Service Endpoints

### Vector Store
- **APP_VECTORSTORE_URL**: URL for the vector store service (default: "http://milvus:19530")
- **APP_VECTORSTORE_NAME**: Type of vector store (default: "milvus")
- **APP_VECTORSTORE_SEARCHTYPE**: Type of vector store search (default: "dense")

### Object Storage
- **MINIO_ENDPOINT**: MinIO service endpoint for storing multimodal content (default: "rag-minio:9000")

## Model Service Endpoints

### LLM Model
- **APP_LLM_SERVERURL**: URL for the LLM model service (default: "nim-llm:8000")
- **APP_LLM_MODELNAME**: Name of the LLM model (default: "nvidia/llama-3.3-nemotron-super-49b-v1.5")

### Query Rewriter Model
- **APP_QUERYREWRITER_SERVERURL**: URL for the query rewriter model service (default: "nim-llm:8000")
- **APP_QUERYREWRITER_MODELNAME**: Name of the query rewriter model (default: "nvidia/llama-3.3-nemotron-super-49b-v1.5")

### Embedding Model
- **APP_EMBEDDINGS_SERVERURL**: URL for the embedding model service (default: "nemo-retriever-embedding-ms:8000")
- **APP_EMBEDDINGS_MODELNAME**: Name of the embedding model (default: "nvidia/llama-3.2-nv-embedqa-1b-v2")

### Reranking Model
- **APP_RANKING_SERVERURL**: URL for the ranking model service (default: "nemo-retriever-reranking-ms:8000")
- **APP_RANKING_MODELNAME**: Name of the ranking model (default: "nvidia/llama-3.2-nv-rerankqa-1b-v2")

### Reflection Model
- **REFLECTION_LLM_SERVERURL**: URL for the reflection LLM service (default: "nim-llm:8000")
- **REFLECTION_LLM**: Name of the reflection model (default: "nvidia/llama-3.3-nemotron-super-49b-v1.5")

## Frontend Endpoints

### API Endpoints
- **VITE_API_CHAT_URL**: Base URL for chat API endpoints (default: "http://rag-server:8081/v1")
- **VITE_MODEL_NAME**: Base URL for vector database API endpoints (default: "http://ingestor-server:8082/v1")

### Model Configuration
- **NEXT_PUBLIC_MODEL_NAME**: Name of the LLM model used in the frontend (default: "nvidia/llama-3.3-nemotron-super-49b-v1.5")
- **VITE_EMBEDDING_MODEL**: Name of the embedding model used in the frontend (default: "nvidia/llama-3.2-nv-embedqa-1b-v2")
- **VITE_RERANKER_MODEL**: Name of the reranker model used in the frontend (default: "nvidia/llama-3.2-nv-rerankqa-1b-v2")

## Monitoring and Tracing Endpoints

### OpenTelemetry
- **APP_TRACING_OTLPHTTPENDPOINT**: HTTP endpoint for OpenTelemetry traces (default: "http://rag-opentelemetry-collector:4318/v1/traces")
- **APP_TRACING_OTLPGRPCENDPOINT**: gRPC endpoint for OpenTelemetry traces (default: "grpc://rag-opentelemetry-collector:4317")

## Ingestor Service Endpoints

### NV-Ingest
- **APP_NVINGEST_MESSAGECLIENTHOSTNAME**: Hostname for NV-Ingest message client (default: "rag-nv-ingest")
- **APP_NVINGEST_MESSAGECLIENTPORT**: Port for NV-Ingest message client (default: "7670")
- **APP_NVINGEST_CAPTIONENDPOINTURL**: Endpoint URL for captioning model (default: "")

### Redis
- **REDIS_HOST**: Redis host (default: "rag-redis-master")
- **REDIS_PORT**: Redis port (default: "6379")
- **REDIS_DB**: Redis database number (default: "0")

## Configuration Notes

1. All model endpoints can be configured to use NVIDIA NIM preview endpoints by leaving the URL empty ("").
2. For local deployments, endpoints should be configured to point to the appropriate service names within the cluster.
3. External endpoints can be configured by providing the full URL including protocol and port.
4. The configuration supports both HTTP and gRPC protocols where applicable.
5. Frontend endpoints are configured through environment variables and can be customized based on your deployment setup.
6. The frontend service is exposed on port 3000 by default and can be configured to use different service types (NodePort, ClusterIP, LoadBalancer).

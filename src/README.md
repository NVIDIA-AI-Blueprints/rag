# Source Code Overview

The `src/nvidia_rag` directory contains the core implementation of the project. Below is an updated overview of the subdirectories and files, along with their purposes:

## Directory Structure

### `ingestor_server/`
This directory contains the implementation for the ingestion server, which handles document ingestion and related tasks.

- **`__init__.py`**: Initializes the module.
- **`Dockerfile`**: Docker configuration for the ingestion server.
- **`main.py`**: Entry point for the ingestion server.
- **`nvingest.py`**: Handles NVIDIA-specific ingestion logic.
- **`server.py`**: Defines the FastAPI server for handling ingestion-related APIs.
- **`task_handler.py`**: Manages ingestion tasks, including task submission and handling.

### `rag_server/`
This directory contains the implementation of the RAG (Retrieval-Augmented Generation) server.

- **`__init__.py`**: Initializes the module.
- **`Dockerfile`**: Docker configuration for the RAG server.
- **`health.py`**: Implements health check APIs for the server.
- **`main.py`**: Entry point for the RAG server.
- **`prompt.yaml`**: Configuration for prompts used in RAG workflows.
- **`reflection.py`**: Implements reflection logic for context relevance and response groundedness.
- **`response_generator.py`**: Handles response generation logic.
- **`server.py`**: Defines the FastAPI server for RAG-related APIs.
- **`tracing.py`**: Enables tracing and instrumentation for the RAG server.
- **`validation.py`**: Implements validation logic for RAG responses.
- **`vlm.py`**: Implements Vision Language Model integration.

### `utils/`
This directory contains utility modules used across the project.

- **`__init__.py`**: Initializes the module.
- **`common.py`**: Provides common utility functions.
- **`configuration.py`**: Handles configuration management.
- **`configuration_wizard.py`**: Provides a wizard for setting up configurations.
- **`embedding.py`**: Implements embedding-related utilities.
- **`llm.py`**: Contains utilities for working with large language models.
- **`minio_operator.py`**: Provides utilities for interacting with MinIO object storage.
- **`reranker.py`**: Implements reranking logic for retrieved documents.
- **`vectorstore.py`**: Manages vector storage for embeddings.

### `observability/`
This directory provides tools for monitoring and instrumentation.

- **`langchain_callback_handler.py`**: Implements callback handlers for Langchain SDK.
- **`langchain_instrumentor.py`**: Provides OpenTelemetry instrumentation for Langchain.
- **`otel_metrics.py`**: Defines metrics collection using OpenTelemetry.

### `__init__.py`
This file initializes the `src` module, making it importable as a package.

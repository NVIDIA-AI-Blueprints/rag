<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deploy NV-Ingest Standalone for NVIDIA RAG Blueprint

This guide explains how to deploy and use NV-Ingest as a standalone service for [NVIDIA RAG Blueprint](readme.md) without deploying the full ingestor server. This is useful when you want to ingest documents directly using Python scripts.

For more details and advanced usage, refer to:
- [NVIDIA/nv-ingest repository](https://github.com/NVIDIA/nv-ingest)
- [Official NV-Ingest Quickstart Guide](https://github.com/NVIDIA/nv-ingest/blob/main/docs/docs/extraction/quickstart-guide.md)

## Limitations

When using NV-Ingest in standalone mode, consider the following limitations:

1. **Citations Disabled**: The RAG server's citation feature will be disabled for documents ingested through standalone NV-Ingest. This is because the citation metadata requires additional processing that is handled by the full ingestor server.

2. **No Web UI**: The standalone deployment does not include the web-based upload interface. All document ingestion must be done through Python scripts.

3. **Manual Collection Management**: Collection management (creation, deletion, etc.) must be handled manually through the Python client, as the management interface is part of the full ingestor server.

## Prerequisites

1. Ensure you have Docker and Docker Compose installed. For details, refer to [Get Started](deploy-docker-self-hosted.md).
2. Have Python 3.12 or later installed
   > ℹ️ If you're using **Python 3.13**, make sure you've `python3.13-dev` installed.
3. [Get an API Key](api-key.md).
4. Install a package manager (either uv or pip):
```bash
# Option 1: Install uv (recommended for faster installation)
pip install uv

# Option 2: Use pip (comes with Python)
# No additional installation needed
```

## Deployment Steps using Docker

1. Follow the steps in the deployment guide for [Self-Hosted Models](deploy-docker-self-hosted.md) or [NVIDIA-Hosted Models](deploy-docker-nvidia-hosted.md), but skip the `ingestor-server` deployment.

The key difference while deploying from `docker-compose-ingestor-server.yaml` file deploy only `nv-ingest-ms-runtime` and `redis` using following command:
```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d nv-ingest-ms-runtime redis
```

2. Create a Python virtual environment and install required packages:

Using uv (recommended):
```bash
# Create and activate virtual environment
uv venv nv-ingest-env
source nv-ingest-env/bin/activate  # On Linux/Mac
# OR
.\nv-ingest-env\Scripts\activate  # On Windows

# Install required packages
uv pip install nv-ingest-api==25.8.0rc6 nv-ingest-client==25.8.0rc6 tritonclient==2.57.0 pymilvus==2.5.8 pymilvus[model] pymilvus[bulk-writer]
```

Using pip:
```bash
# Create and activate virtual environment
python -m venv nv-ingest-env
source nv-ingest-env/bin/activate  # On Linux/Mac
# OR
.\nv-ingest-env\Scripts\activate  # On Windows

# Install required packages
pip install nv-ingest-api==25.8.0rc6 nv-ingest-client==25.8.0rc6 tritonclient==2.57.0 pymilvus==2.5.8 pymilvus[model] pymilvus[bulk-writer]
```

3. Create a Python script to ingest documents. Here's a placeholder script that you can customize:

```python
# ingest_documents.py
from nv_ingest_client.client import Ingestor, NvIngestClient

FILEPATHS = [
    "data/multimodal/multimodal_test.pdf",
    "data/multimodal/woods_frost.pdf"
]

COLLECTION_NAME = "multimodal_data_nvingest"

MILVUS_URI = "http://localhost:19530"
MINIO_ENDPOINT = "localhost:9010"

# Server Mode (Create NV-Ingest client)
client = NvIngestClient(
    message_client_hostname="localhost",
    message_client_port=7670
)

ingestor = Ingestor(client=client)

ingestor = ingestor.files(FILEPATHS)

ingestor = ingestor.extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=False,
                text_depth="page",
                table_output_format="markdown"
            )
ingestor = ingestor.split(
                tokenizer="intfloat/e5-large-unsupervised",
                chunk_size=51,
                chunk_overlap=15,
                params={"split_source_types": ["PDF" ,"text", "html"]},
            )

ingestor = ingestor.embed()

ingestor = ingestor.vdb_upload(
                collection_name=COLLECTION_NAME,
                milvus_uri=MILVUS_URI,
                minio_endpoint=MINIO_ENDPOINT,
                sparse=False,
                enable_images=True,
                recreate=False,
                dense_dim=2048,
                stream=False
            )

results, failures = ingestor.ingest(show_progress=True, return_failures=True)
```

4. Run your ingestion script:
```bash
python ingest_documents.py
```
Post ingestion, you can use the same rag-server to perform inference on the collection name that was used during ingestion.

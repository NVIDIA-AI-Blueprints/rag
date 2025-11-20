<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint with Docker (NVIDIA-Hosted Models)

Use this documentation to deploy the [NVIDIA RAG Blueprint](readme.md) with Docker Compose for a single node deployment, and using NVIDIA-hosted models for testing and experimenting.
For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

:::{tip}
If you want to run the RAG Blueprint with [NVIDIA AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html), use [Quickstart for NVIDIA AI Workbench](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/deploy/workbench/README.md).
:::

:::{note}
When using NVIDIA-hosted endpoints, you might encounter rate limiting with larger file ingestions (>10 files). For details, see [Troubleshoot](troubleshooting.md).
:::



## Prerequisites

1. [Get an API Key](api-key.md).

2. Install Docker Engine. For more information, see [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

3. Install Docker Compose. For more information, see [install the Compose plugin](https://docs.docker.com/compose/install/linux/).

   a. Ensure the Docker Compose plugin version is 2.29.1 or later.

   b. After you get the Docker Compose plugin installed, run `docker compose version` to confirm.

4. To pull images required by the blueprint from NGC, you must first authenticate Docker with nvcr.io. Use the NGC API Key you created in the first step.

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

5. Some containers with are enabled with GPU acceleration, such as Milvus and NVIDIA NIMS deployed on-prem. To configure Docker for GPU-accelerated containers, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), the NVIDIA Container Toolkit.



## Start services using NVIDIA-hosted models

Use the following procedure to start all containers needed for this blueprint.

1. Open `deploy/compose/.env` and uncomment the section `Endpoints for using cloud NIMs`. Then set the environment variables by running the following code.

   ```bash
   source deploy/compose/.env
   ```


2. Start the vector db containers from the repo root.

   ```bash
   docker compose -f deploy/compose/vectordb.yaml up -d
   ```


3. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

   You can check the status of the ingestor-server by running the following code.

   ```bash
   curl -X 'GET' 'http://workstation_ip:8082/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

    You should see output similar to the following.

    ```bash
    {
        "message": "Service is up.",
        "databases": [
            ...
        ],
        "object_storage": [
            ...
        ],
        "nim": [
            {
                "service": "Embeddings",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            },
            {
                "service": "Summary LLM",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            },
            {
                "service": "Caption Model",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            }
        ],
        "processing": [
            {
                "service": "NV-Ingest",
                "status": "healthy",
                ...
            }
        ],
        "task_management": [
            {
                "service": "Redis",
                "status": "healthy",
                ...
            }
        ]
    }
    ```


4. Start the rag containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   You can check the status of the rag-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

    You should see output similar to the following.

    ```bash
    {
        "message": "Service is up.",
        "databases": [
            ...
        ],
        "object_storage": [
            ...
        ],
        "nim": [
            {
                "service": "LLM",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            },
            {
                "service": "Embeddings",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            },
            {
                "service": "Ranking",
                "status": "healthy",
                "message": "Using NVIDIA API Catalog",
                ...
            }
        ]
    }
    ```


5. Check the status of the deployment by running the following code.

   ```bash
   docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

    You should see output similar to the following. Confirm all the following containers are running.

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
   ```


## Experiment with the Web User Interface

After the RAG Blueprint is deployed, you can use the RAG UI to start experimenting with it.

1. Open a web browser and access the RAG UI. You can start experimenting by uploading docs and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).



## Experiment with the Ingestion API Usage Notebook

After the RAG Blueprint is deployed, you can use the Ingestion API Usage notebook to start experimenting with it. For details, refer to [Experiment with the Ingestion API Usage Notebook](notebooks.md#experiment-with-the-ingestion-api-usage-notebook).



## Shut down services

1. To stop all running services.
    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down
    ```


## Advanced Deployment Considerations

After the first time you deploy the RAG Blueprint successfully, you can consider the following advanced deployment options:

- For information about advanced settings, see [Best Practices for Common Settings](accuracy_perf.md).

- To turn on recommended configurations for accuracy optimization, run the following code:

   ```bash
   source deploy/compose/accuracy_profile.env
   ```

- To turn on recommended configurations for performance optimization, run the following code:

   ```bash
   source deploy/compose/perf_profile.env
   ```

- If you don't have a GPU available, you can switch to CPU-only Milvus by following the instructions in [milvus-configuration.md](./milvus-configuration.md).

- If you have a requirement to build the NVIDIA Ingest runtime container from source, you can do it by following instructions [here](https://github.com/NVIDIA/nv-ingest).



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

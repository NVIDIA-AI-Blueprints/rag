<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Get Started With NVIDIA RAG Blueprint

Use the following documentation to get started quickly with the [NVIDIA RAG Blueprint](readme.md).
In this walkthrough you deploy the NVIDIA RAG Blueprint with Docker Compose for a single node deployment, and using self-hosted on-premises models.
For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

:::{tip}
If you want to run the RAG Blueprint with [NVIDIA AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html), use [Quickstart for NVIDIA AI Workbench](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/deploy/workbench/README.md).
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

5. Containers that are enabled with GPU acceleration, such as Milvus and NVIDIA NIMS, deployed on-prem. To configure Docker for GPU-accelerated containers, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

6. Ensure you meet [the hardware requirements](./support-matrix.md).


## Start services using self-hosted on-premises models

Use the following procedure to start all containers needed for this blueprint.

1. Create a directory to cache the models and export the path to the cache as an environment variable.

   ```bash
   mkdir -p ~/.cache/model-cache
   export MODEL_DIRECTORY=~/.cache/model-cache
   ```


2. Export all the required environment variables to use on-prem models. Verify that the section `Endpoints for using cloud NIMs` is commented in this file.

   ```bash
   source deploy/compose/.env
   ```


3. (For A100 SXM and B200 platforms only) Run the following code to allocate 2 available GPUs before you continue with the following steps.

   ```bash
   export LLM_MS_GPU_ID=1,2
   ```


4. List the available [model profiles](model-profiles.md) for your hardware by running the following code.

   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml run nim-llm list-model-profiles
   ```

   The output depends on your hardware. The following example output is for an H100-NVL with 1 GPU allocated.

   ```output
   MODEL PROFILES
   - Compatible with system and runnable:
   - d4910... (vllm-bf16-tp1-pp1-32c3...)
   - e2f00... (vllm)
   - e759b... (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e...)
   - 668b5... (tensorrt_llm)
   - 50e13... (sglang)
   ```


5. Using the list of [model profile](model-profiles.md) from the previous step, set the `NIM_MODEL_PROFILE`. It is ideal to select one of the `tensorrt_llm` profiles for best performance. Because of a known issue, vllm-based profiles are selected, so we recommend that you manually select a `tensorrt_llm` profile before you start the `nim-llm` service.

   ```bash
   export NIM_MODEL_PROFILE="......" # Populate your profile name as per hardware
   ```


6. Start all required NIMs by running the following code.

   :::{warning}
   Do not attempt this step unless you have completed the previous steps.
   :::

   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d
   ```

   The NIM LLM service can take 30 mins to start for the first time as the model is downloaded and cached. Subsequent deployments can take 2-5 minutes, depending on the GPU profile.

   :::{tip}
   The models are downloaded and cached in the path specified by `MODEL_DIRECTORY`.
   :::


7. Check the status of the deployment by running the following code. Wait until all services are up and the `nemoretriever-ranking-ms`, `nemoretriever-embedding-ms` and `nim-llm-ms`  NIMs are in healthy state before proceeding further.

     ```bash
     watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
     ```
    Your output should look similar to the following.

     ```output
        NAMES                                   STATUS

        nemoretriever-ranking-ms                Up 14 minutes (healthy)
        compose-page-elements-1                 Up 14 minutes
        compose-paddle-1                        Up 14 minutes
        compose-graphic-elements-1              Up 14 minutes
        compose-table-structure-1               Up 14 minutes
        nemoretriever-embedding-ms              Up 14 minutes (healthy)
        nim-llm-ms                              Up 14 minutes (healthy)
     ```


8. Start the vector db containers from the repo root.

   ```bash
   docker compose -f deploy/compose/vectordb.yaml up -d
   ```


9. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys them on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

   You can check the status of the ingestor-server and running the following code.

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
                ...
            },
            {
                "service": "Summary LLM",
                "status": "healthy",
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


10. Start the RAG containers from the repo root. This pulls the prebuilt containers from NGC and deploys them on your system.

    ```bash
    docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
    ```

    You can check the status of the rag-server by running the following code.

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
            ...
        },
        {
            "service": "Embeddings",
            "status": "healthy",
            ...
        },
        {
            "service": "Ranking",
            "status": "healthy",
            ...
        }
      ]
    }
    ```


11. Check the status of the deployment by running the following code.

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
    nemoretriever-ranking-ms                Up 38 minutes (healthy)
    compose-page-elements-1                 Up 38 minutes
    compose-paddle-1                        Up 38 minutes
    compose-graphic-elements-1              Up 38 minutes
    compose-table-structure-1               Up 38 minutes
    nemoretriever-embedding-ms              Up 38 minutes (healthy)
    nim-llm-ms                              Up 38 minutes (healthy)
    ```



## Experiment with the Web User Interface

After the RAG Blueprint is deployed, you can use the RAG UI to start experimenting with it.

1. Open a web browser and access the RAG UI. You can start experimenting by uploading docs and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).



## Experiment with the Ingestion API Usage Notebook

After the RAG Blueprint is deployed, you can use the Ingestion API Usage notebook to start experimenting with it. For details, refer to [Experiment with the Ingestion API Usage Notebook](notebooks.md#experiment-with-the-ingestion-api-usage-notebook).



## Shut down services

1. To stop all running services, run the following code.

    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/nims.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down
    ```


## Advanced Deployment Considerations

After the first time you deploy the RAG Blueprint successfully, you can consider the following advanced deployment options:

- For information about advanced settings, see [Best Practices for Common Settings](accuracy_perf.md).

- To turn on recommended configurations for accuracy optimized profile set additional configs by running the following code:

   ```bash
   source deploy/compose/accuracy_profile.env
   ```

- To turn on recommended configurations for performance optimized profile set additional configs by running the following code:

   ```bash
   source deploy/compose/perf_profile.env
   ```

- To start just the services specific to RAG or ingestion add the `--profile rag` or `--profile ingest` flag to the code. For example:

   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d --profile rag
   ```

- If you make code changes and want to redeploy services, add the --build flag to your code.  For example:

   ```bash
   docker compose -f deploy/compose/docker-compose-*-server.yaml up -d --build
   ```

- By default, GPU accelerated Milvus DB is deployed. You can choose the GPU ID to allocate by using the below env variable.

   ```bash
   VECTORSTORE_GPU_DEVICE_ID=0
   ```

- For improved accuracy, consider enabling reasoning mode. For details, refer to [Enable thinking](./enable-nemotron-thinking.md).

- To use NeMo Retriever OCR (Early Access) instead of Paddle OCR, refer to [NeMo Retriever OCR](nemoretriever-ocr.md).

- For advanced users who need direct filesystem access to extraction results, refer to [Ingestor Server Volume Mounting](mount-ingestor-volume.md).

- A single NVIDIA A100-80GB or H100-80GB, B200 GPU can be used to start non-LLM NIMs (nemoretriever-embedding-ms, nemoretriever-ranking-ms, and ingestion services like page-elements, ocr, graphic-elements, and table-structure) for ingestion and RAG workflows. You can control which GPU is used for each service by setting these environment variables in `deploy/compose/.env` file before launching:

   ```bash
   EMBEDDING_MS_GPU_ID=0
   RANKING_MS_GPU_ID=0
   YOLOX_MS_GPU_ID=0
   YOLOX_GRAPHICS_MS_GPU_ID=0
   YOLOX_TABLE_MS_GPU_ID=0
   OCR_MS_GPU_ID=0
   ```

- If the NIMs are deployed in a different workstation or outside the nvidia-rag docker network on the same system, replace the host address of the below URLs with workstation IPs.

   ```bash
   APP_EMBEDDINGS_SERVERURL="workstation_ip:8000"
   APP_LLM_SERVERURL="workstation_ip:8000"
   APP_RANKING_SERVERURL="workstation_ip:8000"
   OCR_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT="workstation_ip:8001"
   ```



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

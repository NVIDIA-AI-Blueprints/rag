<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# NeMo Guardrails Support in NVIDIA RAG Blueprint

This guide provides step-by-step instructions to enable NeMo Guardrails for the [NVIDIA RAG Blueprint](readme.md), enabling you to control and safeguard LLM interactions.

> [!WARNING]
>
> B200 GPUs are not supported for NeMo Guardrails for guardrails at input/output.
> For this feature, use H100 or A100 GPUs instead.


NeMo Guardrails is a framework that provides safety and security measures for LLM applications. When enabled, it provides:

- Content safety filtering
- Topic control to prevent off-topic conversations
- Jailbreak detection to prevent prompt attacks



## Prerequisites

1. Follow the prerequisites for your deployment method:

    - [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
    - [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
    <!-- - [Deploy with Helm](deploy-helm.md) -->



## Current Limitations

- Currently, Helm-based deployment is not supported for NeMo Guardrails.
- Currently, the Jailbreak detection model is not available.
- User queries which attempt to jailbreak the system (asking the bot to behave in a certain way) may not work as expected in the current version. These jailbreak attempts could be better addressed with the [NemoGuard-Jailbreak-Detect](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) NIM Microservice, which currently does not offer out-of-the-box support.
- Both the content-safety and topic-control models are trained on single-turn datasets, meaning they don't handle multi-turn conversations as effectively. When the bot combines multiple queries and previous context, it may inconsistently flag certain phrases as safe or unsafe.
- The current version of Guardrails is tuned to provide simple safe responses, such as "I'm sorry. I can't respond to that."



## Hardware Requirements

You need two extra GPUs (2 x H100 or 2 x A100) for deployment, as each model must be deployed on its own dedicated GPU - one for the Content Safety model and another for the Topic Control model.

The NeMo Guardrails models have specific hardware requirements:

- **Llama 3.1 NemoGuard 8B Content Safety Model**: Requires 48 GB of GPU memory. Refer to [Support Matrix](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/support-matrix.html).
- **Llama 3.1 NemoGuard 8B Topic Control Model**: Requires 48 GB of GPU memory. Refer to [Support Matrix](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/support-matrix.html).


NVIDIA developed and tested these microservices using H100 and A100 GPUs.



## Deployment Option 1: Self-Hosted Microservices (Default)

To deploy all guardrails services on your own dedicated hardware, use the following procedure.

1. The RAG Server must be running before you start NeMo Guardrails services.

    > [!Note]
    > For self-hosted deployment, the default NIM service must be up and running. 
    > If you're unable to run the NIM service locally, 
    > you can use NVIDIA's cloud-hosted LLM by exporting the NIM endpoint URL.
    > 
    > ```bash
    > # Use NVIDIA-hosted LLM
    > export NIM_ENDPOINT_URL=https://integrate.api.nvidia.com/v1
    > # Or provide your own custom NIM endpoint URL
    > # export NIM_ENDPOINT_URL=<your-custom-nim-endpoint-url>
    > ```

2. Set the environment variable to enable guardrails by running the following code.

    ```bash
    export ENABLE_GUARDRAILS=true
    export DEFAULT_CONFIG=nemoguard
    ```

3. After you update the environment variables, you must restart the RAG server by running the following code.

    ```bash
    docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
    ```

4. Create a directory for caching models by running the following code. Ensure that you create a different one than the one used by other models of this blueprint.

    ```bash
    mkdir -p ~/.cache/nemoguard-model-cache
    ```

5. Set the model directory path by running the following code.

    ```bash
    export MODEL_DIRECTORY=~/.cache/nemoguard-model-cache
    ```

6. Check your available GPUs and their IDs by running the following code.

   This displays all available GPUs with their IDs, memory usage, and utilization. 

    ```bash
    nvidia-smi
    ```

7. Use the information in the previous step to export specific GPU IDs for the guardrails services by running the following code.

    ```bash
    # By default, the services use GPU IDs 7 and 6
    # Set these to appropriate values based on your system configuration
    export CONTENT_SAFETY_GPU_ID=0  # Choose GPU ID for content safety model
    export TOPIC_CONTROL_GPU_ID=1   # Choose GPU ID for topic control model
    ```

    > [!Note]
    > Each model requires a dedicated GPU with at least 48GB of memory. Select GPUs with sufficient available memory.

8. Start the NeMo Guardrails service by running the following code.

    ```bash
    USERID=$(id -u) docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml up -d
    ```

    This command starts the following services:

    - NeMo Guardrails microservice
    - Content safety model
    - Topic control model


9. (Optional) The NemoGuard services might take several minutes to fully initialize. You can monitor their status by running the following code. 

    ```bash
    watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "nemoguard|guardrails"'
    ```

    You should see output similar to the following. Wait until all services appear as `healthy` before you proceed to the next step.

    ```
    llama-3.1-nemoguard-8b-topic-control    Up 5 minutes (healthy)
    llama-3.1-nemoguard-8b-content-safety   Up 5 minutes (healthy)
    nemo-guardrails-microservice            Up 4 minutes (healthy)
    ```


## Option 2: NVIDIA-Hosted Deployment

To deploy all guardrails services using NVIDIA-hosted models, use the following procedure.

1. The RAG Server must be running before you start NeMo Guardrails services.

2. Verify that the model names in the configuration file are correct by running the following code.

    ```bash
    cat deploy/compose/nemoguardrails/config-store/nemoguard_cloud/config.yml
    ```

    Ensure that the model names in this file match the models available in your NVIDIA API account. You might need to update these names based on the specific models that you have access to.

3. Enable guardrails by running the following code.

    ```bash
    # Set configuration for cloud deployment
    export ENABLE_GUARDRAILS=true
    export DEFAULT_CONFIG=nemoguard_cloud
    export NIM_ENDPOINT_URL=https://integrate.api.nvidia.com/v1
    ```

4. Start the Guardrails microservice by running the following code.

    ```bash
    docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml up -d --no-deps nemo-guardrails-microservice
    ```

5. Restart the RAG server by running the following code.

    ```bash
    docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
    ```



## Enable Guardrails from the UI or while sending API request

After the services are running, you can enable guardrails from the RAG UI:

1. Open the [RAG UI](user-interface.md).
2. Click **Settings**.
3. In the **Output Preferences** section, toggle **Guardrails** to on.

    <p align="left">
        <img src="assets/toggle_nemo_guardrails.png" >
    </p>

If you are using notebooks or APIs to interact directly with `rag-server`, set `enable_guardrails` to `True` in your /generate request payload.





## Troubleshooting

### GPU Device ID Issues

If you encounter GPU device errors, you can customize the GPU device IDs used by the guardrails services. By default, the services use GPUs 6 and 7, but you can set specific GPUs by setting these environment variables before starting the service:

```bash
# Specify which GPUs to use for guardrail services
export CONTENT_SAFETY_GPU_ID=0  # Default is GPU 0
export TOPIC_CONTROL_GPU_ID=1   # Default is GPU 1
```

This allows you to control which specific GPUs are assigned to each model in multi-GPU systems.

### Service Health Check

To verify if the guardrails services are running properly:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "guardrails|safety|topic"
```

```bash
nemo-guardrails-microservice            Up 19 minutes
llama-3.1-nemoguard-8b-topic-control    Up 19 minutes
llama-3.1-nemoguard-8b-content-safety   Up 19 minutes
```



## Related Topics

- [NeMo Guardrails](https://docs.nvidia.com/nemo-guardrails/)
- [NeMo Guardrails Microservice Overview](https://docs.nvidia.com/nemo/microservices/latest/guardrails/index.html)
- [Integrating with NemoGuard NIM Microservices](https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/integrate-nemoguard-nims.html)

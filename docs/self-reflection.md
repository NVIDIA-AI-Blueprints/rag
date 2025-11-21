<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Self-Reflection for NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) supports self-reflection capabilities to improve response quality through two key mechanisms:

1. Context Relevance Check: Evaluates and potentially improves retrieved document relevance
2. Response Groundedness Check: Ensures generated responses are well-grounded in the retrieved context

For steps to enable reflection in Helm deployment, refer to [Reflection Support via Helm Deployment](#reflection-support-via-helm-deployment).

## Configuration

Enable self-reflection by setting the following environment variables:

```bash
# Enable the reflection feature
ENABLE_REFLECTION=true

# Configure reflection parameters
MAX_REFLECTION_LOOP=3                    # Maximum number of refinement attempts (default: 3)
CONTEXT_RELEVANCE_THRESHOLD=1            # Minimum relevance score 0-2 (default: 1)
RESPONSE_GROUNDEDNESS_THRESHOLD=1        # Minimum groundedness score 0-2 (default: 1)
REFLECTION_LLM="nvidia/llama-3.3-nemotron-super-49b-v1.5"  # Model for reflection (default)
REFLECTION_LLM_SERVERURL="nim-llm:8000"  # Default on-premises endpoint for reflection LLM
```

The reflection feature supports the following deployment options:

- [On-Premises Deployment](#on-premises-deployment-recommended) (Recommended)
- [NVIDIA-Hosted Models](#nvidia-hosted-models-alternative) (Alternative)


## Prerequisites

1. [Get an API Key](api-key.md).

2. Install Docker Engine. For more information, see [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

3. Install Docker Compose. For more information, see [install the Compose plugin](https://docs.docker.com/compose/install/linux/).

   a. Ensure the Docker Compose plugin version is 2.29.1 or later.

   b. After you get the Docker Compose plugin installed, run `docker compose version` to confirm.

4. To pull images required by the blueprint from NGC, you must first authenticate Docker with nvcr.io. Use the NGC API Key you created in [Obtain an API Key](api-key.md).

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

5. Some containers with are enabled with GPU acceleration, such as Milvus and NVIDIA NIMS deployed on-prem. To configure Docker for GPU-accelerated containers, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), the NVIDIA Container Toolkit



## On-Premises Deployment (Recommended)

### On-Premises Deployment Prerequisites

1. Ensure you have completed the [general prerequisites](#prerequisites).

2. Verify you have sufficient GPU resources:
   - **Required**: 8x A100 80GB or H100 80GB GPUs for optimal latency-optimized deployment
   - For detailed GPU requirements and supported model configurations, refer to the [NVIDIA NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html).


### On-Premises Deployment Steps

1. Authenticate Docker with NGC using your NVIDIA API key:

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

2. Create a directory to cache the models:

   ```bash
   mkdir -p ~/.cache/model-cache
   export MODEL_DIRECTORY=~/.cache/model-cache
   ```

3. Start the RAG server with reflection enabled:

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

4. Verify all services are running:

   ```bash
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

5. Open the [RAG UI](user-interface.md) and test the reflection capability.


## NVIDIA-Hosted Models (Alternative)

If you don't have sufficient GPU resources for on-premises deployment, you can use NVIDIA's hosted models:

### NVIDIA-Hosted Models Prerequisites

1. Ensure you have completed the [general prerequisites](#prerequisites).


### NVIDIA-Hosted Models Deployment Steps

1. Set your NVIDIA API key as an environment variable:

   ```bash
   export NGC_API_KEY="nvapi-..."
   ```

2. Configure the environment to use NVIDIA hosted models:

   ```bash
   # Enable reflection feature
   export ENABLE_REFLECTION=true

   # Set empty server URL to use NVIDIA hosted API
   export REFLECTION_LLM_SERVERURL="nim-llm:8000"

   # Choose the reflection model (options below)
   export REFLECTION_LLM="nvidia/llama-3.3-nemotron-super-49b-v1.5"  # Default option
   # export REFLECTION_LLM="meta/llama-3.1-405b-instruct"  # Alternative option
   ```

3. Start the RAG server:

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

4. Verify the service is running:

   ```bash
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

5. Open the [RAG UI](user-interface.md) and test the reflection capability.


:::{note}
When using NVIDIA-hosted models, you must obtain an API key. See [Get an API Key](api-key.md) for instructions.
:::

## Reflection Support via Helm Deployment

You can enable self-reflection through Helm when you deploy the RAG Blueprint.

### Prerequisites

- Only on-premises reflection deployment is supported in Helm
- The model used is: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.

### Deployment Steps

1. Set the following environment variables in your `values.yaml` under `envVars`:

   ```yaml
   # === Reflection ===
   ENABLE_REFLECTION: "True"
   MAX_REFLECTION_LOOP: "3"
   CONTEXT_RELEVANCE_THRESHOLD: "1"
   RESPONSE_GROUNDEDNESS_THRESHOLD: "1"
   REFLECTION_LLM: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
   REFLECTION_LLM_SERVERURL: "nim-llm:8000"
   ```

2. Deploy the RAG Helm chart:

   Follow the steps from [Deploy with Helm](deploy-helm.md) and run:

   ```bash
   helm install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```

## How It Works

### Context Relevance Check

1. The system retrieves initial documents based on the user query
2. A reflection LLM evaluates document relevance on a 0-2 scale:
   - 0: Not relevant
   - 1: Somewhat relevant
   - 2: Highly relevant
3. If relevance is below threshold and iterations remain:
   - The query is rewritten for better retrieval
   - The process repeats with the new query
4. The most relevant context is used for response generation

### Response Groundedness Check

1. The system generates an initial response using retrieved context
2. The reflection LLM evaluates response groundedness on a 0-2 scale:
   - 0: Not grounded in context
   - 1: Partially grounded
   - 2: Well-grounded
3. If groundedness is below threshold and iterations remain:
   - A new response is generated with emphasis on context adherence
   - The process repeats with the new response

## Best Practices

- Start with default thresholds (1) and adjust based on your use case
- Monitor `MAX_REFLECTION_LOOP` to balance quality vs. latency
- Use logging level INFO to observe reflection behavior:
  ```bash
  LOGLEVEL=INFO
  ```
- Feel free to customize the reflection prompts in `src/rag_server/prompt.yaml`:
  ```yaml
  reflection_relevance_check_prompt: # Evaluates context relevance
  reflection_query_rewriter_prompt:  # Rewrites queries for better retrieval
  reflection_groundedness_check_prompt: # Checks response groundedness
  reflection_response_regeneration_prompt: # Regenerates responses for better grounding
  ```

## Limitations

- Each reflection iteration adds latency to the response
- Higher thresholds may result in more iterations
- Response streaming is not supported during response groundedness checks
- For on-premises deployment:
  - Requires significant GPU resources (8x A100/H100 GPUs recommended)
  - Initial model download time may vary based on network bandwidth

For more details on implementation, see the [reflection.py](../src/nvidia_rag/rag_server/reflection.py) source code.
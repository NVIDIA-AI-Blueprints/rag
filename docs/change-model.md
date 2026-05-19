<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Change the LLM or Embedding Model for NVIDIA RAG Blueprint

You can change the LLM or embedding models for the [NVIDIA RAG Blueprint](readme.md) by using the following procedures.

:::{tip}
To navigate this page more easily, click the outline button at the top of the page. ![outline-button](assets/outline-button.png)
:::

## For NVIDIA-Hosted Microservices


### Change the LLM Model

The default LLM is `nvidia/nemotron-3-super-120b-a12b`. To use a different model from the API catalog,
specify the model in the `APP_LLM_MODELNAME` environment variable when you start the RAG Server.

```console
export APP_LLM_MODELNAME='nvidia/nemotron-3-super-120b-a12b'
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

To get a list of valid model names, use one of the following methods:

- Browse the models at <https://build.nvidia.com/>.
  View the sample Python code and get the model name from the `model` argument to the `client.chat.completions.create` method.


:::{tip}
Follow steps in [For Helm Deployments](#for-helm-deployments) to change the inference model for Helm charts.
:::

#### Model-Specific Configuration

##### Nemotron-3-Nano-30B

The `nemotron-3-nano-30b` model has different naming conventions depending on the deployment method:

| Deployment Type | Model Name |
|-----------------|------------|
| NVIDIA-hosted (build.nvidia.com) | `nvidia/nemotron-3-nano-30b-a3b` |
| Self-hosted / Local NIM | `nvidia/nemotron-3-nano` |

Both names refer to the same underlying model. Use the appropriate name based on your deployment type.

##### Nemotron 3 Super

`nvidia/nemotron-3-super-120b-a12b` is the default LLM for this blueprint. For hardware requirements and RTX PRO 6000-specific setup, see the [Nemotron 3 Super deployment guide](nemotron3-super-deployment.md).


### Change the Embedding Model

To change the embedding model to a model from the API catalog,
specify the model in the `APP_EMBEDDINGS_MODELNAME` environment variable when you start the RAG server.
The following example uses the `NVIDIA Embed QA 4` model.

```console
export APP_EMBEDDINGS_MODELNAME='NV-Embed-QA' 
export APP_RANKING_MODELNAME='NV-Embed-QA' 
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
```

As an alternative you can also specify the model names at runtime using `/generate` API call. Refer to the `Generate Answer Endpoint` and `Document Search Endpoint` payload schema in [this](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/retriever_api_usage.ipynb) notebook.

To get a list of valid model names, use one of the following methods:

- Browse the models at <https://build.nvidia.com/explore/retrieval>.
  View the sample Python code and get the model name from the `model` argument to the `client.embeddings.create` method.

- Install the [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/) Python package from PyPi.
  Use the `get_available_models()` method to on an instance of an `NVIDIAEmbeddings` object to list the models.
  Refer to the package web page for sample code to list the models.

:::{tip}
Always use same embedding model or model having same tokinizers for both ingestion and retrieval to yield good accuracy.
:::

### Configure Embedding Dimensions

The default embedding model (`nvidia/llama-nemotron-embed-vl-1b-v2`) uses **2048 dimensions** by default. When changing to a different embedding model, you may need to update the dimensions to match the model's output.

**Important:** Some embedding models have **fixed output dimensions** and do not accept a `dimensions` parameter. For example, `nvidia/nv-embedqa-e5-v5` always outputs 1024-dimensional embeddings. If you use such a model without configuring the dimensions, you may encounter an error like:

```
This model does not support 'dimensions', but a value of '2048' was provided.
```

#### Configure via Environment Variable

```bash
export APP_EMBEDDINGS_DIMENSIONS=1024  # Match your model's output dimensions
export APP_EMBEDDINGS_MODELNAME='nvidia/nv-embedqa-e5-v5'
```

#### Configure via config.yaml (Library Mode)

```yaml
embeddings:
  model_name: "nvidia/nv-embedqa-e5-v5"
  dimensions: 1024  # Must match the model's output dimensions
  server_url: "https://integrate.api.nvidia.com/v1"
```

:::{warning}
**Ingestion and retrieval must use the same embedding model and dimensions.** If you change the embedding model or dimensions after ingesting documents, you must re-ingest your documents to the vector database for accurate retrieval results.
:::

:::{note}
When using models from different providers (e.g., NVIDIA for LLM, Azure OpenAI for embeddings), you can configure service-specific API keys. See [Service-Specific API Keys](api-key.md#service-specific-api-keys) for details.
:::



## For Self-Hosted On Premises Microservices

You can specify the model for NVIDIA NIM containers to use in the [nims.yaml](../deploy/compose/nims.yaml) file.

1. Edit the `deploy/nims.yaml` file and specify an image that includes the model to deploy.

   ```yaml
   services:
     nim-llm:
       container_name: nim-llm-ms
       image: nvcr.io/nim/<image>:<tag>
       ...

     nemotron-embedding-ms:
       container_name: nemotron-embedding-ms
       image: nvcr.io/nim/<image>:<tag>


     nemotron-ranking-ms:
       container_name: nemotron-ranking-ms
       image: nvcr.io/nim/<image>:<tag>
   ```

   To get a list of valid model names, use one of the following methods:

   - Run `ngc registry image list "nim/*"`.

   - Browse the NGC catalog at <https://catalog.ngc.nvidia.com/containers>.

2. Update the corresponding model names using environment variables as required.
   ```bash
   export APP_LLM_MODELNAME=<>
   export APP_RANKING_MODELNAME=<>
   export APP_EMBEDDINGS_MODELNAME=<>
   ```

3. Follow the steps specified [here](deploy-docker-self-hosted.md#start-services-using-self-hosted-on-premises-models) to relaunch the containers with the updated models. Make sure to specify the correct model names using appropriate environment variables as shown in the earlier step.



## For Helm Deployments

Use this procedure to change models when you are running self-hosted NVIDIA NIM microservices. The Helm values map directly to the Docker Compose/self-hosted settings.

1. List the available models and images by running the following code. You can browse the NGC catalog at <https://catalog.ngc.nvidia.com/containers> to learn about the available models.

    ```bash
    ngc registry image list "nim/*"
    ```

2. Set the model names and service URLs used by `rag-server` deployment in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml).

    ```yaml
    # rag-server runtime configuration
    envVars:
      # === LLM ===
      APP_LLM_MODELNAME: "<llm-model-name>"
      # Use the in-cluster NIM LLM service; if empty, NVIDIA-hosted API is used
      APP_LLM_SERVERURL: "nim-llm:8000"

      # === Embeddings ===
      APP_EMBEDDINGS_MODELNAME: "<embedding-model-name>"
      APP_EMBEDDINGS_SERVERURL: "nemotron-embedding-ms:8000/v1"

      # === Reranker ===
      APP_RANKING_MODELNAME: "<reranker-model-name>"
      APP_RANKING_SERVERURL: "nemotron-ranking-ms:8000"
    ```

3. Configure the NIM microservices that host those models. Replace `<image>:<tag>` with the image you selected (format `nvcr.io/nim/<image>:<tag>`) in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml).

    ```yaml
    # LLM NIM
    nimOperator:
      nim-llm:
        enabled: true
        replicas: 1
        service:
          name: "nim-llm"
        image:
          # nvcr.io/nim/<image>:<tag>
          repository: nvcr.io/nim/<image>
          tag: "<tag>"
          pullPolicy: IfNotPresent
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        model:
          engine: tensorrt_llm
        env:
          - name: NIM_HTTP_API_PORT
            value: "8000"
          - name: NIM_TRITON_LOG_VERBOSE
            value: "1"
          - name: NIM_SERVED_MODEL_NAME
            value: "<llm-model-name>"  # Must match APP_LLM_MODELNAME

    # Embedding NIM
    nvidia-nim-llama-nemotron-embed-1b-v2:
      enabled: true
      replicas: 1
      service:
        name: "nemotron-embedding-ms"
      image:
        # nvcr.io/nim/<image>:<tag>
        repository: nvcr.io/nim/<image>
        tag: "<tag>"
        pullPolicy: IfNotPresent
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
      env:
        - name: NIM_HTTP_API_PORT
          value: "8000"
        - name: NIM_TRITON_LOG_VERBOSE
          value: "1"

    # Reranker NIM
    nvidia-nim-llama-nemotron-rerank-1b-v2:
      enabled: true
      replicas: 1
      service:
        name: "nemotron-ranking-ms"
      image:
        # nvcr.io/nim/<image>:<tag>
        repository: nvcr.io/nim/<image>
        tag: "<tag>"
        pullPolicy: IfNotPresent
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
      env: []
    ```

    **Nemotron Nano Models (Thinking budget LLMs) – vLLM profile**

    For these Thinking budget LLMs, only the vLLM profile is supported on H100 and RTX GPUs (for example, RTX 6000 Pro).

    | GPU | Model | Supported profile |
    |-----|-------|-------------------|
    | H100, RTX 6000 Pro | nvidia/nvidia-nemotron-nano-9b-v2 | vllm |
    | H100, RTX 6000 Pro | nvidia/nemotron-3-nano | vllm |

    :::{note}
    **If only the vLLM profile is available**

   When only a vLLM profile is available for a model, such as on H100 and RTX GPUs, you must use the vLLM engine. First [run the list-model-profiles command](model-profiles.md#list-available-profiles) to confirm which profiles are available and then apply the following configurations.
    **For Nemotron Nano Models VLLM profile**
    
    When deploying `nvidia/nvidia-nemotron-nano-9b-v2` or `nvidia/nemotron-3-nano`, check if `tensorrt_llm` profile is available using below command for your required model. 
    
    ```bash
    # Change model name as needed
    USERID=$(id -u) docker run --rm --gpus all \
      nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest \ 
      list-model-profiles
    ```
    
    If only `vllm` profile is available, you must use the **vLLM engine** and add these specific configurations:
    
    ```yaml
    nimOperator:
      nim-llm:
        image:
          repository: nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2
          tag: "latest"
        model:
          engine: vllm  # Required: use vLLM instead of tensorrt_llm
        env:
          - name: NIM_SERVED_MODEL_NAME
            value: "nvidia/nvidia-nemotron-nano-9b-v2"  # Must match APP_LLM_MODELNAME
          # ... other env vars ...
    ```

    Ensure `APP_LLM_MODELNAME` in the `rag-server` section matches `NIM_SERVED_MODEL_NAME`.
    :::

5. After you modify the `values.yaml` file, apply the changes described in [Change a Deployment](deploy-helm.md#change-a-deployment).



## Switch from the VLM Embedder to the Text-Only Embedder

The default embedder is the multimodal `nvidia/llama-nemotron-embed-vl-1b-v2` (2048 dimensions), which embeds both text passages and page images. To revert to the text-only `nvidia/llama-nemotron-embed-1b-v2` (also 2048 dimensions, same vector space size, smaller GPU footprint), follow the steps below. Ingestion and retrieval must use the **same** embedding model — re-ingest your documents after switching.

For multimodal context on what the VLM embedder does, see [Multimodal Retriever](multimodal-retriever.md).

### Docker Compose

The compose files already include the text-only model as a commented-out alternative. Either uncomment it in place, or override at run time via env var.

1. Override via env var (simplest):

   ```bash
   export APP_EMBEDDINGS_MODELNAME="nvidia/llama-nemotron-embed-1b-v2"
   ```

2. Restart the rag and ingestor servers:

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

The text-only embedding NIM (`nemotron-embedding-ms` in [`deploy/compose/nims.yaml`](../deploy/compose/nims.yaml)) is started by the same profile that the VLM embedder uses, so no separate NIM startup is needed.

### Helm

In [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), set the model name in the rag-server and ingestor-server env-var blocks (search for the existing `llama-nemotron-embed-vl-1b-v2` references):

```yaml
envVars:
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-1b-v2"

ingestor-server:
  envVars:
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-1b-v2"
```

The text-only embedding NIM (`nvidia-nim-llama-nemotron-embed-1b-v2` under `nimOperator`) is already configured in `values.yaml`; no image swap is required. Apply with [Change a Deployment](deploy-helm.md#change-a-deployment).

:::{warning}
**Re-ingest after switching.** Vectors produced by the VLM embedder are not directly comparable to vectors from the text-only embedder; retrieval accuracy will degrade until you re-ingest your corpus.
:::



## Switch to the VLM Reranker

The default reranker is the text reranker `nvidia/llama-nemotron-rerank-1b-v2`. To use a multimodal reranker that can re-rank with awareness of cited images, switch to `nvidia/llama-nemotron-rerank-vl-1b-v2`. See [Multimodal Retriever — Part 2: VLM Reranker](multimodal-retriever.md#part-2--vlm-reranker) for what the multimodal reranker does and the `ENABLE_VLM_RERANKER_IMAGE_INPUT` flag in detail. The steps below cover the model swap.

### Docker Compose

1. Start the VLM reranker NIM (`nemotron-ranking-vl-ms`, defined in [`deploy/compose/nims.yaml`](../deploy/compose/nims.yaml) under the `vlm-rerank` and `vlm-rag` profiles):

   ```bash
   export USERID=$(id -u)
   export NGC_API_KEY="nvapi-..."
   export RANKING_VL_MS_GPU_ID=0   # optional GPU pinning

   docker compose -f deploy/compose/nims.yaml --profile vlm-rerank up -d
   ```

   Use `--profile vlm-rag` instead if you also want VLM generation and VLM embedding to come up together.

2. Point the rag-server at the VLM reranker and (optionally) enable image input:

   ```bash
   export ENABLE_RERANKER="True"
   export APP_RANKING_MODELNAME="nvidia/llama-nemotron-rerank-vl-1b-v2"
   export APP_RANKING_SERVERURL="nemotron-ranking-vl-ms:8000"
   export ENABLE_VLM_RERANKER_IMAGE_INPUT="True"   # see multimodal-retriever.md
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   :::{note}
   The rag-server only follows the multimodal reranker code path when `APP_RANKING_MODELNAME` contains `rerank-vl`. With any other model name, `ENABLE_VLM_RERANKER_IMAGE_INPUT` has no effect.
   :::

### Helm

1. In [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), enable the VLM reranker NIM and (optionally) disable the text reranker:

   ```yaml
   nimOperator:
     nvidia-nim-llama-nemotron-rerank-vl-1b-v2:
       enabled: true
     # Optional: free the text reranker's GPU slot
     nvidia-nim-llama-nemotron-rerank-1b-v2:
       enabled: false
   ```

2. Update the rag-server env vars:

   ```yaml
   envVars:
     ENABLE_RERANKER: "True"
     APP_RANKING_MODELNAME: "nvidia/llama-nemotron-rerank-vl-1b-v2"
     APP_RANKING_SERVERURL: "nemotron-ranking-vl-ms:8000"
     ENABLE_VLM_RERANKER_IMAGE_INPUT: "True"
   ```

3. Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).



## VLM Selection for Generation vs. Image Captioning

This blueprint uses two separate VLM NIMs by default:

- **Generation (chat / RAG answering)** — Nemotron Omni (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`), served by the `vlm-ms` service.
- **Ingestion-time image captioning** — Nemotron Nano 12B v2-VL (`nvidia/nemotron-nano-12b-v2-vl`), served by the dedicated `vlm-captioning-ms` service.

### Why two services?

Nemotron Omni produces strong VLM generation results but its reasoning behavior is currently unregulated — for some images it returns only `reasoning_content` with an empty `response_content`, which surfaces in the pipeline as missing captions. NV-Ingest does not yet expose knobs to cap the reasoning budget on the captioning task (tracked as bug **6130091**). Until that lands, captioning runs on Nemotron Nano 12B, which produces stable captions across a wide range of images.

You can opt in to using Nemotron Omni for captioning if you accept that some images may end up with empty captions. Set:

```bash
export APP_NVINGEST_CAPTIONMODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_NVINGEST_CAPTIONENDPOINTURL="http://vlm-ms:8000/v1/chat/completions"
export VLM_CAPTION_MODEL_NAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export VLM_CAPTION_ENDPOINT="http://vlm-ms:8000/v1/chat/completions"
```

In that mode you can stop the standalone `vlm-captioning-ms` to free its GPU.

## Switch Back to Nemotron Nano 12B for Generation

The default generation VLM is **Nemotron Omni**. If you want generation to use **Nemotron Nano 12B** as well (so a single VLM handles both generation and captioning), follow the steps below.

### Docker Compose

1. In `deploy/compose/nims.yaml`, update the `vlm-ms` service image:

   ```yaml
   vlm-ms:
     image: nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:1.6.0
   ```

2. Set the model name and disable Omni-specific reasoning knobs before starting services:

   ```bash
   export APP_VLM_MODELNAME="nvidia/nemotron-nano-12b-v2-vl"
   export APP_VLM_ENABLE_THINKING=false
   export APP_VLM_THINKING_TOKEN_BUDGET=0
   ```

   Captioning is already on Nemotron Nano 12B via `vlm-captioning-ms`, so no captioning-side env changes are needed. If you'd like a single shared VLM container instead, point captioning at `vlm-ms` and stop `vlm-captioning-ms`:

   ```bash
   export APP_NVINGEST_CAPTIONENDPOINTURL="http://vlm-ms:8000/v1/chat/completions"
   export VLM_CAPTION_ENDPOINT="http://vlm-ms:8000/v1/chat/completions"
   ```

3. Restart the affected services:

   ```bash
   docker compose -f deploy/compose/nims.yaml --profile vlm-generation up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

### Helm

1. In `deploy/helm/nvidia-blueprint-rag/values.yaml`, point the generation VLM at Nemotron Nano 12B (captioning is already on `nim-vlm-captioning`, so it stays put):

   ```yaml
   nimOperator:
     nim-vlm:
       image:
         repository: nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl
         tag: "1.6.0"

   envVars:
     APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
     APP_VLM_ENABLE_THINKING: "false"
     APP_VLM_THINKING_TOKEN_BUDGET: "0"
   ```

   If you want a single shared VLM container for both generation and captioning, disable the dedicated captioning service and re-point the captioning endpoints at `nim-vlm`:

   ```yaml
   nimOperator:
     nim-vlm-captioning:
       enabled: false

   ingestor-server:
     envVars:
       APP_NVINGEST_CAPTIONENDPOINTURL: "http://nim-vlm:8000/v1/chat/completions"

   nv-ingest:
     envVars:
       VLM_CAPTION_ENDPOINT: http://nim-vlm:8000/v1/chat/completions
   ```

2. Apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

:::{note}
Nemotron Nano 12B requires 1x H100 GPU for VLM inference. Ensure `APP_VLM_SERVERURL` points to the correct NIM endpoint after switching.
:::

## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Service-Specific API Keys](api-key.md#service-specific-api-keys)

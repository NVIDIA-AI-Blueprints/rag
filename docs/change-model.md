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

To change the inference model to a model from the API catalog,
specify the model in the `APP_LLM_MODELNAME` environment variable when you start the RAG Server.

```console
export APP_LLM_MODELNAME='nvidia/llama-3.3-nemotron-super-49b-v1.5' 
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

To get a list of valid model names, use one of the following methods:

- Browse the models at <https://build.nvidia.com/>.
  View the sample Python code and get the model name from the `model` argument to the `client.chat.completions.create` method.


:::{tip}
Follow steps in [For Helm Deployments](#for-helm-deployments) to change the inference model for Helm charts.
:::


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

As an alternative you can also specify the model names at runtime using `/generate` API call. Refer to the `Generate Answer Endpoint` and `Document Search Endpoint` payload schema in [this](../notebooks/retriever_api_usage.ipynb) notebook.

To get a list of valid model names, use one of the following methods:

- Browse the models at <https://build.nvidia.com/explore/retrieval>.
  View the sample Python code and get the model name from the `model` argument to the `client.embeddings.create` method.

- Install the [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/) Python package from PyPi.
  Use the `get_available_models()` method to on an instance of an `NVIDIAEmbeddings` object to list the models.
  Refer to the package web page for sample code to list the models.

:::{tip}
Always use same embedding model or model having same tokinizers for both ingestion and retrieval to yield good accuracy.
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

     nemoretriever-embedding-ms:
       container_name: nemoretriever-embedding-ms
       image: nvcr.io/nim/<image>:<tag>


     nemoretriever-ranking-ms:
       container_name: nemoretriever-ranking-ms
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
      APP_EMBEDDINGS_SERVERURL: "nemoretriever-embedding-ms:8000"

      # === Reranker ===
      APP_RANKING_MODELNAME: "<reranker-model-name>"
      APP_RANKING_SERVERURL: "nemoretriever-ranking-ms:8000"
    ```

3. Configure the NIM microservices that host those models. Replace `<image>:<tag>` with the image you selected (format `nvcr.io/nim/<image>:<tag>`) in [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml).

    ```yaml
    # LLM NIM
    nim-llm:
      enabled: true
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
        - name: NIM_MODEL_PROFILE
          value: "" # Optional; leave empty for auto profile
      model:
        # Optional: provide NGC API key if private artifacts must be pulled
        ngcAPIKey: ""
        name: "<llm-model-name>"

    # Embedding NIM
    nvidia-nim-llama-32-nv-embedqa-1b-v2:
      enabled: true
      image:
        # nvcr.io/nim/<image>:<tag>
        repository: nvcr.io/nim/<image>
        tag: "<tag>"
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
      nim:
        ngcAPIKey: ""

    # Reranker NIM
    nvidia-nim-llama-32-nv-rerankqa-1b-v2:
      enabled: true
      image:
        # nvcr.io/nim/<image>:<tag>
        repository: nvcr.io/nim/<image>
        tag: "<tag>"
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
      nim:
        ngcAPIKey: ""
    ```

4. After you modify values.yaml, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).



## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

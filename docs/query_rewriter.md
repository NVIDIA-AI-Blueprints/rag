<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable query rewriting support for NVIDIA RAG Blueprint

You can enable query rewriting for the [NVIDIA RAG Blueprint](readme.md). Query rewriting enables higher accuracy for multiturn queries by making an additional LLM call to decontextualize the incoming question, before sending it to the retrieval pipeline.

After you have [deployed the blueprint](readme.md#deployment-options-for-rag-blueprint), to enable query rewriting support, developers have the following options:

  - [Using on-prem model (Recommended)](#using-on-prem-model-recommended)
  - [Using cloud hosted model](#using-cloud-hosted-model)
  - [Using Helm Chart (on-prem only)](#using-helm-chart-on-prem-only)


## Using on-prem model (Recommended)
1. Make sure the nim-llm container is up and in healthy state before proceeding further
   ```bash
   docker ps --filter "name=nim-llm" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   nim-llm                              Up 38 minutes (healthy)
   ```

3. Enable query rewriting
   Export the below environment variable and relaunch the rag-server container.
   ```bash
   export APP_QUERYREWRITER_SERVERURL="nim-llm:8000"
   export ENABLE_QUERYREWRITER="True"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   Alternatively, you can enable this at runtime during retrieval by setting `enable_query_rewriting: True` as part of the schema of POST /generate API, without relaunching the containers. Refer to the [retrieval notebook](../notebooks/retriever_api_usage.ipynb).


## Using cloud hosted model
1. Set the server url to empty string to point towards cloud hosted model
   ```bash
   export APP_QUERYREWRITER_SERVERURL=""
   ```

2. Relaunch the rag-server container by enabling query rewriter.
   ```bash
   export ENABLE_QUERYREWRITER="True"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

:::{tip}
You can change the model name and model endpoint in case of an externally hosted LLM model by setting these two environment variables and restarting the rag services
:::

```bash
export APP_QUERYREWRITER_SERVERURL="<llm_nim_http_endpoint_url>"
export APP_QUERYREWRITER_MODELNAME="<model_name>"
```


## Using Helm Chart (on-prem only)

This section describes how to enable Query Rewriting when you deploy by using Helm, using an on-prem deployment of the LLM model.

:::{note}
Only on-prem deployment of the LLM is supported. The model must be deployed separately using the NIM LLM Helm chart.
:::

### 1. Enable Query Rewriter in `rag-server` Helm deployment
1. Modify the [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, in the `envVars` section, and set the following values.

    ```yaml
       envVars:
          ##===Query Rewriter Model specific configurations===
          APP_QUERYREWRITER_MODELNAME: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
          APP_QUERYREWRITER_SERVERURL: "nim-llm:8000"  # Fully qualified service name
          ENABLE_QUERYREWRITER: "True"
    ```

Follow the steps from [Deploy with Helm](deploy-helm.md) and use the following command to deploy the chart.

```bash
helm install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
   --set ngcApiSecret.password=$NGC_API_KEY \
   -f deploy/helm/nvidia-blueprint-rag/values.yaml
```


<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Get an API Key

You need to generate an API key to access NVIDIA NIM Microservices from the [NVIDIA RAG Blueprint](readme.md). 
You need an API key to access models hosted in the NVIDIA API Catalog, and to download models on-premises. 
For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#ngc-api-keys).

To generate an API key, use the following procedure.

1. Go to https://org.ngc.nvidia.com/setup/api-keys.
2. Click **Generate Personal Key**.
3. Enter a **Key Name**.
4. For **Expiration**, choose **Never Expire**.
5. For **Services Included**, select **NGC Catalog** and **Public API Endpoints**.
6. Click **Generate Personal Key**.
7. Copy your key and save it somewhere safe and private.
8. (Important) Export your key as an environment variable by using the following code.

    ```bash
    export NGC_API_KEY="<your-ngc-api-key>"
    ```



## API Key Expiration

If your API key expires, do one of the following:

- Create a new key by using the previous procedure, and then delete the expired key. 
- Use the **Action** menu to **Rotate** your key. 

You must update the new key information in your environment variables and code.



## Service-Specific API Keys

The RAG Blueprint supports service-specific API keys to provide fine-grained control over API access for different components. This enables separate billing accounts, different rate limits, security segregation, usage tracking per service, or mixed cloud/on-premises deployments.

### How It Works

Service-specific API keys override global `NVIDIA_API_KEY` or `NGC_API_KEY`. The system uses the following fallback order to determine which API key to use:

```text
service-specific key → NVIDIA_API_KEY → NGC_API_KEY → None
```

For example, if you set `APP_LLM_APIKEY`, the LLM service will use that key instead of the global `NVIDIA_API_KEY`.

### Supported Service Keys

| Service Key | Purpose | Used By |
|------------|---------|---------|
| `APP_LLM_APIKEY` | Main language model for RAG responses | RAG Server (generate endpoint) |
| `APP_EMBEDDINGS_APIKEY` | Text embedding generation | Ingestor Server, RAG Server (search) |
| `APP_RANKING_APIKEY` | Document reranking | RAG Server (reranker) |
| `APP_QUERYREWRITER_APIKEY` | Query rewriting and optimization | RAG Server (query decomposition) |
| `APP_FILTEREXPRESSIONGENERATOR_APIKEY` | Metadata filter generation from natural language | RAG Server (search with filters) |
| `APP_VLM_APIKEY` | Vision-Language Model for image understanding | RAG Server (multimodal queries) |
| `SUMMARY_LLM_APIKEY` | Document summarization | Ingestor Server (summary generation) |
| `REFLECTION_LLM_APIKEY` | Self-reflection and answer validation | RAG Server (reflection mode) |


### Library Mode (Python Package)

Configure in `config.yaml`:

```yaml
llm:
  api_key: "your-llm-api-key"  # Optional: overrides NVIDIA_API_KEY
embeddings:
  api_key: "your-embeddings-api-key"  # Optional: overrides NVIDIA_API_KEY
```

### Docker Compose

```bash
export APP_LLM_APIKEY="your-llm-api-key"
export APP_EMBEDDINGS_APIKEY="your-embeddings-api-key"
# Additional service-specific keys can be set as needed
```

### Helm

Use `--set` flags to pass API keys securely via command line:

```bash
helm upgrade --install rag -n rag <chart-path-or-url> \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set apiKeysSecret.llmApiKey=$APP_LLM_APIKEY \
  --set apiKeysSecret.embeddingsApiKey=$APP_EMBEDDINGS_APIKEY
```

Additional service-specific keys can be configured as needed (e.g., `rankingApiKey`, `vlmApiKey`, `summaryLlmApiKey`).

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multi-Turn Conversation Support for NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) supports multi-turn conversations through two configuration options:

1. **CONVERSATION_HISTORY**: Controls how many conversation turns are passed to the LLM for response generation
2. **Query Processing**: Either query rewriting (`ENABLE_QUERYREWRITER`) or simple retrieval (`MULTITURN_RETRIEVER_SIMPLE`)

:::{important}
**For multi-turn conversations to work, you must set `CONVERSATION_HISTORY > 0` (e.g., 3-5 conversation turns).**

Additionally, enable either:
- `ENABLE_QUERYREWRITER=True` (recommended for best accuracy), OR
- `MULTITURN_RETRIEVER_SIMPLE=True` (for lower latency)

Without these settings, each query is processed independently without conversational context.
:::

## How Multi-Turn Conversations Work

### Generation Stage (CONVERSATION_HISTORY)

`CONVERSATION_HISTORY` determines the number of conversation turns (user-assistant pairs) passed to the LLM when generating responses. This provides the LLM with context from previous exchanges.

**Default:** `0` (no conversation history)

**Example:**
```
CONVERSATION_HISTORY=2
```

This passes the last 2 conversation turns (4 messages: 2 user + 2 assistant) to the LLM, providing context from recent exchanges.

### Retrieval Stage

The retrieval stage supports two approaches:

#### Option 1: Query Rewriting (ENABLE_QUERYREWRITER)

Query rewriting makes an additional LLM call to decontextualize the incoming question before sending it to the retrieval pipeline, enabling higher accuracy for multiturn queries.

**Default:** `False` (disabled)

**How it works:**
- Uses an LLM to reformulate the user's query based on conversation context
- Creates a standalone, context-aware query that doesn't require history
- Provides best retrieval accuracy for multi-turn conversations
- Adds latency due to additional LLM call

:::{warning}
If you enable query rewriting (`ENABLE_QUERYREWRITER=True`) but keep `CONVERSATION_HISTORY=0`, query rewriting will be skipped with a warning.
:::

#### Option 2: Simple History Concatenation (MULTITURN_RETRIEVER_SIMPLE)

When `MULTITURN_RETRIEVER_SIMPLE` is enabled, previous user queries from the conversation are concatenated with the current query before retrieving documents from the vector database.

**Default:** `False` (disabled)

**Example:**
```
User Turn 1: "What is NVIDIA?"
User Turn 2: "Tell me about their GPUs"
```

- **When disabled (False)**: Only "Tell me about their GPUs" is used for retrieval
- **When enabled (True)**: "What is NVIDIA?. Tell me about their GPUs" is used for retrieval

**How it works:**
- Concatenates previous user queries with the current query using ". " separator
- Lower latency (no additional LLM call)
- May be less accurate than query rewriting for complex conversational references

:::{note}
`MULTITURN_RETRIEVER_SIMPLE` only applies when query rewriting is disabled. If `ENABLE_QUERYREWRITER` is `True`, query rewriting takes precedence.
:::

## API Usage

The RAG server exposes an OpenAI-compatible API for providing custom conversation history. For full details, see [API - RAG Server Schema](api-rag.md).

Use the `/generate` endpoint to generate responses with custom conversation history.

### Required Parameters

| Parameter   | Description | Type   |
|-------------|-------------|--------|
| messages | A sequence of messages that form a conversation history. Each message contains a `role` field (`user`, `assistant`, or `system`) and a `content` field. | Array |
| use_knowledge_base | `true` to use a knowledge base; otherwise `false`. | Boolean |

### Example API Payload

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant that provides information about FastAPI."
        },
        {
            "role": "user",
            "content": "What is FastAPI?"
        },
        {
            "role": "assistant",
            "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints."
        },
        {
            "role": "user",
            "content": "What are the key features of FastAPI?"
        }
    ],
    "use_knowledge_base": true
}
```

For hands-on examples, refer to the [retriever API usage notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/retriever_api_usage.ipynb).

## Multi-Turn Conversation Strategies

This section lists down different strategies available for enabling multi turn query handling in the pipeline.

### Strategy 1: Query Rewriting (Recommended for Best Accuracy)

**Configuration:**
```bash
ENABLE_QUERYREWRITER="True"
CONVERSATION_HISTORY="5"
```

**When to use:**
- Accuracy is the highest priority
- User queries frequently reference previous conversation turns
- You can tolerate additional latency for better results

### Strategy 2: Simple History Concatenation

**Configuration:**
```bash
MULTITURN_RETRIEVER_SIMPLE="True"
CONVERSATION_HISTORY="5"
```

**When to use:**
- You need multi-turn support with lower latency
- Queries have simple references to previous turns
- Query rewriting adds too much latency for your use case

### Strategy 3: Single-Turn Mode (No History)

**Configuration:**
```bash
CONVERSATION_HISTORY="0"
```

**When to use:**
- This is the default setting
- Queries are independent and don't reference previous turns
- Minimizing token usage and latency is critical
- Building a Q&A system without conversational memory


## Docker Deployment

### Prerequisites

Follow the deployment guide for [Self-Hosted Models](deploy-docker-self-hosted.md) or [NVIDIA-Hosted Models](deploy-docker-nvidia-hosted.md).

### Enable Query Rewriting with On-Prem Model (Recommended)

1. Verify the nim-llm container is healthy:
   ```bash
   docker ps --filter "name=nim-llm" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output:*
   ```
   NAMES                                   STATUS
   nim-llm                              Up 38 minutes (healthy)
   ```

2. Enable query rewriting:
   ```bash
   export APP_QUERYREWRITER_SERVERURL="nim-llm:8000"
   export ENABLE_QUERYREWRITER="True"
   export CONVERSATION_HISTORY="5"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

:::{tip}
You can enable query rewriting at runtime by setting `enable_query_rewriting: True` in the POST /generate API schema without relaunching containers. Refer to the [retrieval notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/retriever_api_usage.ipynb). Note that `CONVERSATION_HISTORY` must still be > 0.
:::

### Enable Query Rewriting with Cloud-Hosted Model

1. Configure for cloud-hosted model:
   ```bash
   export APP_QUERYREWRITER_SERVERURL=""
   export ENABLE_QUERYREWRITER="True"
   export CONVERSATION_HISTORY="5"
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

:::{tip}
For externally hosted LLM models, customize the endpoint and model name:
```bash
export APP_QUERYREWRITER_SERVERURL="<llm_nim_http_endpoint_url>"
export APP_QUERYREWRITER_MODELNAME="<model_name>"
```
:::

### Enable Simple History Concatenation

```bash
export MULTITURN_RETRIEVER_SIMPLE="True"
export CONVERSATION_HISTORY="5"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Disable All Multi-Turn Features (Single-Turn Mode)

```bash
export CONVERSATION_HISTORY="0"
export MULTITURN_RETRIEVER_SIMPLE="False"
export ENABLE_QUERYREWRITER="False"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## Helm Deployment

For details on Helm deployment, see [Deploy with Helm](deploy-helm.md).

### Enable Query Rewriting with On-Prem Model (Recommended)

:::{note}
Only on-prem deployment of the LLM is supported for Helm. The model must be deployed separately using the NIM LLM Helm chart.
:::

1. Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable query rewriting:

   ```yaml
   # Environment variables for rag-server
   envVars:
     # ... existing configurations ...
     
     # === Query Rewriter Model specific configurations ===
     APP_QUERYREWRITER_MODELNAME: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
     APP_QUERYREWRITER_SERVERURL: "nim-llm:8000"  # Fully qualified service name
     ENABLE_QUERYREWRITER: "True"
     CONVERSATION_HISTORY: "5"
   ```

2. Deploy or upgrade the chart:

   After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

   For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### Enable Simple History Concatenation

1. Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable simple history concatenation:

   ```yaml
   # Environment variables for rag-server
   envVars:
     # ... existing configurations ...
     
     # === Simple Multi-Turn (History Concatenation) ===
     MULTITURN_RETRIEVER_SIMPLE: "True"
     CONVERSATION_HISTORY: "5"
   ```

2. Upgrade the deployment:

   After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

   For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

## Configuration Summary

| Environment Variable | Stage | Default | Required For | Description |
|---------------------|-------|---------|--------------|-------------|
| `CONVERSATION_HISTORY` | Generation | `0` | All multi-turn features | Number of conversation turns to pass to LLM (0 = no history) |
| `ENABLE_QUERYREWRITER` | Retrieval | `False` | Advanced multi-turn | Enable AI-powered query rewriting for better retrieval accuracy |
| `MULTITURN_RETRIEVER_SIMPLE` | Retrieval | `False` | Simple multi-turn | Concatenate conversation history with current query for document retrieval |
| `APP_QUERYREWRITER_SERVERURL` | Retrieval | - | Query rewriting | Server URL for query rewriter model (empty string for cloud-hosted) |
| `APP_QUERYREWRITER_MODELNAME` | Retrieval | - | Query rewriting | Model name for query rewriter |

## Related Topics

- [API - RAG Server Schema](api-rag.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy on Kubernetes with Helm](deploy-helm.md)

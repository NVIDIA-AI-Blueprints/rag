<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Conversation History Configuration for NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) provides two configuration options to control how conversation history is used in multiturn conversations:

1. **MULTITURN_RETRIEVAL_SIMPLE**: Controls whether conversation history is concatenated with the current query for document retrieval
2. **CONVERSATION_HISTORY**: Controls how many conversation turns are passed to the LLM for response generation

:::{important}
For multi-turn conversations to work, you must set **CONVERSATION_HISTORY > 0**. This is required for both query rewriting and conversation history concatenation features.
:::

## How Conversation History Works

### Generation Stage (CONVERSATION_HISTORY)

`CONVERSATION_HISTORY` determines the number of conversation turns (user-assistant pairs) passed to the LLM when generating responses. This provides the LLM with context from previous exchanges.

**Default:** `0` (no conversation history)

**Example:**
```
CONVERSATION_HISTORY=2
```

This passes the last 2 conversation turns (4 messages: 2 user + 2 assistant) to the LLM, providing context from recent exchanges.

:::{warning}
**Query Rewriting Requires CONVERSATION_HISTORY > 0**

If you enable query rewriting (`ENABLE_QUERYREWRITER=True`) but keep `CONVERSATION_HISTORY=0`, query rewriting will be skipped with a warning. Query rewriting needs conversation history to reformulate queries based on conversational context.
:::

### Retrieval Stage (MULTITURN_RETRIEVAL_SIMPLE)

When `MULTITURN_RETRIEVAL_SIMPLE` is enabled, previous user queries from the conversation are concatenated with the current query before retrieving documents from the vector database. This helps retrieve more contextually relevant documents when the current query references previous conversation turns.

**Default:** `False` (disabled)

**Example:**
```
User Turn 1: "What is NVIDIA?"
User Turn 2: "Tell me about their GPUs"
```

- **When disabled (False)**: Only "Tell me about their GPUs" is used for retrieval
- **When enabled (True)**: "What is NVIDIA?. Tell me about their GPUs" is used for retrieval

:::{note}
`MULTITURN_RETRIEVAL_SIMPLE` only applies when query rewriting is disabled. If `ENABLE_QUERYREWRITER` is `True`, query rewriting takes precedence.
:::

## Docker Deployment

### Prerequisites

Follow the deployment guide for [Self-Hosted Models](deploy-docker-self-hosted.md) or [NVIDIA-Hosted Models](deploy-docker-nvidia-hosted.md).

### Enable Multi-Turn Conversations with Query Rewriting (Recommended)

For best accuracy in multi-turn conversations, enable query rewriting along with conversation history:

```bash
export ENABLE_QUERYREWRITER="True"
export CONVERSATION_HISTORY="5"  # Required for query rewriting to work
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Enable Multi-Turn Conversations with Simple Retrieval

To use simple history concatenation without query rewriting:

```bash
export MULTITURN_RETRIEVAL_SIMPLE="True"
export CONVERSATION_HISTORY="5"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Disable All Multi-Turn Features (Single-Turn Mode)

To use only the current query without any conversation history:

```bash
export CONVERSATION_HISTORY="0"
export MULTITURN_RETRIEVAL_SIMPLE="False"
export ENABLE_QUERYREWRITER="False"
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## Helm Deployment

You can configure conversation history when deploying RAG using Helm for Kubernetes environments. For details, see [Deploy with Helm](deploy-helm.md).

### Enable Multi-Turn Conversations with Query Rewriting (Recommended)

Modify the [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file in the `envVars` section:

```yaml
envVars:
  ENABLE_QUERYREWRITER: "True"
  CONVERSATION_HISTORY: "5"
```

Then upgrade the deployment:

```bash
helm upgrade rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=${NGC_API_KEY} \
  --set ngcApiSecret.password=${NGC_API_KEY} \
  --set envVars.ENABLE_QUERYREWRITER="True" \
  --set envVars.CONVERSATION_HISTORY="5" \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

### Enable Multi-Turn Conversations with Simple Retrieval

Modify the [values.yaml](../deploy/helm/nvidia-blueprint-rag/values.yaml) file in the `envVars` section:

```yaml
envVars:
  MULTITURN_RETRIEVAL_SIMPLE: "True"
  CONVERSATION_HISTORY: "5"
```

Then upgrade the deployment:

```bash
helm upgrade rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=${NGC_API_KEY} \
  --set ngcApiSecret.password=${NGC_API_KEY} \
  --set envVars.MULTITURN_RETRIEVAL_SIMPLE="True" \
  --set envVars.CONVERSATION_HISTORY="5" \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

## Configuration Summary

| Environment Variable | Stage | Default | Required For | Description |
|---------------------|-------|---------|--------------|-------------|
| `CONVERSATION_HISTORY` | Generation | `0` | Query Rewriting, Multi-turn | Number of conversation turns to pass to LLM (0 = no history) |
| `MULTITURN_RETRIEVAL_SIMPLE` | Retrieval | `False` | Simple multi-turn retrieval | Concatenate conversation history with current query for document retrieval |
| `ENABLE_QUERYREWRITER` | Retrieval | `False` | Advanced multi-turn | Enable AI-powered query rewriting for better retrieval accuracy |

## Multi-Turn Conversation Strategies

### Strategy 1: Query Rewriting (Recommended for Best Accuracy)

**Configuration:**
```bash
ENABLE_QUERYREWRITER="True"
CONVERSATION_HISTORY="5"
```

**How it works:**
- Uses an LLM call to reformulate the user's query based on conversation context
- Creates a standalone, context-aware query that doesn't require history
- Provides best retrieval accuracy for multi-turn conversations
- Adds latency due to additional LLM call

### Strategy 2: Simple History Concatenation

**Configuration:**
```bash
MULTITURN_RETRIEVAL_SIMPLE="True"
CONVERSATION_HISTORY="5"
```

**How it works:**
- Concatenates previous user queries with the current query using ". " separator
- Lower latency (no additional LLM call)
- May be less accurate than query rewriting for complex conversational references

### Strategy 3: Single-Turn Mode (No History)

**Configuration:**
```bash
CONVERSATION_HISTORY="0"
```

**How it works:**
- Each query is processed independently
- No conversation context is used
- Lowest latency and token usage
- Best for independent, single-turn queries

## When to Use Each Strategy

### Use Query Rewriting When:
- Accuracy is the highest priority
- User queries frequently reference previous conversation turns
- You can tolerate additional latency for better results
- **Remember:** Set `CONVERSATION_HISTORY > 0` (e.g., 3-5) for query rewriting to work

### Use Simple History Concatenation When:
- You need multi-turn support with lower latency
- Queries have simple references to previous turns
- Query rewriting adds too much latency for your use case

### Use Single-Turn Mode When:
- Queries are independent and don't reference previous turns
- Minimizing token usage and latency is critical
- Building a Q&A system without conversational memory

## Related Topics

- [Multi-Turn Conversation Support](multiturn.md)
- [Query Rewriting](query_rewriter.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy on Kubernetes with Helm](deploy-helm.md)

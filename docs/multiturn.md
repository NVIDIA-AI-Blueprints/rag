<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multi-Turn Conversation Support in NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) supports multi-turn conversations through two complementary features that must be configured together:

1. **Conversation History** (`CONVERSATION_HISTORY`): Controls how many conversation turns are passed to the system
2. **Query Processing**: Either query rewriting (`ENABLE_QUERYREWRITER`) or simple retrieval (`MULTITURN_RETRIEVAL_SIMPLE`)

:::{important}
**For multi-turn conversations to work effectively, you must:**
- Set `CONVERSATION_HISTORY > 0` (e.g., 3-5 conversation turns)
- Enable either `ENABLE_QUERYREWRITER=True` (recommended) OR `MULTITURN_RETRIEVAL_SIMPLE=True`

Without these settings, each query is processed independently without conversational context.
:::

The RAG server exposes an OpenAI-compatible API for providing custom conversation history. For full details, see [APIs for RAG Server](api-rag.md).

Use the `/generate` endpoint in the RAG server of a RAG pipeline to generate responses to prompts using custom conversation history.

To support multi-turn conversations, include the following parameters in the request body.


| Parameter   | Description | Type   |
|-------------|-------------|--------|
| messages | A sequence of messages that form a conversation history. Each message contains a `role` field, which can be `user`, `assistant`, or `system`, and a `content` field that contains the message text. | Array |
| use_knowledge_base | `true` to use a knowledge base; otherwise `false`. | Boolean |



## Example payload for customization

The following example payload includes a `messages` parameter that passes a custom conversation history to `/generate` endpoint for better contextual answers. You can include or change the following parameters in the request body while trying out the generate API using [this notebook](../notebooks/retriever_api_usage.ipynb).


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

:::{tip}
**To enable multi-turn conversations in your deployment:**
1. Set `CONVERSATION_HISTORY > 0` (e.g., 5) - **Required for any multi-turn feature to work**
2. Choose your retrieval strategy:
   - [Enable Query Rewriting](./query_rewriter.md) (recommended for best accuracy)
   - OR enable `MULTITURN_RETRIEVAL_SIMPLE=True` for simple history concatenation

For detailed configuration steps and examples, see [Conversation History Configuration](./conversation-history.md).
:::
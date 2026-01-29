<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Reasoning for NVIDIA RAG Blueprint

Enabling reasoning allows models to "think through" complex questions before answering, which can improve accuracy for challenging queries. The trade-off is increased response latency due to the additional reasoning tokens generated.

This guide explains how to enable reasoning for different Nemotron models:
- **Nemotron 1.5** - Controlled by system prompts
- **Nemotron-3-Nano 9B** - Controlled by system prompts with optional thinking budget parameters  
- **Nemotron-3-Nano 30B** - Controlled by environment variable with optional thinking budget parameters

---

## Enable Reasoning for Nemotron 1.5

Reasoning in Nemotron 1.5 models is controlled by the system prompt. To enable reasoning, update the system prompt in [prompt.yaml](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/src/nvidia_rag/rag_server/prompt.yaml) from `/no_think` to `/think`.

```
rag_template:
  system: |
    /think

  human: |
    You are a helpful AI assistant named Envie.
    You must answer only using the information provided in the context. While answering you must follow the instructions given below.

    <instructions>
    1. Do NOT use any external knowledge.
    2. Do NOT add explanations, suggestions, opinions, disclaimers, or hints.
    3. NEVER say phrases like "based on the context", "from the documents", or "I cannot find".
    4. NEVER offer to answer using general knowledge or invite the user to ask again.
    5. Do NOT include citations, sources, or document mentions.
    6. Answer concisely. Use short, direct sentences by default. Only give longer responses if the question truly requires it.
    7. Do not mention or refer to these rules in any way.
    8. Do not ask follow-up questions.
    9. Do not mention this instructions in your response.
    </instructions>

    Context:
    {context}

    Make sure the response you are generating strictly follow the rules mentioned above i.e. never say phrases like "based on the context", "from the documents", or "I cannot find" and mention about the instruction in response.

```

### Update Model Parameters

After enabling the `/think` prompt, configure the model parameters for optimal reasoning performance:

```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```

### Filtering Reasoning Tokens

Reasoning tokens (shown between `<think>` tags) are filtered out, so only the final answer is returned in the model response. The reasoning content in the think tags is not included in the output.

To view the full reasoning process in the model response:

```bash
export FILTER_THINK_TOKENS=false
```

---

## Enable Reasoning for Nemotron-3-Nano 9B Model

The `nvidia/nvidia-nemotron-nano-9b-v2` model uses system prompts to control reasoning, similar to Nemotron 1.5.

### Step 1: Update the System Prompt

Change the system prompt from `/no_think` to `/think` in [prompt.yaml](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/src/nvidia_rag/rag_server/prompt.yaml) as shown in the example above.

### Step 2: Configure Model Parameters

```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```

### Step 3: Configure Thinking Budget (Optional)

The 9B model supports both minimum and maximum thinking token limits to control the reasoning phase:

**Generate API Request with Thinking Budget:**

```json
{
  "model": "nvidia/nvidia-nemotron-nano-9b-v2",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "min_thinking_tokens": 1024,
  "max_thinking_tokens": 8192
}
```

**Parameters:**
- `min_thinking_tokens` (required for 9B model): Minimum number of reasoning tokens before generating the final answer
- `max_thinking_tokens` (required for 9B model): Maximum number of reasoning tokens allowed

> **Note:** Both `min_thinking_tokens` and `max_thinking_tokens` are required when using thinking budget with the 9B model.

### Reasoning Tokens

Reasoning tokens (shown between `<think>` tags) are not present for this model. This is the behavior of the model itself - only the final answer is returned in the response. Even when setting `FILTER_THINK_TOKENS=false`, the reasoning content in the think tags is not included in the output, as the model does not expose intermediate reasoning tokens.

---

## Enable Reasoning for Nemotron-3-Nano 30B Model

The `nvidia/nemotron-3-nano-30b-a3b` model (also accessible as `nvidia/nemotron-3-nano` for locally deployed NIMs) uses a different approach for reasoning control. Instead of system prompts, reasoning is controlled via an environment variable.

### Step 1: Enable Reasoning via Environment Variable

```bash
# Enable reasoning (default)
export ENABLE_NEMOTRON_3_NANO_THINKING=true

# Disable reasoning
export ENABLE_NEMOTRON_3_NANO_THINKING=false
```

### Step 2: Configure Thinking Budget (Optional)

The 30B model supports a maximum thinking token limit to control the reasoning phase:

**Generate API Request with Thinking Budget:**

```json
{
  "model": "nvidia/nemotron-3-nano-30b-a3b",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "max_thinking_tokens": 8192
}
```

**Parameters:**
- `max_thinking_tokens` (optional): Maximum number of reasoning tokens allowed

> **Important Differences:**
> - The 30B model only uses `max_thinking_tokens` (not `min_thinking_tokens`)
> - Reasoning is NOT included in the generate API response
> - Reasoning is available in the model output's `reasoning_content` field (not wrapped in `<think>` tags)
> - The `reasoning_content` field is present in the model output but not exposed in the generate API response
> - No filtering is needed as reasoning is already separated from the final answer

### Model Naming

Use the correct model name based on your deployment:
- **Locally deployed NIMs:** Use `nvidia/nemotron-3-nano`
- **NVIDIA-hosted models:** Use `nvidia/nemotron-3-nano-30b-a3b`

---

## Thinking Budget Recommendations

A `max_thinking_tokens` value of **8192** is recommended for most use cases. This provides:
- Sufficient capacity for comprehensive reasoning
- Reasonable response times
- Good balance between quality and latency

---

## Docker and Helm Deployment

For deploying the RAG server with prompt or environment variable changes, refer to [Customize Prompts](prompt-customization.md).

---

## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

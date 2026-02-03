<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Reasoning for NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) supports reasoning capabilities that allow models to "think through" complex questions before answering. This feature improves accuracy for challenging queries but increases response latency due to additional reasoning tokens.

:::{tip}
Reasoning is particularly beneficial for the following:

- Complex multi-step questions
- Queries requiring logical deduction
- Technical or mathematical problem-solving
- Scenarios where accuracy is more important than response speed
:::

This guide explains how to enable reasoning for different Nemotron models, each using a different control mechanism.

| Model | Control Method | Thinking Budget Parameters |
|-------|----------------|----------------------------|
| Nemotron 1.5 | System prompts | None |
| Nemotron-3-Nano 9B | System prompts | min/max thinking tokens |
| Nemotron-3-Nano 30B | Environment variable | max thinking tokens only |

## Enable Reasoning for Nemotron 1.5

Reasoning in Nemotron 1.5 models (such as `nvidia/llama-3.3-nemotron-super-49b-v1.5`) is controlled through system prompts. The model switches between reasoning and non-reasoning modes using `/think` and `/no_think` directives.

### Update the System Prompt

To enable reasoning, update the system prompt from `/no_think` to `/think` in [prompt.yaml](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/src/nvidia_rag/rag_server/prompt.yaml), as shown in the following code.

```yaml
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

### Configure Model Parameters

After you enable the `/think` prompt, configure the model parameters for optimal reasoning performance:

```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```

### Filter Reasoning Tokens

By default, reasoning tokens (shown between `<think>` tags) are filtered out so only the final answer is returned in the model response.

To view the full reasoning process including the `<think>` tags in the model response, use the following code.

```bash
export FILTER_THINK_TOKENS=false
```

:::{note}
For most production use cases, keep `FILTER_THINK_TOKENS=true` (default) to provide cleaner responses to end users.
:::

## Enable Reasoning for Nemotron-3-Nano 9B

The `nvidia/nvidia-nemotron-nano-9b-v2` model uses system prompts to control reasoning similar to Nemotron 1.5. It also adds support for thinking budget parameters to control the extent of reasoning.

### Update the System Prompt

Change the system prompt from `/no_think` to `/think` in [prompt.yaml](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/src/nvidia_rag/rag_server/prompt.yaml) as shown in the previous Nemotron 1.5 example.

### Configure Model Parameters

```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```

### Configure Thinking Budget (Optional)

The 9B model supports both minimum and maximum thinking token limits to control the reasoning phase. You can include these parameters in API requests to the model:

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

**Thinking budget parameters:**

**`min_thinking_tokens`**
: Minimum number of reasoning tokens before generating the final answer.

**`max_thinking_tokens`**
: Maximum number of reasoning tokens allowed before generating the final answer.

:::{important}
The key differences for the 9B model are the following:


- Requires both `min_thinking_tokens` and `max_thinking_tokens` parameters
- Reasoning is available in the model output's `reasoning_content` field (not wrapped in `<think>` tags)
- The `reasoning_content` field is present in the model output but isn't exposed in the generate API response
- No filtering is needed because reasoning is already separated from the final answer
:::

## Enable Reasoning for Nemotron-3-Nano 30B

The `nvidia/nemotron-3-nano-30b-a3b` model uses a different approach for reasoning control. Instead of system prompts, you control reasoning through an environment variable.

### Enable Reasoning Through an Environment Variable

Set the environment variable to enable or disable reasoning:

```bash
# Enable reasoning (default)
export ENABLE_NEMOTRON_3_NANO_THINKING=true

# Disable reasoning
export ENABLE_NEMOTRON_3_NANO_THINKING=false
```

### Configure Thinking Budget (Optional)

The 30B model supports a maximum thinking token limit to control the reasoning phase:

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

**Thinking budget parameters:**

**`max_thinking_tokens`**
: Maximum number of reasoning tokens allowed before generating the final answer.

:::{important}
The key differences for the 30B model are the following:

- Uses only `max_thinking_tokens` (not `min_thinking_tokens`)
- Reasoning is available in the model output's `reasoning_content` field (not wrapped in `<think>` tags)
- The `reasoning_content` field is present in the model output but isn't exposed in the generate API response
- No filtering is needed because reasoning is already separated from the final answer
:::

### Model Naming

Use the correct model name based on your deployment:

**Locally deployed NIMs**
: `nvidia/nemotron-3-nano`

**NVIDIA-hosted models**
: `nvidia/nemotron-3-nano-30b-a3b`

## Deploy with Reasoning Enabled

After you configure reasoning settings in `prompt.yaml` or environment variables, redeploy your services:

### Docker Compose

```bash
# For prompt changes, rebuild and restart the RAG server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build

# For environment variable changes only
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

### Helm

For Helm deployments with custom prompts or environment variables, refer to [Customize Prompts](prompt-customization.md) for detailed instructions.

## Thinking Budget Recommendations

For models that support thinking budget parameters, a `max_thinking_tokens` value of **8192** is recommended for most use cases. This value provides:

- Sufficient capacity for comprehensive reasoning
- Reasonable response times
- Good balance between quality and latency

:::{tip}
Adjust the thinking budget based on your use case:

- **Lower values (1024-4096)**: Faster responses for simpler questions
- **Higher values (8192-16384)**: More thorough reasoning for complex queries
:::

## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md)
- [Customize Prompts](prompt-customization.md)
- [Change the LLM or Embedding Model](change-model.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

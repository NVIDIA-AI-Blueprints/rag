<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Enable Reasoning for NVIDIA RAG Blueprint

By default, reasoning is disabled in the [NVIDIA RAG Blueprint](readme.md). 
If your application can accept increased latency, enabling reasoning is an easy change to get an accuracy boost. 

Reasoning in Nemotron 1.5 is controlled by the system prompt. To enable reasoning for your use case, 
you can update the system prompt in [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml) from `/no_think` to `/think`. 
For example, to enable reasoning in RAG, update the system prompt from `/no_think` to `/think` as shown in the following code. 
You can update other prompts as well.

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

After you update the prompt, update the temperature and top_p to the recommended values by using the following environment variables.

```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```


## Docker and Helm Deployment

For details about how to deploy the RAG server with prompt changes, refer to [Customize Prompts](prompt-customization.md).



## Filtering Reasoning Tokens
By default, we filter out reasoning tokens and only provide the final response from the LLM. If you want to see the reasoning tokens as well, you can set `FILTER_THINK_TOKENS` to false.

```bash
export FILTER_THINK_TOKENS=false
```


## Nemotron-3-Nano Reasoning Configuration

For the `nemotron-3-nano-30b-a3b` model (also accessible as `nvidia/nemotron-3-nano` for local NIMs), reasoning is controlled via an environment variable. There is no need to filter thinking tokens for this model as the thinking content is returned in a separate `reasoning_content` key in the model response.

```bash
# Enable reasoning (default)
export ENABLE_NEMOTRON_3_NANO_THINKING=true

# Disable reasoning
export ENABLE_NEMOTRON_3_NANO_THINKING=false
```

This controls the `enable_thinking` flag in the model's `chat_template_kwargs`.

> **Note - Model Naming:**
> - **For locally deployed NIMs:** Use model name `nvidia/nemotron-3-nano`
> - **For NVIDIA-hosted models:** Use model name `nvidia/nemotron-3-nano-30b-a3b`

For other models, reasoning can be enabled by following the system prompt steps described at the beginning of this document.

## LLM Thinking Budget

The **Thinking Budget** feature allows you to control the number of tokens a model generates during its reasoning phase before producing a final answer. This is useful for managing latency and computational costs while still benefiting from the model's reasoning capabilities.

When the thinking budget is enabled, the model monitors the token count within the thinking region. Once the specified token limit is reached, the model concludes the reasoning phase and proceeds to generate the final answer.

### Supported Models

The following models support the Thinking Budget feature:

- `nvidia/nvidia-nemotron-nano-9b-v2`
- `nvidia/nemotron-3-nano-30b-a3b` (also accessible as `nvidia/nemotron-3-nano`)

For the latest supported models, refer to the [NIM Thinking Budget Control documentation](https://docs.nvidia.com/nim/large-language-models/latest/thinking-budget-control.html).

### Enabling Thinking Budget on RAG

To use the thinking budget, reasoning should be enabled by following the steps described above. Enable the thinking budget feature by including the `max_thinking_tokens` parameter in your API request:

**Example API request:**

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

A `max_thinking_tokens` value of **8192** is recommended to provide sufficient capacity for comprehensive reasoning while maintaining reasonable response times.


## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

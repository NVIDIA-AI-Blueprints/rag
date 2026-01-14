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



## Accuracy Improvement Example

Accuracy improvements from enabling reasoning across datasets average approximately 5%, 
with several cases demonstrating dramatic corrections.

For example, using the [ADOBE_2017_10Kpdf](https://github.com/patronus-ai/financebench/blob/main/pdfs/ADOBE_2017_10K.pdf) from [FinanceBench](https://github.com/patronus-ai/financebench/), 
and the following question: 

```text
What is the FY2017 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places. Please utilize information provided primarily within the balance sheet and the cash flow statement. 
```

Before enabling reasoning, the baseline model incorrectly computed Adobe's FY2017 operating cash flow ratio as 2.91. 
After enabling reasoning, the model produced the correct answer (0.83), demonstrating precise contextual understanding. 
The answer is found on 2 separate pages of the PDF; page 57 and page 61.



## Docker and Helm Deployment

For details about how to deploy the RAG server with prompt changes, refer to [Customize Prompts](prompt-customization.md).



## Filtering Reasoning Tokens
By default, we filter out reasoning tokens and only provide the final response from the LLM. If you want to see the reasoning tokens as well, you can set `FILTER_THINK_TOKENS` to false.

```bash
export FILTER_THINK_TOKENS=false
```


## Previous Models

For the previous `llama-3.3-nemotron-super-49b-v1` model, you set the environment variable `ENABLE_NEMOTRON_THINKING` to `true` to enable reasoning.

```bash 
export ENABLE_NEMOTRON_THINKING=true
```

For the previous `llama-3.3-nemotron-super-49b-v1` model, you can also add it to `services: rag-server: environment:` in `docker-compose-rag-server.yaml`.

```yaml
services:
  rag-server:
    environment:
      # Enable Nemotron thinking/reasoning for llama-3.3-nemotron-super-49b-v1 model
      ENABLE_NEMOTRON_THINKING: ${ENABLE_NEMOTRON_THINKING:-true}
```


## LLM Thinking Budget

The **Thinking Budget** feature allows you to control the number of tokens a model generates during its reasoning phase before producing a final answer. This is useful for managing latency and computational costs while still benefiting from the model's reasoning capabilities.

### Overview

When the thinking budget is enabled, the model monitors the token count within the thinking region. Once the specified token limit is reached, the model concludes the reasoning phase and proceeds to generate the final answer. This provides a balance between reasoning depth and response time.

### Supported Models

As of NIM version 1.12, the Thinking Budget feature is supported on the following models:

- **nvidia/nvidia-nemotron-nano-9b-v2**
- **nvidia/nemotron-3-nano-30b-a3b**

For the latest supported models, refer to the [NIM Thinking Budget Control documentation](https://docs.nvidia.com/nim/large-language-models/latest/thinking-budget-control.html).

### Enabling Thinking Budget on RAG

After enabling the reasoning as per the steps mentioned above, enable the thinking budget feature in RAG by including the following parameters in your API request:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_thinking_tokens` | 1 | Minimum number of thinking tokens to allocate for reasoning models. |
| `max_thinking_tokens` | 8192 | Maximum number of thinking tokens to allocate for reasoning models. |

> **Note for `nvidia/nemotron-3-nano-30b-a3b`**  
> This model only uses the `max_thinking_tokens` parameter.  
> - `min_thinking_tokens` is ignored for this model.  
> - Thinking budget is enabled by passing a positive `max_thinking_tokens` value in the request.

**Example API requests:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the FY2017 operating cash flow ratio for Adobe?"
    }
  ],
  "min_thinking_tokens": 1,
  "max_thinking_tokens": 8192,
  "model": "nvidia/nvidia-nemotron-nano-9b-v2"
}
```

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the FY2017 operating cash flow ratio for Adobe?"
    }
  ],
  "max_thinking_tokens": 8192,
  "model": "nvidia/nemotron-3-nano-30b-a3b"
}
```

**Recommendation:** A `max_thinking_tokens` value of **8192 tokens** is recommended to provide sufficient capacity for comprehensive reasoning while maintaining reasonable response times.


## Related Topics

- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)

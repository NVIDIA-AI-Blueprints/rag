<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Query Decomposition for NVIDIA RAG Blueprint

You can use query decomposition with the [NVIDIA RAG Blueprint](readme.md). Query decomposition is an advanced RAG (Retrieval-Augmented Generation) technique that breaks down complex, multi-faceted queries into simpler, focused subqueries. Each subquery is processed independently to gather comprehensive context, which is then synthesized into a final comprehensive response.


## Key Benefits
- **Better Context Coverage** – Captures multiple aspects of complex queries.
- **Iterative Refinement** – Follows up with additional questions based on initial findings.
- **Multi-perspective Analysis** – Approaches queries from different angles.

## Core Components
- **Subquery Generation** – LLM-powered breakdown of complex queries.
- **Iterative Processing** – Multi-round refinement with follow-up questions.
- **Response Synthesis** – Combines insights from all subqueries.


## Deployment Options
You can enable query decomposition in both Docker and Helm deployments:

1. **Docker Deployment** – Use Docker to deploy RAG on your hardware
2. **Helm Deployment** – Deploys as a Helm chart on a Kubernetes cluster (local deployment only)

## Option 1: Docker Deployment (Default)

### Step 1: Deploy all the dependent services

Follow the deployment guide for [Self-Hosted Models](deploy-docker-self-hosted.md) or [NVIDIA-Hosted Models](deploy-docker-nvidia-hosted.md).

### Step 2: Enable query decomposition

Set the environment variable to enable query decomposition:

```bash
export ENABLE_QUERY_DECOMPOSITION=true
export MAX_RECURSION_DEPTH=3
```

After setting these environment variables, you must restart the RAG server for `ENABLE_QUERY_DECOMPOSITION` to take effect:

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## Option 2: Helm Deployment (Local Deployment Only)

Alternatively, you can deploy RAG with query decomposition using Helm for Kubernetes environments. For details, see [Deploy with Helm](deploy-helm.md).

### Update RAG Blueprint Deployment with Query Decomposition

Use the Helm upgrade command below to enable query decomposition in RAG Blueprint by setting `ENABLE_QUERY_DECOMPOSITION` and `MAX_RECURSION_DEPTH`:

```bash
helm upgrade rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=${NGC_API_KEY} \
  --set ngcApiSecret.password=${NGC_API_KEY} \
  --set envVars.ENABLE_QUERY_DECOMPOSITION="true" \
  --set envVars.MAX_RECURSION_DEPTH="3"
```

## When to Use Query Decomposition

Query decomposition is especially valuable for multi-hop or complex queries that involve multiple steps or aspects. By breaking down a complex question into smaller, focused subqueries, the system can generate intermediate answers for each part. These intermediate responses are then combined to produce a comprehensive and accurate final answer.

**Example: Complex Multihop Query**

```
"If my future wife has the same first name as the 15th first lady of the United States' mother
and her surname is the same as the second assassinated president's mother's maiden name,
what is my future wife's name?"

This query requires multiple interconnected steps:
1. Identify the 15th first lady of the United States
2. Find the 15th first lady's mother's first name
3. Identify the second assassinated president
4. Find that president's mother's maiden name
5. Combine the two names to form the final answer
```

## When NOT to Use Query Decomposition

**Simple Factual Questions**
```
"What is the capital of France?"
(Simple lookup, no decomposition needed)
```

**Single-concept Queries**
```
"Define machine learning"
(Direct definition, no sub-aspects needed)
```

**Time-sensitive Queries**
```
"What's the current stock price of NVIDIA?"
(Real-time data, decomposition adds latency)
```

**Highly Specific Technical Questions**
```
"What's the syntax for creating a Python dictionary?"
(Specific syntax, no multiple perspectives needed)
```

## How Query Decomposition Works
To visualize how query decomposition works, see the diagram below:

```{figure} assets/query_decomposition.jpeg

Query Decomposition Flow — The system breaks down a complex query into subqueries, processes each iteratively, and synthesizes a comprehensive answer.
```


### Core Algorithm

1. **Subquery Generation**: Break down into focused questions
2. **Iterative Processing**: For each recursion depth:
   - Process each subquery independently
   - Rewrite queries with accumulated context
   - Retrieve and rank relevant documents
   - Generate focused answers
   - Collect contexts for synthesis
3. **Follow-up Generation**: Create follow-up questions for information missing from original query
4. **Termination Check**: Stop if no follow-up needed or max depth reached
5. **Final Synthesis**: Generate comprehensive response from all contexts and conversation



:::{important}
Query decomposition is not available for direct LLM calls (when `use_kb=false`). This feature requires the knowledge base integration to process subqueries and retrieve relevant documents. For direct LLM interactions, queries are processed without decomposition.
:::

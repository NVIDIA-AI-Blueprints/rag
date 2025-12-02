<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Evaluate Your NVIDIA RAG Blueprint System

After you [deploy your NVIDIA RAG Blueprint system](readme.md#deployment-options-for-rag-blueprint),
you can evaluate it by using [Ragas](https://docs.ragas.io/en/stable/) metrics specifically designed for Large Language Model (LLM) Applications.


## Ragas Metrics

Ragas include a set of [NVIDIA-specific metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/) 
that you can use to evaluate the performance of your LLM application. 
These metrics are designed to help you objectively measure the performance of your application, and include the following:

- **Answer Accuracy** – Measures the agreement between a model’s response and a reference ground truth for a given question.
- **Context Relevancy** – Evaluates whether the retrieved contexts (chunks or passages) are pertinent to the user input.
- **Response Groundedness** – Measures how well a response is supported or "grounded" by the retrieved contexts. It assesses whether each claim in the response can be found, either wholly or partially, in the provided contexts.

For more information, refer to the notebook [Evaluate Your RAG Pipeline with Ragas: Answer Accuracy, Context Relevancy, and Groundedness](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/evaluation_01_ragas.ipynb).


## Recall

You can also evaluate how well the retrieval system performs by using the [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) metric:

- **Recall** – Measures the proportion of relevant documents that are successfully retrieved at different top-k cutoffs (1, 3, 5, 10).

For more information, refer to the notebook [Evaluate Your RAG Pipeline with Ragas: Recall](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/evaluation_02_recall.ipynb).



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Get Started](deploy-docker-self-hosted.md)
- [Notebooks](notebooks.md)

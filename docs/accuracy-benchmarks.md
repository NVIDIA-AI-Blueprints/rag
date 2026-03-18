<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking RAG Accuracy: Evaluating LLM Reasoning and VLM Integration

In the fast-moving world of Retrieval-Augmented Generation (RAG), the gap between a “good” system and one that’s truly production-ready often depends on how effectively the pipeline manages complex reasoning and multimodal data. To measure these advancements, our team conducted extensive benchmarks across multiple configurations, examining the influence of LLM reasoning (“Think” mode) and Vision-Language Models (VLM).

## Benchmarked Datasets

Our analysis centered on seven major public datasets encompassing a broad range of challenges, from financial reasoning to intricate structural document parsing.

| Dataset | Domain | Corpus Language | Main Modalities | # Pages | # Queries |
|---|---|---|---|---|---|
| [RagBattlepacket](https://www.eyelevel.ai/post/most-accurate-rag) | Finance, Tax & Consulting | English | Text, Tables, Charts, Infographics | 1,141 | 92 |
| [KG-RAG](https://github.com/docugami/KG-RAG-datasets/tree/main/sec-10-q/data/v1) | Finance (SEC 10-Q) | English | Text, Tables | 1,037 | 195 |
| [Financebench](https://github.com/patronus-ai/financebench) | Finance (Public Equity) | English | Text, Tables | 54,057 | 150 |
| [DC767](https://digitalcorpora.org/) | General (Gov, NGO, Health) | English | Text, Tables | 54,730 | 488 |
| [HotPotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | Wikipedia-based question-answer pairs | English | Text | 2,673 (txt files) | 979 |
| [Google Frames](https://huggingface.co/datasets/google/frames-benchmark) | History, Sports, Science, Animals, Health | English | Text | 31,708 | 824 |

### [Vidore Dataset](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3#public-datasets)

| Dataset | Domain | Corpus Language | Main Modalities | # Pages | # Queries (with translations) |
|---|---|---|---|---|---|
| French Public Company Annual Reports | Finance-FR | French | Text, Table, Charts | 2,384 | 1,920 |
| U.S. Public Company Annual Reports | Finance-EN | English | Text, Table | 2,942 | 1,854 |
| Computer Science Textbooks | Computer Science | English | Text, Infographic, Tables | 1,360 | 1,290 |
| HR Reports from EU | HR | English | Text, Table, Charts | 1,110 | 1,908 |
| French Governmental Energy Reports | Energy | French | Text, Charts | 2,229 | 1,848 |
| USAF Technical Orders | Industrial | English | Text, Tables, Infographics, Images | 5,244 | 1,698 |
| FDA Reports | Pharmaceuticals | English | Text, Charts, Images, Infographic, Tables | 2,313 | 2,184 |
| French Physics Lectures | Physics | French | Text, Images, Infographics | 1,674 | 1,812 |


## Evaluation Methodology

Our primary evaluation metric is end-to-end RAG answer accuracy, measured using the [NVIDIA Answer Accuracy metric from RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/). Each response is rated on a 0–4 scale by an LLM judge, with scores normalized to a range for reporting. We chose [mistralai/Mixtral-8x22B-Instruct-v0.1](https://build.nvidia.com/mistralai/mixtral-8x22b-instruct) as the LLM judge, guided by performance on the [Judge’s Verdict](https://huggingface.co/spaces/nvidia/judges-verdict) benchmark.

Full evaluation pipeline: [evaluation_01_ragas.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/evaluation_01_ragas.ipynb)

- Metric: Accuracy, defined as the degree to which generated responses align with the ground truth answers.
- Pipeline configuration: All experiments were run using the default configuration.
- Generation models:
  - LLM: nvidia/llama-3.3-nemotron-super-49b-v1.5
  - VLM: nvidia/nemotron-nano-vl-12b-v2
- Judge model: mistralai/Mixtral-8x22B-Instruct-v0.1

## Configuration and Accuracy Results

We tested four main configurations to evaluate how ["Reasoning" (Think On)](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/enable-nemotron-thinking.md) and ["Vision Language Model" (VLM)](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/vlm.md) features influence accuracy. In the VLM-based generation pipeline, image captioning was enabled during data ingestion. For text-only datasets, we excluded the VLM-based generation setup from evaluation.

| Dataset | LLM (Reasoning Off) | LLM (Reasoning On) | VLM (Reasoning Off) | VLM (Reasoning On) |
|---|---|---|---|---|
| FinanceBench | 0.612 | 0.668 | 0.622 | 0.697 |
| KG-RAG | 0.569 | 0.593 | 0.596 | 0.643 |
| RAGBattle | 0.812 | 0.818 | 0.867 | 0.842 |
| DC767 | 0.906 | 0.899 | 0.907 | 0.897 |
| Hotpotqa | 0.672 | 0.676 | n/a | n/a |
| Google Frames | 0.486 | 0.597 | n/a | n/a |

The table in the following section summarizes the accuracy scores for each dataset across our experimental configurations.

### Vidore-V3 Results

For the Vidore-v3 evaluation, we combined all domains into a single collection and then performed domain-specific evaluations.

| Dataset subsets | LLM (Reasoning Off) | LLM (Reasoning On) | VLM (Reasoning Off) | VLM (Reasoning On) |
|---|---|---|---|---|
| Computer Science | 0.894 | 0.882 | 0.927 | 0.931 |
| Energy | 0.751 | 0.765 | 0.802 | 0.824 |
| Finance EN | 0.699 | 0.718 | 0.758 | 0.766 |
| Pharmaceuticals | 0.759 | 0.775 | 0.849 | 0.858 |
| HR | 0.726 | 0.735 | 0.767 | 0.804 |
| Industrial | 0.677 | 0.674 | 0.733 | 0.758 |
| Physics | 0.840 | 0.806 | 0.903 | 0.910 |
| Finance FR | 0.639 | 0.647 | 0.683 | 0.687 |


## Key Results

The following sections describe the key results from our analysis.

### The "Reasoning Dividend" in FinanceBench and KG-RAG

For FinanceBench and KG-RAG datasets we have observed improved accuracy with reasoning on.

Why it makes sense

- FinanceBench is heavily table-centric—about 75% of queries involve tables—and many of these require mathematical operations or extracting data across multiple line items. Simple retrieval is not sufficient; the model must perform an explicit reasoning step to carry out the necessary arithmetic and cross-referencing to match the human-annotated ground truth.

- KG-RAG requires temporal reasoning (for example, comparing Q3 2022 with Q1 2023). Without reasoning enabled, the model may retrieve the correct company but the wrong fiscal quarter. Turning Reasoning On lets the LLM check dates and periods before finalizing its answer.

### The Multimodal Unlock: Decoding Visual Complexity in ViDoRe and RAGBattlePacket

Across both the ViDoRe benchmark and RAGBattlePacket, we saw best results when moving from a text-only LLM to a VLM. RAGBattlePacket reached its highest baseline accuracy (0.867) simply by enabling the VLM, and ViDoRe showed broad gains across nearly all of its diverse sub-domains.

Why it makes sense

- Preserving Spatial Layouts (ViDoRe): Sub-domains like Finance and Pharmaceuticals depend on rigid tables and charts that text-only pipelines often fail to capture. A VLM can directly “see” and preserve these structures, leading to higher accuracy on this benchmark.
- Targeting Visual Queries (RAGBattlePacket): About 10% of RAGBattlePacket queries focus on charts, bar graphs, and customer journey diagrams, which standard pipelines often hallucinate on or ignore. A VLM can directly interpret these visuals, returning precise percentages and preserving the underlying structure.

### Semantic Robustness in DC767

This dataset showed the highest overall stability, maintaining roughly 0.90 or higher accuracy across almost all configurations.

Why it makes sense

Because the dataset is about 70% text-based prose, it relies heavily on high-quality embeddings and semantic search. Our core retriever is clearly optimized for dense text retrieval, as adding Vision or Reasoning produced only a marginal gain (about a 1.1% change). This suggests that our base RAG engine is already very strong for standard retriever-focused tasks.

### Reasoning as the Catalyst in Google Frames

This dataset demonstrated the true impact of active reasoning on complex, multi-hop queries. By turning reasoning on, the model achieved a massive leap in overall performance. This gain represents our most significant improvement driven purely by logical processing.

Why it makes sense

Google Frames targets complex queries that require synthesizing facts across multiple documents while tracking overlapping constraints. A standard LLM often struggles to keep all these parameters in mind in a single pass. Turning on reasoning enables the model to systematically decompose multi-step logic and verify dependencies, which is essential for accurate factual extraction.

## Related Topics

- [Evaluate Your NVIDIA RAG Blueprint System](evaluate.md)
- [Enable Reasoning in Nemotron LLM Models](enable-nemotron-thinking.md)
- [VLM-Based Inferencing in RAG](vlm.md)
- [Image Captioning Support](image_captioning.md)
- [Best Practices for Common Settings](accuracy_perf.md)
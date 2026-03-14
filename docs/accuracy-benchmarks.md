<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Benchmarking RAG Accuracy: Evaluating LLM Reasoning and VLM Integration

In the rapidly evolving landscape of Retrieval-Augmented Generation (RAG), the difference between a "good" and a "production-ready" system often lies in how the pipeline handles complex reasoning and multimodal data. To quantify these improvements, our team performed extensive benchmarking across various configurations, focusing on the impact of LLM reasoning ("Think" mode) and Vision-Language Models (VLM).

## Benchmarked Datasets

We focused our analysis on seven key public datasets that represent diverse challenges, from financial analysis to complex structural document parsing.

| Dataset | Domain | Corpus Language | Main Modalities | # Pages | # Queries |
|---|---|---|---|---|---|
| RagBattlepacket | Finance, Tax & Consulting | English | Text, Tables, Charts, Infographics | 1,141 | 92 |
| KG-RAG | Finance (SEC 10-Q) | English | Text, Tables | 1,037 | 195 |
| Financebench | Finance (Public Equity) | English | Text, Tables | 54,057 | 150 |
| DC767 | General (Gov, NGO, Health) | English | Text, Tables | 54,730 | 488 |
| HotPotQA | Wikipedia-based question-answer pairs | English | Text | 2,673 (txt files) | 979 |
| Google Frames | History, Sports, Science, Animals, Health | English | Text | 31,708 | 824 |

### Vidore Dataset

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

We use end-to-end RAG answer accuracy as the primary metric via the [NVIDIA Answer Accuracy metric from RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/). Each response is scored on a 0–4 scale using a LLM as Judge. Scores are normalized to [0, 1] for reporting. We have selected `mistralai/Mixtral-8x22B-Instruct-v0.1` as the LLM judge based on the Judge's verdict benchmark results.

> Full evaluation pipeline: [evaluation_01_ragas.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/evaluation_01_ragas.ipynb)

**Metric:** We use accuracy, specifically measuring the correctness of response with respect to ground truth answer.

**Pipeline configuration:** All experiments have been conducted with default configuration.

**Generation models:**

| Role | Model |
|---|---|
| LLM | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| VLM | `nvidia/nemotron-nano-vl-12b-v2` |
| Judge | `mistralai/Mixtral-8x22B-Instruct-v0.1` |

## Configuration & Accuracy Results

We experimented with four primary configurations to see how "Reasoning" (Think On) and "Vision Language Model" (VLM) capabilities moved the needle on accuracy. For VLM based generation pipeline, we have also enabled VLM based image captioning during ingestion. Also for text only datasets we have not evaluated the VLM based generation configuration.

| Dataset | LLM (Reasoning Off) | LLM (Reasoning On) | VLM (Reasoning Off) | VLM (Reasoning On) |
|---|---|---|---|---|
| FinanceBench | 0.612 | 0.668 | 0.622 | 0.697 |
| KG-RAG | 0.569 | 0.593 | 0.596 | 0.643 |
| RAGBattle | 0.812 | 0.818 | 0.867 | 0.842 |
| DC767 | 0.906 | 0.899 | 0.907 | 0.897 |
| Hotpotqa | 0.672 | 0.676 | n/a | n/a |
| Google Frames | 0.486 | 0.597 | n/a | n/a |

The table below summarizes the accuracy scores for each dataset across our experimental configurations.

### Vidore-V3 Results

For Vidore-v3 evaluation we have ingested the full corpus consisting of all the domains in a single collection and then carried out the domain specific evaluations.

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


## Key Takeaways

### The "Reasoning Dividend" in FinanceBench & KG-RAG

For FinanceBench and KG-RAG datasets we have observed improved accuracy with reasoning on.

**Why it makes sense**

FinanceBench is dominated by tables (75% of queries) that often require mathematical operations or data extraction from multiple line items. Simple retrieval isn't enough; the model needs the reasoning step to perform the arithmetic and cross-referencing required by the human-annotated ground truth.

KG-RAG requires temporal reasoning (comparing Q3 2022 vs. Q1 2023). Without reasoning enabled, a model might retrieve the correct company but the wrong fiscal quarter. The "Reasoning On" mode allows the LLM to verify dates and periods before finalizing the answer.

### The Multimodal Unlock: Decoding Visual Complexity (ViDoRe & RAGBattlePacket)

Across both ViDoRe benchmark and RAGBattlePacket, peak performance was consistently achieved when shifting from a text-only LLM to a VLM. RAGBattlePacket hit its highest baseline accuracy (0.867) purely by engaging the VLM, while ViDoRe saw sweeping improvements across almost all of its diverse sub-domains.

**Why it makes sense**

- **Preserving Spatial Layouts (ViDoRe):** Sub-domains like Finance and Pharmaceuticals rely on rigid tables and charts that text based pipeline misses. The VLM natively "sees" and preserves these structures, overall improving accuracy in this dataset.
- **Targeting Visual Queries (RAGBattlePacket):** With 10% of RAGBattlePacket queries specifically targeting charts, bar graphs, and customer journey diagrams, standard pipelines often "hallucinate" or skip the data entirely. The VLM explicitly reads these infographics, providing precise percentage readouts and structural context.

### Semantic Robustness in DC767

This dataset showed the highest overall stability, maintaining ~0.90+ accuracy across almost all configurations.

**Why it makes sense**

The dataset is 70% text-based prose, it relies heavily on high-quality embeddings and semantic search. Our core retriever is clearly highly optimized for dense text retrieval, as the addition of Vision or Reasoning provided only marginal gains (1.1% change). It proves our "base" RAG engine is exceptionally strong for standard retriever tasks.

### Reasoning as the Catalyst in Google Frames

This dataset demonstrated the true impact of active reasoning on complex, multi-hop queries. By turning reasoning on, the model achieved a massive leap in overall performance. This gain represents our most significant improvement driven purely by logical processing.

**Why it makes sense**

Google Frames focuses on complex queries that require synthesizing facts across multiple documents and tracking overlapping constraints. A standard LLM struggles to hold all these parameters in a single pass. Activating reasoning allows the model to systematically break down multi-step logic and verify dependencies, which is critical for rigorous factual extraction.


## Related Topics

- [Evaluate Your NVIDIA RAG Blueprint System](evaluate.md)
- [Enable Reasoning in Nemotron LLM Models](enable-nemotron-thinking.md)
- [VLM-Based Inferencing in RAG](vlm.md)
- [Image Captioning Support](image_captioning.md)
- [Best Practices for Common Settings](accuracy_perf.md)

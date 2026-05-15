---
name: rag-evaluate-quality
description: Evaluate NVIDIA RAG Blueprint answer quality, retrieval recall, RAGAS metrics, accuracy benchmarks, performance benchmarks, and evaluation notebooks. Use when the user asks to run RAG evaluation, score answer quality, measure retrieval recall, compare configurations, or validate accuracy and performance.
version: "1.0.0"
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
  tags:
    - rag
    - evaluation
    - ragas
---

# RAG Evaluate Quality

## Overview

Run or guide quality, recall, accuracy, and performance evaluations for RAG
deployments and configuration changes.

## Prerequisites

- Confirm target deployment, dataset, metrics, and evaluation surface.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Ask the user to classify the dataset and confirm the target environment is
  approved before sending confidential data to external endpoints.
- Read `references/evaluation.md` before selecting a workflow.
- Do not send confidential datasets to unapproved endpoints.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate dataset, ground truth availability, deployment mode, and metrics.
2. Prepare notebook, script, or benchmark inputs.
3. Execute the evaluation workflow.
4. Verify outputs are complete and tied to the intended config.
5. Report metrics, configuration, dataset, and actionable interpretation.

## Reference

- `references/evaluation.md`
- `../../docs/evaluate.md`
- `../../docs/accuracy-benchmarks.md`
- `../../docs/perf-benchmarks.md`
- `../../docs/accuracy_perf.md`
- `../../notebooks/evaluation_01_ragas.ipynb`
- `../../notebooks/evaluation_02_recall.ipynb`

## Error Handling

If evals fail, separate dataset issues, dependency issues, endpoint issues, and
metric-calculation issues. Do not compare runs unless the dataset, config, and
model versions are recorded.

## Examples

- "Run RAGAS evaluation."
- "Measure retrieval recall."
- "Compare accuracy profile versus performance profile."
- "Explain the eval notebook results."

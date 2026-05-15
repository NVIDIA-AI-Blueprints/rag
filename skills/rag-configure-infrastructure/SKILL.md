---
name: rag-configure-infrastructure
description: "Configure RAG infrastructure settings: LLM, embedding, reranking, OCR, parse, vector database, endpoints, model profiles, GPU assignment, ports, and API key wiring. Use when the user asks to change models, switch hosted/local NIMs, configure Milvus or Elasticsearch, update model profiles, or tune service endpoints."
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
---

# RAG Configure Infrastructure

## Overview

Configure the model and infrastructure layer for RAG deployments: LLMs,
embedding models, rerankers, OCR/parse models, vector databases, endpoints,
profiles, ports, and GPU assignments.

## Prerequisites

- Detect deployment mode before editing config.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Read `references/infrastructure.md` for source-of-truth files and docs.
- Check API key presence without printing key values.
- Verify GPU availability before assigning local NIMs.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate active deployment mode and config source.
2. Identify which component changes: LLM, embedding, reranker, OCR, parse,
   vector DB, endpoint, key, model profile, port, or GPU assignment.
3. Prepare changes in the active config file.
4. Restart only affected services when possible.
5. Verify health and run a small query or ingestion action that exercises the
   changed component.
6. Report changed settings without exposing secrets.

Ask for explicit confirmation before changing vector database backends, schema
settings, service ports, GPU assignments, or production endpoint configuration.

## Reference

- `references/infrastructure.md`
- `../../docs/change-model.md`
- `../../docs/model-profiles.md`
- `../../docs/change-vectordb.md`
- `../../docs/milvus-configuration.md`
- `../../docs/nemoretriever-ocr.md`
- `../../docs/nemotron-parse-extraction.md`
- `../../docs/llm-params.md`
- `../../docs/service-port-gpu-reference.md`

## Error Handling

If config and live service environment disagree, report stale runtime config and
restart requirements. If a local NIM cannot start, inspect GPU, port, image, and
API key issues before changing models again.

## Examples

- "Switch the LLM endpoint to NVIDIA-hosted."
- "Change the embedding model."
- "Use Elasticsearch instead of Milvus."
- "Move the reranker to GPU 1."

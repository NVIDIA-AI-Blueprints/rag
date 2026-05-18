---
name: rag-enable-vlm
description: Enable, configure, or verify RAG Vision Language Model support, VLM embeddings, multimodal query, and image captioning. Use when the user asks to query images, use VLM embeddings, caption images, process multimodal PDFs, or enable image plus text retrieval in the RAG Blueprint.
version: "1.0.0"
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
  tags:
    - rag
    - vlm
    - multimodal
---

# RAG Enable VLM

## Overview

Enable and validate multimodal RAG features: VLM inference, VLM embeddings,
image captioning, multimodal query, and image-aware ingestion/retrieval.

## Prerequisites

- Confirm deployment mode and hardware support.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Ask the user to classify image/document content and confirm the target
  environment is approved before sending confidential multimodal data.
- Read `references/vlm.md` before changing config.
- Check API key presence without printing values.
- Verify GPU and endpoint constraints in the support matrix.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate whether the request is about VLM generation, VLM embeddings, image
   captioning, multimodal query, or multimodal ingestion.
2. Identify the active deployment config.
3. Prepare endpoint, model, GPU, and env changes.
4. Restart affected RAG, ingestor, and NIM services.
5. Verify with a small image or multimodal query.
6. Report enabled components and any unsupported hardware limitations.

## Reference

- `references/vlm.md`
- `../../docs/vlm.md`
- `../../docs/vlm-embed.md`
- `../../docs/multimodal-query.md`
- `../../docs/image_captioning.md`
- `../../docs/support-matrix.md`
- `../../docs/service-port-gpu-reference.md`
- `../../notebooks/image_input.ipynb`

## Error Handling

If VLM verification fails, check hardware support, endpoint reachability, model
profile, API key presence, image payload shape, and whether ingestion used
compatible multimodal settings.

Route broad failures, logs, and unknown unhealthy-service issues to
`rag-troubleshoot-blueprint`; return here only when the root cause is VLM
configuration.

## Examples

- "Enable image queries in RAG."
- "Use VLM embeddings for multimodal retrieval."
- "Caption images during ingestion."
- "Verify the VLM endpoint after configuration."

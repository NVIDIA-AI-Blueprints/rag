---
name: rag-configure-retrieval
description: Configure RAG retrieval, hybrid search, multi-collection retrieval, reranking, metadata filters, natural-language filter generation, retrieval topK, thresholds, and accuracy or performance profiles. Use when the user asks to tune search quality, enable hybrid search, query multiple collections, or fix retrieval relevance.
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
---

# RAG Configure Retrieval

## Overview

Configure retrieval behavior for RAG collections, including hybrid search,
multi-collection retrieval, filters, reranking, and performance/accuracy
profiles.

## Prerequisites

- Confirm active deployment mode and config source.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Read `references/retrieval.md` before making retrieval changes.
- Determine whether existing collections must be re-created or re-ingested.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate current retrieval settings and user goal.
2. Identify whether the change affects ingestion, query-time payloads, RAG
   server config, ingestor config, or vector DB schema.
3. Prepare config changes and warn when re-ingestion is required.
4. Restart affected services.
5. Verify with a search or generate query and inspect citations/chunks.
6. Report settings, verification output, and any re-ingestion requirement.

Ask for explicit confirmation before re-creating collections, changing vector
schema/index settings, or triggering bulk re-ingestion.

## Reference

- `references/retrieval.md`
- `../../docs/hybrid_search.md`
- `../../docs/multi-collection-retrieval.md`
- `../../docs/custom-metadata.md`
- `../../docs/accuracy_perf.md`
- `../../docs/python-client.md`
- `../../notebooks/retriever_api_usage.ipynb`
- `../../notebooks/nb_metadata.ipynb`

## Error Handling

If relevance is poor, check ingestion state, collection selection, filter syntax,
reranker state, topK, chunking, and whether the query asks for information not
present in the documents.

## Examples

- "Enable hybrid search."
- "Tune retrieval for accuracy."
- "Query five collections with reranking."
- "Fix metadata filter generation."

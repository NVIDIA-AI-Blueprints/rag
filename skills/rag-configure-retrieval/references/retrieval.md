# Retrieval Reference

## Routing

| Goal | Source docs |
|---|---|
| Hybrid dense + sparse search | `docs/hybrid_search.md` |
| Multi-collection retrieval | `docs/multi-collection-retrieval.md` |
| Custom metadata and filters | `docs/custom-metadata.md` |
| Accuracy and performance profiles | `docs/accuracy_perf.md` |
| Python search client | `docs/python-client.md` |

## Notes

- Hybrid search can require re-creating or re-ingesting collections.
- Multi-collection retrieval has collection-count and reranker constraints; read
  the current docs before changing payloads.
- Metadata filter syntax depends on the vector database backend.
- Accuracy profile changes can affect latency and GPU usage.

## Verification

Use a known query against a known collection before and after the change. Compare
retrieved chunks, citations, and scores where available. Report whether quality
changed, not only whether the API returned 200.


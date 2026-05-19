# VLM Reference

## Routing

| Goal | Source docs |
|---|---|
| VLM generation and config | `docs/vlm.md` |
| VLM embeddings | `docs/vlm-embed.md` |
| Image plus text query | `docs/multimodal-query.md` |
| Image captioning | `docs/image_captioning.md` |
| Hardware and ports | `docs/support-matrix.md`, `docs/service-port-gpu-reference.md` |

## Verification

Use the smallest representative image workflow:

1. Verify RAG and ingestor health.
2. Verify VLM endpoint health or container state.
3. Submit a known image or multimodal query.
4. Confirm the response includes image-grounded content or expected retrieved
   multimodal chunks.

Report unsupported GPU or deployment combinations instead of trying unrelated
fallbacks.


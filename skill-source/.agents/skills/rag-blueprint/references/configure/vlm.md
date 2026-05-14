# VLM, VLM Embeddings & Image Captioning

## When to Use
User wants image understanding, visual content analysis, VLM inference, multimodal embeddings, VLM reranking, VLM reasoning output, or image captioning during ingestion.

## Restrictions
- Not available on B200 GPUs. Source docs list H100 or A100 for VLM inference and multimodal query; do not recommend RTX PRO 6000 for VLM unless the current model support matrix explicitly confirms it.
- Requires extra GPU (GPU 1+ for 2-GPU systems, GPU 2+ for 3+ GPUs with fallback)
- VLM embeddings: experimental and PDF-only. `docs/multimodal-retriever.md` says summary generation does not work when VLM embedding is enabled; `docs/multimodal-query.md` includes `SUMMARY_LLM*` examples for deployments without `nim-llm`. If asked about summaries with VLM embeddings, flag this documentation conflict and validate on the target deployment instead of promising support.
- Image captioning on Helm: on-prem only (modify `values.yaml` to enable)

## Process
1. Detect the deployment mode (Docker / Helm / Library). Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read the relevant source doc for detailed steps:
   - VLM generation: `docs/vlm.md`
   - VLM embeddings and VLM reranker: `docs/multimodal-retriever.md`
   - Image captioning: `docs/image_captioning.md`
3. Start VLM NIM (self-hosted) or configure cloud endpoint (NVIDIA-hosted)
4. Set the required variables in the active config:
   - Enabling: `ENABLE_VLM_INFERENCE=true` and `APP_NVINGEST_EXTRACTIMAGES=True`
   - Disabling: re-comment those variables in the env file
5. Restart affected services and verify with a health check + image-containing document query

## Decision Table

| Goal | Source Doc | Docker Profile | Notes |
|------|-----------|---------------|-------|
| VLM replaces LLM | `docs/vlm.md` | `--profile vlm-generation` | LLM not started; set `VLM_TO_LLM_FALLBACK=false` |
| VLM + LLM fallback | `docs/vlm.md` | `--profile vlm-only` | Needs 3+ GPUs; both VLM and LLM running |
| VLM embeddings | `docs/multimodal-retriever.md` | `--profile vlm-embed` | Experimental; requires re-ingestion |
| VLM reranker | `docs/multimodal-retriever.md` | `--profile vlm-rerank` or `--profile vlm-rag` | Set `APP_RANKING_MODELNAME` to `rerank-vl` model and `ENABLE_VLM_RERANKER_IMAGE_INPUT=True` |
| Image captioning | `docs/image_captioning.md` | `--profile vlm-only` | Requires VLM NIM; Helm: on-prem only |
| Multimodal query | `docs/multimodal-query.md` | (depends on VLM mode) | Image + text querying |

## Agent-Specific Notes

- `--profile vlm-generation` skips the LLM entirely — use `--profile vlm-only` for fallback mode
- `VLM_TO_LLM_FALLBACK` defaults to `true`, but `vlm-generation` profile does not start LLM
- Helm VLM: disable `nim-llm` and enable `nim-vlm` (VLM uses LLM's GPU allocation)
- Helm fallback: keep both `nim-vlm` and `nim-llm` enabled, set `VLM_TO_LLM_FALLBACK: "true"`
- VLM context window is limited — keep queries self-contained
- VLM reasoning streams final answer in `content` and reasoning in `reasoning_content`; `VLM_FILTER_THINK_TOKENS` is retained for compatibility and no longer wraps reasoning in text sentinels
- Image queries bypass reranking, including VLM reranking
- Image captioning known issue: files without graphs/charts/tables/plots fail to ingest when captioning is enabled

### Key Env Vars (always needed)
- `ENABLE_VLM_INFERENCE=true` — master toggle
- `APP_NVINGEST_EXTRACTIMAGES=True` — extract images during ingestion
- `VLM_MS_GPU_ID=<gpu-id>` — self-hosted GPU assignment
- `APP_VLM_ENABLE_THINKING`, `APP_VLM_THINKING_TOKEN_BUDGET`, `VLM_FILTER_THINK_TOKENS` — VLM reasoning behavior

## Notebooks
- `notebooks/image_input.ipynb` — Multimodal queries with VLM (text + image)

## Source Documentation
- `docs/vlm.md` — VLM generation (self-hosted, NVIDIA-hosted, Helm, Library)
- `docs/multimodal-retriever.md` — VLM embeddings (experimental)
- `docs/image_captioning.md` — Image captioning during ingestion
- `docs/multimodal-query.md` — Image + text querying
- `docs/service-port-gpu-reference.md` — default GPU assignments for VLM and other NIMs

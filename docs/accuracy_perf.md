<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Best Practices for Common NVIDIA RAG Blueprint Settings

Use this documentation to learn how to configure the performance of the [NVIDIA RAG Blueprint](readme.md) according to your specific use-case.
Default values are set to balance between accuracy and performance.
Change the setting if you want different behavior.


## Ingestion and Chunking

| Name                 | Default    | Description         | Advantages           | Disadvantages            |
|----------------------|------------|---------------------|----------------------|--------------------------|
| `APP_NVINGEST_CHUNKOVERLAP` | `150` | Increase overlap to ensure smooth transitions between chunks. | - Larger overlap provides smoother transitions between chunks. <br/>  | - Might increase processing overhead. <br/> |
| `APP_NVINGEST_CHUNKSIZE` | `512` | Increase chunk size for more context. | - Larger chunks retain more context, improving coherence. <br/> | - Larger chunks increase embedding size, slowing retrieval. <br/> - Longer chunks might increase latency due to larger prompt size. <br/> |
| `APP_NVINGEST_ENABLEPDFSPLITTER` | `true` | Set to `true` to perform chunk-based splitting of pdfs after the default page-level extraction occurs. Recommended for PDFs that are mostly text content. | - Provides more granular content segmentation. <br/> | - Can increase the number of chunks and slow down the ingestion process. <br/> |
| `APP_NVINGEST_EXTRACTCHARTS` | `true` | Set to `true` to extract charts. | - Improves accuracy for documents that contain charts. <br/> | - Increases ingestion time. <br/> |
| `APP_NVINGEST_EXTRACTIMAGES` | `false` | Set to `true` to enable image captioning during ingestion. For details, refer to [Image Captioning Support](image_captioning.md). | - Enhances multimodal retrieval accuracy for documents having images. <br/> | - Increased processing time during ingestion. <br/> - Requires additional GPU resources for VLM model deployment. <br/> |
| `APP_NVINGEST_EXTRACTINFOGRAPHICS` | `false` | Set to `true` to extract infographics and text-as-images. | - Improves accuracy for documents that contain text in image format. <br/> | - Increases ingestion time. <br/> |
| `APP_NVINGEST_EXTRACTTABLES` | `true` | Set to `true` to extract tables. | - Improves accuracy for documents that contain tables. <br/> | - Increases ingestion time. <br/> |
| `APP_NVINGEST_PDFEXTRACTMETHOD` | `pdfium` | Set to `nemoretriever_parse` to use nemoretriever_parse to extract pdfs. For details, refer to [PDF extraction with Nemoretriever Parse](nemoretriever-parse-extraction.md). | - Provides enhanced PDF parsing and structure understanding. <br/> - Better extraction of complex PDF layouts and content. <br/> | - Requires additional GPU resources for the Nemoretriever Parse service. <br/> - Only supports PDF format documents. <br/> - Not supported on NVIDIA B200 GPUs. <br/> |
| `APP_NVINGEST_SEGMENTAUDIO` | `false` | Set to `true` to enable audio segmentation. For details, refer to [Audio Ingestion Support](audio_ingestion.md). | - Segments audio files based on commas and other punctuation marks for more granular audio chunks. <br/> - Improves downstream processing and retrieval accuracy for audio content. <br/> | - Might increase processing time during audio ingestion. <br/> |
| `INGEST_DISABLE_DYNAMIC_SCALING` | `true` | Set to `true` to disable dynamic scaling. | - When disabled, provides better ingestion performance and throughput. <br/> - When disabled, more predictable resource allocation and processing behavior. <br/> | - When disabled, higher memory utilization as resources are statically allocated. <br/> - When disabled, less efficient memory usage when processing smaller workloads. <br/> |



## Retrieval and Generation

| Name                 | Default    | Description         | Advantages           | Disadvantages            |
|----------------------|------------|---------------------|----------------------|--------------------------|
| - `APP_LLM_MODELNAME` <br/> - `APP_EMBEDDINGS_MODELNAME` <br/> - `APP_RANKING_MODELNAME` <br/> | See description | The default models are the following: <br/>- `nvidia/llama-3.3-nemotron-super-49b-v1.5` <br/> - `nvidia/llama-3.2-nv-embedqa-1b-v2` <br/> - `nvidia/llama-3.2-nv-rerankqa-1b-v2` <br/><br/>You can use larger models.  For details, refer to [Change the Inference or Embedding Model](change-model.md). | - Higher accuracy with better reasoning and a larger context length. <br/> | - Slower response time. <br/> - Higher inference cost. <br/> - Higher GPU requirement. <br/>  |
| `APP_VECTORSTORE_SEARCHTYPE` | `dense` | Set to `hybrid` to enable hybrid search. For details, refer to [Hybrid Search Support](hybrid_search). | - Can provide better retrieval accuracy for domain-specific content. <br/> | - Can induce higher latency for large number of documents. <br/> |
| `ENABLE_GUARDRAILS` | `false` | Set to `true` to enable NeMo Guardrails. For details, refer to [Nemo Guardrails Support](nemo-guardrails.md). | - Applies input/output constraints for better safety and consistency. <br/> | - Significant increased processing overhead for additional LLM calls. <br/> - Needs additional GPUs to deploy guardrails-specific models locally. <br/> |
| `ENABLE_QUERYREWRITER` | `false` | Set to `true` to enable query rewriting.  For details, refer to [Query Rewriting Support](query_rewriter.md). | - Enhances retrieval accuracy for multi-turn scenarios by rephrasing the query. <br/> | - Adds an extra LLM call, increasing latency. <br/> |
| `ENABLE_REFLECTION` | `false` | Set to `true` to enable self-reflection. For details, refer to [Self-Reflection Support](self-reflection.md). | - Can improve the response quality by refining intermediate retrieval and final LLM output. <br/> | - Significantly higher latency due to multiple iterations of LLM model call. <br/> - You might need to deploy a separate judge LLM model, increasing GPU requirement. <br/> |
| `ENABLE_RERANKER`    | `true` | Set to `true` to use the reranking model.    | - Improves accuracy by selecting better documents for response generation. <br/> | - Increases latency due to additional processing. <br/> - Additional hardware requirements for self-hosted on premises deployment. <br/>   |
| `ENABLE_VLM_INFERENCE` | `false`    | Set to `true` to use the Vision-Language Model (VLM) for response generation. For details, refer to [VLM for Generation](vlm.md).  | - Enables analysis of retrieved images alongside text for richer, multimodal responses. <br/> - Can process up to 4 images per citation. <br/> - Useful for document Q&A, visual search, and multimodal chatbots. <br/> | - Requires additional GPU resources for VLM model deployment. <br/> - Increases latency due to image processing. <br/> |
| Reasoning in `llama-3.3-nemotron-super-49b-v1.5` | `/no_think` | Use `/think` to enable reasoning. For details, refer to [Enable Reasoning](enable-nemotron-thinking.md). | - Improves response quality through enhanced reasoning capabilities. <br/> - Yields more precise responses. The default model is verbose and works best with reasoning enabled. <br/> | - Can increase response latency due to additional thinking process. <br/> - Can increase token usage and computational overhead. <br/> |
| `RERANKER_CONFIDENCE_THRESHOLD` | `0.0` | Filters out retrieved chunks if reranker relevance is lower than this threshold. We recommend that you set this value between `0.3` and `0.5` to balance quality and coverage. For details, refer to [Use the Python Package](python-client.md). | - Faster retrieval by processing fewer documents. <br/> - Can improve accuracy by excluding low-relevance documents. <br/> | - Requires `ENABLE_RERANKER` set to `true` for effective filtering. <br/> - Might filter out too many chunks if the threshold is set high, causing no response from the RAG server. <br/> |
| `RERANKER TOP K` | 10 | Increase `reranker TOP K` to increase the probability of relevant context being part of the top-k contexts. | Increasing the value can improve accuracy. | Increasing the value can increase latency. |
| `VDB TOP K` | 100 | Increase `VDB TOP K` to provide a larger candidate pool for reranking. | Increasing the value can improve accuracy. | Increasing the value can increase latency. |



## Advanced Ingestion Batch Mode Optimization

By default, the ingestion server processes files in parallel batches, distributing the workload to multiple workers for efficient ingestion.
This parallel processing architecture helps optimize throughput while managing system resources effectively.
You can use the following environment variables to configure the batch processing behavior.

> [!CAUTION]
> These variables are not "set it and forget it" variables.
> These variables require trial and error tuning for optimal performance.


| Name                 | Default    | Description         | Advantages           | Disadvantages            |
|----------------------|------------|---------------------|----------------------|--------------------------|
| `NV_INGEST_CONCURRENT_BATCHES` | 4 | Controls the number of parallel batch processing streams. | - You can increase this for systems with high memory capacity. <br/> | - Higher values require more system memory. <br/> - Requires careful tuning based on available system resources. <br/> |
| `NV_INGEST_FILES_PER_BATCH` | 16 | Controls how many files are processed in a single batch during ingestion. | - Adjust this to helps optimize memory usage and processing efficiency. <br/> | - Setting this too high can cause memory pressure. <br/> - Setting this too low can reduce throughput. <br/> |

> [!TIP]
> For optimal resource utilization, `NV_INGEST_CONCURRENT_BATCHES` times `NV_INGEST_FILES_PER_BATCH` should approximately equal `MAX_INGEST_PROCESS_WORKERS`.



## Related Topics

- [Model Profiles](model-profiles.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Deploy with Helm and MIG Support](mig-deployment.md)
- [Deploy with NIM Operator](deploy-nim-operator.md)

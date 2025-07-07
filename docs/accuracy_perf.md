<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Best practices for common NVIDIA RAG Blueprint settings

These parameters allow fine-tuning RAG performance based on specific accuracy vs. latency trade-offs. Choose the configurations based on use case needs! The default values are kept considering a balance between accuracy and performance. The default values and the environment variables controlling these settings are mentioned in (brackets)

## Retrieval and Ranking

- **Reranking model**
  - ✅ This improves accuracy by selecting better documents for response generation.
  - ❌ Increases latency due to additional processing. Additional model deployment will be needed for on-prem setting of NIMS.
  - Controlled using `ENABLE_RERANKER` environment variable. Default is on.

- **Enable Vision-Language Model (VLM) for response generation**
  - ✅ Enables analysis of retrieved images alongside text for richer, multimodal responses
  - ✅ Can process up to 4 images per citation
  - ✅ Useful for document Q&A, visual search, and multimodal chatbots
  - ❌ Requires additional GPU resources for VLM model deployment
  - ❌ Increases latency due to image processing
  - Controlled via `ENABLE_VLM_INFERENCE` environment variable. Default is off.
  - Check out [this](./vlm.md) section to learn more.

- **Enable Nemoretriever Parse for PDF Extraction**
  - ✅ Provides enhanced PDF parsing and structure understanding
  - ✅ Better extraction of complex PDF layouts and content
  - ❌ Requires additional GPU resources for the Nemoretriever Parse service
  - ❌ Only supports PDF format documents
  - ❌ Not supported on NVIDIA B200 GPUs
  - Controlled via `APP_NVINGEST_PDFEXTRACTMETHOD` environment variable. Default is pdfium.
  - Check out [this](./nemoretriever-parse-extraction.md) section to learn more.

- **Increase VDB TOP K and reranker TOP K**
  - ✅ VDB TOP K provides a larger candidate pool for reranking, which may improve accuracy. Reranker TOP K increases the probability of relevant context being part of the top-k contexts.
  - ❌ May increase retrieval latency as the value of TOP K increases.
  - Controlled using `VECTOR_DB_TOPK` and `APP_RETRIEVER_TOPK` environment variable.

- **Use a larger LLM model**
  - ✅ Higher accuracy with better reasoning and a larger context length.
  - ❌ Slower response time and higher inference cost. Also will have a higher GPU requirement.
  - Check out [this](./change-model.md) section to understand how to switch the inference models.

- **Enable Query rewriting**
  - ✅ Enhances retrieval accuracy for multi-turn scenario by rephrasing the query.
  - ❌ Adds an extra LLM call, increasing latency.
  - Check out [this](./query_rewriter.md) section to learn more. Default is off.

- **Enable Self-reflection**
  - ✅ May improve the response quality by refining intermediate retrieval and final LLM output.
  - ❌ Significantly higher latency due to multiple iterations of LLM model call.
  - ❌ May need a seperate judge LLM model to be deployed increasing GPU requirement.
  - Check out [this](./self-reflection.md) section to learn more. Default is off.

- **Enable Hybrid search in Milvus**
  - ✅ May provide better retrieval accuracy for domain-specific content
  - ❌ May induce slightly higher latency for large number of documents; default setting is dense search.

- **Vector Store Retriever Consistency Level**
  - ✅ **Bounded** : Searches with tolerance for some data loss, providing better performance for most use cases. If you are using a pre-ingested knowledge base which is not frequently updated, then use this consistency level to achieve better latency at the cost of no accuracy drop.
  - ✅ **Strong** : Waits for the latest data to be available before executing searches, ensuring highest accuracy. This is the recommended setting for continuous data ingestion pipelines, where retrieval time is within few minutes after ingestion is completed to achieve best accuracy.
  - ✅ **Session**: Searches include all data inserted by the current client session
  - Controlled via `APP_VECTORSTORE_CONSISTENCYLEVEL` environment variable. Default is "Strong".
  - **Trade-off**: Use "Bounded" consistency level for better performance, and "Strong" to ensure highest accuracy
  - **Reference**: [Milvus Consistency Level Documentation](https://milvus.io/docs/consistency.md)

- **Enable NeMo Guardrails**
  - ✅ Applies input/output constraints for better safety and consistency
  - ❌ Significant increased processing overhead for additional LLM calls. It always needs additional GPUs to deploy the guardrails specific models on-prem.
  - Check out [this](./nemo-guardrails.md) section to learn more. Default is off.

## Ingestion and Chunking

- **Extracting infographics**
  - ✅ Improves accuracy for documents containing text in image format (e.g. infographics in PDFs/PPTs)
  - ❌ Increases ingestion time. Can be disabled if documents don't contain text-as-images
  - Controlled via `APP_NVINGEST_EXTRACTINFOGRAPHICS` environment variable. Default is off

- **Extracting tables and charts**
  - ✅ Improves accuracy for documents having images of tables and charts.
  - ❌ Increases ingestion time. You can turn these off in case there are no images present in the ingested doc. refer to [this](./text_only_ingest.md) section.
  - Controlled via `APP_NVINGEST_EXTRACTTABLES` and `APP_NVINGEST_EXTRACTCHARTS` environment variables. Default is on for both.

- **Enable Image captioning during ingestion**
  - ✅ Enhances multimodal retrieval accuracy for documents having images.
  - ❌ Additional call to a vision-language model increases processing time during ingestion and also requires additional GPU to be available for on-prem deployment of the VLM model.
  - Check out [this](./image_captioning.md) section to learn more. Default is off.

- **Customize Chunk size**
  - ✅ Larger chunks retain more context, improving coherence
  - ❌ Larger increases embedding size, slowing retrieval
  - ❌ Longer chunks may increase latency due to larger prompt size
  - Controlled via `APP_NVINGEST_CHUNKSIZE` environment variable. Default value is 512.

- **Customize Chunk overlap**
  - ✅ More overlap ensures smooth transitions between chunks
  - ❌ May slightly increase processing overhead
  - Controlled via `APP_NVINGEST_CHUNKOVERLAP` environment variable. Default value is 150.

- **Customize PDF Splitting**
  - ✅ PDFs are extracted at the page level by default. When PDF splitting is enabled, chunk-based splitting is performed after page-level extraction for more granular content segmentation. Recommended for PDFs with pages with more text content
  - ❌ This may increase number of chunks and slightly slow down ingestion process
  - Controlled by `APP_NVINGEST_ENABLEPDFSPLITTER` environment variable. Default value is `True`.

## Ingestion Batch Mode Optimization

The ingestor server processes files in parallel batches by default, distributing the workload to nv-ingest workers for efficient ingestion. This parallel processing architecture helps optimize throughput while managing system resources effectively. The following parameters help fine-tune this batch processing behavior:

- **Files per batch (`NV_INGEST_FILES_PER_BATCH`)**
  - ✅ Controls how many files are processed in a single batch during ingestion
  - ✅ Helps optimize memory usage and processing efficiency
  - ❌ Setting too high may cause memory pressure
  - ❌ Setting too low may reduce throughput

- **Concurrent batches (`NV_INGEST_CONCURRENT_BATCHES`)**
  - ✅ Controls number of parallel batch processing streams
  - ✅ Can be increased for systems with high memory capacity
  - ❌ Higher values require more system memory
  - ❌ Requires careful tuning based on available system resources
  - ⚠️ Note: `NV_INGEST_CONCURRENT_BATCHES * NV_INGEST_FILES_PER_BATCH` should approximately equal `MAX_INGEST_PROCESS_WORKERS` for optimal resource utilization
  - ⚠️ Advanced users only: These parameters require trial and error tuning for optimal performance

These parameters allow fine-tuning RAG performance based on specific accuracy vs. latency trade-offs. Choose the configurations based on use case needs!

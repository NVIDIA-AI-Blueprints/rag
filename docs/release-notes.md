<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Release Notes for NVIDIA RAG Blueprint

This documentation contains the release notes for [NVIDIA RAG Blueprint](readme.md).



<!-- Beginning with version 2.4.0 -->
<!-- Delete all previous versions from this page -->
<!-- And populate the Previous Versions section at the end of this page with version 2.3.0 -->

## Release 2.3.2 (2025-12-25)

This release focuses on updating NIM dependencies.

It includes the following version upgrades:

- llama-3.2-nv-embedqa-1b-v2 → 1.10.1
- llama-3.3-nemotron-super-49b-v1.5 (nim-llm) → 1.14.0

All updates are applied to both Docker Compose files and Helm chart values, ensuring that all referenced image tags are aligned with the new versions.



## Release 2.3.0 (2025-10-14)

This release adds RTX6000 platform support, adds deployment by using NIM operator, improves vector database pluggability with the blueprint, and other changes.

### Highlights 

This release contains the following key changes:

- You can now deploy the RAG Blueprint on [RTX Pro 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).
- Migrated to [`llama-3.3-nemotron-super-49b-v1.5`](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) as the default LLM model.
- Added support to deploy the helm chart by using NVIDIA NIM operator. For details, refer to [Deploy NVIDIA RAG Blueprint with NIM Operator](deploy-nim-operator.md).
- Updated all NIMs, NVIDIA Ingest and third party dependencies to latest versions.
- Refactored to support custom 3rd-party vector database integration in a streamlined manner. For details, refer to [Building Custom Vector Database Operators](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/building_rag_vdb_operator.ipynb).
- Added support for [elasticsearch vector DB as an alternate to milvus](change-vectordb.md).
- Added opt-in [query decomposition support](query_decomposition.md).
- Added opt-in [nemoretriever-ocr support](nemoretriever-ocr.md).
- Added opt-in [VLM embedding support](vlm-embed.md)
- Custom metadata enhancements including the following. For details, refer to [Advanced Metadata Filtering](custom-metadata.md).
  - Added support for more data types.
  - Added opt-in support to generate filters using LLM yielding better accuracy.
  - Added an interactive notebook that showcases the new features. For details, refer to [Notebooks](notebooks.md).
- Added dependency check support for ingestor server /health API.
- Added support for configurable confidence threshold for retrieval from API layer.
- Added support to store NV-Ingest extraction results [directly from the filesystem](mount-ingestor-volume.md).
- Logging enhancements
- Added better latency data reporting for RAG server
  - API level enhancements for component level latency
  - Added dedicated Prometheus metric endpoint
- Added independent script to [showcase batch ingestion](https://github.com/NVIDIA-AI-Blueprints/rag/blob/release-v2.3.0/scripts/README.md)
- Enabled support for [GPU indexing with CPU search](milvus-configuration.md#gpu-indexing-with-cpu-search)
  - Exposed `APP_VECTORSTORE_EF` as a configurable parameter
- Added environment variables to control llm parameters LLM_MAX_TOKENS, LLM_TEMPERATURE and LLM_TOP_P
- Added notebooks for showcasing RAG evaluation using common metrics. For details, refer to [Notebooks](notebooks.md).
- Added unit tests and pre-commit hooks for maintaining code quality.
- Optimized container sizes by removing unnecessary packages and improving security.
- Refactored the rag-playground code including the following changes. For details, refer to [User Interface](user-interface.md).
  - Use React end to end. Next.js dependencies were deprecated.
  - More developer friendly and intuitive look and feel.
  - The `rag-playground` service is renamed to `rag-frontend`.
- Refactored helm chart support including the following changes. For details, refer to [Deploy with Helm](deploy-helm.md).
  - Expanded and reorganized Helm chart configuration, enabling granular control over service components, resource settings, and observability (tracing, metrics).
  - Introduced ConfigMap and service definitions to facilitate improved application deployment flexibility.
  - Implemented refined service account and secret management in Helm templates.
  - Added a new Helm values file for nim-operator to configure LLM model environment and component toggles.


### Removed
- Removed consistency level configuration support for Milvus.
- Removed `EMBEDDING_NIM_ENDPOINT` and `EMBEDDING_NIM_MODEL_NAME` environment variables for nv-ingest.
- Removed `ENABLE_MULTITURN` environment variable from rag-server.
- Removed `ENABLE_NEMOTRON_THINKING` environment variable from rag-server.


### Fixed Known Issues

The following are the known issues that are fixed in this version:

- Fixed support for long audio file ingestion.
- Fixed support to ingest images without charts/tables.
- Fixed requirement of rebuilding rag frontend container when LLM model name was changed.

For the full list of known issues, see [Known Issues](#all-known-issues).



## Release 2.2.1 (2025-07-22)

This is a minor patch release. 
This release updates to the latest `nv-ingest-client` version 25.6.3 to fix breaking changes introduced by [pypdfium2](https://github.com/pypdfium2-team/pypdfium2). 
For details, refer to [NVIDIA NV Ingest 25.6.3](https://github.com/NVIDIA/nv-ingest/releases/tag/25.6.3).



## Release 2.2.0 (2025-07-08)

This release adds B200 platform support, a native Python API, and major enhancements for multimodal and metadata features. It also improves deployment flexibility and customization across the RAG blueprint.

### Highlights 

This release contains the following key changes:

- You can now deploy the RAG Blueprint on [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/).
- You can use Multi Instance GPUs to reduce the GPU requirements to 3xH100. For details, refer to [Deploy on Kubernetes with Helm and MIG Support](mig-deployment.md).
- The RAG Blueprint project now uses `uv` as the package manager.
- Added support for [NVIDIA AI Workbench](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/deploy/workbench/README.md).
- Added support for native python API. For details, refer to [Python Client Package](python-client.md).
- Added support for custom metadata for files and metadata-based filtering. For details, refer to [Advanced Metadata Filtering](custom-metadata.md).
- Added support for multi collection based retrieval. For details, refer to [Multi-Collection Retrieval](multi-collection-retrieval.md).
- Added support for .mp3 and .wav files. For details, refer to [Audio Ingestion](audio_ingestion.md).
- Added support for vision language model-based inferencing for charts and images. For details, refer to [Vision Language Model for Generation](vlm.md).
- Added support for generating summaries of uploaded files. For details, refer to [Document Summarization](summarization.md).
- Added support for configurable vector store consistency levels (Bounded/Strong/Session) to optimize retrieval performance and accuracy trade-offs.
- Added support for non-blocking file upload to the user interface. For details, refer to [User Interface](user-interface.md).
- Added more efficient error reporting to the user interface for ingestion failures. For details, refer to [User Interface](user-interface.md).
- Added support for customizing prompts without rebuilding images. For details, refer to [Customize Prompts](prompt-customization.md).
- Added support to enable infographics, which improves accuracy for documents containing text in image format. For details, refer to [Ingestion and Chunking](accuracy_perf.md#ingestion-and-chunking).
- Optimized batch mode ingestion support to improve performance for multi-user concurrent file upload. For details, refer to [Advanced Ingestion Batch Mode Optimization](accuracy_perf.md#advanced-ingestion-batch-mode-optimization).
- Added support for enhanced pdf extraction. For details, refer to [Nemoretriever Parse](nemoretriever-parse-extraction.md).
- Added support for running CPU-based Milvus. For details, refer to [Milvus Configuration](milvus-configuration.md).
- Added support for running NV-Ingest as a standalone service for the RAG Blueprint. For more information, refer to [Deploy NV-Ingest Standalone](nv-ingest-standalone.md).
- Updated the API, including the following changes. For details, refer to [Migration Guide](migration_guide.md).
  - POST /collections is replaced by POST /collection for `ingestor-server`.
  - New endpoint GET /summary added for rag-server.
  - Metadata information available as part of GET /collections and GET /documents API.



## Release 2.1.0 (2025-05-13)

This release reduces the overall GPU requirement for the deployment of the RAG Blueprint. 
This release also improves the performance and stability for both Docker- and Helm-based deployments.

### Highlights 

This release contains the following key changes:

- The overall GPU requirement is now reduced to 2xH100 / 3xA100. For details, refer to [Support Matrix](support-matrix.md).
  - Changed the default LLM model to [llama-3_3-nemotron-super-49b-v1](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1). This reduces overall GPU needed to deploy LLM model to 1xH100 / 2xA100.
  - Changed the default GPU needed for all other NIMs (ingestion and reranker) to 1xH100 / 1xA100.
- The Helm chart is now published on the NGC Public registry. For details, refer to [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/blueprint/helm-charts/nvidia-blueprint-rag).
- Helm chart customization is now available for many optional features. For details, refer to [Deploy with Helm](deploy-helm.md).
- Changed the default chunk size to 512 to reduce LLM context size and in turn reduce RAG server response latency.
- Exposed config to split PDFs after chunking. You can control this by using the `APP_NVINGEST_ENABLEPDFSPLITTER` environment variable. For details, refer to [Best Practices](accuracy_perf.md).
- Added batch-based ingestion which can help manage memory usage of `ingestor-server` more effectively. You can control this by using the `ENABLE_NV_INGEST_BATCH_MODE` and `NV_INGEST_FILES_PER_BATCH` variables. For details, refer to [Best Practices](accuracy_perf.md).
- Removed `extract_options` from API level of `ingestor-server`.
- Made security and stability improvements.
- Added non-blocking async support to upload documents API. For details, refer to [Migration Guide](migration_guide.md).
  - Added a new field `blocking: bool` to control this behavior from the client side. The default is set to `true`. 
  - Added a new API `/status` to monitor state or completion status of uploaded docs.


### Fixed Known Issues

The following are the known issues that are fixed in this version:

- Issues processing very large files have been fixed.
- Resolved an issue during bulk ingestion where the ingestion job failed if ingestion of a single file failed.

For the full list of known issues, see [Known Issues](#all-known-issues).



## Release 2.0.0 (2025-03-18)

This release adds support for multimodal documents including the ability to parse PDF, Word, and PowerPoint documents. 
This release also significantly improves accuracy and performance considerations by refactoring the APIs and architecture. 
There is also a new developer-friendly user interface.

### Highlights 

This release contains the following key changes:

- The RAG Blueprint now uses two separate microservices to manage ingestion and retrieval/generation.
- The RAG Blueprint now uses [Retriever Extraction](https://github.com/NVIDIA/nv-ingest) instead of unstructured.io for the ingestion pipeline.
- Default settings are now configured to achieve a balance between accuracy and perf. For details, refer to [Best Practices](accuracy_perf.md).
- Added support for observability and telemetry. For details, refer to [Observability](observability.md).
- Added new react and nodeJS-based user interface to showcase runtime configurations. For details, refer to [User Interface](user-interface.md).
- Query rewriting now uses a smaller llama3.1-8b-instruct model and is turned off by default. For details, refer to [Query Rewriting](query_rewriter.md).
- Added support for the following optional features to improve accuracy and reliability of the pipeline. These are turned off by default. For details, refer to [Best Practices](accuracy_perf.md).
  - [Self Reflection](self-reflection.md)
  - [NeMo Guardrails](nemo-guardrails.md)
  - [Hybrid Search](hybrid_search.md)
- Support to use conversation history during retrieval for low-latency  multiturn support.
- Added a deployment-ready notebook intended to run in a [Brev environment](https://console.brev.dev/environment/new). For details, refer to [Notebooks](notebooks.md).
- Helm charts are now modularized, with separate helm charts provided for each distinct microservice. For details, refer to [Deploy with Helm](deploy-helm.md).
- The default docker deployment flow now uses on-premises models. Alternatively, you can deploy with Docker and using NVIDIA-Hosted Models. For details, refer to the following:
  - [Get Started With the NVIDIA RAG Blueprint](deploy-docker-self-hosted.md)
  - [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- Made security and stability improvements.
- Updated the API, including the following changes. For details, refer to [Migration Guide](migration_guide.md).
  - Support runtime configuration of all common parameters. 
  - Multimodal citation support.
  - New dedicated endpoints for deleting collection, creating collections and re-ingestion of documents.



## Release 1.0.0 (2025-01-15)

This is the initial release of the NVIDIA RAG Blueprint.







<!-- REQUIRED: Add any new know issues in this release -->
<!-- REQUIRED: Remove any know issues fixed in this release -->
## All Known Issues

The following are the known issues for the NVIDIA RAG Blueprint:

- Optional features reflection and image captioning are not available in Helm-based deployment.
- Currently, Helm-based deployment is not supported for [NeMo Guardrails](nemo-guardrails.md).
- The Blueprint responses can have significant latency when using [NVIDIA API Catalog cloud hosted models](deploy-docker-nvidia-hosted.md).
- The accuracy of the pipeline is optimized for certain file types like `.pdf`, `.txt`, `.docx`. The accuracy may be poor for other file types supported by NV-Ingest, since image captioning is disabled by default.
- When updating model configurations in Kubernetes `values.yaml` (for example, changing from 70B to 8B models), the RAG UI automatically detects and displays the new model configuration from the backend. No container rebuilds are required - simply redeploy the Helm chart with updated values and refresh the UI to see the new model settings in the Settings panel.
- The NeMo LLM microservice can take 5-6 minutes to start for every deployment.
- B200 GPUs are not supported for the following advanced features. For these features, use H100 or A100 GPUs instead.
  - Image captioning support for ingested documents
  - NeMo Guardrails for guardrails at input/output
  - VLM-based inferencing in RAG
  - PDF extraction with Nemoretriever Parse
- Sometimes when HTTP cloud NIM endpoints are used from `deploy/compose/.env`, the `nv-ingest-ms-runtime` still logs gRPC environment variables. Following log entries can be ignored.
- For MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing This affects the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.
- Individual file uploads are limited to a maximum size of 400 MB during ingestion. Files exceeding this limit are rejected and must be split into smaller segments before ingesting.
- `llama-3.3-nemotron-super-49b-v1.5` model provides more verbose responses in non-reasoning mode compared to v1.0. For some queries the LLM model may respond with information not available in given context. Also for out of domain queries the model may provide responses based on it's own knowledge. Developers are strongly advised to [tune the prompt](prompt-customization.md) for their use cases to avoid these scenarios.
- The auto selected NIM-LLM profile for llama-3.3-nemotron-super-49b-v1.5 may not work for some GPUs. Follow steps outlined in the appropriate [deployment guide](model-profiles.md) to select an optimized profile using `NIM_MODEL_PROFILE` before deploying.
- Slow VDB upload is observed in Helm deployments for Elasticsearch.



<!-- ADD THIS SECTION Starting with version 2.4.0 -->
<!-- ## Release Notes for Previous Versions -->
<!--                                        -->
<!-- | link | link | link | -->



## Related Topics

- [Known Issues and Troubleshooting the RAG UI](user-interface.md#known-issues-and-troubleshooting-the-rag-ui)
- [Troubleshoot NVIDIA RAG Blueprint](troubleshooting.md)
- [Migration Guide](migration_guide.md)
- [Get Started with NVIDIA RAG Blueprint](deploy-docker-self-hosted.md)

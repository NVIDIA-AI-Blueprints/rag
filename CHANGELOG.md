## Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.


## [2.2.0] - 2025-07-08

This release adds B200 platform support, a native Python API, and major enhancements for multimodal and metadata features. It also improves deployment flexibility and customization across the RAG blueprint.

### Added
- Support deploying the blueprint on B200 platform.
- Support for [native python API](./docs/python-client.md)
  - Refactoring code and directory to support python API
  - Better modularization for easier customization
  - Moved to `uv` as the package manager for this project
- Added support for configurable vector store consistency levels (Bounded/Strong/Session) to optimize retrieval performance vs accuracy trade-offs.
- [Capability to add custom metadata](./docs/custom-metadata.md) for files and metadata based filtering
- Documentation of [using Multi Instance GPUs](./docs/mig-deployment.md). Reduces minimum GPU requirement for helm charts to 3xH100.
- [Multi collection based retrieval](./docs/multi-collection-retrieval.md) support
- [Audio files (.mp3 and .wav) support](./docs/audio_ingestion.md)
- Support of using [Vision Language Model](./docs/vlm.md) based generation for charts and images
- Support for [generating summaries](./docs/summarization.md) of uploaded files
- Sample user interface enhancements
  - Support for non-blocking file upload
  - More efficient error reporting for ingestion failures
- [Prompt customization](./docs/prompt-customization.md) support without rebuilding images
- Added support to enable infographics, which improves accuracy for documents containing text in image format.
  - See [this guide](./docs/accuracy_perf.md#ingestion-and-chunking) for details
- New customizations
  - How to support non nvingest based ingestion + retrieval
  - How to enable [CPU based milvus](./docs/milvus-configuration.md)
  - How to enable [nemoretriever-parse](./docs/nemoretriever-parse-extraction.md) as an alternate PDF parser
  - How to use [standalone nv-ingest python client](./docs/nv-ingest-standalone.md) to do ingestion
- [Nvidia AI Workbench support](./deploy/workbench/)

### Changed
- [Changed API schema](./docs/api_reference/) to support newly added features
  - POST /collections to be deprecated in favour of POST /collection for ingestor-server
  - New endpoint GET /summary added for rag-server
  - Metadata information available as part of GET /collections and GET /documents API
  - Check out [migration guide](./docs/migration_guide.md#migration-guide-rag-v210-to-rag-v220) for detailed changes at API level
- [Optimized batch mode](./docs/accuracy_perf.md#ingestion-batch-mode-optimization) ingestion support to improve perf for multi user concurrent file upload.

### Known Issues
Check out [this section](./docs/troubleshooting.md#known-issues) to understand the known issues present for this release.

## [2.1.0] - 2025-05-13

This release reduces overall GPU requirement for the deployment of the blueprint. It also improves the performance and stability for both docker and helm based deployments.

### Added
- Added non-blocking async support to upload documents API
  - Added a new field `blocking: bool` to control this behaviour from client side. Default is set to `true`
  - Added a new API `/status` to monitor state or completion status of uploaded docs
- Helm chart is published on NGC Public registry.
- Helm chart customization guide is now available for all optional features under [documentation](./README.md#available-customizations).
- Issues with very large file upload has been fixed.
- Security enhancements and stability improvements.

### Changed
- Overall GPU requirement reduced to 2xH100/3xA100.
  - Changed default LLM model to [llama-3_3-nemotron-super-49b-v1](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1). This reduces overall GPU needed to deploy LLM model to 1xH100/2xA100
  - Changed default GPU needed for all other NIMs (ingestion and reranker NIMs) to 1xH100/1xA100
- Changed default chunk size to 512 in order to reduce LLM context size and in turn reduce RAG server response latency.
- Exposed config to split PDFs post chunking. Controlled using `APP_NVINGEST_ENABLEPDFSPLITTER` environment variable in ingestor-server. Default value is set to `True`.
- Added batch-based ingestion which can help manage memory usage of `ingestor-server` more effectively. Controlled using `ENABLE_NV_INGEST_BATCH_MODE` and `NV_INGEST_FILES_PER_BATCH` variables. Default value is `True` and `100` respectively.
- Removed `extract_options` from API level of `ingestor-server`.
- Resolved an issue during bulk ingestion, where ingestion job failed if ingestion of a single file fails.

### Known Issues
- The `rag-playground` container needs to be rebuild if the `APP_LLM_MODELNAME`, `APP_EMBEDDINGS_MODELNAME` or `APP_RANKING_MODELNAME` environment variable values are changed.
- While trying to upload multiple files at the same time, there may be a timeout error `Error uploading documents: [Error: aborted] { code: 'ECONNRESET' }`. Developers are encouraged to use API's directly for bulk uploading, instead of using the sample rag-playground. The default timeout is set to 1 hour from UI side, while uploading.
- In case of failure while uploading files, error messages may not be shown in the user interface of rag-playground. Developers are encouraged to check the `ingestor-server` logs for details.

A detailed guide is available [here](./docs/migration_guide.md) for easing developers experience, while migrating from older versions.

## [2.0.0] - 2025-03-18

This release adds support for multimodal documents using [Nvidia Ingest](https://github.com/NVIDIA/nv-ingest) including support for parsing PDFs, Word and PowerPoint documents. It also significantly improves accuracy and perf considerations by refactoring the APIs, architecture as well as adds a new developer friendly UI.

### Added
- Integration with Nvingest for ingestion pipeline, the unstructured.io based pipeline is now deprecated.
- OTEL compatible [observability and telemetry support](./docs/observability.md).
- API refactoring. Updated schemas [here](./docs/api_reference/).
  - Support runtime configuration of all common parameters. 
  - Multimodal citation support.
  - New dedicated endpoints for deleting collection, creating collections and reingestion of documents
- [New react + nodeJS based UI](./frontend/) showcasing runtime configurations
- Added optional features to improve accuracy and reliability of the pipeline, turned off by default. Best practices [here](./docs/accuracy_perf.md)
  - [Self reflection support](./docs/self-reflection.md)
  - [NeMo Guardrails support](./docs/nemo-guardrails.md)
  - [Hybrid search support using Milvus](./docs/hybrid_search.md)
- [Brev dev](https://developer.nvidia.com/brev) compatible [notebook](./notebooks/launchable.ipynb)
- Security enhancements and stability improvements

### Changed
- - In **RAG v1.0.0**, a single server managed both **ingestion** and **retrieval/generation** APIs. In **RAG v2.0.0**, the architecture has evolved to utilize **two separate microservices**.
- [Helm charts](./deploy/helm/) are now modularized, seperate helm charts are provided for each distinct microservice.
- Default settings configured to achieve a balance between accuracy and perf.
  - [Default flow uses on-prem models](./docs/quickstart.md#deploy-with-docker-compose) with option to switch to API catalog endpoints for docker based flow.
  - [Query rewriting](./docs/query_rewriter.md) uses a smaller llama3.1-8b-instruct and is turned off by default.
  - Support to use conversation history during retrieval for low-latency  multiturn support.

### Known Issues
- The `rag-playground` container needs to be rebuild if the `APP_LLM_MODELNAME`, `APP_EMBEDDINGS_MODELNAME` or `APP_RANKING_MODELNAME` environment variable values are changed.
- Optional features reflection, nemoguardrails and image captioning are not available in helm based deployment.
- Uploading large files with .txt extension may fail during ingestion, we recommend splitting such files into smaller parts, to avoid this issue.

A detailed guide is available [here](./docs/migration_guide.md) for easing developers experience, while migrating from older versions.

## [1.0.0] - 2025-01-15

### Added

- First release.


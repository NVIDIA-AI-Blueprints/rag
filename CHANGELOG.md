# Changelog

All notable changes to the [NVIDIA RAG Blueprint](README.md) will be documented in this file.

To upgrade your version, see [Migration Guide](docs/migration_guide.md).


## Version 2.3.0 (2025-10-14)

This release adds RTX6000 platform support, adds deployment by using NIM operator, improves vector database pluggability with the blueprint, and other changes.

### Added
- Support deploying the blueprint on RTX6000 platform.
- Migrated to [`llama-3.3-nemotron-super-49b-v1.5`](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) as the default LLM model.
- Added support to deploy the helm chart by using NVIDIA NIM operator. For details, refer to [Deploy NVIDIA RAG Blueprint with NIM Operator](docs/deploy-nim-operator.md).
- Updated all NIMs, NVIDIA Ingest and third party dependencies to latest versions.
- Refactoring to support custom 3rd party vector DB integration in a streamlined manner.
  - Interactive notebook showcasing integration with library mode [here](./notebooks/building_rag_vdb_operator.ipynb).
- Added support for [elasticsearch vector DB as an alternate to milvus](./docs/change-vectordb.md).
- Added opt-in [query decomposition support](./docs/query_decomposition.md).
- Added opt-in [nemoretriever-ocr support](./docs/nemoretriever-ocr.md).
- Added opt-in [VLM embedding support](./docs/vlm-embed.md)
- Custom metadata enhancments. Detailed doc [here](./docs/custom-metadata.md).
  - Added support for more datatypes.
  - Added opt-in support to generate filters using LLM yielding better accuracy.
  - Added an [interactive notebook](./notebooks/nb_metadata.ipynb) showcasing new features.
- Added dependency check support for ingestor server /health API.
- Added support for configurable confidence threashold for retrieval from API layer.
- Added support to store NV-Ingest extraction results [directly from the filesystem](./docs/mount-ingestor-volume.md).
- Logging enhancements
- Added better latency data reporting for RAG server
  - API level enhancements for component level latency
  - Added dedicated Prometheus metric endpoint
- Added independent script to [showcase batch ingestion](./scripts/README.md)
- Enabled support for [GPU indexing with CPU search](./docs/milvus-configuration.md#gpu-indexing-with-cpu-search)
  - Exposed `APP_VECTORSTORE_EF` as a configurable parameter
- Added environment variables to control llm parameters LLM_MAX_TOKENS, LLM_TEMPERATURE and LLM_TOP_P
- Added notebooks for showcasing RAG evaluation using common metrics
  - [Notebook 1 - evaluation using RAGAS](./notebooks/evaluation_01_ragas.ipynb)
  - [Notebook 2 - Recall calculation](./notebooks/evaluation_02_recall.ipynb)
- Added [unit tests](./tests/unit/) and [pre-commit](./LINTING.md) hooks for maintaining code quality.
- Optimized container sizes by removing unnecessary packages and improving security.

### Changed
- Migrated default LLM model for reflection to `llama-3.3-nemotron-super-49b` instead of `mixtral-8x22b-instruct-v01`.
- Refactored [rag-playground](./frontend/) code
  - Use React end to end. Next.js dependencies were deprecated.
  - More developer friendly and intuitive look and feel.
  - `rag-playground` service is renamed to `rag-frontend`
- Refactored [helm chart support](./deploy/helm/)
  - Expanded and reorganized Helm chart configuration, enabling granular control over service components, resource settings, and observability (tracing, metrics).
  - Introduced ConfigMap and service definitions to facilitate improved application deployment flexibility.
  - Implemented refined service account and secret management in Helm templates.
  - Added a new Helm values file for nim-operator to configure LLM model environment and component toggles.

### Fixed
- Fixed support for long audio file ingestion.
- Fixed support to ingest images without charts/tables.
- Fixed requirement of rebuilding rag frontend container when LLM model name was changed.

### Removed
- Removed consistency level configuration support for Milvus.
- Removed `EMBEDDING_NIM_ENDPOINT` and `EMBEDDING_NIM_MODEL_NAME` environment variables for nvingest.
- Removed unused `ENABLE_MULTITURN` environment variable from rag-server.
- Removed `ENABLE_NEMOTRON_THINKING` environment variable from rag-server.


### Known Issues

For the full list of known issues, see [Known Issues](#all-known-issues).



## Version 2.2.1 (2025-07-22)

This is a minor patch release that updates to the latest nv-ingest-client version 25.6.3 to fix breaking changes introduced by pypdfium.
For details, refer to [NVIDIA NV Ingest 25.6.3](https://github.com/NVIDIA/nv-ingest/releases/tag/25.6.3).



## Version 2.2.0 (2025-07-08)

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
  - Refer to the [migration guide](./docs/migration_guide.md#migration-guide-rag-v210-to-rag-v220) for detailed changes at API level
- [Optimized batch mode](./docs/accuracy_perf.md#ingestion-batch-mode-optimization) ingestion support to improve perf for multi user concurrent file upload.


### Known Issues

For the full list of known issues, see [Known Issues](#all-known-issues).



## Version 2.1.0 (2025-05-13)

This release reduces overall GPU requirement for the deployment of the blueprint. It also improves the performance and stability for both docker and helm based deployments.

### Added
- Added non-blocking async support to upload documents API
  - Added a new field `blocking: bool` to control this behaviour from client side. Default is set to `true`
  - Added a new API `/status` to monitor state or completion status of uploaded docs
- Helm chart is published on NGC Public registry.
- Helm chart customization guide is now available for many optional features. For details, refer to [NVIDIA RAG Blueprint Documentation](docs/readme.md).
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

The following are the new known issues in this version:

- While trying to upload multiple files at the same time, there may be a timeout error `Error uploading documents: [Error: aborted] { code: 'ECONNRESET' }`. Developers are encouraged to use API's directly for bulk uploading, instead of using the sample rag-frontend. The default timeout is set to 1 hour from UI side, while uploading.
- In case of failure while uploading files, error messages may not be shown in the user interface of rag-frontend. Developers are encouraged to check the `ingestor-server` logs for details.

For the full list of known issues, see [Known Issues](#all-known-issues).



## Version 2.0.0 (2025-03-18)

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
- [Helm charts](./deploy/helm/) are now modularized, separate helm charts are provided for each distinct microservice.
- Default settings configured to achieve a balance between accuracy and perf.
  - [Default Docker deployment flow now uses on-prem models](docs/deploy-docker-self-hosted.md). Alternatively, you can [Deploy NVIDIA RAG Blueprint with Docker (NVIDIA-Hosted Models)](docs/deploy-docker-nvidia-hosted.md).
  - [Query rewriting](./docs/query_rewriter.md) uses a smaller llama3.1-8b-instruct and is turned off by default.
  - Support to use conversation history during retrieval for low-latency  multiturn support.

### Known Issues

The following are the new known issues in this version:

- Optional features reflection, nemoguardrails and image captioning are not available in helm based deployment.
- Uploading large files with .txt extension may fail during ingestion, we recommend splitting such files into smaller parts, to avoid this issue.

For the full list of known issues, see [Known Issues](#all-known-issues).


## Version 1.0.0 (2025-01-15)

This is the initial release of the NVIDIA RAG Blueprint.


## All Known Issues

The following are the known issues for RAG Blueprint:

- Currently, Helm-based deployment is not supported for [NeMo Guardrails](docs/nemo-guardrails.md).
- The Blueprint responses can have significant latency when using [NVIDIA API Catalog cloud hosted models](./docs/deploy-docker-nvidia-hosted.md).
- The accuracy of the pipeline is optimized for certain file types like `.pdf`, `.txt`, `.docx`. The accuracy may be poor for other file types supported by NvIngest, since image captioning is disabled by default.
- The UI file upload interface has a hard limit of 100 files per upload batch. When selecting more than 100 files, only the first 100 are processed. For bulk uploads beyond this limit, use multiple upload batches or the [programmatic API](./notebooks/ingestion_api_usage.ipynb).
- When updating model configurations in Kubernetes `values.yaml` (for example, changing from 70B to 8B models), the RAG UI automatically detects and displays the new model configuration from the backend. No container rebuilds are required - simply redeploy the Helm chart with updated values and refresh the UI to see the new model settings in the Settings panel.
- The NeMo LLM microservice can take 5-6 minutes to start for every deployment.
- B200 GPUs are not supported for the following advanced features. For these features, use H100 or A100 GPUs instead.
  - Image captioning support for ingested documents
  - NeMo Guardrails for guardrails at input/output
  - VLM based inferencing in RAG
  - PDF extraction with Nemoretriever Parse
- Sometimes when HTTP cloud NIM endpoints are used from `deploy/compose/.env`, the `nv-ingest-ms-runtime` still logs gRPC environment variables. Following log entries can be ignored.
- If one of the file in a bulk ingestion job is of type svg, which is a unsupported format, the full bulk ingestion job fails.
- Complicated filter expressions with custom metadata while sending a query, are not supported from the [RAG UI](./docs/user-interface.md).
- For MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing This affects the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.
- Individual file uploads are limited to a maximum size of 400 MB during ingestion. Files exceeding this limit are rejected and must be split into smaller segments before ingesting.
- `llama-3.3-nemotron-super-49b-v1.5` model provides more verbose responses in non-reasoning mode compared to v1.0. For some queries the LLM model may respond with information not available in given context. Also for out of domain queries the model may provide responses based on it's own knowledge. Developers are strongly advised to [tune the prompt](./docs/prompt-customization.md) for their usecases to avoid these scenarios.
- The auto selected NIM-LLM profile for llama-3.3-nemotron-super-49b-v1.5 may not work for some GPUs. Follow steps outlined in the appropriate [deployment guide](docs/model-profiles.md) to select an optimized profile using `NIM_MODEL_PROFILE` before deploying.
- Slow VDB upload is observed in Helm deployments for Elasticsearch.
- Immediately after document ingestion, there might be a delay before the [RAG UI](docs/user-interface.md) accurately reflects the number of Milvus entities in a collection. Although the count that appears might be temporarily inconsistent, the presence of a document in the RAG UI confirms its successful ingestion.



## Related Topics

- [Troubleshooting](docs/troubleshooting.md)
- [Debugging](docs/debugging.md)

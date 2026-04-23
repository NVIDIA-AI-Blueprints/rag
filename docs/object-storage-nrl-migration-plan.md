# Object Storage Migration Plan on NRL Baseline

## Goal

Migrate the app from MinIO-specific application code to:

- SeaweedFS as the default S3-compatible object store
- any S3-compatible bucket as a supported replacement
- filesystem-backed storage only if it passes real NRL end-to-end verification

Constraints:

- base all work on clean `origin/develop` with NRL already integrated
- keep the diff intentionally small
- remove app-level MinIO compatibility completely
- do not make unrelated docs/workflow/notebook changes
- every storage mode we keep must be proven by live verification, not only unit tests

This plan assumes a stable storage contract so we do not need to revisit the same code paths repeatedly:

- one shared S3-compatible runtime path for SeaweedFS and any other S3-compatible bucket
- one URI-driven retrieval path in app code (`s3://...` or `file://...`)
- filesystem support only if NRL ingestion and app retrieval both work on the real path

## Progress

- Phase 0: complete
  - `pytest-asyncio` added and async pytest baseline restored
- Phase 1: complete
  - app-level object-store API neutralized away from MinIO-specific naming
- Phase 2: complete for Docker defaults and Helm config
  - SeaweedFS is now the default S3-compatible backend in Docker Compose, Workbench, and Helm values/templates
  - Docker persistence was live-verified through the real object-store code path across SeaweedFS restart
- Phase 3: complete on this branch
  - filesystem backend config, operator, store params, retrieval path, and health checks are in code
  - live library-mode NRL verification passed against cloud endpoints with filesystem-backed artifact persistence and URI-based readback
- Helm live verification: pending by plan; do last

## Current Baseline on `origin/develop`

### NRL storage architecture

- NRL ingestion is in-process via `NemoRetrieverHandler` and `GraphIngestor`.
- Active NRL ingest path only supports `LanceDB` for vectors.
- NRL visual artifacts are stored separately through `StoreParams`.
- `make_store_params()` currently emits S3-only settings:
  - `storage_uri = s3://default-bucket/<collection>/images`
  - `public_base_url = storage_uri`
  - `storage_options.client_kwargs.endpoint_url = http://<MINIO_ENDPOINT>`
- NRL citation retrieval in `response_generator.py` still assumes `stored_image_uri` is an S3/MinIO URI and fetches bytes through the MinIO operator.

### App-level MinIO coupling that should be removed

- `src/nvidia_rag/utils/configuration.py`
  - `MinioConfig`
  - `NvidiaRAGConfig.minio`
  - `MINIO_*` app env names
- `src/nvidia_rag/utils/minio_operator.py`
  - MinIO-specific operator naming
- `src/nvidia_rag/ingestor_server/nemo_retriever/params.py`
  - `config.minio.*`
- `src/nvidia_rag/rag_server/response_generator.py`
  - MinIO singleton and S3-only citation fetch path

### Deploy baseline

- Compose and Helm still default to a `minio` service and `MINIO_*` app env wiring.
- Milvus and `nv-ingest` containers also use `MINIO_*` env names, but those are upstream/runtime integration names rather than app API names.

### Important architectural boundary

App-level MinIO compatibility should be removed.

Upstream/runtime-owned env names may still need to remain where required by:

- Milvus
- `nv-ingest`
- other third-party images/configs that still speak in `MINIO_*`

That is not backward compatibility for our app API. It is compatibility with upstream services we do not control.

## Target Architecture

### 1. App object-store API

Use neutral app naming everywhere:

- `ObjectStoreConfig`
- `config.object_store`
- `object_store.py`
- `get_object_store_operator()`

Supported app backends:

- `s3`
- `filesystem`

App env contract:

- `OBJECTSTORE_BACKEND=s3|filesystem`
- `OBJECTSTORE_ENDPOINT`
- `OBJECTSTORE_ACCESSKEY`
- `OBJECTSTORE_SECRETKEY`
- `OBJECTSTORE_LOCAL_PATH`
- `OBJECTSTORE_LOCAL_INGEST_PATH`

Do not keep:

- `MinioConfig`
- `config.minio`
- app fallback from `OBJECTSTORE_*` to `MINIO_*`
- `get_minio_operator()`
- `MinioOperator`

### 2. SeaweedFS / generic S3 mode

This is the default shipping mode.

Behavior:

- SeaweedFS becomes the default deploy-time S3-compatible object store
- any S3-compatible endpoint should work by changing `OBJECTSTORE_*`
- NRL artifact storage remains S3-shaped
- citation retrieval must fetch by storage URI, not by backend-specific key parsing
- Docker and Helm must both use persistent storage for SeaweedFS by default

### 3. Filesystem mode

Filesystem is an additional backend, not the default.

Required behavior:

- NRL ingest stores visual artifacts to a real filesystem path
- retrieved `stored_image_uri` values remain usable by the app through the same URI-driven retrieval path
- citations can read those stored artifacts back
- artifacts remain on disk across process restart

Important rule:

- if real NRL ingest plus retrieval does not work in filesystem mode, filesystem support must not be declared complete
- if NRL itself does not support the needed `file://` path cleanly, we either implement a repo-side workaround deliberately or drop filesystem from the final scope

## Minimal File Scope

Only touch files needed for runtime, deploy, tests, and minimal supporting docs.

### Runtime

- `src/nvidia_rag/utils/configuration.py`
- `src/nvidia_rag/utils/minio_operator.py` -> replace with neutral object-store module
- `src/nvidia_rag/ingestor_server/nemo_retriever/params.py`
- `src/nvidia_rag/rag_server/response_generator.py`
- `src/nvidia_rag/ingestor_server/health.py`
- `src/nvidia_rag/rag_server/health.py`
- `src/nvidia_rag/utils/vdb/__init__.py`
- only other runtime files if a real storage call path requires them

### Deploy

- compose files actually used by the storage path
- Helm values/templates required for SeaweedFS default and persistence

### Tests

- unit tests for config, params, retrieval, deploy config, and VDB wiring
- integration tests only where they directly prove storage behavior

### Docs

- only minimal docs if needed to explain supported env names or an operational footgun
- do not carry audit logs or planning history into the final branch

## Execution Plan

### Phase 0. Restore trustworthy test and verification baseline

Changes:

- add `pytest-asyncio` to the dev/test toolchain
- configure pytest async execution explicitly
- verify the current rebased branch has a stable async baseline before storage changes expand

Success criteria:

- async-heavy unit slices run without "async def functions are not natively supported" failures
- later failures represent real regressions, not missing test infrastructure

### Phase 1. Replace app-level MinIO API with neutral object-store API

Changes:

- introduce `ObjectStoreConfig`
- introduce neutral operator module
- switch app runtime code from `config.minio` to `config.object_store`
- remove app-level `MINIO_*` config names
- change retrieval to fetch assets by storage URI rather than S3-object-key-only assumptions

Success criteria:

- no app runtime code depends on `config.minio`
- no app runtime code imports `minio_operator.py`
- tests updated to the new object-store API

### Phase 2. Make SeaweedFS the default S3-compatible backend

Changes:

- replace default deploy-time MinIO service with SeaweedFS in Compose and Helm
- ensure mounted persistent data is actually used
- use a checked-in S3 config file instead of brittle inline shell config generation
- fix object-store endpoint wiring for app services
- keep upstream `MINIO_*` env names only where third-party services require them
- switch `YOLOX_PAGE_IMAGE_FORMAT` default to `PNG` if the current naming path still assumes `.png`

Success criteria:

- deploy defaults point at SeaweedFS
- Compose persistence is real
- Helm defaults are correct for persistence and object-store addressing
- generic S3 endpoint replacement still works through app `OBJECTSTORE_*` config

### Phase 3. Add filesystem backend only through the real NRL path

Changes:

- add `OBJECTSTORE_BACKEND`
- add filesystem operator implementation
- update NRL `StoreParams` builder to support filesystem mode
- update citation retrieval to read assets from either `s3://` or `file://`
- update health checks for filesystem mode

Success criteria:

- NRL ingest writes non-empty image/table/chart artifacts to disk
- app retrieval can read those artifacts back through `stored_image_uri`
- artifacts remain present after app process restart

If this phase fails under live verification:

- do not merge partial filesystem support
- instead document the blocker and either implement a deliberate workaround or drop filesystem from the branch

### Phase 4. Remove stale references and keep the branch small

Changes:

- remove stale app MinIO references from tests, env examples, deploy comments, and code
- leave upstream `MINIO_*` only where runtime integrations require them
- do not include unrelated doc/workflow/notebook churn

Success criteria:

- branch diff is storage-focused
- no leftover half-migrated naming in kept files

## Live Verification Requirements

This is mandatory for every mode we keep.

## Verification Policy

Every implementation phase must close with both:

- targeted automated verification
- live verification against the real runtime path affected by that phase

Do not treat a phase as complete on unit tests alone.

For this migration, the rule is:

1. make the smallest runtime/deploy/test change needed for the phase
2. run the narrowest useful automated tests for that change
3. run live verification for the affected path using real services and real library or API calls
4. only then move to the next phase

If live verification fails:

- do not mark the phase complete
- either fix the implementation or reduce scope explicitly
- do not carry speculative support forward

### General verification shape

- deploy required dependencies with containers
- create a separate Python virtual environment for library-mode verification
- call real library APIs from that venv
- use cloud model endpoints where possible to avoid local GPU requirements
- verify persisted data survives the restart/reschedule claim for that phase

### SeaweedFS default verification

Required proof:

1. bring up SeaweedFS and required supporting services
2. run library-mode NRL ingestion against a real sample PDF with images/tables/charts
3. confirm LanceDB ingest completes
4. confirm visual artifacts are written to SeaweedFS and are non-empty
5. read one or more stored artifacts back through the real app/object-store path
6. restart the relevant app process and re-read the stored artifact
7. restart SeaweedFS and confirm the artifact still exists

### Generic S3 verification

Required proof:

- same app code path works when `OBJECTSTORE_ENDPOINT` points to a non-SeaweedFS S3-compatible endpoint
- this can be satisfied by proving the app code is endpoint-agnostic plus at least one live S3-compatible run

### Filesystem verification

Required proof:

1. run library-mode NRL ingestion with filesystem backend enabled
2. confirm artifacts are written under the configured filesystem root
3. confirm files are non-empty
4. read one or more artifacts back through the real retrieval path
5. recreate the library object / process and confirm files are still readable

Latest verification status:

- passed in library mode with `scripts/verify_nrl_filesystem_object_store.py`
- verified non-empty filesystem artifacts under `OBJECTSTORE_LOCAL_PATH`
- verified LanceDB retrieval returns docs with `stored_image_uri=file://...`
- verified `prepare_citations_nrl()` reads persisted artifacts back through the real object-store URI path
- verified the same stored URI remains readable after recreating library objects

Verification command used:

```bash
PYTHONPATH=src uv run python scripts/verify_nrl_filesystem_object_store.py \
  --pdf data/multimodal/functional_validation.pdf \
  --work-root tmp/fs-object-store-live-verify \
  --keep
```

Environment note:

- in this environment, NRL's image-caption stage requires local `torch`
- the verification therefore uses multimodal/page/table artifacts without enabling caption-based image extraction
- filesystem-backed object storage itself is verified; if full caption-image verification is required in a future environment, install `torch` and rerun with image extraction enabled

### Helm verification

This should be done last.

Required proof:

1. deploy the chart with SeaweedFS enabled
2. use cloud endpoints to avoid local model GPU pressure where possible
3. verify SeaweedFS persistence across pod restart or pod replacement
4. verify real NRL ingest through the deployed stack
5. verify stored artifacts can still be read after the SeaweedFS restart check

## Test Plan

Unit tests to add or update only where they protect the migration:

- config tests for `ObjectStoreConfig`
- NRL `StoreParams` tests
- citation retrieval tests for URI-based fetch
- deploy config tests for SeaweedFS defaults and persistence-sensitive settings
- VDB wiring tests for any object-store env/config handoff that remains

Integration tests:

- one focused SeaweedFS smoke test is useful
- do not add broad integration coverage that does not directly prove storage behavior

Phase gates:

- Phase 0 gate:
  - install and sync async pytest support
  - rerun representative async-heavy test files successfully
- Phase 1 gate:
  - targeted unit tests for config/operator/call-site changes
  - live verification that the app runtime can initialize and use the neutral object-store path against a real S3-compatible backend
- Phase 2 gate:
  - targeted deploy/config tests
  - live SeaweedFS verification in Docker plus library-mode ingestion/retrieval
  - live persistence verification across restart
- Phase 3 gate:
  - targeted unit tests for filesystem config/operator/retrieval
  - live filesystem NRL ingestion and retrieval verification with non-empty persisted artifacts
- Phase 4 gate:
  - targeted regression tests for any renamed or cleaned references
  - final live sanity check that SeaweedFS default mode still works after cleanup

## Order of Work

1. implement Phase 0 on clean `origin/develop`
2. verify the async baseline
3. implement Phase 1
4. verify Phase 1 with targeted tests and live S3-compatible runtime checks
5. implement Phase 2
6. live-verify SeaweedFS default in Docker and library mode, including persistence
7. implement Phase 3
8. live-verify filesystem in library mode, including persisted non-empty artifacts
9. trim stale references and finalize the branch
10. do Helm verification last

## Non-Goals

- preserving app-level MinIO backward compatibility
- carrying forward unrelated branch changes
- claiming filesystem support without passing live NRL ingest and retrieval verification

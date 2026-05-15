---
name: rag-ingest-documents
description: Ingest documents, folders, PDFs, text files, audio or video files, mounted volumes, and batch inputs into NVIDIA RAG Blueprint collections. Use when the user asks to upload documents, ingest a corpus, add files to a collection, run batch ingestion, or configure ingestion options.
version: "1.0.0"
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
  tags:
    - rag
    - ingestion
    - documents
---

# RAG Ingest Documents

## Overview

Ingest content into RAG collections through the ingestor API, UI, Python client,
batch scripts, or mounted-volume workflows.

## Prerequisites

- Confirm a RAG deployment is running or route to `rag-deploy-blueprint`.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Ask the user to classify the corpus and confirm the target environment is
  approved before uploading confidential data. Pause on restricted data without
  documented controls.
- Validate user-provided paths and collection names.
- Read `references/ingestion.md` before choosing an ingestion path.
- Do not upload restricted or confidential data to unapproved endpoints.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate target deployment, ingestor health, collection name, file path, and
   ingestion mode.
2. Prepare metadata, collection settings, extraction options, and summary flags.
3. Execute with the smallest appropriate surface:
   - UI for small interactive uploads.
   - Ingestor REST API for direct service workflows.
   - Python client for library or scripted workflows.
   - Batch script for folders and larger corpora.
   - Mounted volume for extraction-output workflows.
4. Verify using `/status`, document counts, failed document lists, and a sample
   query or search.
5. Report task IDs, completed/failed documents, validation errors, and next
   retrieval step.

Ask for explicit confirmation before deleting documents, collections, or
generated extraction output.

## Reference

- `references/ingestion.md`
- `../../docs/api-ingestor.md`
- `../../docs/text_only_ingest.md`
- `../../docs/audio_ingestion.md`
- `../../docs/nemotron-parse-extraction.md`
- `../../docs/nemoretriever-ocr.md`
- `../../docs/mount-ingestor-volume.md`
- `../../docs/nv-ingest-standalone.md`
- `../../notebooks/ingestion_api_usage.ipynb`

## Error Handling

If ingestion fails, inspect the task status before retrying. Separate validation
errors from extraction failures, service health failures, and vector database
write failures.

Do not retry destructive collection operations automatically.

## Examples

- "Upload these PDFs into a new collection."
- "Batch ingest this folder."
- "Enable text-only ingestion."
- "Check why my ingestion task failed."

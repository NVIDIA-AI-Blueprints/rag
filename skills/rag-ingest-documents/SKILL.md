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

## Routing guard

Only activate for tasks that add, upload, or ingest new content into a
collection. If the request is to search, query, retrieve, or ask a question
about existing knowledge-base content, do not activate this skill — route to
`rag-query-knowledge` instead. Do not use `curl` or any other tool to query the
RAG server directly; that is outside this skill's scope.

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
   Only make network calls to the ingestor endpoint (`INGESTOR_URL` or
   `http://localhost:8082`). Do not call any other network endpoints during
   ingestion — in particular, do not query the RAG server or any external URL.
4. After submitting documents, poll the task status endpoint until each task
   reaches a terminal state (completed or failed). Do not declare success until
   all submitted tasks have resolved. Report a per-file or per-task summary:
   how many succeeded, how many failed, and the reason for any failure.
5. Report task IDs, completed/failed document counts, validation errors, and
   the recommended next step (e.g., query with `rag-query-knowledge`).

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

## Response style

Keep the final report to a structured short summary: files submitted, completed
count, failed count with reasons, and next step. Do not reproduce full file
contents or API response bodies in the reply.

## Examples

- "Upload these PDFs into a new collection."
- "Batch ingest this folder."
- "Enable text-only ingestion."
- "Check why my ingestion task failed."

# Ingestion Reference

## Source Docs

| Goal | Source |
|---|---|
| API schema and status tracking | `docs/api-ingestor.md` |
| UI ingestion limits | `docs/user-interface.md` |
| Python client ingestion | `docs/python-client.md` |
| Text-only mode | `docs/text_only_ingest.md` |
| Audio/video ingestion | `docs/audio_ingestion.md` |
| Nemotron Parse | `docs/nemotron-parse-extraction.md` |
| OCR selection | `docs/nemoretriever-ocr.md` |
| Mounted extraction output | `docs/mount-ingestor-volume.md` |
| Standalone NV-Ingest | `docs/nv-ingest-standalone.md` |
| Ingestion tuning | `docs/accuracy_perf.md` |

## Input Validation

- Collection names must be explicit and should not contain shell metacharacters.
- Paths must stay inside the user's intended workspace. Reject `..` traversal
  unless the user explicitly gives an absolute path and the host policy permits
  it.
- URLs must use expected schemes and domains.
- Metadata must be valid JSON-compatible data.

## Verification

Use the task ID returned by document upload and poll `/status` until the task is
`FINISHED`, `FAILED`, or clearly stuck. Report:

- total documents
- completed documents
- failed documents
- validation errors
- extraction status when available

After a successful ingest, run a small retrieval or generation query against the
target collection when the user wants end-to-end validation.


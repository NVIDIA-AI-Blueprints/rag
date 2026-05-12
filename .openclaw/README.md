# RAG OpenClaw Plugin

This directory is the starting point for a native RAG OpenClaw package. It
includes `openclaw.plugin.json`, the native manifest required by current
OpenClaw plugin discovery, and package-local skill links under `skills/`.

Use the files under `workspace/` as the RAG agent identity and operating manual.
Operational workflows should route back to the canonical skills in `../skills/`
instead of duplicating instructions here.

## Intended Package Shape

```text
.openclaw/
  openclaw.plugin.json
  README.md
  workspace/
    identity.md
    overview.md
    manual.md
  skills/
    README.md
    rag-* -> ../../skills/rag-*
```

## Install

From the RAG repository root:

```bash
openclaw plugins install ./.openclaw/
```

The manifest declares skill directories relative to the plugin root. The
`skills/rag-*` entries are symlinks back to the canonical `../skills/rag-*`
catalog, so use a checkout or archive format that preserves symlinks.

## Validation Targets

- Deploy RAG on Brev or a local GPU host.
- Ingest a small document set.
- Query the ingested collection.
- Troubleshoot an unhealthy service.
- Run one RAG quality evaluation.

Manifest reference: https://docs.openclaw.ai/plugins/manifest

---
name: rag-troubleshoot-blueprint
description: Troubleshoot, debug, diagnose, or fix NVIDIA RAG Blueprint deployments, unhealthy services, failed containers, failed pods, API errors, ingestion failures, query failures, GPU issues, port conflicts, and vector database problems.
version: "1.0.0"
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
  tags:
    - rag
    - troubleshoot
    - debug
---

# RAG Troubleshoot Blueprint

## Overview

Diagnose and resolve RAG service, deployment, ingestion, retrieval, model, GPU,
port, and configuration failures.

## Prerequisites

- Read `references/troubleshoot.md` before running diagnostics.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Prefer read-only diagnostics first.
- Sanitize logs before showing them to the user.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate the deployment mode and failure surface.
2. Collect health, container/pod state, recent logs, ports, GPU status, config
   source, and API error response.
3. Classify the issue: deploy, model endpoint, ingestion, retrieval, vector DB,
   guardrails, VLM, networking, or UI.
4. Apply the smallest safe fix or route to the specialized skill.
5. Verify the failing workflow again.
6. Report root cause, evidence, fix, and remaining risk.

Ask for explicit confirmation before destructive cleanup, collection deletion,
or volume removal.

## Reference

- `references/troubleshoot.md`
- `../../docs/troubleshooting.md`
- `../../docs/debugging.md`
- `../../docs/support-matrix.md`
- `../../docs/service-port-gpu-reference.md`
- `../../docs/api-ingestor.md`
- `../../docs/api-rag.md`

## Error Handling

If the issue cannot be fixed safely, report the exact blocker and the next
manual action. Do not mask unresolved failures with a generic success summary.

## Examples

- "RAG is unhealthy."
- "The ingestor is failing."
- "Queries return irrelevant answers."
- "Docker says the NIM container is unhealthy."

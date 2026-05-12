---
name: rag-enable-guardrails
description: Enable, configure, or verify NeMo Guardrails for NVIDIA RAG Blueprint queries and responses. Use when the user asks to add guardrails, content safety, topic control, policy checks, safe responses, or guardrail policy validation.
owner: nvidia-rag-team
service: nvidia-rag-guardrails
version: "0.1.0"
reviewed: "2026-05-11"
license: Apache-2.0
data_classification: internal
metadata:
  github-url: "https://github.com/NVIDIA-AI-Blueprints/rag"
  tags: "nvidia rag guardrails safety"
---

# RAG Enable Guardrails

## Overview

Enable and validate NeMo Guardrails for RAG query and response workflows.

## Prerequisites

- Confirm a compatible RAG deployment is running.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Read `references/guardrails.md` before changing config.
- Check model endpoint and key presence without printing values.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate deployment mode, guardrails service state, and desired policy scope.
2. Prepare config and service changes in the active deployment files.
3. Start or restart guardrails and affected RAG services.
4. Verify with allowed and disallowed prompts.
5. Report behavior, policy files touched, and residual policy gaps.

## Reference

- `references/guardrails.md`
- `../../docs/nemo-guardrails.md`
- `../../docs/query-to-answer-pipeline.md`
- `../../docs/support-matrix.md`

## Error Handling

If guardrails blocks expected prompts, inspect policy config and service logs. If
unsafe prompts pass through, stop and report the policy gap rather than claiming
coverage.

Route broad failures, logs, and unknown unhealthy-service issues to
`rag-troubleshoot-blueprint`; return here only when the root cause is guardrails
configuration.

## Examples

- "Enable NeMo Guardrails."
- "Add content safety to RAG responses."
- "Test whether guardrails block off-topic prompts."
- "Verify guardrails service startup."

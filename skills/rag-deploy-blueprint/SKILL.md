---
name: rag-deploy-blueprint
description: Deploy, start, verify, shut down, or tear down the NVIDIA RAG Blueprint. Use when the user says deploy RAG, start RAG, set up RAG, run RAG on Docker, deploy RAG with Helm, use library mode, stop RAG, or clean up a RAG deployment.
owner: nvidia-rag-team
service: nvidia-rag-blueprint
version: "0.1.0"
reviewed: "2026-05-11"
license: Apache-2.0
data_classification: internal
metadata:
  github-url: "https://github.com/NVIDIA-AI-Blueprints/rag"
  tags: "nvidia blueprint deployment rag"
---

# RAG Deploy Blueprint

## Overview

Deploy and manage the NVIDIA RAG Blueprint across Docker Compose, Helm, and
library modes. Prefer host and repo discovery over user questions.

## Prerequisites

- Work from the RAG repository root.
- Resolve `RAG_REPO_ROOT` first. If the skill was copied rather than symlinked,
  ask the user to set it to the repository checkout path.
- Read `references/deployment.md` before making deployment decisions.
- Read `../../docs/support-matrix.md` and
  `../../docs/service-port-gpu-reference.md` for current requirements.
- Check API keys without printing values.

## Usage

Follow `Validate -> Prepare -> Execute -> Verify -> Report`.

1. Validate GPU, driver, Docker, Compose, Python, disk, ports, repo state, and
   whether RAG services are already running.
2. Route the deployment mode:
   - Docker self-hosted for hosts that satisfy GPU, driver, disk, and NVIDIA
     Container Toolkit requirements.
   - Docker NVIDIA-hosted when Docker is available but local inference is not
     appropriate.
   - Docker retrieval-only when the user asks for search/retrieve without LLM
     generation.
   - Helm when the user asks for Kubernetes, Helm, or MIG slicing.
   - Library mode when the user asks for Python usage or no Docker.
3. Prepare config in the correct source-of-truth file:
   - Docker: `deploy/compose/.env` or `deploy/compose/nvdev.env`.
   - Helm: values files under `deploy/helm/`.
   - Library: `notebooks/config.yaml` or user-provided config.
4. Execute the documented deployment path.
5. Verify health through RAG and ingestor health endpoints.
6. Report deployment mode, endpoints, health, and unresolved blockers.

Ask for explicit confirmation before destructive cleanup, volume removal, or
collection deletion.

## Reference

- `references/deployment.md` for routing and verification.
- `../../docs/deploy-docker-self-hosted.md`
- `../../docs/deploy-docker-nvidia-hosted.md`
- `../../docs/retrieval-only-deployment.md`
- `../../docs/deploy-helm.md`
- `../../docs/mig-deployment.md`
- `../../docs/python-client.md`
- `../../docs/troubleshooting.md`

## Error Handling

Collect all blockers before returning. Do not stop at the first missing
dependency if more checks can be run safely. If a deployment fails, inspect logs
and health endpoints before suggesting fixes.

Never echo `NGC_API_KEY`, `NVIDIA_API_KEY`, or values from env files.

## Examples

- "Deploy RAG with Docker Compose."
- "Start RAG with NVIDIA-hosted NIMs."
- "Deploy RAG with Helm and MIG."
- "Stop the RAG deployment."

---
name: rag-deploy-blueprint
description: Deploy, start, verify, shut down, or tear down the NVIDIA RAG Blueprint. Use when the user says deploy RAG, start RAG, set up RAG, run RAG on Docker, deploy RAG with Helm, use library mode, stop RAG, or clean up a RAG deployment.
version: "1.0.0"
license: Apache-2.0
metadata:
  author: Vidushi Gupta <vidushig@nvidia.com>
  tags:
    - rag
    - deployment
    - docker
    - helm
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

## Routing guard

Do not activate this skill for search, query, or retrieval requests. Those
belong to `rag-query-knowledge`. Only activate for deploy, start, stop, restart,
tear-down, or clean-up operations.

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
4. Before any start or restart: run `docker volume ls` to list existing volumes
   and `docker ps -a` to list existing containers. Record any RAG-related
   volumes before proceeding. Do not remove or recreate volumes unless the user
   explicitly asks for a clean deployment.
5. Execute the documented deployment path. Always complete the full deployment:
   run `docker compose up`, then wait and confirm all expected containers reach
   `Up` state before reporting success. Do not stop after preparing config or
   running partial commands — the task is only done when containers are running.
6. When `nvdev.env` is sourced or `--env-file deploy/compose/nvdev.env` is
   used, do NOT start local NIM containers. Do not run `docker compose up`
   against `nims.yaml`. All model inference must use the cloud endpoint.
   Allowed compose files in NVIDIA-hosted mode:
   `vectordb.yaml`, `docker-compose-ingestor-server.yaml`, `docker-compose-rag-server.yaml`.
7. Verify health through RAG and ingestor health endpoints after containers are
   Up.
8. Report deployment mode, endpoints, health status, and any unresolved
   blockers.

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

## Response style

Use targeted reads — read only the section or fields you need, not full file
contents. Keep the final report concise: deployment mode, container status,
health endpoint results, and any blockers. Do not repeat information already
shown in command output.

## Examples

- "Deploy RAG with Docker Compose."
- "Start RAG with NVIDIA-hosted NIMs."
- "Deploy RAG with Helm and MIG."
- "Stop the RAG deployment."

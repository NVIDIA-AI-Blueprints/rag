#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live verification for NRL + filesystem-backed object storage.

This script exercises the real library-mode NRL ingestion path with:
- LanceDB as the local vector store
- filesystem-backed object storage for persisted visual artifacts
- NVIDIA-hosted endpoints for extraction/caption/embedding

Verification steps:
1. create a temporary LanceDB collection
2. ingest a multimodal PDF through ``NvidiaRAGIngestor`` with ``INGESTOR_BACKEND=nrl``
3. confirm filesystem artifacts exist and are non-empty
4. retrieve documents through the real LanceDB retrieval path
5. build citations through ``prepare_citations_nrl`` and confirm one visual asset is readable
6. recreate library objects and confirm the same stored URI is still readable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from nvidia_rag import NvidiaRAGIngestor
from nvidia_rag.ingestor_server.main import Mode
from nvidia_rag.rag_server.response_generator import (
    configure_object_store_operator,
    get_object_store_operator_instance,
    prepare_citations_nrl,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.vdb import _get_vdb_op

DEFAULT_PDF = Path("data/multimodal/functional_validation.pdf")
DEFAULT_QUERIES = [
    "table",
    "chart",
    "image",
    "figure",
    "diagram",
    "visual summary",
]


def _require_api_key() -> None:
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY"):
        return
    raise RuntimeError(
        "NVIDIA_API_KEY or NGC_API_KEY must be set for cloud verification."
    )


def _build_config(work_root: Path, collection_name: str) -> NvidiaRAGConfig:
    config = NvidiaRAGConfig()

    config.vector_store.name = "lancedb"
    config.vector_store.url = str((work_root / "lancedb").resolve())
    config.vector_store.default_collection_name = collection_name
    config.vector_store.enable_gpu_index = False
    config.vector_store.enable_gpu_search = False
    config.vector_store.search_type = "dense"

    config.nv_ingest.backend = "nrl"
    config.nv_ingest.nrl_run_mode = "inprocess"
    config.nv_ingest.extract_text = True
    config.nv_ingest.extract_tables = True
    config.nv_ingest.extract_charts = True
    config.nv_ingest.extract_infographics = True
    # Keep image captioning off for verification in environments without local
    # torch; tables/charts/page images still exercise stored visual artifacts.
    config.nv_ingest.extract_images = False
    config.nv_ingest.extract_page_as_image = True
    config.nv_ingest.page_elements_invoke_url = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
    )
    config.nv_ingest.graphic_elements_invoke_url = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1"
    )
    config.nv_ingest.ocr_invoke_url = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"
    )
    config.nv_ingest.table_structure_invoke_url = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
    )
    config.nv_ingest.caption_endpoint_url = (
        "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    config.nv_ingest.caption_model_name = "nvidia/nemotron-nano-12b-v2-vl"

    config.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
    config.embeddings.model_name = "nvidia/llama-nemotron-embed-1b-v2"

    config.ranking.enable_reranker = False
    config.query_rewriter.enable_query_rewriter = False
    config.filter_expression_generator.enable_filter_generator = False

    config.object_store.backend = "filesystem"
    config.object_store.local_path = str((work_root / "object-store").resolve())

    config.enable_citations = True
    return config


def _collect_nonempty_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    if not root.exists():
        return files

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= 0:
            continue
        files.append({"path": str(path), "size": size})
    return files


async def _run(pdf_path: Path, work_root: Path, queries: list[str]) -> dict[str, object]:
    _require_api_key()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    work_root.mkdir(parents=True, exist_ok=True)
    collection_name = f"fs_verify_{uuid.uuid4().hex[:8]}"
    config = _build_config(work_root, collection_name)
    configure_object_store_operator(config)

    vdb_endpoint = str((work_root / "lancedb").resolve())
    object_store_root = Path(config.object_store.local_path).resolve()

    ingestor = NvidiaRAGIngestor(mode=Mode.LIBRARY, config=config)
    create_response = ingestor.create_collection(
        collection_name=collection_name,
        vdb_endpoint=vdb_endpoint,
    )

    upload_response = await ingestor.upload_documents(
        collection_name=collection_name,
        vdb_endpoint=vdb_endpoint,
        blocking=True,
        filepaths=[str(pdf_path.resolve())],
        generate_summary=False,
    )
    if upload_response.get("message") != "Document upload job successfully completed.":
        raise RuntimeError(
            f"Ingestion did not complete successfully: {upload_response}"
        )
    if upload_response.get("failed_documents"):
        raise RuntimeError(
            f"Ingestion reported failed documents: {upload_response['failed_documents']}"
        )

    artifact_files = _collect_nonempty_files(object_store_root)
    if not artifact_files:
        raise RuntimeError(
            f"No non-empty filesystem artifacts found under {object_store_root}"
        )

    embedding_model = get_embedding_model(
        model=config.embeddings.model_name,
        url=config.embeddings.server_url,
        config=config,
    )
    vdb_op = _get_vdb_op(
        vdb_endpoint=vdb_endpoint,
        collection_name=collection_name,
        embedding_model=embedding_model,
        config=config,
    )

    retrieved_docs = []
    selected_query = None
    for query in queries:
        docs = vdb_op.retrieval_langchain(
            query=query,
            collection_name=collection_name,
            top_k=20,
        )
        visual_docs = [doc for doc in docs if doc.metadata.get("stored_image_uri")]
        if visual_docs:
            retrieved_docs = visual_docs
            selected_query = query
            break

    if not retrieved_docs:
        raise RuntimeError(
            "Retrieved documents did not include any visual artifacts with stored_image_uri."
        )

    citations = prepare_citations_nrl(
        retrieved_docs,
        force_citations=True,
        enable_citations=True,
    )
    if citations.total_results <= 0:
        raise RuntimeError("Citation preparation produced no results for visual docs.")

    first_uri = str(retrieved_docs[0].metadata.get("stored_image_uri") or "")
    if not first_uri:
        raise RuntimeError("First retrieved visual document had no stored_image_uri.")

    first_read = get_object_store_operator_instance(config).get_object_from_uri(first_uri)
    if not first_read:
        raise RuntimeError(f"Stored artifact at {first_uri} was empty on first read.")

    recreated_ingestor = NvidiaRAGIngestor(mode=Mode.LIBRARY, config=config)
    configure_object_store_operator(config)
    second_read = get_object_store_operator_instance().get_object_from_uri(first_uri)
    if not second_read:
        raise RuntimeError(f"Stored artifact at {first_uri} was empty after recreation.")

    return {
        "status": "ok",
        "pdf": str(pdf_path.resolve()),
        "work_root": str(work_root.resolve()),
        "collection_name": collection_name,
        "create_collection": create_response,
        "upload_message": upload_response.get("message"),
        "upload_total_documents": upload_response.get("total_documents"),
        "object_store_root": str(object_store_root),
        "artifact_file_count": len(artifact_files),
        "artifact_files_preview": artifact_files[:10],
        "retrieval_query": selected_query,
        "retrieved_visual_docs": len(retrieved_docs),
        "citations_total_results": citations.total_results,
        "first_stored_uri": first_uri,
        "first_read_bytes": len(first_read),
        "second_read_bytes": len(second_read),
        "recreated_operator_available": recreated_ingestor.object_store_operator is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live-verify filesystem-backed object storage via library-mode NRL."
    )
    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF),
        help="Path to a multimodal PDF to ingest.",
    )
    parser.add_argument(
        "--work-root",
        default="tmp/fs-object-store-live-verify",
        help="Scratch directory for LanceDB and filesystem object-store artifacts.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Retrieval query to try. May be passed multiple times.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the work directory after the run.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    work_root = Path(args.work_root)
    queries = args.queries or list(DEFAULT_QUERIES)

    try:
        result = asyncio.run(_run(pdf_path, work_root, queries))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "pdf": str(pdf_path.resolve()),
                    "work_root": str(work_root.resolve()),
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    finally:
        if work_root.exists() and not args.keep:
            shutil.rmtree(work_root, ignore_errors=True)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

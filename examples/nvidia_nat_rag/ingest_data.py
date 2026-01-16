#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Document ingestion script for NVIDIA RAG using NV-Ingest.

This script demonstrates containerless document ingestion using:
- Milvus Lite (embedded vector database)
- NV-Ingest pipeline in subprocess mode
- NVIDIA cloud APIs for embeddings

Usage:
    python ingest_data.py --collection <name> --files <file1> <file2> ...

Example:
    python ingest_data.py --collection my_docs --files ./data/doc1.pdf ./data/doc2.pdf
"""

import argparse
import logging
import os
import platform
import sys
import time

# Suppress verbose logs for cleaner output
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_api_key():
    """Validate that NVIDIA API key is set in environment."""
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")

    if not api_key:
        logger.error("=" * 60)
        logger.error("ERROR: NVIDIA API key not found!")
        logger.error("=" * 60)
        logger.error("")
        logger.error("Please set one of the following environment variables:")
        logger.error("  export NVIDIA_API_KEY='nvapi-...'")
        logger.error("  export NGC_API_KEY='nvapi-...'")
        logger.error("")
        logger.error("You can obtain an API key from:")
        logger.error("  https://build.nvidia.com/")
        logger.error("=" * 60)
        sys.exit(1)

    if not api_key.startswith("nvapi-"):
        logger.error("=" * 60)
        logger.error("ERROR: Invalid NVIDIA API key format!")
        logger.error("=" * 60)
        logger.error("")
        logger.error("The API key must start with 'nvapi-'")
        logger.error(f"Your key starts with: '{api_key[:10]}...'")
        logger.error("")
        logger.error("Please check your API key and try again.")
        logger.error("=" * 60)
        sys.exit(1)

    # Ensure both env vars are set for compatibility
    os.environ["NVIDIA_API_KEY"] = api_key
    os.environ["NGC_API_KEY"] = api_key
    logger.info("NVIDIA API key validated successfully")


def apply_macos_patch():
    """Apply compatibility patch for macOS if needed."""
    if platform.system() == "Darwin":
        logger.info("Detected macOS - applying compatibility patch for nv-ingest...")
        import nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners as runners

        def _noop_set_pdeathsig():
            """No-op replacement for Linux-only set_pdeathsig on macOS."""
            pass

        runners.set_pdeathsig = _noop_set_pdeathsig
        logger.info("macOS compatibility patch applied")
    else:
        logger.info(f"Running on {platform.system()} - no patch needed")


def start_pipeline_and_client(startup_wait: int = 20):
    """
    Start NV-Ingest pipeline and connect client.

    Args:
        startup_wait: Seconds to wait for pipeline to stabilize

    Returns:
        NvIngestClient instance
    """
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        run_pipeline,
        PipelineCreationSchema,
    )
    from nv_ingest_client.client import NvIngestClient
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

    logger.info("Bootstrapping NeMo Retriever Pipeline (Local Library Mode)...")
    logger.info(f"This initializes worker processes. Please wait ~{startup_wait} seconds.")

    config = PipelineCreationSchema()

    # Launch the Pipeline Service in a non-blocking subprocess
    run_pipeline(
        config,
        block=False,
        disable_dynamic_scaling=True,
        run_in_subprocess=True,
    )

    # Wait for Ray actors to stabilize
    logger.info(f"Waiting {startup_wait}s for pipeline to stabilize...")
    time.sleep(startup_wait)

    # Connect the Client using SimpleBroker protocol on default port 7671
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost"
    )
    logger.info("Pipeline active. Client connected.")

    return client


def ingest_documents(
    client,
    collection_name: str,
    filepaths: list[str],
    milvus_uri: str = "./milvus-lite.db",
    chunk_size: int = 1024,
    chunk_overlap: int = 150,
    extract_tables: bool = True,
    extract_charts: bool = False,
    recreate_collection: bool = False,
):
    """
    Ingest documents into Milvus collection.

    Args:
        client: NvIngestClient instance
        collection_name: Name of the collection to create/use
        filepaths: List of file paths to ingest
        milvus_uri: Path to Milvus Lite database file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        extract_tables: Whether to extract tables from documents
        extract_charts: Whether to extract charts from documents
        recreate_collection: Whether to recreate the collection if it exists
    """
    from nv_ingest_client.client.interface import Ingestor

    logger.info("=" * 60)
    logger.info("DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Documents: {filepaths}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Vector DB: {milvus_uri}")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    logger.info("")

    # Build the ingestion pipeline
    ingestor = (
        Ingestor(client=client)
        .files(filepaths)
        .extract(
            extract_text=True,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            text_depth="page",
            table_output_format="markdown"
        )
        .split(
            tokenizer="meta-llama/Llama-3.2-1B",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        .embed()  # Uses llama-3.2-nv-embedqa-1b-v2 (2048 dimensions)
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=False,
            dense_dim=2048,
            recreate=recreate_collection
        )
    )

    logger.info("Starting ingestion pipeline...")
    logger.info("Stages: Extract → Split → Embed → VDB Upload")
    logger.info("")

    t0 = time.time()
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    duration = time.time() - t0

    if failures:
        logger.warning(f"{len(failures)} failures occurred:")
        for source, error in failures[:5]:
            logger.warning(f"  - {source}: {error}")
    else:
        logger.info(f"Ingested into '{milvus_uri}' in {duration:.2f}s")

    return results, failures


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Milvus using NV-Ingest pipeline"
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="test_collection",
        help="Collection name (default: test_collection)",
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        required=True,
        help="File paths to ingest",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./milvus-lite.db",
        help="Path to Milvus Lite database (default: ./milvus-lite.db)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for text splitting (default: 1024)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for text splitting (default: 150)",
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        default=True,
        help="Extract tables from documents (default: True)",
    )
    parser.add_argument(
        "--no-extract-tables",
        action="store_false",
        dest="extract_tables",
        help="Don't extract tables from documents",
    )
    parser.add_argument(
        "--extract-charts",
        action="store_true",
        default=False,
        help="Extract charts from documents (default: False)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="Recreate collection if it exists (default: False)",
    )
    parser.add_argument(
        "--startup-wait",
        type=int,
        default=20,
        help="Seconds to wait for pipeline startup (default: 20)",
    )

    args = parser.parse_args()

    # Validate files exist
    for filepath in args.files:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            sys.exit(1)

    # Validate API key
    validate_api_key()

    # Apply macOS patch if needed
    apply_macos_patch()

    # Start pipeline and get client
    try:
        client = start_pipeline_and_client(startup_wait=args.startup_wait)
    except Exception as e:
        logger.error(f"Pipeline startup failed: {e}")
        logger.error("Tip: Ensure no other Ray instances are running (`ray stop` in terminal).")
        sys.exit(1)

    # Run ingestion
    try:
        results, failures = ingest_documents(
            client=client,
            collection_name=args.collection,
            filepaths=args.files,
            milvus_uri=args.db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            extract_tables=args.extract_tables,
            extract_charts=args.extract_charts,
            recreate_collection=args.recreate,
        )

        if failures:
            logger.warning("Ingestion completed with some failures")
            sys.exit(1)
        else:
            logger.info("Ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

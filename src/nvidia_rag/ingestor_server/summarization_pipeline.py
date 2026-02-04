# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Document summarization pipeline module for RAG ingestion.

This module provides functions for generating document summaries using both
shallow extraction (text-only) and deep extraction (multimodal) workflows.

Functions:
    ingest_document_summary: Trigger parallel summary generation for documents
    process_shallow_batch: Process shallow extraction for a batch of files
    perform_shallow_extraction_workflow: Orchestrate shallow extraction in batches
    perform_shallow_extraction: Perform text-only extraction using NV-Ingest
"""

import asyncio
import logging
import os
import time
from typing import Any

from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.ingestor_server.nvingest import get_nv_ingest_ingestor
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    trace_function,
)
from nvidia_rag.utils.summarization import generate_document_summaries
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER

logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.summarization_pipeline")


@trace_function("ingestor.summarization_pipeline.ingest_document_summary", tracer=TRACER)
async def ingest_document_summary(
    results: list[list[dict[str, str | dict]]],
    collection_name: str,
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    page_filter: list[list[int]] | str | None = None,
    summarization_strategy: str | None = None,
    is_shallow: bool = False,
) -> None:
    """
    Trigger parallel summary generation for documents with optional page filtering.

    Args:
        results: List of document extraction results from nv-ingest
        collection_name: Name of the collection
        config: Configuration object
        prompts: Prompts dictionary for LLM calls
        page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
        summarization_strategy: Strategy for summarization ('single', 'hierarchical') or None for default
        is_shallow: Whether this is shallow extraction (text-only, uses simplified prompt)
    """
    try:
        stats = await generate_document_summaries(
            results=results,
            collection_name=collection_name,
            page_filter=page_filter,
            summarization_strategy=summarization_strategy,
            config=config,
            is_shallow=is_shallow,
            prompts=prompts,
        )

        if stats["failed"] > 0:
            logger.warning(f"Failed summaries for {collection_name}:")
            for file_name, file_stats in stats["files"].items():
                if file_stats["status"] == "FAILED":
                    logger.warning(
                        f"  - {file_name}: {file_stats.get('error', 'unknown error')}"
                    )

    except Exception as e:
        logger.error(
            f"Summary batch failed for {collection_name}: {e}", exc_info=True
        )


async def process_shallow_batch(
    filepaths: list[str],
    collection_name: str,
    split_options: dict[str, Any],
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    state_manager: IngestionStateManager,
    perform_async_nv_ingest_ingestion_func,
    background_tasks: set[asyncio.Task],
    page_filter: list[list[int]] | str | None = None,
    summarization_strategy: str | None = None,
    batch_num: int = 0,
) -> set[str]:
    """
    Process shallow extraction for a batch of files and start summary task.

    Args:
        filepaths: List of file paths to process
        collection_name: Name of the collection
        split_options: Options for splitting documents
        config: Configuration object
        prompts: Prompts dictionary for LLM calls
        state_manager: State manager for the ingestion process
        perform_async_nv_ingest_ingestion_func: Function to perform async NV-Ingest ingestion
        background_tasks: Set to track background tasks
        page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
        summarization_strategy: Strategy for summarization
        batch_num: Batch number for logging

    Returns:
        Set of filenames that failed during shallow extraction
    """
    shallow_failed_files: set[str] = set()

    shallow_results, shallow_failures = await perform_shallow_extraction(
        filepaths=filepaths,
        split_options=split_options,
        config=config,
        prompts=prompts,
        state_manager=state_manager,
        perform_async_nv_ingest_ingestion_func=perform_async_nv_ingest_ingestion_func,
        batch_number=batch_num,
    )

    # Mark per-file shallow extraction failures immediately
    if shallow_failures:
        for failed_path, error in shallow_failures:
            file_name = os.path.basename(str(failed_path))
            shallow_failed_files.add(file_name)
            SUMMARY_STATUS_HANDLER.update_progress(
                collection_name=collection_name,
                file_name=file_name,
                status="FAILED",
                error=str(error),
            )

    if shallow_results:
        task = asyncio.create_task(
            ingest_document_summary(
                results=shallow_results,
                collection_name=collection_name,
                config=config,
                prompts=prompts,
                page_filter=page_filter,
                summarization_strategy=summarization_strategy,
                is_shallow=True,
            )
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    else:
        # No shallow results at all: mark every file in the batch as failed (if not already marked)
        for filepath in filepaths:
            file_name = os.path.basename(filepath)
            if file_name in shallow_failed_files:
                continue
            shallow_failed_files.add(file_name)
            SUMMARY_STATUS_HANDLER.update_progress(
                collection_name=collection_name,
                file_name=file_name,
                status="FAILED",
                error="Shallow extraction failed - no text-only results returned",
            )

    return shallow_failed_files


@trace_function("ingestor.summarization_pipeline.perform_shallow_extraction_workflow", tracer=TRACER)
async def perform_shallow_extraction_workflow(
    filepaths: list[str],
    collection_name: str,
    split_options: dict[str, Any],
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    state_manager: IngestionStateManager,
    perform_async_nv_ingest_ingestion_func,
    background_tasks: set[asyncio.Task],
    summary_options: dict[str, Any] | None = None,
) -> None:
    """
    Perform shallow extraction workflow for fast summary generation.
    Runs shallow extraction and starts summary tasks.
    Handles both single batch and multi-batch modes.
    Respects ENABLE_NV_INGEST_BATCH_MODE and ENABLE_NV_INGEST_PARALLEL_BATCH_MODE.

    Args:
        filepaths: List of file paths to process
        collection_name: Name of the collection
        split_options: Options for splitting documents
        config: Configuration object
        prompts: Prompts dictionary for LLM calls
        state_manager: State manager for the ingestion process
        perform_async_nv_ingest_ingestion_func: Function to perform async NV-Ingest ingestion
        background_tasks: Set to track background tasks
        summary_options: Advanced options for summary
    """
    # Extract options (summary_options is guaranteed to be non-None when this is called)
    page_filter = summary_options.get("page_filter") if summary_options else None
    summarization_strategy = (
        summary_options.get("summarization_strategy") if summary_options else None
    )

    # Determine processing mode
    if not config.nv_ingest.enable_batch_mode:
        # Single batch mode
        logger.info("Starting shallow extraction for %d files", len(filepaths))
        failed_files = await process_shallow_batch(
            filepaths=filepaths,
            collection_name=collection_name,
            split_options=split_options,
            config=config,
            prompts=prompts,
            state_manager=state_manager,
            perform_async_nv_ingest_ingestion_func=perform_async_nv_ingest_ingestion_func,
            background_tasks=background_tasks,
            page_filter=page_filter,
            summarization_strategy=summarization_strategy,
            batch_num=0,
        )
        if failed_files:
            logger.warning(
                "Shallow extraction failed for %d files", len(failed_files)
            )
        logger.info("Shallow extraction complete, starting deep ingestion")
    else:
        # Batch mode
        num_batches = (
            len(filepaths) + state_manager.files_per_batch - 1
        ) // state_manager.files_per_batch

        logger.info(
            "Starting shallow extraction for %d files across %d batches",
            len(filepaths),
            num_batches,
        )

        if not config.nv_ingest.enable_parallel_batch_mode:
            # Sequential batch processing
            total_failed = 0
            for i in range(
                0, len(filepaths), state_manager.files_per_batch
            ):
                sub_filepaths = filepaths[
                    i : i + state_manager.files_per_batch
                ]
                batch_num = i // state_manager.files_per_batch + 1

                failed_files = await process_shallow_batch(
                    filepaths=sub_filepaths,
                    collection_name=collection_name,
                    split_options=split_options,
                    config=config,
                    prompts=prompts,
                    state_manager=state_manager,
                    perform_async_nv_ingest_ingestion_func=perform_async_nv_ingest_ingestion_func,
                    background_tasks=background_tasks,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                    batch_num=batch_num,
                )
                total_failed += len(failed_files)

            if total_failed > 0:
                logger.warning(
                    "Shallow extraction failed for %d files across all batches",
                    total_failed,
                )
        else:
            # Parallel batch processing with worker pool
            tasks = []
            semaphore = asyncio.Semaphore(state_manager.concurrent_batches)

            async def process_shallow_batch_parallel(sub_filepaths, batch_num):
                async with semaphore:
                    return await process_shallow_batch(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        split_options=split_options,
                        config=config,
                        prompts=prompts,
                        state_manager=state_manager,
                        perform_async_nv_ingest_ingestion_func=perform_async_nv_ingest_ingestion_func,
                        background_tasks=background_tasks,
                        page_filter=page_filter,
                        summarization_strategy=summarization_strategy,
                        batch_num=batch_num,
                    )

            for i in range(
                0, len(filepaths), state_manager.files_per_batch
            ):
                sub_filepaths = filepaths[
                    i : i + state_manager.files_per_batch
                ]
                batch_num = i // state_manager.files_per_batch + 1
                task = process_shallow_batch_parallel(sub_filepaths, batch_num)
                tasks.append(task)

            # Wait for all shallow extraction tasks to complete
            batch_results = await asyncio.gather(*tasks)

            # Count total failed files from all batches
            total_failed = sum(len(failed_files) for failed_files in batch_results)
            if total_failed > 0:
                logger.warning(
                    "Shallow extraction failed for %d files across all batches",
                    total_failed,
                )

        logger.info("Shallow extraction complete, starting deep ingestion")


@trace_function("ingestor.summarization_pipeline.perform_shallow_extraction", tracer=TRACER)
async def perform_shallow_extraction(
    filepaths: list[str],
    split_options: dict[str, Any],
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    state_manager: IngestionStateManager,
    perform_async_nv_ingest_ingestion_func,
    batch_number: int = 0,
) -> tuple[list[list[dict[str, str | dict]]], list[tuple[str, Exception]]]:
    """
    Perform text-only extraction using NV-Ingest for fast summary generation.

    Extracts only text content without multimodal elements (tables, images, charts).
    Does not generate embeddings or upload to VDB.
    Does not perform text splitting - summarization will handle its own splitting.

    Args:
        filepaths: List of file paths to extract
        split_options: Options for splitting documents (unused in shallow extraction)
        config: Configuration object
        prompts: Prompts dictionary for LLM calls
        state_manager: State manager for the ingestion process
        perform_async_nv_ingest_ingestion_func: Function to perform async NV-Ingest ingestion
        batch_number: Batch number for logging

    Returns:
        Tuple of (results, failures) where failures is list of (filepath, exception) tuples
    """
    extract_override = {
        "extract_text": True,
        "extract_infographics": False,
        "extract_tables": False,
        "extract_charts": False,
        "extract_images": False,
        "extract_method": config.nv_ingest.pdf_extract_method,
        "text_depth": config.nv_ingest.text_depth,
        "table_output_format": "pseudo_markdown",
        "extract_audio_params": {
            "segment_audio": config.nv_ingest.segment_audio
        },
        "extract_page_as_image": False,
    }

    try:
        # Get nv_ingest_client from config or create a new one
        # Note: This assumes nv_ingest_client is passed through config or available globally
        from nvidia_rag.ingestor_server.nvingest import get_nv_ingest_client

        nv_ingest_client = get_nv_ingest_client(config)

        nv_ingest_ingestor = get_nv_ingest_ingestor(
            nv_ingest_client_instance=nv_ingest_client,
            filepaths=filepaths,
            split_options=None,  # Skip splitting for shallow extraction
            vdb_op=None,
            extract_override=extract_override,
            config=config,
            enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
            pdf_split_processing_options=state_manager.pdf_split_processing_options,
            prompts=prompts,
        )

        start_time = time.time()
        results, failures = await perform_async_nv_ingest_ingestion_func(
            nv_ingest_ingestor=nv_ingest_ingestor,
            state_manager=state_manager,
            nv_ingest_traces=True,
            trace_context=create_nv_ingest_trace_context(
                span_namespace=f"nv_ingest.shallow_batch_{batch_number}",
                batch_number=batch_number,
            ),
        )
        total_time = time.time() - start_time

        logger.debug(
            "Shallow extraction batch %d: %.2fs, %d results, %d failures",
            batch_number,
            total_time,
            len(results) if results else 0,
            len(failures) if failures else 0,
        )

        if failures:
            logger.debug(
                "Shallow extraction: %d failures in batch %d",
                len(failures),
                batch_number,
            )

        # Normalize return values to empty lists instead of None
        return results or [], failures or []

    except Exception as e:
        logger.error(
            "Shallow extraction failed for batch %d: %s",
            batch_number,
            str(e),
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        # Treat every file in this batch as a failure
        failure_records = [(filepath, e) for filepath in filepaths]
        return [], failure_records

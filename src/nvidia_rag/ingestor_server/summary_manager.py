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

import logging
import asyncio
from typing import Any
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER


from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    process_nv_ingest_traces,
    trace_function,
)
from nvidia_rag.utils.summarization import generate_document_summaries

logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.summary_manager")

class SummaryManager:
    def __init__(
        self,
        config: dict,
        prompts: dict,
    ):
        self.config = config
        self.prompts = prompts
    
    @trace_function("ingestor.summary_manager.ingest_document_summary", tracer=TRACER)
    async def ingest_document_summary(
        self,
        results: list[list[dict[str, str | dict]]],
        collection_name: str,
        page_filter: list[list[int]] | str | None = None,
        summarization_strategy: str | None = None,
        is_shallow: bool = False,
    ) -> None:
        """
        Trigger parallel summary generation for documents with optional page filtering.

        Args:
            results: List of document extraction results from nv-ingest
            collection_name: Name of the collection
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
                config=self.config,
                is_shallow=is_shallow,
                prompts=self.prompts,
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
    
    @trace_function("ingestor.summary_manager.perform_shallow_extraction_workflow", tracer=TRACER)
    async def perform_shallow_extraction_workflow(
        self,
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any],
        summary_options: dict[str, Any] | None,
        state_manager: IngestionStateManager,
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
            summary_options: Advanced options for summary
        """
        # Extract options (summary_options is guaranteed to be non-None when this is called)
        page_filter = summary_options.get("page_filter") if summary_options else None
        summarization_strategy = (
            summary_options.get("summarization_strategy") if summary_options else None
        )

        # Determine processing mode
        if not self.config.nv_ingest.enable_batch_mode:
            # Single batch mode
            logger.info("Starting shallow extraction for %d files", len(filepaths))
            failed_files = await self.__process_shallow_batch(
                filepaths=filepaths,
                collection_name=collection_name,
                split_options=split_options,
                page_filter=page_filter,
                summarization_strategy=summarization_strategy,
                batch_num=0,
                state_manager=state_manager,
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

            if not self.config.nv_ingest.enable_parallel_batch_mode:
                # Sequential batch processing
                total_failed = 0
                for i in range(
                    0, len(filepaths), state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + state_manager.files_per_batch
                    ]
                    batch_num = i // state_manager.files_per_batch + 1

                    failed_files = await self.__process_shallow_batch(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        split_options=split_options,
                        page_filter=page_filter,
                        summarization_strategy=summarization_strategy,
                        batch_num=batch_num,
                        state_manager=state_manager,
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
                        return await self.__process_shallow_batch(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            split_options=split_options,
                            page_filter=page_filter,
                            summarization_strategy=summarization_strategy,
                            batch_num=batch_num,
                            state_manager=state_manager,
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
    
    @trace_function("ingestor.summary_manager.process_shallow_batch", tracer=TRACER)
    async def __process_shallow_batch(
        self,
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any],
        page_filter: list[list[int]] | str | None,
        summarization_strategy: str | None,
        batch_num: int,
        state_manager: IngestionStateManager,
    ) -> set[str]:
        """
        Process shallow extraction for a batch of files and start summary task.

        Args:
            filepaths: List of file paths to process
            collection_name: Name of the collection
            split_options: Options for splitting documents
            page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
            summarization_strategy: Strategy for summarization
            batch_num: Batch number for logging

        Returns:
            Set of filenames that failed during shallow extraction
        """
        shallow_failed_files: set[str] = set()

        shallow_results, shallow_failures = await self._perform_shallow_extraction(
            filepaths,
            split_options,
            batch_num,
            state_manager=state_manager,
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
                self.__ingest_document_summary(
                    shallow_results,
                    collection_name=collection_name,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                    is_shallow=True,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
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

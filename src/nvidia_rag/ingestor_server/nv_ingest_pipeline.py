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
NV-Ingest Pipeline Functions Module

This module contains the core NV-Ingest batched ingestion pipeline functions extracted from the main module.
It handles:
1. Batched document ingestion orchestration
2. NV-Ingest pipeline execution and processing
3. Asynchronous NV-Ingest ingestion with status polling
4. File extension-based ingestion splitting (PDF vs non-PDF)

Functions:
- run_nvingest_batched_ingestion: Orchestrates batched ingestion workflow
- nv_ingest_ingestion_pipeline: Executes the NV-Ingest extraction and embedding pipeline
- perform_async_nv_ingest_ingestion: Performs async NV-Ingest ingestion with status tracking
- perform_file_ext_based_nv_ingest_ingestion: Splits ingestion by file type (PDF/non-PDF)
"""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Callable

from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_ingestor,
)
from nvidia_rag.ingestor_server.task_handler import INGESTION_TASK_HANDLER
from nvidia_rag.ingestor_server.document_processor import (
    build_ingestion_response,
    log_result_info,
)
from nvidia_rag.ingestor_server.validation import (
    remove_unsupported_files,
    split_pdf_and_non_pdf_files,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    process_nv_ingest_traces,
    trace_function,
)
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.nv_ingest_pipeline")


@trace_function("ingestor.nv_ingest_pipeline.run_nvingest_batched_ingestion", tracer=TRACER)
async def run_nvingest_batched_ingestion(
    filepaths: list[str],
    collection_name: str,
    config: NvidiaRAGConfig,
    nv_ingest_client,
    prompts: dict[str, str],
    get_documents_func: Callable,
    put_content_to_minio_callback: Callable | None = None,
    ingest_document_summary_callback: Callable | None = None,
    background_tasks: set | None = None,
    mode: str | None = None,
    vdb_op: VDBRag | None = None,
    split_options: dict[str, Any] | None = None,
    generate_summary: bool = False,
    summary_options: dict[str, Any] | None = None,
    state_manager: IngestionStateManager | None = None,
    perform_shallow_extraction_workflow_callback: Callable | None = None,
) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
    """
    Wrapper function to ingest documents in chunks using NV-ingest

    Args:
        - filepaths: List[str] - List of absolute filepaths
        - collection_name: str - Name of the collection in the vector database
        - config: NvidiaRAGConfig - Configuration object
        - nv_ingest_client: NV-Ingest client instance
        - prompts: dict[str, str] - Prompts for various operations
        - get_documents_func: Callable - Function to get documents from VDB
        - put_content_to_minio_callback: Callable - Callback for putting content to MinIO
        - ingest_document_summary_callback: Callable - Callback for ingesting document summaries
        - background_tasks: set - Set of background tasks
        - mode: str - Application mode
        - vdb_op: VDBRag - VDB operator instance
        - split_options: SplitOptions - Options for splitting documents
        - generate_summary: bool - Whether to generate summaries
        - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
        - state_manager: IngestionStateManager - State manager for the ingestion process
        - perform_shallow_extraction_workflow_callback: Callable - Callback for shallow extraction workflow
    """
    # Extract summary options
    shallow_summary = False
    if summary_options:
        shallow_summary = summary_options.get("shallow_summary", False)

    # Set PENDING status for all files if summary generation is enabled
    if generate_summary:
        logger.debug("Setting PENDING status for %d files", len(filepaths))
        for filepath in filepaths:
            file_name = os.path.basename(filepath)
            SUMMARY_STATUS_HANDLER.set_status(
                collection_name=collection_name,
                file_name=file_name,
                status_data={
                    "status": "PENDING",
                    "queued_at": datetime.now(UTC).isoformat(),
                    "file_name": file_name,
                    "collection_name": collection_name,
                },
            )

        # Perform shallow extraction workflow if enabled
        if shallow_summary and perform_shallow_extraction_workflow_callback:
            await perform_shallow_extraction_workflow_callback(
                filepaths=filepaths,
                collection_name=collection_name,
                split_options=split_options,
                summary_options=summary_options,
                state_manager=state_manager,
            )

    if not config.nv_ingest.enable_batch_mode:
        # Single batch mode
        logger.info(
            "== Performing ingestion in SINGLE batch for collection_name: %s with %d files ==",
            collection_name,
            len(filepaths),
        )
        results, failures = await nv_ingest_ingestion_pipeline(
            filepaths=filepaths,
            collection_name=collection_name,
            config=config,
            nv_ingest_client=nv_ingest_client,
            prompts=prompts,
            get_documents_func=get_documents_func,
            put_content_to_minio_callback=put_content_to_minio_callback,
            ingest_document_summary_callback=ingest_document_summary_callback,
            background_tasks=background_tasks,
            mode=mode,
            vdb_op=vdb_op,
            split_options=split_options,
            generate_summary=generate_summary,
            summary_options=summary_options,
            state_manager=state_manager,
        )
        return results, failures

    else:
        # BATCH_MODE
        logger.info(
            f"== Performing ingestion in BATCH_MODE for collection_name: {collection_name} "
            f"with {len(filepaths)} files =="
        )

        # Process batches sequentially
        if not config.nv_ingest.enable_parallel_batch_mode:
            logger.info("Processing batches sequentially")
            all_results = []
            all_failures = []
            for i in range(
                0, len(filepaths), state_manager.files_per_batch
            ):
                sub_filepaths = filepaths[
                    i : i + state_manager.files_per_batch
                ]
                batch_num = i // state_manager.files_per_batch + 1
                total_batches = (
                    len(filepaths) + state_manager.files_per_batch - 1
                ) // state_manager.files_per_batch
                logger.info(
                    f"=== Batch Processing Status - Collection: {collection_name} - "
                    f"Processing batch {batch_num} of {total_batches} - "
                    f"Documents in current batch: {len(sub_filepaths)} ==="
                )
                results, failures = await nv_ingest_ingestion_pipeline(
                    filepaths=sub_filepaths,
                    collection_name=collection_name,
                    config=config,
                    nv_ingest_client=nv_ingest_client,
                    prompts=prompts,
                    get_documents_func=get_documents_func,
                    put_content_to_minio_callback=put_content_to_minio_callback,
                    ingest_document_summary_callback=ingest_document_summary_callback,
                    background_tasks=background_tasks,
                    mode=mode,
                    vdb_op=vdb_op,
                    batch_number=batch_num,
                    split_options=split_options,
                    generate_summary=generate_summary,
                    summary_options=summary_options,
                    state_manager=state_manager,
                )
                all_results.extend(results)
                all_failures.extend(failures)

            if (
                hasattr(vdb_op, "csv_file_path")
                and vdb_op.csv_file_path is not None
            ):
                os.remove(vdb_op.csv_file_path)
                logger.debug(
                    f"Deleted temporary custom metadata csv file: {vdb_op.csv_file_path} "
                    f"for collection: {collection_name}"
                )

            return all_results, all_failures

        else:
            # Process batches in parallel with worker pool
            logger.info(
                f"Processing batches in parallel with concurrency: {state_manager.concurrent_batches}"
            )
            all_results = []
            all_failures = []
            tasks = []
            semaphore = asyncio.Semaphore(
                state_manager.concurrent_batches
            )  # Limit concurrent tasks

            async def process_batch(sub_filepaths, batch_num):
                async with semaphore:
                    if len(filepaths) % state_manager.files_per_batch == 0:
                        total_batches = len(filepaths) // state_manager.files_per_batch
                    else:
                        total_batches = len(filepaths) // state_manager.files_per_batch + 1
                    logger.info(
                        f"=== Processing Batch - Collection: {collection_name} - "
                        f"Batch {batch_num} of {total_batches} - "
                        f"Documents in batch: {len(sub_filepaths)} ==="
                    )
                    return await nv_ingest_ingestion_pipeline(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        config=config,
                        nv_ingest_client=nv_ingest_client,
                        prompts=prompts,
                        get_documents_func=get_documents_func,
                        put_content_to_minio_callback=put_content_to_minio_callback,
                        ingest_document_summary_callback=ingest_document_summary_callback,
                        background_tasks=background_tasks,
                        mode=mode,
                        vdb_op=vdb_op,
                        batch_number=batch_num,
                        split_options=split_options,
                        generate_summary=generate_summary,
                        summary_options=summary_options,
                        state_manager=state_manager,
                    )

            for i in range(
                0, len(filepaths), state_manager.files_per_batch
            ):
                sub_filepaths = filepaths[
                    i : i + state_manager.files_per_batch
                ]
                batch_num = i // state_manager.files_per_batch + 1
                task = process_batch(sub_filepaths, batch_num)
                tasks.append(task)

            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*tasks)

            # Combine results from all batches
            for results, failures in batch_results:
                all_results.extend(results)
                all_failures.extend(failures)

            if (
                hasattr(vdb_op, "csv_file_path")
                and vdb_op.csv_file_path is not None
            ):
                os.remove(vdb_op.csv_file_path)
                logger.debug(
                    f"Deleted temporary custom metadata csv file: {vdb_op.csv_file_path} "
                    f"for collection: {collection_name}"
                )

            return all_results, all_failures


@trace_function("ingestor.nv_ingest_pipeline.run_nv_ingest_ingestion_pipeline", tracer=TRACER)
async def nv_ingest_ingestion_pipeline(
    filepaths: list[str],
    collection_name: str,
    config: NvidiaRAGConfig,
    nv_ingest_client,
    prompts: dict[str, str],
    get_documents_func: Callable,
    put_content_to_minio_callback: Callable | None = None,
    ingest_document_summary_callback: Callable | None = None,
    background_tasks: set | None = None,
    mode: str | None = None,
    vdb_op: VDBRag | None = None,
    batch_number: int = 0,
    split_options: dict[str, Any] | None = None,
    generate_summary: bool = False,
    summary_options: dict[str, Any] | None = None,
    state_manager: IngestionStateManager | None = None,
) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
    """
    This methods performs following steps:
    - Perform extraction and splitting using NV-ingest ingestor (NV-Ingest)
    - Embeds and add documents to Vectorstore collection (NV-Ingest)
    - Put content to MinIO (Ingestor Server)
    - Update batch progress with the ingestion response (Ingestor Server)

    Arguments:
        - filepaths: List[str] - List of absolute filepaths
        - collection_name: str - Name of the collection in the vector database
        - config: NvidiaRAGConfig - Configuration object
        - nv_ingest_client: NV-Ingest client instance
        - prompts: dict[str, str] - Prompts for various operations
        - get_documents_func: Callable - Function to get documents
        - put_content_to_minio_callback: Callable - Callback to put content to MinIO
        - ingest_document_summary_callback: Callable - Callback to ingest document summary
        - background_tasks: set - Set of background tasks
        - mode: str - Application mode (LITE/SERVER/LIBRARY)
        - vdb_op: VDBRag - VDB operator instance
        - batch_number: int - Batch number for the ingestion process
        - split_options: SplitOptions - Options for splitting documents
        - generate_summary: bool - Whether to generate summaries
        - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
        - state_manager: IngestionStateManager - State manager for the ingestion process
    """
    if split_options is None:
        split_options = {
            "chunk_size": config.nv_ingest.chunk_size,
            "chunk_overlap": config.nv_ingest.chunk_overlap,
        }

    # Extract summary options
    page_filter = None
    shallow_summary = False
    summarization_strategy = None
    if summary_options:
        page_filter = summary_options.get("page_filter")
        shallow_summary = summary_options.get("shallow_summary", False)
        summarization_strategy = summary_options.get("summarization_strategy")

    filtered_filepaths = await remove_unsupported_files(filepaths)

    if len(filtered_filepaths) == 0:
        logger.error("No files to ingest after filtering.")
        results, failures = [], []
        return results, failures

    results, failures = await perform_file_ext_based_nv_ingest_ingestion(
        batch_number=batch_number,
        filtered_filepaths=filtered_filepaths,
        split_options=split_options,
        config=config,
        nv_ingest_client=nv_ingest_client,
        prompts=prompts,
        vdb_op=vdb_op,
        state_manager=state_manager,
    )

    # Start summary task only if not shallow_summary (already started in batch wrapper)
    if generate_summary and not shallow_summary and ingest_document_summary_callback:
        if background_tasks is not None:
            task = asyncio.create_task(
                ingest_document_summary_callback(
                    results,
                    collection_name=collection_name,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                )
            )
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            logger.info(
                "Started summary generation after full ingestion for batch %d",
                batch_number,
            )

    if not results:
        error_message = "NV-Ingest ingestion failed with no results."
        logger.error(error_message)

        # Update FAILED status only if not shallow_summary
        if generate_summary and not shallow_summary:
            for filepath in filtered_filepaths:
                file_name = os.path.basename(filepath)
                SUMMARY_STATUS_HANDLER.update_progress(
                    collection_name=collection_name,
                    file_name=file_name,
                    status="FAILED",
                    error="Ingestion failed - no results returned from NV-Ingest",
                )
            logger.warning(
                "Marked %d files as FAILED for batch %d due to ingestion failure",
                len(filtered_filepaths),
                batch_number,
            )

        if len(failures) > 0:
            return results, failures
        raise Exception(error_message)

    try:
        if mode != "lite" and put_content_to_minio_callback:
            start_time = time.time()
            put_content_to_minio_callback(
                results=results, collection_name=collection_name
            )
            end_time = time.time()
            logger.info(
                f"== MinIO upload for collection_name: {collection_name} "
                f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
            )
        start_time = time.time()
        batch_progress_response = await build_ingestion_response(
            results=results,
            failures=failures,
            filepaths=filepaths,
            is_final_batch=False,
            collection_name=state_manager.collection_name,
            custom_metadata=state_manager.custom_metadata,
            failed_validation_documents=state_manager.failed_validation_documents,
            validation_errors=state_manager.validation_errors,
            get_documents_func=get_documents_func,
            vdb_op=vdb_op,
        )
        end_time = time.time()
        logger.info(
            f"== Build ingestion response and adding document info for collection_name: {collection_name} "
            f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
        )
        ingestion_state = await state_manager.update_batch_progress(
            batch_progress_response=batch_progress_response,
        )
        await INGESTION_TASK_HANDLER.set_task_status_and_result(
            task_id=state_manager.get_task_id(),
            status="PENDING",
            result=ingestion_state,
        )
    except Exception as e:
        logger.error(
            "Failed to put content to minio: %s, citations would be disabled for collection: %s",
            str(e),
            collection_name,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )

    return results, failures


@trace_function("ingestor.nv_ingest_pipeline.perform_async_nv_ingest_ingestion", tracer=TRACER)
async def perform_async_nv_ingest_ingestion(
    nv_ingest_ingestor,
    state_manager,
    nv_ingest_traces: bool = False,
    trace_context: dict[str, Any] | None = None,
):
    """
    Perform NV-Ingest ingestion asynchronously using .ingest_async() method
    Also, poll the ingestion status until it is complete and update the ingestion status using state_manager

    Arguments:
        - nv_ingest_ingestor: Ingestor - NV-Ingest ingestor instance

    Returns:
        - tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]] - Results and failures
    """
    ingest_start_ns = time.time_ns()
    future = nv_ingest_ingestor.ingest_async(
        return_failures=True,
        show_progress=logger.getEffectiveLevel() <= logging.DEBUG,
        return_traces=nv_ingest_traces,
    )
    # Convert concurrent.futures.Future to asyncio.Future
    async_future = asyncio.wrap_future(future)

    while True:
        status_dict = await asyncio.to_thread(nv_ingest_ingestor.get_status)
        filename_status_map = {}
        # Normalize the status to a dictionary of filename to status
        for filepath, file_status in status_dict.items():
            filename = os.path.basename(filepath)
            filename_status_map[filename] = file_status
        nv_ingest_status = await state_manager.update_nv_ingest_status(
            filename_status_map
        )
        await INGESTION_TASK_HANDLER.set_task_state_dict(
            state_manager.get_task_id(),
            {"nv_ingest_status": nv_ingest_status},
        )

        await asyncio.sleep(1)

        if future.done():
            break

    if nv_ingest_traces:
        results, failures, traces = await async_future

        if trace_context is not None:
            process_nv_ingest_traces(
                traces,
                tracer=TRACER,
                span_namespace=trace_context.get("span_namespace", "nv_ingest"),
                collection_name=trace_context.get("collection_name"),
                batch_number=trace_context.get("batch_number"),
                reference_time_ns=trace_context.get(
                    "reference_time_ns", ingest_start_ns
                ),
            )

        return results, failures

    results, failures = await async_future
    return results, failures


@trace_function(
    "ingestor.nv_ingest_pipeline.perform_file_ext_based_nv_ingest_ingestion", tracer=TRACER
)
async def perform_file_ext_based_nv_ingest_ingestion(
    batch_number: int,
    filtered_filepaths: list[str],
    split_options: dict[str, Any],
    config: NvidiaRAGConfig,
    nv_ingest_client,
    prompts: dict[str, str],
    vdb_op: VDBRag,
    state_manager: IngestionStateManager,
):
    """
    Perform ingestion using NV-Ingest ingestor based on file extension
    - If pdf extract method is None, perform ingestion for all files
    - If pdf extract method is not None, split the files into PDF and non-PDF files and perform ingestion for PDF files
        - Perform ingestion for non-PDF files with remove_extract_method=True

    Arguments:
        - batch_number: int - Batch number for the ingestion process
        - filtered_filepaths: list[str] - List of filtered filepaths
        - split_options: dict[str, Any] - Options for splitting documents
        - config: NvidiaRAGConfig - Configuration object
        - nv_ingest_client: NV-Ingest client instance
        - prompts: dict[str, str] - Prompts for various operations
        - vdb_op: VDBRag - Vector database operation instance

    Returns:
        - tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]] - Results and failures
    """
    if config.nv_ingest.pdf_extract_method is None:
        nv_ingest_ingestor = get_nv_ingest_ingestor(
            nv_ingest_client_instance=nv_ingest_client,
            filepaths=filtered_filepaths,
            split_options=split_options,
            vdb_op=vdb_op,
            config=config,
            enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
            pdf_split_processing_options=state_manager.pdf_split_processing_options,
            prompts=prompts,
        )
        start_time = time.time()
        logger.info(
            f"Performing ingestion for batch {batch_number} with parameters: {split_options}"
        )
        results, failures = await perform_async_nv_ingest_ingestion(
            nv_ingest_ingestor=nv_ingest_ingestor,
            state_manager=state_manager,
            nv_ingest_traces=True,
            trace_context=create_nv_ingest_trace_context(
                span_namespace=f"nv_ingest.batch_{batch_number}",
                collection_name=vdb_op.collection_name,
                batch_number=batch_number,
            ),
        )
        total_ingestion_time = time.time() - start_time
        document_info = log_result_info(
            batch_number, results, failures, total_ingestion_time
        )
        vdb_op.add_document_info(
            info_type="collection",
            collection_name=vdb_op.collection_name,
            document_name="NA",
            info_value=document_info,
        )
        return results, failures
    else:
        pdf_filepaths, non_pdf_filepaths = await split_pdf_and_non_pdf_files(
            filtered_filepaths
        )
        logger.info(
            f"Split PDF and non-PDF files for batch {batch_number}: "
            f"Count of PDF files: {len(pdf_filepaths)}, Count of non-PDF files: {len(non_pdf_filepaths)}"
        )

        results, failures = [], []
        # Perform ingestion for PDF files
        if len(pdf_filepaths) > 0:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=nv_ingest_client,
                filepaths=pdf_filepaths,
                split_options=split_options,
                vdb_op=vdb_op,
                config=config,
                enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                pdf_split_processing_options=state_manager.pdf_split_processing_options,
                prompts=prompts,
            )
            start_time = time.time()
            logger.info(
                f"Performing ingestion for PDF files for batch {batch_number} with parameters: {split_options}"
            )
            (
                results_pdf,
                failures_pdf,
            ) = await perform_async_nv_ingest_ingestion(
                nv_ingest_ingestor=nv_ingest_ingestor,
                state_manager=state_manager,
                nv_ingest_traces=True,
                trace_context=create_nv_ingest_trace_context(
                    span_namespace=f"nv_ingest.batch_{batch_number}.pdf",
                    collection_name=vdb_op.collection_name,
                    batch_number=batch_number,
                ),
            )
            total_ingestion_time = time.time() - start_time
            document_info = log_result_info(
                batch_number,
                results,
                failures,
                total_ingestion_time,
                additional_summary="PDF files ingestion completed",
            )
            results.extend(results_pdf)
            failures.extend(failures_pdf)

        # Perform ingestion for non-PDF files
        if len(non_pdf_filepaths) > 0:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=nv_ingest_client,
                filepaths=non_pdf_filepaths,
                split_options=split_options,
                vdb_op=vdb_op,
                remove_extract_method=True,
                config=config,
                enable_pdf_split_processing=state_manager.enable_pdf_split_processing,
                pdf_split_processing_options=state_manager.pdf_split_processing_options,
                prompts=prompts,
            )
            start_time = time.time()
            logger.info(
                f"Performing ingestion for non-PDF files for batch {batch_number} with parameters: {split_options}"
            )
            (
                results_non_pdf,
                failures_non_pdf,
            ) = await perform_async_nv_ingest_ingestion(
                nv_ingest_ingestor=nv_ingest_ingestor,
                state_manager=state_manager,
                nv_ingest_traces=True,
                trace_context=create_nv_ingest_trace_context(
                    span_namespace=f"nv_ingest.batch_{batch_number}.non_pdf",
                    collection_name=vdb_op.collection_name,
                    batch_number=batch_number,
                ),
            )
            total_ingestion_time = time.time() - start_time
            document_info = log_result_info(
                batch_number,
                results_non_pdf,
                failures_non_pdf,
                total_ingestion_time,
                additional_summary="Non-PDF files ingestion completed",
            )
            results.extend(results_non_pdf)
            failures.extend(failures_non_pdf)

        vdb_op.add_document_info(
            info_type="collection",
            collection_name=vdb_op.collection_name,
            document_name="NA",
            info_value=document_info,
        )

        return results, failures

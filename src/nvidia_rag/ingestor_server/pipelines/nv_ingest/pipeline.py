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
This module contains the NV-Ingest ingestion pipeline.
""" 

import logging
import os
import asyncio
import time
from pathlib import Path
from datetime import datetime, UTC
from typing import Any

from nvidia_rag.ingestor_server.task_handler import INGESTION_TASK_HANDLER
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER

from nvidia_rag.utils.observability.tracing import (
    create_nv_ingest_trace_context,
    get_tracer,
    process_nv_ingest_traces,
    trace_function,
)
from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
)
from nvidia_rag.utils.common import (
    create_catalog_metadata,
    create_document_metadata,
    derive_boolean_flags,
    get_current_timestamp,
    perform_document_info_aggregation,
)
from nvidia_rag.ingestor_server.constants import Mode
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager
from nvidia_rag.ingestor_server.response_builder import ResponseBuilder
from nvidia_rag.ingestor_server.summary_manager import SummaryManager
from nvidia_rag.utils.minio_operator import MinioOperator

logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.pipelines.nv_ingest.pipeline")

class NVIngestIngestionPipeline():
    
    def __init__(
        self,

        # Configuration arguments
        config: dict,
        vdb_op: VDBRag,
        state_manager: IngestionStateManager,
        mode: Mode,
        minio_operator: MinioOperator,
        prompts: dict,

        # Pipeline arguments
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
    ):
        self.config = config
        self.vdb_op = vdb_op
        self.state_manager = state_manager
        self.mode = mode
        self.response_builder = ResponseBuilder(
            vdb_op=vdb_op,
            config=self.config,
            minio_operator=minio_operator,
        )
        self.nv_ingest_client = get_nv_ingest_client(
            config=self.config, get_lite_client=self.mode == Mode.LITE
        )
        self.prompts = prompts
        self.summary_manager = SummaryManager(
            config=self.config,
            prompts=self.prompts,
        )

        # Pipeline arguments
        self.filepaths = filepaths
        self.collection_name = collection_name
        self.split_options = split_options
        self.generate_summary = generate_summary
        self.summary_options = summary_options

    async def run(self):
        return await self.__run_nvingest_batched_ingestion(
            filepaths=self.filepaths,
            collection_name=self.collection_name,
            split_options=self.split_options,
            generate_summary=self.generate_summary,
            summary_options=self.summary_options,
        )

    @trace_function("ingestor.pipelines.nv_ingest.pipeline.run_nvingest_batched_ingestion", tracer=TRACER)
    async def __run_nvingest_batched_ingestion(
        self,
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
    ) -> tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]]:
        """
        Wrapper function to ingest documents in chunks using NV-ingest

        Args:
            - filepaths: List[str] - List of absolute filepaths
            - collection_name: str - Name of the collection in the vector database
            - split_options: SplitOptions - Options for splitting documents
            - generate_summary: bool - Whether to generate summaries
            - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
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
            if shallow_summary:
                await self.summary_manager.perform_shallow_extraction_workflow(
                    filepaths=filepaths,
                    collection_name=collection_name,
                    split_options=split_options,
                    summary_options=summary_options,
                    state_manager=self.state_manager,
                )

        if not self.config.nv_ingest.enable_batch_mode:
            # Single batch mode
            logger.info(
                "== Performing ingestion in SINGLE batch for collection_name: %s with %d files ==",
                collection_name,
                len(filepaths),
            )
            results, failures = await self.__nv_ingest_ingestion_pipeline(
                filepaths=filepaths,
                collection_name=collection_name,
                split_options=split_options,
                generate_summary=generate_summary,
                summary_options=summary_options,
            )
            return results, failures

        else:
            # BATCH_MODE
            logger.info(
                f"== Performing ingestion in BATCH_MODE for collection_name: {collection_name} "
                f"with {len(filepaths)} files =="
            )

            # Process batches sequentially
            if not self.config.nv_ingest.enable_parallel_batch_mode:
                logger.info("Processing batches sequentially")
                all_results = []
                all_failures = []
                for i in range(
                    0, len(filepaths), self.state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + self.state_manager.files_per_batch
                    ]
                    batch_num = i // self.state_manager.files_per_batch + 1
                    total_batches = (
                        len(filepaths) + self.state_manager.files_per_batch - 1
                    ) // self.state_manager.files_per_batch
                    logger.info(
                        f"=== Batch Processing Status - Collection: {collection_name} - "
                        f"Processing batch {batch_num} of {total_batches} - "
                        f"Documents in current batch: {len(sub_filepaths)} ==="
                    )
                    results, failures = await self.__nv_ingest_ingestion_pipeline(
                        filepaths=sub_filepaths,
                        collection_name=collection_name,
                        batch_number=batch_num,
                        split_options=split_options,
                        generate_summary=generate_summary,
                        summary_options=summary_options,
                    )
                    all_results.extend(results)
                    all_failures.extend(failures)

                if (
                    hasattr(self.vdb_op, "csv_file_path")
                    and self.vdb_op.csv_file_path is not None
                ):
                    os.remove(self.vdb_op.csv_file_path)
                    logger.debug(
                        f"Deleted temporary custom metadata csv file: {self.vdb_op.csv_file_path} "
                        f"for collection: {collection_name}"
                    )

                return all_results, all_failures

            else:
                # Process batches in parallel with worker pool
                logger.info(
                    f"Processing batches in parallel with concurrency: {self.state_manager.concurrent_batches}"
                )
                all_results = []
                all_failures = []
                tasks = []
                semaphore = asyncio.Semaphore(
                    self.state_manager.concurrent_batches
                )  # Limit concurrent tasks

                async def process_batch(sub_filepaths, batch_num):
                    async with semaphore:
                        if len(filepaths) % self.state_manager.files_per_batch == 0:
                            total_batches = len(filepaths) // self.state_manager.files_per_batch
                        else:
                            total_batches = len(filepaths) // self.state_manager.files_per_batch + 1
                        logger.info(
                            f"=== Processing Batch - Collection: {collection_name} - "
                            f"Batch {batch_num} of {total_batches} - "
                            f"Documents in batch: {len(sub_filepaths)} ==="
                        )
                        return await self.__nv_ingest_ingestion_pipeline(
                            filepaths=sub_filepaths,
                            collection_name=collection_name,
                            batch_number=batch_num,
                            split_options=split_options,
                            generate_summary=generate_summary,
                            summary_options=summary_options,
                        )

                for i in range(
                    0, len(filepaths), self.state_manager.files_per_batch
                ):
                    sub_filepaths = filepaths[
                        i : i + self.state_manager.files_per_batch
                    ]
                    batch_num = i // self.state_manager.files_per_batch + 1
                    task = process_batch(sub_filepaths, batch_num)
                    tasks.append(task)

                # Wait for all tasks to complete
                batch_results = await asyncio.gather(*tasks)

                # Combine results from all batches
                for results, failures in batch_results:
                    all_results.extend(results)
                    all_failures.extend(failures)

                if (
                    hasattr(self.vdb_op, "csv_file_path")
                    and self.vdb_op.csv_file_path is not None
                ):
                    os.remove(self.vdb_op.csv_file_path)
                    logger.debug(
                        f"Deleted temporary custom metadata csv file: {self.vdb_op.csv_file_path} "
                        f"for collection: {collection_name}"
                    )

                return all_results, all_failures

    @trace_function("ingestor.main.run_nv_ingest_ingestion_pipeline", tracer=TRACER)
    async def __nv_ingest_ingestion_pipeline(
        self,
        filepaths: list[str],
        collection_name: str,
        batch_number: int = 0,
        split_options: dict[str, Any] | None = None,
        generate_summary: bool = False,
        summary_options: dict[str, Any] | None = None,
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
            - batch_number: int - Batch number for the ingestion process
            - split_options: SplitOptions - Options for splitting documents
            - generate_summary: bool - Whether to generate summaries
            - summary_options: SummaryOptions - Advanced options for summary (page_filter, shallow_summary, summarization_strategy)
        """
        if split_options is None:
            split_options = {
                "chunk_size": self.config.nv_ingest.chunk_size,
                "chunk_overlap": self.config.nv_ingest.chunk_overlap,
            }

        # Extract summary options
        page_filter = None
        shallow_summary = False
        summarization_strategy = None
        if summary_options:
            page_filter = summary_options.get("page_filter")
            shallow_summary = summary_options.get("shallow_summary", False)
            summarization_strategy = summary_options.get("summarization_strategy")

        filtered_filepaths = await self.response_builder.remove_unsupported_files(filepaths)

        if len(filtered_filepaths) == 0:
            logger.error("No files to ingest after filtering.")
            results, failures = [], []
            return results, failures

        results, failures = await self._perform_file_ext_based_nv_ingest_ingestion(
            batch_number=batch_number,
            filtered_filepaths=filtered_filepaths,
            split_options=split_options,
        )

        # Start summary task only if not shallow_summary (already started in batch wrapper)
        if generate_summary and not shallow_summary:
            task = asyncio.create_task(
                self.summary_manager.ingest_document_summary(
                    results,
                    collection_name=collection_name,
                    page_filter=page_filter,
                    summarization_strategy=summarization_strategy,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
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
            if self.mode != Mode.LITE:
                start_time = time.time()
                self.response_builder.put_content_to_minio(
                    results=results, collection_name=collection_name
                )
                end_time = time.time()
                logger.info(
                    f"== MinIO upload for collection_name: {collection_name} "
                    f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
                )
            start_time = time.time()
            batch_progress_response = await self.__build_ingestion_response(
                results=results,
                failures=failures,
                filepaths=filepaths,
                state_manager=self.state_manager,
                is_final_batch=False,
                vdb_op=self.vdb_op,
            )
            end_time = time.time()
            logger.info(
                f"== Build ingestion response and adding document info for collection_name: {collection_name} "
                f"for batch {batch_number} is complete! Time taken: {end_time - start_time} seconds =="
            )
            ingestion_state = await self.state_manager.update_batch_progress(
                batch_progress_response=batch_progress_response,
            )
            await INGESTION_TASK_HANDLER.set_task_status_and_result(
                task_id=self.state_manager.get_task_id(),
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
    
    @trace_function(
        "ingestor.pipelines.nv_ingest.pipeline.perform_file_ext_based_nv_ingest_ingestion", tracer=TRACER
    )
    async def _perform_file_ext_based_nv_ingest_ingestion(
        self,
        batch_number: int,
        filtered_filepaths: list[str],
        split_options: dict[str, Any],
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

        Returns:
            - tuple[list[list[dict[str, str | dict]]], list[dict[str, Any]]] - Results and failures
        """
        if self.config.nv_ingest.pdf_extract_method is None:
            nv_ingest_ingestor = get_nv_ingest_ingestor(
                nv_ingest_client_instance=self.nv_ingest_client,
                filepaths=filtered_filepaths,
                split_options=split_options,
                vdb_op=self.vdb_op,
                config=self.config,
                enable_pdf_split_processing=self.state_manager.enable_pdf_split_processing,
                pdf_split_processing_options=self.state_manager.pdf_split_processing_options,
                prompts=self.prompts,
            )
            start_time = time.time()
            logger.info(
                f"Performing ingestion for batch {batch_number} with parameters: {split_options}"
            )
            results, failures = await self.__perform_async_nv_ingest_ingestion(
                nv_ingest_ingestor=nv_ingest_ingestor,
                state_manager=self.state_manager,
                nv_ingest_traces=True,
                trace_context=create_nv_ingest_trace_context(
                    span_namespace=f"nv_ingest.batch_{batch_number}",
                    collection_name=self.vdb_op.collection_name,
                    batch_number=batch_number,
                ),
            )
            total_ingestion_time = time.time() - start_time
            document_info = self._log_result_info(
                batch_number, results, failures, total_ingestion_time
            )
            self.vdb_op.add_document_info(
                info_type="collection",
                collection_name=self.vdb_op.collection_name,
                document_name="NA",
                info_value=document_info,
            )
            return results, failures
        else:
            pdf_filepaths, non_pdf_filepaths = await self.__split_pdf_and_non_pdf_files(
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
                    nv_ingest_client_instance=self.nv_ingest_client,
                    filepaths=pdf_filepaths,
                    split_options=split_options,
                    vdb_op=self.vdb_op,
                    config=self.config,
                    enable_pdf_split_processing=self.state_manager.enable_pdf_split_processing,
                    pdf_split_processing_options=self.state_manager.pdf_split_processing_options,
                    prompts=self.prompts,
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for PDF files for batch {batch_number} with parameters: {split_options}"
                )
                (
                    results_pdf,
                    failures_pdf,
                ) = await self.__perform_async_nv_ingest_ingestion(
                    nv_ingest_ingestor=nv_ingest_ingestor,
                    state_manager=self.state_manager,
                    nv_ingest_traces=True,
                    trace_context=create_nv_ingest_trace_context(
                        span_namespace=f"nv_ingest.batch_{batch_number}.pdf",
                        collection_name=self.vdb_op.collection_name,
                        batch_number=batch_number,
                    ),
                )
                total_ingestion_time = time.time() - start_time
                document_info = self._log_result_info(
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
                    nv_ingest_client_instance=self.nv_ingest_client,
                    filepaths=non_pdf_filepaths,
                    split_options=split_options,
                    vdb_op=self.vdb_op,
                    remove_extract_method=True,
                    config=self.config,
                    enable_pdf_split_processing=self.state_manager.enable_pdf_split_processing,
                    pdf_split_processing_options=self.state_manager.pdf_split_processing_options,
                    prompts=self.prompts,
                )
                start_time = time.time()
                logger.info(
                    f"Performing ingestion for non-PDF files for batch {batch_number} with parameters: {split_options}"
                )
                (
                    results_non_pdf,
                    failures_non_pdf,
                ) = await self.__perform_async_nv_ingest_ingestion(
                    nv_ingest_ingestor=nv_ingest_ingestor,
                    state_manager=self.state_manager,
                    nv_ingest_traces=True,
                    trace_context=create_nv_ingest_trace_context(
                        span_namespace=f"nv_ingest.batch_{batch_number}.non_pdf",
                        collection_name=self.vdb_op.collection_name,
                        batch_number=batch_number,
                    ),
                )
                total_ingestion_time = time.time() - start_time
                document_info = self._log_result_info(
                    batch_number,
                    results_non_pdf,
                    failures_non_pdf,
                    total_ingestion_time,
                    additional_summary="Non-PDF files ingestion completed",
                )
                results.extend(results_non_pdf)
                failures.extend(failures_non_pdf)

            self.vdb_op.add_document_info(
                info_type="collection",
                collection_name=self.vdb_op.collection_name,
                document_name="NA",
                info_value=document_info,
            )

            return results, failures

    @staticmethod
    @trace_function("ingestor.pipelines.nv_ingest.pipeline.perform_async_nv_ingest_ingestion", tracer=TRACER)
    async def __perform_async_nv_ingest_ingestion(
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

    @trace_function("ingestor.pipelines.nv_ingest.pipeline.log_result_info", tracer=TRACER)
    def _log_result_info(
        self,
        batch_number: int,
        results: list[list[dict[str, str | dict]]],
        failures: list[dict[str, Any]],
        total_ingestion_time: float,
        additional_summary: str = "",
    ) -> dict[str, Any]:
        """Log the results info with document type counts.

        Returns:
            dict[str, Any]: Document info with metrics
        """
        doc_type_counts, total_documents, total_elements, raw_text_elements_size = (
            self.response_builder.get_document_type_counts(results)
        )

        document_info = {
            "doc_type_counts": doc_type_counts,
            "total_elements": total_elements,
            "raw_text_elements_size": raw_text_elements_size,
            "number_of_files": total_documents,
            **derive_boolean_flags(doc_type_counts),
            "last_indexed": get_current_timestamp(),
            "ingestion_status": "Success"
            if not failures
            else ("Partial" if len(results) > 0 else "Failed"),
            "last_ingestion_error": (
                str(failures[0][1]) if failures and len(failures[0]) > 1 else ""
            ),
        }

        summary_parts = []
        for doc_type in doc_type_counts.keys():
            count = doc_type_counts.get(doc_type, 0)
            if count > 0:
                summary_parts.append(f"{doc_type}:{count}")
        if raw_text_elements_size > 0:
            summary_parts.append(
                f"Raw text elements size: {raw_text_elements_size} bytes"
            )

        summary = (
            f"Successfully processed {total_documents} document(s) with {total_elements} element(s) • "
            + " • ".join(summary_parts)
        )
        if failures:
            summary += f", {len(failures)} files failed ingestion"

        if additional_summary:
            summary += f" • {additional_summary}"

        logger.info(
            f"== Batch {batch_number} Ingestion completed in {total_ingestion_time:.2f} seconds • Summary: {summary} =="
        )
        return document_info
    
    @trace_function("ingestor.pipelines.nv_ingest.pipeline.split_pdf_and_non_pdf_files", tracer=TRACER)
    async def __split_pdf_and_non_pdf_files(
        self, filepaths: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split PDF and non-PDF files from the list of filepaths"""
        pdf_filepaths = []
        non_pdf_filepaths = []
        for filepath in filepaths:
            if os.path.splitext(filepath)[1].lower() == ".pdf":
                pdf_filepaths.append(filepath)
            else:
                non_pdf_filepaths.append(filepath)
        return pdf_filepaths, non_pdf_filepaths

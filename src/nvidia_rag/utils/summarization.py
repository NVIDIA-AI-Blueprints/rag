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
Document summarization utilities with parallel processing and Redis coordination.

This module provides document summarization functionality with:
1. generate_document_summaries: Main entry point for parallel summarization of multiple documents.
2. get_summarization_semaphore: Get or create event-loop-aware semaphore for local concurrency.
3. acquire_global_summary_slot: Acquire a slot in the global summary queue via Redis.
4. release_global_summary_slot: Release a slot in the global summary queue via Redis.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nvidia_rag.utils.common import ConfigProxy
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.minio_operator import get_unique_thumbnail_id
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER

logger = logging.getLogger(__name__)
CONFIG = ConfigProxy()

# Module-level semaphore storage (per event loop)
_event_loop_semaphores = {}

# Redis key for global rate limiting
REDIS_GLOBAL_SUMMARY_KEY = "summary:global:active_count"


def get_summarization_semaphore() -> asyncio.Semaphore:
    """Get or create local semaphore for current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            "No running event loop - cannot create summarization semaphore"
        ) from e

    loop_id = id(loop)

    if loop_id not in _event_loop_semaphores:
        # High capacity local semaphore (real limiting happens in Redis)
        _event_loop_semaphores[loop_id] = asyncio.Semaphore(1000)
        logger.info(f"Initialized summary semaphore (event loop {loop_id})")

    return _event_loop_semaphores[loop_id]


async def acquire_global_summary_slot() -> bool:
    """
    Acquire a slot in the global summary queue via Redis.

    Returns:
        bool: True if slot acquired, False if should retry
    """
    if not SUMMARY_STATUS_HANDLER.is_available():
        return True

    try:
        max_global = CONFIG.summarizer.max_parallelization
        redis_client = SUMMARY_STATUS_HANDLER._redis_client

        # Atomic increment
        current_count = await asyncio.to_thread(
            redis_client.incr, REDIS_GLOBAL_SUMMARY_KEY
        )

        if current_count <= max_global:
            logger.debug(f"Acquired global slot {current_count}/{max_global}")
            return True
        else:
            # Over limit - decrement and return False
            await asyncio.to_thread(redis_client.decr, REDIS_GLOBAL_SUMMARY_KEY)
            logger.debug(f"Global limit reached ({max_global}), waiting...")
            return False

    except Exception as e:
        logger.warning(f"Redis error in global rate limiting, proceeding: {e}")
        return True


async def release_global_summary_slot() -> None:
    """Release a slot in the global summary queue via Redis."""
    if not SUMMARY_STATUS_HANDLER.is_available():
        return

    try:
        redis_client = SUMMARY_STATUS_HANDLER._redis_client
        await asyncio.to_thread(redis_client.decr, REDIS_GLOBAL_SUMMARY_KEY)
    except Exception as e:
        logger.warning(f"Redis error releasing global slot: {e}")


async def generate_document_summaries(
    results: list[list[dict[str, str | dict]]],
    collection_name: str,
) -> dict[str, Any]:
    """
    Generate summaries for multiple documents in parallel with global rate limiting.

    Args:
        results: NV-Ingest extraction results (nested list structure)
        collection_name: Collection name for status tracking

    Returns:
        dict: Statistics with total_files, successful, failed, duration_seconds, files
    """
    start_time = time.time()

    logger.info(f"Starting summary generation for collection: {collection_name}")

    if not SUMMARY_STATUS_HANDLER.is_available():
        logger.warning("Redis unavailable - summary status tracking disabled")

    semaphore = get_summarization_semaphore()

    # Extract unique files from NV-Ingest results
    file_results = []
    for result_list in results:
        if not result_list:
            continue

        first_element = result_list[0]
        source_id = (
            first_element.get("metadata", {})
            .get("source_metadata", {})
            .get("source_id", "")
        )
        file_name = os.path.basename(source_id) if source_id else "unknown"

        if file_name and file_name != "unknown":
            file_results.append(
                {
                    "result_element": first_element,
                    "file_name": file_name,
                }
            )

    total_files = len(file_results)
    logger.info(f"Found {total_files} files to summarize")

    if total_files == 0:
        logger.warning("No files to summarize")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "duration_seconds": time.time() - start_time,
            "files": {},
        }

    # Create async tasks for all files (parallel processing)
    tasks = [
        _process_single_file_summary(
            file_data=file_data,
            collection_name=collection_name,
            results=results,
            semaphore=semaphore,
        )
        for file_data in file_results
    ]

    # Wait for all summaries to complete
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect statistics
    stats = {
        "total_files": total_files,
        "successful": 0,
        "failed": 0,
        "duration_seconds": time.time() - start_time,
        "files": {},
    }

    for result in completed_results:
        if isinstance(result, Exception):
            stats["failed"] += 1
            logger.error(f"Unexpected exception in summary task: {result}")
        elif isinstance(result, dict):
            file_name = result.get("file_name", "unknown")
            stats["files"][file_name] = result

            if result.get("status") == "SUCCESS":
                stats["successful"] += 1
            else:
                stats["failed"] += 1

    logger.info(
        f"Summary completed: {stats['successful']}/{stats['total_files']} successful "
        f"in {stats['duration_seconds']:.2f}s"
    )

    return stats


async def _process_single_file_summary(
    file_data: dict[str, Any],
    collection_name: str,
    results: list[list[dict[str, str | dict]]],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Process summary for a single file with global rate limiting.

    Args:
        file_data: Dict with "file_name" and "result_element"
        collection_name: Collection name for status tracking
        results: Full results list for document preparation
        semaphore: Semaphore for concurrency control

    Returns:
        dict: Result with status, duration, and optional error
    """
    file_name = file_data["file_name"]
    result_element = file_data["result_element"]

    file_start_time = time.time()

    # Set IN_PROGRESS status before acquiring global slot
    SUMMARY_STATUS_HANDLER.update_progress(
        collection_name=collection_name,
        file_name=file_name,
        status="IN_PROGRESS",
        progress={"current": 0, "total": 0, "message": "Queued..."},
    )

    slot_acquired = False
    try:
        async with semaphore:
            # Wait for global slot availability (coordinated via Redis)
            while not await acquire_global_summary_slot():
                await asyncio.sleep(0.5)

            slot_acquired = True

            logger.info(f"Starting summary: {file_name}")

            # Lazily load document content
            document = await _prepare_single_document(
                result_element=result_element,
                results=results,
                collection_name=collection_name,
            )

            progress_callback = partial(
                _update_file_progress,
                collection_name=collection_name,
                file_name=file_name,
            )

            # Generate summary
            summary_doc = await _generate_single_document_summary(
                document=document,
                progress_callback=progress_callback,
            )

            # Store in MinIO
            await _store_summary_in_minio(summary_doc)

            # Update Redis status
            SUMMARY_STATUS_HANDLER.update_progress(
                collection_name=collection_name,
                file_name=file_name,
                status="SUCCESS",
            )

            duration = time.time() - file_start_time
            logger.info(f"Summary completed: {file_name} ({duration:.2f}s)")

            return {
                "file_name": file_name,
                "status": "SUCCESS",
                "duration": duration,
            }

    except Exception as e:
        error_msg = str(e)
        SUMMARY_STATUS_HANDLER.update_progress(
            collection_name=collection_name,
            file_name=file_name,
            status="FAILED",
            error=error_msg,
        )

        duration = time.time() - file_start_time
        logger.error(f"Summary failed: {file_name} - {error_msg}")

        return {
            "file_name": file_name,
            "status": "FAILED",
            "duration": duration,
            "error": error_msg,
        }
    finally:
        # Release global slot only if it was acquired
        if slot_acquired:
            await release_global_summary_slot()


async def _prepare_single_document(
    result_element: dict[str, str | dict],
    results: list[list[dict[str, str | dict]]],
    collection_name: str,
) -> Document:
    """
    Prepare a single document for summarization by lazily loading content.

    Args:
        result_element: Single result element with file metadata
        results: Full results list to search for all chunks of this file
        collection_name: Collection name for metadata

    Returns:
        Document: LangChain document with full content and metadata
    """
    source_id = (
        result_element.get("metadata", {})
        .get("source_metadata", {})
        .get("source_id", "")
    )
    file_name = os.path.basename(source_id)

    # Find all content chunks for this file
    content_parts = []
    for result_list in results:
        for elem in result_list:
            elem_source = (
                elem.get("metadata", {}).get("source_metadata", {}).get("source_id", "")
            )

            if os.path.basename(elem_source) == file_name:
                content = _extract_content_from_element(elem)
                if content:
                    content_parts.append(content)

    # Concatenate all content
    full_content = " ".join(content_parts)

    logger.debug(
        f"Prepared document {file_name}: {len(full_content)} chars "
        f"from {len(content_parts)} chunks"
    )

    return Document(
        page_content=full_content,
        metadata={
            "filename": file_name,
            "collection_name": collection_name,
        },
    )


def _extract_content_from_element(elem: dict[str, Any]) -> str | None:
    """
    Extract text content from element based on type and config settings.

    Args:
        elem: Result element with document_type and metadata

    Returns:
        str | None: Extracted text content or None
    """
    doc_type = elem.get("document_type")
    metadata = elem.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")

    elif doc_type == "structured":
        # Tables/charts - respect config flags
        structured_content = metadata.get("table_metadata", {}).get("table_content")
        subtype = metadata.get("content_metadata", {}).get("subtype")

        if subtype == "table" and CONFIG.nv_ingest.extract_tables:
            return structured_content
        elif subtype == "chart" and CONFIG.nv_ingest.extract_charts:
            return structured_content

    elif doc_type == "image" and CONFIG.nv_ingest.extract_images:
        # Image captions - respect config flag
        return metadata.get("image_metadata", {}).get("caption")

    elif doc_type == "audio":
        # Audio transcripts - always included
        return metadata.get("audio_metadata", {}).get("audio_transcript")

    return None


async def _generate_single_document_summary(
    document: Document,
    progress_callback: Callable | None = None,
) -> Document:
    """
    Generate summary for a single document using single-pass or iterative chunking.

    Args:
        document: LangChain document with page_content and metadata
        progress_callback: Optional callback for progress updates

    Returns:
        Document: Same document with summary added to metadata
    """
    file_name = document.metadata.get("filename", "unknown")
    document_text = document.page_content
    total_chars = len(document_text)

    max_chunk_chars = CONFIG.summarizer.max_chunk_length
    chunk_overlap = CONFIG.summarizer.chunk_overlap

    logger.info(
        f"Summarizing {file_name}: {total_chars} chars (threshold: {max_chunk_chars})"
    )

    llm = _get_summary_llm()
    prompts = get_prompts()
    initial_chain, iterative_chain = _create_llm_chains(llm, prompts)

    if total_chars <= max_chunk_chars:
        logger.info(f"Using single-pass for {file_name}")

        if progress_callback:
            await progress_callback(current=0, total=1)

        summary = await initial_chain.ainvoke(
            {"document_text": document_text},
            config={"run_name": f"summary-{file_name}"},
        )

        if progress_callback:
            await progress_callback(current=1, total=1)

    else:
        logger.info(f"Using iterative chunking for {file_name}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        text_chunks = text_splitter.split_text(document_text)
        total_chunks = len(text_chunks)

        logger.info(f"Split {file_name} into {total_chunks} chunks")

        if progress_callback:
            await progress_callback(current=0, total=total_chunks)

        summary = await initial_chain.ainvoke(
            {"document_text": text_chunks[0]},
            config={"run_name": f"summary-{file_name}-chunk-1"},
        )

        if progress_callback:
            await progress_callback(current=1, total=total_chunks)

        for i, chunk in enumerate(text_chunks[1:], start=1):
            logger.debug(f"Processing chunk {i + 1}/{total_chunks} for {file_name}")

            summary = await iterative_chain.ainvoke(
                {"previous_summary": summary, "new_chunk": chunk},
                config={"run_name": f"summary-{file_name}-chunk-{i + 1}"},
            )

            if progress_callback:
                await progress_callback(current=i + 1, total=total_chunks)

    document.metadata["summary"] = summary
    logger.debug(f"Summary generated for {file_name}: {summary[:100]}...")

    return document


def _get_summary_llm():
    """Get configured LLM for summarization."""
    llm_params = {
        "model": CONFIG.summarizer.model_name,
        "temperature": CONFIG.summarizer.temperature,
        "top_p": CONFIG.summarizer.top_p,
    }

    if CONFIG.summarizer.server_url:
        llm_params["llm_endpoint"] = CONFIG.summarizer.server_url

    return get_llm(**llm_params)


def _create_llm_chains(llm, prompts):
    """Create LangChain chains for initial and iterative summarization."""
    document_summary_prompt = prompts.get("document_summary_prompt")
    iterative_summary_prompt_config = prompts.get("iterative_summary_prompt")

    initial_summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", document_summary_prompt["system"]),
            ("human", document_summary_prompt["human"]),
        ]
    )

    iterative_summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", iterative_summary_prompt_config["system"]),
            ("human", iterative_summary_prompt_config["human"]),
        ]
    )

    initial_chain = initial_summary_prompt | llm | StrOutputParser()
    iterative_chain = iterative_summary_prompt | llm | StrOutputParser()

    return initial_chain, iterative_chain


async def _update_file_progress(
    collection_name: str,
    file_name: str,
    current: int,
    total: int,
):
    """Update chunk-level progress for a file in Redis."""
    SUMMARY_STATUS_HANDLER.update_progress(
        collection_name=collection_name,
        file_name=file_name,
        status="IN_PROGRESS",
        progress={
            "current": current,
            "total": total,
            "message": f"Processing chunk {current}/{total}",
        },
    )


async def _store_summary_in_minio(document: Document):
    """Store document summary in MinIO."""
    # Import here to avoid circular dependency
    from nvidia_rag.ingestor_server.main import get_minio_operator_instance

    summary = document.metadata["summary"]
    file_name = document.metadata["filename"]
    collection_name = document.metadata["collection_name"]

    unique_thumbnail_id = get_unique_thumbnail_id(
        collection_name=f"summary_{collection_name}",
        file_name=file_name,
        page_number=0,
        location=[],
    )

    get_minio_operator_instance().put_payload(
        payload={
            "summary": summary,
            "file_name": file_name,
            "collection_name": collection_name,
        },
        object_name=unique_thumbnail_id,
    )

    logger.debug(f"Stored summary for {file_name} in MinIO")

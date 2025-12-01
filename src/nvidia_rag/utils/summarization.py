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
from transformers import AutoTokenizer

from nvidia_rag.rag_server.response_generator import get_minio_operator_instance
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.minio_operator import get_unique_thumbnail_id
from nvidia_rag.utils.summary_status_handler import SUMMARY_STATUS_HANDLER

logger = logging.getLogger(__name__)

# Module-level semaphore storage (per event loop)
_event_loop_semaphores = {}

# Redis key for global rate limiting
REDIS_GLOBAL_SUMMARY_KEY = "summary:global:active_count"

# Cache for tokenizer to avoid reloading
_tokenizer_cache = None


def _reset_global_summary_counter() -> None:
    """
    Reset the global summary counter in Redis to 0.

    This should be called on server startup to clear any stale counter values
    from crashed or restarted processes. Runs synchronously on module import.
    """
    if not SUMMARY_STATUS_HANDLER.is_available():
        logger.debug("Redis not available, skipping summary counter reset")
        return

    try:
        redis_client = SUMMARY_STATUS_HANDLER._redis_client
        redis_client.delete(REDIS_GLOBAL_SUMMARY_KEY)
        logger.debug(
            f"✅ Reset Redis summary counter '{REDIS_GLOBAL_SUMMARY_KEY}' "
            f"at {SUMMARY_STATUS_HANDLER._redis_host}:{SUMMARY_STATUS_HANDLER._redis_port}"
        )
    except Exception as e:
        logger.warning(
            f"⚠️  Could not reset Redis summary counter: {e}. "
            "This is OK if Redis is not configured."
        )


# Reset counter on module import (runs once when server starts)
_reset_global_summary_counter()


def _get_tokenizer(config: NvidiaRAGConfig):
    """Get or create cached tokenizer instance.

    Uses the same tokenizer as configured in nv-ingest for consistency.
    The tokenizer is cached to avoid reloading on subsequent calls.

    Args:
        config: NvidiaRAGConfig instance

    Returns:
        AutoTokenizer: The cached tokenizer instance

    Raises:
        Exception: If tokenizer fails to load
    """
    global _tokenizer_cache
    if _tokenizer_cache is None:
        tokenizer_name = config.nv_ingest.tokenizer
        _tokenizer_cache = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Loaded tokenizer for summarization: {tokenizer_name}")
    return _tokenizer_cache


def _token_length(text: str, config: NvidiaRAGConfig) -> int:
    """Calculate text length in tokens.

    Uses the same tokenizer as nv-ingest for consistent token counting.

    Args:
        text: Input text to measure
        config: NvidiaRAGConfig instance

    Returns:
        int: Number of tokens in the text

    Raises:
        Exception: If tokenizer fails to load or encode
    """
    tokenizer = _get_tokenizer(config)
    return len(tokenizer.encode(text, add_special_tokens=False))


def matches_page_filter(
    page_num: int,
    page_filter: list[list[int]] | str | None,
    total_pages: int | None = None,
) -> bool:
    """Check if page number matches filter criteria.

    Args:
        page_num: Page number to check (1-indexed)
        page_filter: Filter specification - either list of ranges [[start,end],...] or string ('even'/'odd')
        total_pages: Total pages in document (required for negative index resolution)

    Returns:
        True if page matches filter, False otherwise
    """
    if not page_filter:
        return True

    # Handle string filters: "even" or "odd"
    if isinstance(page_filter, str):
        page_filter_lower = page_filter.lower()
        if page_filter_lower == "even":
            return page_num % 2 == 0
        elif page_filter_lower == "odd":
            return page_num % 2 != 0
        else:
            logger.error(
                f"Invalid page filter string: '{page_filter}'. "
                f"Allowed values: 'even', 'odd'. "
                f"Please check your page_filter configuration."
            )
            return False

    # Handle ranges: [[1, 10], [20, 30]] or with negative indices [[-10, -1]]
    if isinstance(page_filter, list):
        try:
            for start, end in page_filter:
                # Resolve negative indices if total_pages provided
                if total_pages is not None:
                    resolved_start = start if start > 0 else total_pages + start + 1
                    resolved_end = end if end > 0 else total_pages + end + 1
                    # Clamp to valid range
                    resolved_start = max(1, min(resolved_start, total_pages))
                    resolved_end = max(1, min(resolved_end, total_pages))
                else:
                    resolved_start = start
                    resolved_end = end

                if resolved_start <= page_num <= resolved_end:
                    return True
            return False
        except (ValueError, TypeError) as e:
            logger.error(
                f"Error processing page filter ranges: {e}. "
                f"Expected format: [[start, end], ...]. "
                f"Got: {page_filter}"
            )
            return False

    # Invalid format
    logger.error(
        f"Invalid page filter format: {type(page_filter).__name__}. "
        f"Allowed formats: list of ranges [[1,10],[20,30]] or string 'even'/'odd'. "
        f"Please check your page_filter configuration."
    )
    return False


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


async def acquire_global_summary_slot(config: NvidiaRAGConfig) -> bool:
    """
    Acquire a slot in the global summary queue via Redis.

    Args:
        config: NvidiaRAGConfig instance

    Returns:
        bool: True if slot acquired, False if should retry
    """
    if not SUMMARY_STATUS_HANDLER.is_available():
        return True

    try:
        max_global = config.summarizer.max_parallelization
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
    page_filter: list[list[int]] | str | None = None,
    summarization_strategy: str | None = None,
    config: NvidiaRAGConfig | None = None,
    is_shallow: bool = False,
) -> dict[str, Any]:
    """
    Generate summaries for multiple documents in parallel with global rate limiting.

    Args:
        results: NV-Ingest extraction results (nested list structure)
        collection_name: Collection name for status tracking
        page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')
        summarization_strategy: Strategy for summarization ('single', 'hierarchical') or None for default iterative
        config: NvidiaRAGConfig instance. If None, creates a new one from environment.
        is_shallow: Whether this is shallow extraction (text-only, uses simplified prompt)

    Returns:
        dict: Statistics with total_files, successful, failed, duration_seconds, files
    """
    if config is None:
        config = NvidiaRAGConfig()

    start_time = time.time()

    logger.info(f"Starting summary generation for collection: {collection_name}")

    if not SUMMARY_STATUS_HANDLER.is_available():
        logger.warning("Redis unavailable - summary status tracking disabled")

    semaphore = get_summarization_semaphore()

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

    if page_filter:
        logger.info(f"Global page filter: {page_filter}")

    if total_files == 0:
        logger.warning("No files to summarize")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "duration_seconds": time.time() - start_time,
            "files": {},
        }

    tasks = [
        _process_single_file_summary(
            file_data=file_data,
            collection_name=collection_name,
            results=results,
            semaphore=semaphore,
            config=config,
            page_filter=page_filter,
            summarization_strategy=summarization_strategy,
            is_shallow=is_shallow,
        )
        for file_data in file_results
    ]

    completed_results = await asyncio.gather(*tasks, return_exceptions=True)

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
    config: NvidiaRAGConfig,
    page_filter: list[list[int]] | str | None = None,
    summarization_strategy: str | None = None,
    is_shallow: bool = False,
) -> dict[str, Any]:
    """
    Process summary for a single file with global rate limiting.

    Args:
        file_data: Dict with "file_name" and "result_element"
        collection_name: Collection name for status tracking
        results: Full results list for document preparation
        semaphore: Semaphore for concurrency control
        config: NvidiaRAGConfig instance
        page_filter: Global page filter for all files
        summarization_strategy: Strategy for summarization ('single', 'hierarchical') or None for default iterative
        is_shallow: Whether this is shallow extraction (text-only, uses simplified prompt)

    Returns:
        dict: Result with status, duration, and optional error
    """
    file_name = file_data["file_name"]
    result_element = file_data["result_element"]

    effective_filter = page_filter

    file_start_time = time.time()

    SUMMARY_STATUS_HANDLER.update_progress(
        collection_name=collection_name,
        file_name=file_name,
        status="IN_PROGRESS",
        progress={"current": 0, "total": 0, "message": "Queued..."},
    )

    slot_acquired = False
    try:
        async with semaphore:
            while not await acquire_global_summary_slot(config):
                await asyncio.sleep(0.5)

            slot_acquired = True

            document = await _prepare_single_document(
                result_element=result_element,
                results=results,
                collection_name=collection_name,
                page_filter=effective_filter,
                config=config,
            )

            progress_callback = partial(
                _update_file_progress,
                collection_name=collection_name,
                file_name=file_name,
            )

            summary_doc = await _generate_single_document_summary(
                document=document,
                progress_callback=progress_callback,
                summarization_strategy=summarization_strategy,
                config=config,
                is_shallow=is_shallow,
            )

            await _store_summary_in_minio(summary_doc)

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
        if slot_acquired:
            await release_global_summary_slot()


async def _prepare_single_document(
    result_element: dict[str, str | dict],
    results: list[list[dict[str, str | dict]]],
    collection_name: str,
    config: NvidiaRAGConfig,
    page_filter: list[list[int]] | str | None = None,
) -> Document:
    """Prepare document for summarization by loading content with optional page filtering.

    Args:
        result_element: Single result element with file metadata
        results: Full results list to search for all chunks of this file
        collection_name: Collection name for metadata
        config: NvidiaRAGConfig instance
        page_filter: Optional page filter - either list of ranges [[start,end],...] or string ('even'/'odd')

    Returns:
        LangChain document with full content and metadata
    """
    source_id = (
        result_element.get("metadata", {})
        .get("source_metadata", {})
        .get("source_id", "")
    )
    file_name = os.path.basename(source_id)

    # Collect pages with their content - nv-ingest provides sorted pages
    pages_data = []  # List of (page_num, content) tuples in order
    seen_pages = set()

    for result_list in results:
        for elem in result_list:
            elem_source = (
                elem.get("metadata", {}).get("source_metadata", {}).get("source_id", "")
            )

            if os.path.basename(elem_source) == file_name:
                page_num = (
                    elem.get("metadata", {})
                    .get("content_metadata", {})
                    .get("page_number", 1)
                )

                content = _extract_content_from_element(elem, config)
                if content:
                    pages_data.append((page_num, content))
                    seen_pages.add(page_num)

    if not pages_data:
        raise ValueError(f"No content found in document '{file_name}'")

    # Apply page filter if specified
    if page_filter:
        total_pages = max(seen_pages)

        # Filter pages with negative index resolution
        pages_data = [
            (page_num, content)
            for page_num, content in pages_data
            if matches_page_filter(page_num, page_filter, total_pages)
        ]

        if not pages_data:
            raise ValueError(
                f"No content found for file '{file_name}' with page filter: {page_filter}"
            )

    # Concatenate content - already in correct order from nv-ingest
    content_parts = [content for _, content in pages_data]
    full_content = " ".join(content_parts)

    return Document(
        page_content=full_content,
        metadata={
            "filename": file_name,
            "collection_name": collection_name,
        },
    )


def _extract_content_from_element(
    elem: dict[str, Any], config: NvidiaRAGConfig
) -> str | None:
    """Extract text content from element based on type and config settings.

    Args:
        elem: Result element with document_type and metadata
        config: NvidiaRAGConfig instance

    Returns:
        Extracted text content or None
    """
    doc_type = elem.get("document_type")
    metadata = elem.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")

    elif doc_type == "structured":
        # Tables/charts - respect config flags
        structured_content = metadata.get("table_metadata", {}).get("table_content")
        subtype = metadata.get("content_metadata", {}).get("subtype")

        if subtype == "table" and config.nv_ingest.extract_tables:
            return structured_content
        elif subtype == "chart" and config.nv_ingest.extract_charts:
            return structured_content

    elif doc_type == "image" and config.nv_ingest.extract_images:
        # Image captions - respect config flag
        return metadata.get("image_metadata", {}).get("caption")

    elif doc_type == "audio":
        # Audio transcripts - always included
        return metadata.get("audio_metadata", {}).get("audio_transcript")

    return None


async def _generate_single_document_summary(
    document: Document,
    config: NvidiaRAGConfig,
    progress_callback: Callable | None = None,
    summarization_strategy: str | None = None,
    is_shallow: bool = False,
) -> Document:
    """Generate summary for a single document using configured strategy."""
    file_name = document.metadata.get("filename", "unknown")

    if summarization_strategy is None:
        summarization_strategy = "iterative"

    logger.info(f"Summarizing {file_name} using strategy: {summarization_strategy}")

    if summarization_strategy == "single":
        return await _summarize_single_pass(
            document, config, progress_callback, is_shallow
        )
    elif summarization_strategy == "iterative":
        return await _summarize_iterative(
            document, config, progress_callback, is_shallow
        )
    elif summarization_strategy == "hierarchical":
        return await _summarize_hierarchical(
            document, config, progress_callback, is_shallow
        )
    else:
        raise ValueError(
            f"Unknown summarization_strategy: {summarization_strategy}. "
            f"Supported: 'single', 'hierarchical', or None for default 'iterative'"
        )


async def _summarize_single_pass(
    document: Document,
    config: NvidiaRAGConfig,
    progress_callback: Callable | None = None,
    is_shallow: bool = False,
) -> Document:
    """Summarize entire document in one pass, truncating if needed."""
    file_name = document.metadata.get("filename", "unknown")
    document_text = document.page_content
    total_tokens = _token_length(document_text, config)
    max_chunk_tokens = config.summarizer.max_chunk_length

    if total_tokens > max_chunk_tokens:
        logger.warning(
            f"Document {file_name} has {total_tokens} tokens (max: {max_chunk_tokens}) - truncating for single-pass"
        )
        # Rough character-based truncation (tokens are roughly 3-4 chars)
        approx_chars = max_chunk_tokens * 4
        document_text = document_text[:approx_chars]

    logger.info(f"Single-pass summarization for {file_name}: {total_tokens} tokens")

    llm = _get_summary_llm(config)
    prompts = get_prompts()
    initial_chain, _ = _create_llm_chains(llm, prompts, is_shallow)

    if progress_callback:
        await progress_callback(current=0, total=1)

    summary = await initial_chain.ainvoke(
        {"document_text": document_text},
        config={"run_name": f"summary-{file_name}"},
    )

    if progress_callback:
        await progress_callback(current=1, total=1)

    document.metadata["summary"] = summary
    logger.debug(f"Summary generated for {file_name}: {summary[:100]}...")

    return document


async def _summarize_iterative(
    document: Document,
    config: NvidiaRAGConfig,
    progress_callback: Callable | None = None,
    is_shallow: bool = False,
) -> Document:
    """Iterative sequential summarization - processes chunks one by one."""
    file_name = document.metadata.get("filename", "unknown")
    document_text = document.page_content
    total_tokens = _token_length(document_text, config)

    max_chunk_tokens = config.summarizer.max_chunk_length
    chunk_overlap = config.summarizer.chunk_overlap

    logger.info(
        f"Iterative summarization for {file_name}: {total_tokens} tokens (threshold: {max_chunk_tokens})"
    )

    llm = _get_summary_llm(config)
    prompts = get_prompts()
    initial_chain, iterative_chain = _create_llm_chains(llm, prompts, is_shallow)

    if total_tokens <= max_chunk_tokens:
        logger.info(f"Using single-pass for {file_name} (fits in one chunk)")

        if progress_callback:
            await progress_callback(current=0, total=1)

        summary = await initial_chain.ainvoke(
            {"document_text": document_text},
            config={"run_name": f"summary-{file_name}"},
        )

        if progress_callback:
            await progress_callback(current=1, total=1)

    else:
        # Token-based splitting with recursive character splitting at semantic boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens,
            chunk_overlap=chunk_overlap,
            length_function=partial(_token_length, config=config),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        text_chunks = text_splitter.split_text(document_text)
        total_chunks = len(text_chunks)

        logger.info(
            f"Split {file_name} into {total_chunks} chunks for sequential processing"
        )

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


async def _summarize_hierarchical(
    document: Document,
    config: NvidiaRAGConfig,
    progress_callback: Callable | None = None,
    is_shallow: bool = False,
) -> Document:
    """Hierarchical parallel summarization with token-based chunking."""
    file_name = document.metadata.get("filename", "unknown")
    document_text = document.page_content
    total_tokens = _token_length(document_text, config)
    max_chunk_tokens = config.summarizer.max_chunk_length

    logger.info(
        f"Hierarchical summarization for {file_name}: {total_tokens} tokens (threshold: {max_chunk_tokens})"
    )

    if total_tokens <= max_chunk_tokens:
        logger.info(f"Document fits in one chunk, using single-pass for {file_name}")
        return await _summarize_single_pass(
            document, config, progress_callback, is_shallow
        )

    # Token-based splitting with recursive character splitting at semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_tokens,
        chunk_overlap=config.summarizer.chunk_overlap,
        length_function=partial(_token_length, config=config),
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    text_chunks = text_splitter.split_text(document_text)
    total_chunks = len(text_chunks)

    logger.info(f"Split {file_name} into {total_chunks} chunks for parallel processing")

    llm = _get_summary_llm(config)
    prompts = get_prompts()
    initial_chain, iterative_chain = _create_llm_chains(llm, prompts, is_shallow)

    chunk_summaries = await asyncio.gather(
        *[
            initial_chain.ainvoke(
                {"document_text": chunk},
                config={"run_name": f"summary-{file_name}-chunk-{i}"},
            )
            for i, chunk in enumerate(text_chunks, 1)
        ]
    )

    if progress_callback:
        await progress_callback(current=total_chunks, total=total_chunks)

    current_summaries = chunk_summaries
    level = 2

    while len(current_summaries) > 1:
        batched_summaries = _batch_summaries_by_length(
            current_summaries, max_chunk_tokens
        )

        next_level_summaries = await asyncio.gather(
            *[
                _combine_summaries_batch(batch, iterative_chain, file_name, level, i)
                for i, batch in enumerate(batched_summaries)
            ]
        )

        current_summaries = next_level_summaries
        level += 1

    final_summary = current_summaries[0]
    document.metadata["summary"] = final_summary
    logger.debug(f"Summary generated for {file_name}: {final_summary[:100]}...")

    return document


def _batch_summaries_by_length(
    summaries: list[str],
    max_chunk_chars: int,
) -> list[list[str]]:
    """Group summaries into batches respecting max character length."""
    batches = []
    current_batch = []
    current_length = 0

    for summary in summaries:
        summary_length = len(summary)

        if current_batch and (current_length + summary_length) > max_chunk_chars:
            batches.append(current_batch)
            current_batch = [summary]
            current_length = summary_length
        else:
            current_batch.append(summary)
            current_length += summary_length

    if current_batch:
        batches.append(current_batch)

    return batches


async def _combine_summaries_batch(
    summaries: list[str],
    iterative_chain,
    file_name: str,
    level: int,
    batch_idx: int,
) -> str:
    """Combine a batch of summaries using iterative aggregation."""
    if len(summaries) == 1:
        return summaries[0]

    combined = summaries[0]
    for i, summary in enumerate(summaries[1:], 1):
        combined = await iterative_chain.ainvoke(
            {"previous_summary": combined, "new_chunk": summary},
            config={"run_name": f"summary-{file_name}-L{level}-B{batch_idx}-{i}"},
        )

    return combined


def _get_summary_llm(config: NvidiaRAGConfig):
    """Get configured LLM for summarization."""
    llm_params = {
        "model": config.summarizer.model_name,
        "temperature": config.summarizer.temperature,
        "top_p": config.summarizer.top_p,
    }

    if config.summarizer.server_url:
        llm_params["llm_endpoint"] = config.summarizer.server_url

    return get_llm(**llm_params)


def _create_llm_chains(llm, prompts, is_shallow: bool = False):
    """Create LangChain chains for initial and iterative summarization."""
    # Use shallow prompt for text-only extraction, otherwise use full document summary prompt
    if is_shallow:
        initial_prompt_config = prompts.get("shallow_summary_prompt")
    else:
        initial_prompt_config = prompts.get("document_summary_prompt")

    iterative_summary_prompt_config = prompts.get("iterative_summary_prompt")

    initial_summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", initial_prompt_config["system"]),
            ("human", initial_prompt_config["human"]),
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

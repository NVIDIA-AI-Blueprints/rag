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
Batch processing utilities for the ingestor server.

This module provides utilities for calculating optimal batch parameters
based on file characteristics and system resources.
"""

import os
import logging
from pathlib import Path

from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)


# Text-like file extensions that process quickly
# These are mapped to TXT or HTML in EXTENSION_TO_DOCUMENT_TYPE
TEXT_LIKE_EXTENSIONS = frozenset({
    "txt",
    "md",
    "json",
    "sh",
    "html",
})

# Optimal batch parameters for text-like files
# Text files process quickly, so we use larger batches with sequential processing
TEXT_FILE_CONCURRENT_BATCHES = 4 # Fixed number of concurrent batches for text-like files
TEXT_FILE_MEMORY_OVERHEAD_FACTOR = 2.0 # Factor to account for memory overhead of other file types
TEXT_ALLOWED_BATCH_SIZES = [16, 32, 64, 128, 250] # Allowed batch sizes for text-like files

# Threshold percentage to determine if workload is text-heavy
TEXT_FILE_PERCENTAGE_THRESHOLD = 50.0 # Threshold percentage to determine if workload is text file heavy

# Size-aware batching for non-text-like (multimodal) workloads.
#
# When the workload is dominated by complex files (pptx, pdf, images, media)
# the prior fallback returned the unmodified configured defaults
# (files_per_batch / concurrent_batches). With heterogeneous datasets containing
# one or more very large files (e.g. a 100+MB multimodal PDF whose summary or
# multimodal extraction can take 10+ minutes) a single slow file pins every
# other file in its batch behind it, and because per-batch VDB writes are
# serialized the slowdown blocks sibling batches as well. The result observed
# in NVBug 6191293 was a 53-file bulk ingest staying "pending" past 30 minutes.
#
# To address this without changing the public API or batching architecture, we
# extend the non-text-like fallback to consider file sizes (max + average) and
# automatically reduce files_per_batch when very large files are present. This
# matches the "File sizes and complexity" future-enhancement note already in
# the docstring of calculate_dynamic_batch_parameters().
MULTIMODAL_VERY_LARGE_FILE_MB = 100.0  # Any single file above this MB triggers aggressive batch reduction
MULTIMODAL_LARGE_FILE_MB = 50.0        # Any single file above this MB triggers moderate batch reduction
MULTIMODAL_AVG_LARGE_FILE_MB = 25.0    # Average size above this MB triggers moderate batch reduction
MULTIMODAL_MIN_FILES_PER_BATCH = 2     # Floor when shrinking batches automatically


def calculate_dynamic_batch_parameters(
    filepaths: list[str],
    config: NvidiaRAGConfig,
) -> tuple[int, int]:
    """
    Calculate optimal batch parameters dynamically based on file characteristics.

    Analyzes file extensions and sizes to determine optimal batching strategy:
    - Text-like files (html, json, md, sh, txt): Faster processing, larger batches with less concurrency
      Uses files_per_batch={TEXT_FILE_BATCH_SIZE}, concurrent_batches={TEXT_FILE_CONCURRENT_BATCHES}
    - Complex files (pdf, images, docx, pptx, media): Starts from configured defaults, then
      automatically reduces files_per_batch when one or more files are large (max > 50/100 MB
      or average > 25 MB) so that a single slow file does not pin its whole batch and stall
      sibling batches via VDB write serialization (see MULTIMODAL_* constants above).

    The function can be enhanced in the future with additional factors:
    - File complexity beyond raw size (number of pages, resolution, etc.)
    - Available system resources (memory, CPU)
    - Historical performance metrics
    - Target latency and throughput requirements

    Args:
        filepaths: List of file paths to be processed
        config: NvidiaRAGConfig instance containing batch configuration

    Returns:
        Tuple of (files_per_batch, concurrent_batches)
            - files_per_batch: Number of files to include in each batch
            - concurrent_batches: Number of batches to process concurrently
    """
    # Check if dynamic batching is enabled
    if not config.nv_ingest.enable_dynamic_batching:
        # Return default configured values without analysis
        logger.info(
            f"Dynamic batching is disabled. Using default configuration parameters for files processing: "
            f"files_per_batch={config.nv_ingest.files_per_batch}, concurrent_batches={config.nv_ingest.concurrent_batches}"
        )
        return config.nv_ingest.files_per_batch, config.nv_ingest.concurrent_batches

    # Analyze file extensions
    if not filepaths:
        logger.warning("Empty filepaths list provided to dynamic batch calculator")
        return config.nv_ingest.files_per_batch, config.nv_ingest.concurrent_batches

    # Extract extensions and count text-like files
    text_file_count = 0
    total_files = len(filepaths)
    extension_counts = {}

    for filepath in filepaths:
        # Extract extension (lowercase, without dot)
        ext = Path(filepath).suffix.lstrip(".").lower()
        
        # Track extension distribution
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # Check if it's a text-like file
        if ext in TEXT_LIKE_EXTENSIONS:
            text_file_count += 1

    # Calculate percentage of text-like files
    text_file_percentage = (text_file_count / total_files) * 100 if total_files > 0 else 0

    # Log file distribution for debugging
    logger.debug(
        f"File distribution analysis: Total={total_files}, "
        f"Text-like={text_file_count} ({text_file_percentage:.1f}%), "
        f"Extensions={dict(extension_counts)}"
    )

    # Decision logic: If majority (>TEXT_FILE_PERCENTAGE_THRESHOLD%) are text-like files, optimize for them
    if text_file_percentage > TEXT_FILE_PERCENTAGE_THRESHOLD:
        files_per_batch, concurrent_batches = calculate_text_like_batch_params(filepaths, config)
        logger.info(
            f"Dynamic batching: Detected {text_file_percentage:.1f}% text-like files. "
            f"Using optimized parameters for text processing: "
            f"files_per_batch={files_per_batch}, concurrent_batches={concurrent_batches}"
        )
    else:
        # Use default configuration as the starting point for non-text-like
        # (multimodal) workloads, then apply size-aware adjustments below.
        files_per_batch = config.nv_ingest.files_per_batch
        concurrent_batches = config.nv_ingest.concurrent_batches

        # Size-aware adjustment: if any file in the workload is very large, or
        # the average file size is large, shrink files_per_batch so a single
        # slow file does not pin too many siblings.  This addresses the long
        # "pending" stall observed in NVBug 6191293, where one 100+MB
        # multimodal PDF in a 16-file batch blocked the entire ingest task.
        adjusted_files_per_batch = calculate_multimodal_size_aware_batch_size(
            filepaths=filepaths,
            default_files_per_batch=files_per_batch,
        )

        if adjusted_files_per_batch != files_per_batch:
            logger.info(
                f"Dynamic batching: Detected {text_file_percentage:.1f}% text-like files "
                f"with large multimodal content. Reducing files_per_batch from "
                f"{files_per_batch} to {adjusted_files_per_batch} to limit the impact "
                f"of slow per-file processing on sibling files in the same batch. "
                f"concurrent_batches={concurrent_batches}"
            )
            files_per_batch = adjusted_files_per_batch
        else:
            logger.info(
                f"Dynamic batching: Detected {text_file_percentage:.1f}% text-like files. "
                f"Using default configuration parameters for files processing: "
                f"files_per_batch={files_per_batch}, concurrent_batches={concurrent_batches}"
            )

    return files_per_batch, concurrent_batches


def calculate_multimodal_size_aware_batch_size(
    filepaths: list[str],
    default_files_per_batch: int,
) -> int:
    """
    Compute a size-aware ``files_per_batch`` for non-text-like (multimodal)
    workloads.

    When the workload contains very large files, a single slow file can pin
    every other file in its batch behind it. Combined with the existing VDB
    write serialization, that pin-blocks sibling batches as well. This helper
    inspects the actual on-disk file sizes and proposes a smaller
    ``files_per_batch`` when the workload looks risky:

    - Any single file > MULTIMODAL_VERY_LARGE_FILE_MB (100 MB):
      shrink aggressively (``default // 4``, floor MULTIMODAL_MIN_FILES_PER_BATCH).
    - Any single file > MULTIMODAL_LARGE_FILE_MB (50 MB) OR average size >
      MULTIMODAL_AVG_LARGE_FILE_MB (25 MB): shrink moderately
      (``default // 2``, floor MULTIMODAL_MIN_FILES_PER_BATCH).
    - Otherwise leave ``files_per_batch`` unchanged.

    Args:
        filepaths: List of file paths actually on disk (already validated by
            the ingestor server before this is called).
        default_files_per_batch: The configured ``files_per_batch`` to start
            from.

    Returns:
        The (possibly reduced) files_per_batch. Never returns less than
        ``MULTIMODAL_MIN_FILES_PER_BATCH``; never returns more than
        ``default_files_per_batch``.
    """
    if not filepaths:
        return default_files_per_batch

    sizes_bytes: list[int] = []
    for filepath in filepaths:
        try:
            sizes_bytes.append(os.path.getsize(filepath))
        except OSError as exc:
            # Stay conservative: if we cannot stat a file, skip it for the
            # size computation rather than aborting the batching decision.
            logger.debug(
                "calculate_multimodal_size_aware_batch_size: could not stat "
                "%s (%s); skipping for size analysis",
                filepath,
                exc,
            )

    if not sizes_bytes:
        return default_files_per_batch

    bytes_per_mb = 1024 * 1024
    max_size_mb = max(sizes_bytes) / bytes_per_mb
    avg_size_mb = (sum(sizes_bytes) / len(sizes_bytes)) / bytes_per_mb

    logger.debug(
        "Multimodal size analysis: max_size_mb=%.2f, avg_size_mb=%.2f, "
        "num_files=%d, default_files_per_batch=%d",
        max_size_mb,
        avg_size_mb,
        len(sizes_bytes),
        default_files_per_batch,
    )

    if max_size_mb > MULTIMODAL_VERY_LARGE_FILE_MB:
        return max(MULTIMODAL_MIN_FILES_PER_BATCH, default_files_per_batch // 4)

    if (
        max_size_mb > MULTIMODAL_LARGE_FILE_MB
        or avg_size_mb > MULTIMODAL_AVG_LARGE_FILE_MB
    ):
        return max(MULTIMODAL_MIN_FILES_PER_BATCH, default_files_per_batch // 2)

    return default_files_per_batch


def calculate_text_like_batch_params(
    filepaths: list[str],
    config: NvidiaRAGConfig,
) -> tuple[int, int]:
    """
    Calculate batch size and concurrent batches for text-like files.
    """
    # Calculate average file size
    avg_file_size_bytes = calculate_average_file_size(filepaths)

    # Calculate average embedding size
    avg_embedding_size_bytes = calculate_average_embedding_size(
        avg_file_size_bytes=avg_file_size_bytes,
        num_files=len(filepaths),
        chunk_size=config.nv_ingest.chunk_size,
        chunk_overlap=config.nv_ingest.chunk_overlap,
        embedding_dimensions=config.embeddings.dimensions,
    )

    # Calculate memory required per file
    memory_per_file_bytes = (avg_file_size_bytes + avg_embedding_size_bytes) * TEXT_FILE_MEMORY_OVERHEAD_FACTOR

    # Calculate max concurrent files in a single ingestion job (i.e batch size x concurrent batches)
    max_concurrent_files = config.nv_ingest.max_memory_budget_mb * 1024 * 1024 // memory_per_file_bytes
    
    logger.debug(
        f"Text-like file batching parameters: "
        f"avg_file_size_bytes={avg_file_size_bytes}, "
        f"avg_embedding_size_bytes={avg_embedding_size_bytes}, "
        f"memory_per_file_bytes={memory_per_file_bytes}, "
        f"max_concurrent_files={max_concurrent_files}"
    )

    for batch_size in sorted(TEXT_ALLOWED_BATCH_SIZES, reverse=True):
        if batch_size * TEXT_FILE_CONCURRENT_BATCHES <= len(filepaths):
            if batch_size * TEXT_FILE_CONCURRENT_BATCHES <= max_concurrent_files:
                return batch_size, TEXT_FILE_CONCURRENT_BATCHES

    return min(TEXT_ALLOWED_BATCH_SIZES), TEXT_FILE_CONCURRENT_BATCHES


def calculate_average_file_size(filepaths: list[str]) -> int:
    """
    Calculate average file size in bytes.
    """
    total_size = 0
    for filepath in filepaths:
        total_size += os.path.getsize(filepath)
    return total_size // len(filepaths)

def calculate_average_embedding_size(
    avg_file_size_bytes: int,
    num_files: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_dimensions: int,
) -> int:
    """
    Calculate average embedding size in bytes.
    """
    total_token_count = avg_file_size_bytes * num_files // 4
    total_chunk_count = total_token_count // (chunk_size - chunk_overlap) + 1
    total_embedding_size_bytes = total_chunk_count * embedding_dimensions * 4
    return total_embedding_size_bytes // num_files

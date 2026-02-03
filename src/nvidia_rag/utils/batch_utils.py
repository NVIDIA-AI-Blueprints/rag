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
TEXT_FILE_BATCH_SIZE = 200
TEXT_FILE_CONCURRENT_BATCHES = 1

# Threshold percentage to determine if workload is text-heavy
TEXT_FILE_PERCENTAGE_THRESHOLD = 50.0


def calculate_dynamic_batch_parameters(
    filepaths: list[str],
    config: NvidiaRAGConfig,
) -> tuple[int, int]:
    """
    Calculate optimal batch parameters dynamically based on file characteristics.

    Analyzes file extensions to determine optimal batching strategy:
    - Text-like files (html, json, md, sh, txt): Faster processing, larger batches with less concurrency
      Uses files_per_batch={TEXT_FILE_BATCH_SIZE}, concurrent_batches={TEXT_FILE_CONCURRENT_BATCHES}
    - Complex files (pdf, images, docx, pptx, media): Smaller batches with higher concurrency
      Uses default configured values

    The function can be enhanced in the future with additional factors:
    - File sizes and complexity (number of pages, resolution, etc.)
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
        files_per_batch = TEXT_FILE_BATCH_SIZE
        concurrent_batches = TEXT_FILE_CONCURRENT_BATCHES
        logger.info(
            f"Dynamic batching: Detected {text_file_percentage:.1f}% text-like files. "
            f"Using optimized parameters for text processing: "
            f"files_per_batch={files_per_batch}, concurrent_batches={concurrent_batches}"
        )
    else:
        # Use default configuration for other files (PDFs, images, media, etc.)
        files_per_batch = config.nv_ingest.files_per_batch
        concurrent_batches = config.nv_ingest.concurrent_batches
        logger.info(
            f"Dynamic batching: Detected {text_file_percentage:.1f}% text-like files. "
            f"Using default configuration parameters for files processing: "
            f"files_per_batch={files_per_batch}, concurrent_batches={concurrent_batches}"
        )

    return files_per_batch, concurrent_batches

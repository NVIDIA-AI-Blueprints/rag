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
Validation functions for the RAG ingestion pipeline.

This module provides validation utilities for:
1. File type validation against supported extensions
2. Directory traversal attack detection
3. Custom metadata validation against schemas
4. File path validation and filtering

Functions:
    validate_directory_traversal_attack: Validates file paths to prevent directory traversal attacks
    get_non_supported_files: Identifies files with unsupported extensions
    remove_unsupported_files: Filters out files with unsupported extensions
    split_pdf_and_non_pdf_files: Separates PDF files from other file types
    validate_custom_metadata: Validates custom metadata against collection schemas
"""

import logging
import os
from pathlib import Path
from typing import Any

from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.metadata_validation import (
    MetadataSchema,
    MetadataValidator,
)
from nvidia_rag.utils.observability.tracing import (
    get_tracer,
    trace_function,
)

# Initialize logger
logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.validation")

# Supported file types (excluding SVG)
SUPPORTED_FILE_TYPES = set(EXTENSION_TO_DOCUMENT_TYPE.keys()) - set({"svg"})


@trace_function("ingestor.validation.validate_directory_traversal_attack", tracer=TRACER)
async def validate_directory_traversal_attack(file: str) -> None:
    """
    Validate file path to prevent directory traversal attacks.

    This function uses Path.resolve(strict=True) to obtain the absolute and normalized
    path, with the added condition that the path must physically exist on the filesystem.
    If a directory traversal attack is attempted, the resulting path after the resolve
    will be invalid and an exception will be raised.

    Args:
        file: File path to validate

    Raises:
        ValueError: If file not found or directory traversal attack detected
    """
    try:
        # Path.resolve(strict=True) is a method used to
        # obtain the absolute and normalized path, with
        # the added condition that the path must physically
        # exist on the filesystem. If a directory traversal
        # attack is tried, resulting path after the resolve
        # will be invalid.
        if file:
            _ = Path(file).resolve(strict=True)
    except Exception as e:
        raise ValueError(
            f"File not found or a directory traversal attack detected! Filepath: {file}"
        ) from e


@trace_function("ingestor.validation.get_non_supported_files", tracer=TRACER)
async def get_non_supported_files(filepaths: list[str]) -> list[str]:
    """
    Get filepaths of non-supported file extensions.

    Args:
        filepaths: List of file paths to check

    Returns:
        List of file paths with unsupported extensions
    """
    non_supported_files = []
    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [
            "." + supported_ext for supported_ext in SUPPORTED_FILE_TYPES
        ]:
            non_supported_files.append(filepath)
    return non_supported_files


async def remove_unsupported_files(
    filepaths: list[str],
) -> list[str]:
    """
    Remove unsupported files from the list of filepaths.

    Args:
        filepaths: List of file paths to filter

    Returns:
        List of file paths with only supported extensions
    """
    non_supported_files = await get_non_supported_files(filepaths)
    return [
        filepath for filepath in filepaths if filepath not in non_supported_files
    ]


@trace_function("ingestor.validation.split_pdf_and_non_pdf_files", tracer=TRACER)
async def split_pdf_and_non_pdf_files(
    filepaths: list[str]
) -> tuple[list[str], list[str]]:
    """
    Split PDF and non-PDF files from the list of filepaths.

    Args:
        filepaths: List of file paths to split

    Returns:
        Tuple of (pdf_filepaths, non_pdf_filepaths)
    """
    pdf_filepaths = []
    non_pdf_filepaths = []
    for filepath in filepaths:
        if os.path.splitext(filepath)[1].lower() == ".pdf":
            pdf_filepaths.append(filepath)
        else:
            non_pdf_filepaths.append(filepath)
    return pdf_filepaths, non_pdf_filepaths


@trace_function("ingestor.validation.validate_custom_metadata", tracer=TRACER)
async def validate_custom_metadata(
    custom_metadata: list[dict[str, Any]],
    collection_name: str,
    metadata_schema_data: list[dict[str, Any]],
    filepaths: list[str],
    config: NvidiaRAGConfig,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Validate custom metadata against schema and return validation status and errors.

    This function performs comprehensive validation of user-provided metadata:
    1. Validates that metadata filenames match the files being ingested
    2. Validates metadata values against the collection's schema (if one exists)
    3. Normalizes datetime values according to schema definitions
    4. Checks for missing required metadata fields

    Args:
        custom_metadata: User-provided metadata list. Each item should contain:
            - filename: Name of the file
            - metadata: Dictionary of metadata key-value pairs
        collection_name: Name of the collection
        metadata_schema_data: Metadata schema from VDB as a list of field definitions
        filepaths: List of file paths being ingested
        config: Configuration object for the RAG system

    Returns:
        Tuple[bool, List[Dict[str, Any]]]: (validation_status, validation_errors)
        - validation_status: True if all validation passes, False otherwise
        - validation_errors: List of error dictionaries in the format:
            {
                "error": "Error message",
                "metadata": {"filename": "...", "file_metadata": {...}}
            }

    Note:
        If validation passes, the custom_metadata list may be modified in-place
        to include normalized datetime values.
    """
    logger.info(
        f"Metadata schema for collection {collection_name}: {metadata_schema_data}"
    )
    # Validate that metadata filenames match the files being ingested
    filenames = {os.path.basename(filepath) for filepath in filepaths}

    # Setup validation if schema exists
    validator = None
    metadata_schema = None
    if metadata_schema_data:
        logger.debug(
            f"Using metadata schema for collection '{collection_name}' with {len(metadata_schema_data)} fields"
        )
        validator = MetadataValidator(config)
        metadata_schema = MetadataSchema(schema=metadata_schema_data)
    else:
        logger.info(
            f"No metadata schema found for collection {collection_name}. Skipping schema validation."
        )

    filename_to_metadata = {
        item.get("filename"): item.get("metadata", {}) for item in custom_metadata
    }

    validation_errors = []
    validation_status = True

    # Process all metadata items and validate them
    for custom_metadata_item in custom_metadata:
        filename = custom_metadata_item.get("filename", "")
        metadata = custom_metadata_item.get("metadata", {})

        # Check if the filename is provided in the ingestion request
        if filename not in filenames:
            validation_errors.append(
                {
                    "error": f"Filename: {filename} is not provided in the ingestion request",
                    "metadata": {"filename": filename, "file_metadata": metadata},
                }
            )
            validation_status = False
            continue

        if validator and metadata_schema:
            (
                is_valid,
                field_errors,
                normalized_metadata,
            ) = validator.validate_and_normalize_metadata_values(
                metadata, metadata_schema
            )
            logger.debug(
                f"Metadata validation for '{filename}': {'PASSED' if is_valid else 'FAILED'}"
            )
            if not is_valid:
                validation_status = False
                # Convert new validator format to original format for backward compatibility
                for error in field_errors:
                    error_message = error.get("error", "Validation error")
                    validation_errors.append(
                        {
                            "error": f"File '{filename}': {error_message}",
                            "metadata": {
                                "filename": filename,
                                "file_metadata": metadata,
                            },
                        }
                    )
            else:
                # Update the metadata with normalized datetime values
                custom_metadata_item["metadata"] = normalized_metadata
                logger.debug(
                    f"Updated metadata for file '{filename}' with normalized datetime values"
                )
        else:
            # No schema - just do basic validation (ensure it's a dict)
            if not isinstance(metadata, dict):
                validation_errors.append(
                    {
                        "error": f"Metadata for file '{filename}' must be a dictionary",
                        "metadata": {
                            "filename": filename,
                            "file_metadata": metadata,
                        },
                    }
                )
                validation_status = False

    # Check for files without metadata that require it
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if filename not in filename_to_metadata:
            if validator and metadata_schema:
                required_fields = metadata_schema.required_fields
                if required_fields:
                    validation_errors.append(
                        {
                            "error": f"File '{filename}': No metadata provided but schema requires fields: {required_fields}",
                            "metadata": {"filename": filename, "file_metadata": {}},
                        }
                    )
                    validation_status = False
            else:
                logger.debug(
                    f"File '{filename}': No metadata provided, but no required fields in schema"
                )

    if not validation_status:
        logger.error(
            f"Custom metadata validation failed: {len(validation_errors)} errors"
        )
    else:
        logger.debug("Custom metadata validated and normalized successfully.")

    return validation_status, validation_errors

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
MinIO Content Handler Module.

This module provides functionality for handling MinIO operations in the ingestion pipeline,
specifically for uploading NV-Ingest image, table, and chart content to MinIO storage.

Functions:
    put_content_to_minio: Upload NV-Ingest image/table/chart content to MinIO.
"""

import logging
import os

from nvidia_rag.utils.minio_operator import get_unique_thumbnail_id_from_result

logger = logging.getLogger(__name__)


def put_content_to_minio(
    results: list[list[dict[str, str | dict]]],
    collection_name: str,
    minio_operator,
    enable_citations: bool,
) -> None:
    """
    Put nv-ingest image/table/chart content to minio

    Arguments:
        results: List of lists containing result elements with document metadata
        collection_name: Name of the collection in the vector database
        minio_operator: MinIO operator instance for handling uploads
        enable_citations: Flag to enable/disable MinIO insertion
    """
    if not enable_citations:
        logger.info(f"Skipping minio insertion for collection: {collection_name}")
        return  # Don't perform minio insertion if captioning is disabled

    payloads = []
    object_names = []

    for result in results:
        for result_element in result:
            if result_element.get("document_type") in ["image", "structured"]:
                # Extract required fields
                metadata = result_element.get("metadata", {})
                content = result_element.get("metadata").get("content")

                file_name = os.path.basename(
                    result_element.get("metadata")
                    .get("source_metadata")
                    .get("source_id")
                )
                page_number = (
                    result_element.get("metadata")
                    .get("content_metadata")
                    .get("page_number")
                )
                location = (
                    result_element.get("metadata")
                    .get("content_metadata")
                    .get("location")
                )

                # Get unique_thumbnail_id using the centralized function
                # Try with extracted location first, fallback to content_metadata if None
                unique_thumbnail_id = get_unique_thumbnail_id_from_result(
                    collection_name=collection_name,
                    file_name=file_name,
                    page_number=page_number,
                    location=location,
                    metadata=metadata,
                )

                if unique_thumbnail_id is not None:
                    # Pull content from result_element
                    payloads.append({"content": content})
                    object_names.append(unique_thumbnail_id)
                # If unique_thumbnail_id is None, the item is skipped
                # (warning already logged in get_unique_thumbnail_id_from_result)

    if minio_operator is not None:
        if os.getenv("ENABLE_MINIO_BULK_UPLOAD", "True") in ["True", "true"]:
            logger.info(f"Bulk uploading {len(payloads)} payloads to MinIO")
            try:
                minio_operator.put_payloads_bulk(
                    payloads=payloads, object_names=object_names
                )
            except Exception as e:
                logger.warning(f"Failed to bulk upload to MinIO: {e}")
        else:
            logger.info(f"Sequentially uploading {len(payloads)} payloads to MinIO")
            for payload, object_name in zip(payloads, object_names, strict=False):
                try:
                    minio_operator.put_payload(
                        payload=payload, object_name=object_name
                    )
                except Exception as e:
                    logger.warning(f"Failed to upload {object_name} to MinIO: {e}")
    else:
        logger.warning(
            f"MinIO unavailable - skipping upload of {len(payloads)} payloads"
        )

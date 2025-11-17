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

"""Minio operator Module to store metadata and assocated utilities.
1. MinioOperator: Class to store metadata using Minio-client.
2. get_minio_operator: Get the MinioOperator object.
3. get_unique_thumbnail_id_collection_prefix: Get the unique thumbnail id collection prefix.
4. get_unique_thumbnail_id_file_name_prefix: Get the unique thumbnail id file name prefix.
5. get_unique_thumbnail_id: Get the unique thumbnail id.
6. extract_location_from_metadata: Extract location from metadata based on document type.
7. get_unique_thumbnail_id_from_result: Generate unique thumbnail ID from nv-ingest result element.
"""

import json
import logging
from io import BytesIO

from minio import Minio
from minio.commonconfig import SnowballObject

from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)
DEFAULT_BUCKET_NAME = "default-bucket"


class MinioOperator:
    """Minio operator Class to store metadata using Minio-client"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        default_bucket_name: str = DEFAULT_BUCKET_NAME,
    ):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=False
        )
        self.default_bucket_name = default_bucket_name
        self._make_bucket(bucket_name=self.default_bucket_name)

    def _make_bucket(self, bucket_name: str):
        """Create new bucket if doesn't exists"""
        if not self.client.bucket_exists(bucket_name):
            logger.info(f"Creating bucket: {bucket_name}")
            self.client.make_bucket(bucket_name)
            logger.info(f"Bucket created: {bucket_name}")
        else:
            logger.info(f"Bucket already exists: {bucket_name}")

    def put_payload(self, payload: dict, object_name: str):
        """Put dictionary to S3 storage using minio client"""
        # Convert payload dictionary to JSON bytes
        json_data = json.dumps(payload).encode("utf-8")

        # Upload JSON data to MinIO
        self.client.put_object(
            self.default_bucket_name,
            object_name,
            BytesIO(json_data),
            len(json_data),
            content_type="application/json",
        )

    def put_payloads_bulk(self, payloads: list[dict], object_names: list[str]):
        """Put list of dictionaries to S3 storage using minio client"""
        json_datas = [json.dumps(payload).encode("utf-8") for payload in payloads]

        snowball_objects = []
        for object_name, json_data in zip(object_names, json_datas, strict=False):
            snowball_objects.append(
                SnowballObject(
                    object_name, data=BytesIO(json_data), length=len(json_data)
                )
            )

        # Bulk upload objects to MinIO
        self.client.upload_snowball_objects(self.default_bucket_name, snowball_objects)

    def get_payload(self, object_name: str) -> dict:
        """Get dictionary from S3 storage using minio client"""
        # Retrieve JSON from MinIO

        try:
            response = self.client.get_object(self.default_bucket_name, object_name)

            # Read and decode the JSON data
            retrieved_data = json.loads(response.read().decode("utf-8"))
            return retrieved_data
        except Exception as e:
            logger.warning(
                f"Error while getting object from Minio! Object name: {object_name}"
            )
            logger.debug(f"Error while getting object from Minio: {e}")
            return {}

    def list_payloads(self, prefix: str = "") -> list[str]:
        """List payloads from S3 storage using minio client"""
        list_of_objects = []
        for obj in self.client.list_objects(
            self.default_bucket_name, prefix=prefix, recursive=True
        ):
            list_of_objects.append(obj.object_name)
        return list_of_objects

    def delete_payloads(self, object_names: list[str]) -> None:
        """Delete payloads from S3 storage using minio client"""
        for object_name in object_names:
            self.client.remove_object(self.default_bucket_name, object_name)


def get_minio_operator(
    default_bucket_name: str = DEFAULT_BUCKET_NAME,
    config: NvidiaRAGConfig | None = None,
) -> MinioOperator:
    """
    Prepares and return MinioOperator object

    Args:
        default_bucket_name: Default bucket name
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        - minio_operator: MinioOperator
    """
    if config is None:
        config = NvidiaRAGConfig()

    minio_operator = MinioOperator(
        endpoint=config.minio.endpoint,
        access_key=config.minio.access_key.get_secret_value(),
        secret_key=config.minio.secret_key.get_secret_value(),
        default_bucket_name=default_bucket_name,
    )
    return minio_operator


def get_unique_thumbnail_id_collection_prefix(
    collection_name: str,
) -> str:
    """
    Prepares unique thumbnail id prefix based on input collection name
    Returns:
        - unique_thumbnail_id_prefix: str
    """
    prefix = f"{collection_name}_::"
    return prefix


def get_unique_thumbnail_id_file_name_prefix(
    collection_name: str,
    file_name: str,
) -> str:
    """
    Prepares unique thumbnail id prefix based on input collection name and file name
    Returns:
        - unique_thumbnail_id_prefix: str
    """
    collection_prefix = get_unique_thumbnail_id_collection_prefix(collection_name)
    prefix = f"{collection_prefix}_{file_name}_::"
    return prefix


def get_unique_thumbnail_id(
    collection_name: str,
    file_name: str,
    page_number: int,
    location: list[float],  # Bbox information
) -> str:
    """
    Prepares unique thumbnail id based on input arguments
    Returns:
        - unique_thumbnail_id: str
    """
    # Round bbox values to reduce precision
    rounded_bbox = [round(coord, 4) for coord in location]
    prefix = get_unique_thumbnail_id_file_name_prefix(collection_name, file_name)
    # Create a string representation
    unique_thumbnail_id = f"{prefix}_{page_number}_" + "_".join(map(str, rounded_bbox))
    return unique_thumbnail_id


def extract_location_from_metadata(
    document_type: str,
    metadata: dict,
) -> list[float] | None:
    """
    Extract location (bounding box) from metadata based on document type.

    This function handles the complexity of finding location information
    across different document types and their respective metadata structures.

    Args:
        document_type: Type of document ("image", "structured", etc.)
        metadata: The metadata dictionary (can be from nv-ingest or VDB)

    Returns:
        Location as [x1, y1, x2, y2] list or None if not found

    Supported document types:
        - "image": Extracts from image_metadata.image_location or content_metadata.location
        - "structured": Extracts from table_metadata.table_location or content_metadata.location
                       (works for both tables and charts)
    """
    # First check if location is in content_metadata (VDB format)
    content_metadata_dict = metadata.get("content_metadata", {})
    location = content_metadata_dict.get("location") if content_metadata_dict else None

    # If not found in content_metadata, extract from document-type-specific metadata (nv-ingest format)
    if location is None:
        # Extract location from image_metadata, table_metadata, and chart_metadata
        image_metadata = metadata.get("image_metadata", {}) or {}
        table_metadata = metadata.get("table_metadata", {}) or {}
        chart_metadata = metadata.get("chart_metadata", {}) or {}

        location = (
            image_metadata.get("image_location", [])
            + table_metadata.get("table_location", [])
            + chart_metadata.get("chart_location", [])
        )

        if location:
            logger.info("Extracted location is %s", location)

    return location


def get_unique_thumbnail_id_from_result(
    collection_name: str,
    file_name: str,
    page_number: int,
    location: list[float] | None = None,
    metadata: dict | None = None,
) -> str | None:
    """
    Generate unique thumbnail ID with fallback to metadata if location is not provided.

    This function tries to use the provided location first. If location is None,
    it will attempt to extract location from metadata based on document type.

    Args:
        collection_name: Name of the collection
        file_name: Name of the file
        page_number: Page number
        location: Bounding box location [x1, y1, x2, y2] (optional)
        metadata: Content metadata dict containing image_metadata or table_metadata (optional)

    Returns:
        Unique thumbnail ID string or None if location cannot be extracted

    Examples:
        # With location already extracted
        thumbnail_id = get_unique_thumbnail_id_from_result(
            collection_name="my_collection",
            file_name="doc.pdf",
            page_number=0,
            location=[35, 0, 575, 539]
        )

        # Without location, fallback to metadata
        thumbnail_id = get_unique_thumbnail_id_from_result(
            collection_name="my_collection",
            file_name="doc.pdf",
            page_number=0,
            location=None,
            metadata={}
        )
    """
    try:
        # If location is not provided, try to extract from metadata
        if not location and metadata:
            content_metadata = metadata.get("content_metadata", {})
            document_type = content_metadata.get("type")
            if document_type:
                logger.debug(
                    "Attempting to extract location from metadata for %s (type: %s)",
                    file_name,
                    document_type,
                )
                location = extract_location_from_metadata(document_type, metadata)

        # If still no location, cannot generate thumbnail ID
        if location is None:
            logger.warning(
                f"Skipping page {page_number} of {file_name}: "
                f"No location information found"
            )
            return None

        # Generate and return unique thumbnail ID
        return get_unique_thumbnail_id(
            collection_name=collection_name,
            file_name=file_name,
            page_number=page_number,
            location=location,
        )

    except Exception as e:
        logger.warning("Failed to generate thumbnail ID for %s: %s", file_name, str(e))
        return None

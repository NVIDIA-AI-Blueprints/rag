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

"""Object-store utilities for S3-compatible backends."""

import json
import logging
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from minio.commonconfig import SnowballObject

from nvidia_rag.utils.common import object_key_from_storage_uri
from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
DEFAULT_BUCKET_NAME = "default-bucket"


class S3ObjectStoreOperator:
    """Object-store operator backed by an S3-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        default_bucket_name: str = DEFAULT_BUCKET_NAME,
        secure: bool = False,
    ):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self.default_bucket_name = default_bucket_name
        self.endpoint = endpoint
        self.secure = secure
        try:
            self._make_bucket(bucket_name=self.default_bucket_name)
        except Exception as e:
            logger.warning(
                "Object store unavailable at %s - bucket operations will fail at runtime: %s",
                endpoint,
                e,
            )

    def _make_bucket(self, bucket_name: str):
        """Create a bucket if it does not exist."""
        if not self.client.bucket_exists(bucket_name):
            logger.info("Creating bucket: %s", bucket_name)
            self.client.make_bucket(bucket_name)
            logger.info("Bucket created: %s", bucket_name)
        else:
            logger.info("Bucket already exists: %s", bucket_name)

    def put_payload(self, payload: dict, object_name: str):
        """Store a JSON payload."""
        json_data = json.dumps(payload).encode("utf-8")
        self.client.put_object(
            self.default_bucket_name,
            object_name,
            BytesIO(json_data),
            len(json_data),
            content_type="application/json",
        )

    def put_payloads_bulk(self, payloads: list[dict], object_names: list[str]):
        """Store multiple JSON payloads."""
        json_datas = [json.dumps(payload).encode("utf-8") for payload in payloads]
        snowball_objects = []
        for object_name, json_data in zip(object_names, json_datas, strict=False):
            snowball_objects.append(
                SnowballObject(
                    object_name, data=BytesIO(json_data), length=len(json_data)
                )
            )

        self.client.upload_snowball_objects(self.default_bucket_name, snowball_objects)

    def get_object(self, object_name: str) -> bytes:
        """Fetch raw object bytes."""
        response = self.client.get_object(self.default_bucket_name, object_name)
        return response.read()

    def get_object_from_uri(self, uri: str) -> bytes:
        """Fetch raw bytes from an S3 URI."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Unsupported storage URI scheme: {parsed.scheme}")

        bucket_name = parsed.netloc
        object_name = object_key_from_storage_uri(uri)
        if not bucket_name or not object_name:
            raise ValueError("Invalid S3 URI")

        response = self.client.get_object(bucket_name, object_name)
        return response.read()

    def get_payload(self, object_name: str) -> dict:
        """Fetch a JSON payload."""
        try:
            response = self.client.get_object(self.default_bucket_name, object_name)
            return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.warning(
                "Error while getting object from object store! Object name: %s",
                object_name,
            )
            logger.debug("Error while getting object from object store: %s", e)
            return {}

    def list_payloads(self, prefix: str = "") -> list[str]:
        """List payload object names below a prefix."""
        return [
            obj.object_name
            for obj in self.client.list_objects(
                self.default_bucket_name, prefix=prefix, recursive=True
            )
        ]

    def delete_payloads(self, object_names: list[str]) -> None:
        """Delete payloads."""
        for object_name in object_names:
            self.client.remove_object(self.default_bucket_name, object_name)


class FilesystemObjectStoreOperator:
    """Object-store operator backed by the local filesystem."""

    def __init__(
        self,
        root_path: str | Path,
        default_bucket_name: str = DEFAULT_BUCKET_NAME,
    ):
        self.root_path = Path(root_path).expanduser().resolve()
        self.default_bucket_name = default_bucket_name
        self.endpoint = self.root_path.as_uri()
        self.secure = False
        self._make_bucket(bucket_name=self.default_bucket_name)

    def _bucket_root(self, bucket_name: str) -> Path:
        return self.root_path / bucket_name

    def _object_path(self, object_name: str, bucket_name: str | None = None) -> Path:
        active_bucket = bucket_name or self.default_bucket_name
        return self._bucket_root(active_bucket) / object_name

    def _make_bucket(self, bucket_name: str):
        self._bucket_root(bucket_name).mkdir(parents=True, exist_ok=True)

    def put_payload(self, payload: dict, object_name: str):
        object_path = self._object_path(object_name)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_text(json.dumps(payload), encoding="utf-8")

    def put_payloads_bulk(self, payloads: list[dict], object_names: list[str]):
        for payload, object_name in zip(payloads, object_names, strict=False):
            self.put_payload(payload, object_name)

    def get_object(self, object_name: str) -> bytes:
        return self._object_path(object_name).read_bytes()

    def get_object_from_uri(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"Unsupported storage URI scheme: {parsed.scheme}")
        return Path(parsed.path).read_bytes()

    def get_payload(self, object_name: str) -> dict:
        try:
            return json.loads(self._object_path(object_name).read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                "Error while getting object from object store! Object name: %s",
                object_name,
            )
            logger.debug("Error while getting object from object store: %s", e)
            return {}

    def list_payloads(self, prefix: str = "") -> list[str]:
        bucket_root = self._bucket_root(self.default_bucket_name)
        if not bucket_root.exists():
            return []

        payloads: list[str] = []
        for candidate in bucket_root.rglob("*"):
            if not candidate.is_file():
                continue
            relative_name = candidate.relative_to(bucket_root).as_posix()
            if relative_name.startswith(prefix):
                payloads.append(relative_name)
        return payloads

    def delete_payloads(self, object_names: list[str]) -> None:
        bucket_root = self._bucket_root(self.default_bucket_name)
        for object_name in object_names:
            object_path = self._object_path(object_name)
            if object_path.exists():
                object_path.unlink()
            parent = object_path.parent
            while parent != bucket_root and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent


ObjectStoreOperator = S3ObjectStoreOperator | FilesystemObjectStoreOperator


def get_object_store_operator(
    default_bucket_name: str = DEFAULT_BUCKET_NAME,
    config: NvidiaRAGConfig | None = None,
) -> S3ObjectStoreOperator | FilesystemObjectStoreOperator:
    """Prepare and return the configured object-store operator."""
    if config is None:
        config = NvidiaRAGConfig()

    if config.object_store.backend == "filesystem":
        return FilesystemObjectStoreOperator(
            root_path=config.object_store.storage_root,
            default_bucket_name=default_bucket_name,
        )

    return S3ObjectStoreOperator(
        endpoint=config.object_store.endpoint,
        access_key=config.object_store.access_key.get_secret_value(),
        secret_key=config.object_store.secret_key.get_secret_value(),
        default_bucket_name=default_bucket_name,
        secure=config.object_store.secure,
    )


def get_unique_thumbnail_id_collection_prefix(
    collection_name: str,
) -> str:
    """Prepare unique thumbnail id prefix based on input collection name."""
    return f"{collection_name}_::"


def get_unique_thumbnail_id_file_name_prefix(
    collection_name: str,
    file_name: str,
) -> str:
    """Prepare unique thumbnail id prefix based on input collection name and file name."""
    collection_prefix = get_unique_thumbnail_id_collection_prefix(collection_name)
    return f"{collection_prefix}_{file_name}_::"


def get_unique_thumbnail_id(
    collection_name: str,
    file_name: str,
    page_number: int,
    location: list[float],
) -> str:
    """Prepare unique thumbnail id based on input arguments."""
    rounded_bbox = [round(coord, 4) for coord in location]
    prefix = get_unique_thumbnail_id_file_name_prefix(collection_name, file_name)
    return f"{prefix}_{page_number}_" + "_".join(map(str, rounded_bbox))


def extract_location_from_metadata(
    document_type: str,
    metadata: dict,
) -> list[float] | None:
    """
    Extract location (bounding box) from metadata based on document type.

    This function handles the complexity of finding location information
    across different document types and their respective metadata structures.
    """
    content_metadata_dict = metadata.get("content_metadata", {})
    location = content_metadata_dict.get("location") if content_metadata_dict else None

    if location is None:
        image_metadata = metadata.get("image_metadata", {}) or {}
        table_metadata = metadata.get("table_metadata", {}) or {}
        chart_metadata = metadata.get("chart_metadata", {}) or {}

        location = (
            image_metadata.get("image_location", [])
            + table_metadata.get("table_location", [])
            + chart_metadata.get("chart_location", [])
        )

        if location:
            logger.debug("Extracted location is %s", location)

    return location


def get_unique_thumbnail_id_from_result(
    collection_name: str,
    file_name: str,
    page_number: int,
    location: list[float] | None = None,
    metadata: dict | None = None,
) -> str | None:
    """Generate unique thumbnail ID with fallback to metadata if location is absent."""
    try:
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

        if location is None:
            logger.warning(
                "Skipping page %s of %s: No location information found",
                page_number,
                file_name,
            )
            return None

        return get_unique_thumbnail_id(
            collection_name=collection_name,
            file_name=file_name,
            page_number=page_number,
            location=location,
        )
    except Exception as e:
        logger.warning("Failed to generate thumbnail ID for %s: %s", file_name, str(e))
        return None

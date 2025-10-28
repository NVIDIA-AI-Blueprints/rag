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

"""Module for handling summary generation status tracking.

This module provides Redis-based status tracking for document summary generation.
It enables cross-service communication between the ingestor server (which generates
summaries) and the RAG server (which retrieves them), allowing users to query the
status of summary generation tasks.

Classes:
    SummaryStatusHandler: Manages summary generation status using Redis.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any

from redis import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)

# Redis connection configuration constants
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2
REDIS_SOCKET_TIMEOUT_SECONDS = 2
REDIS_STATUS_TTL_SECONDS = 86400  # 24 hours
REDIS_MAX_CONNECTIONS = 50  # Connection pool size


class SummaryStatusHandler:
    """
    Handles summary generation status tracking using Redis.

    This handler provides cross-service status tracking for document summarization.
    Redis is used to enable communication between the ingestor server (which writes
    status during generation) and the RAG server (which reads status for retrieval).

    The handler automatically detects Redis availability and logs appropriate warnings
    if the connection cannot be established. Status tracking is transparently disabled
    if Redis is unavailable.

    Attributes:
        _redis_host (str): Redis server hostname from REDIS_HOST env var
        _redis_port (int): Redis server port from REDIS_PORT env var
        _redis_db (int): Redis database number from REDIS_DB env var
        _redis_available (bool): Whether Redis connection is active
        _redis_pool (ConnectionPool): Redis connection pool
        _redis_client (Redis): Redis client instance using the connection pool
    """

    _redis_host: str = os.getenv("REDIS_HOST", "localhost")
    _redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    _redis_db: int = int(os.getenv("REDIS_DB", 0))
    _redis_available: bool = False
    _redis_pool: ConnectionPool | None = None
    _redis_client: Redis | None = None

    def __init__(self):
        """Initialize Redis connection and test connectivity."""
        self._check_redis_connection()

    def _check_redis_connection(self) -> None:
        """Test Redis connection and create connection pool."""
        try:
            self._redis_pool = ConnectionPool(
                host=self._redis_host,
                port=self._redis_port,
                db=self._redis_db,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                max_connections=REDIS_MAX_CONNECTIONS,
                decode_responses=False,
            )

            self._redis_client = Redis(connection_pool=self._redis_pool)
            self._redis_client.ping()
            self._redis_available = True
            logger.info(
                f"Connected to Redis at {self._redis_host}:{self._redis_port} "
                f"(pool: {REDIS_MAX_CONNECTIONS})"
            )
        except (ConnectionError, RedisError, OSError) as e:
            self._redis_available = False
            self._redis_pool = None
            self._redis_client = None
            logger.warning(
                f"Redis unavailable at {self._redis_host}:{self._redis_port} - "
                f"summary status tracking disabled: {e}"
            )

    def is_available(self) -> bool:
        """
        Check if Redis is available for status tracking.

        Returns:
            bool: True if Redis connection is active, False otherwise
        """
        return self._redis_available

    def _get_key(self, collection_name: str, file_name: str) -> str:
        """
        Generate Redis key for summary status.

        Args:
            collection_name: Name of the document collection
            file_name: Name of the file (basename only)

        Returns:
            str: Redis key in format "summary_status:{collection}:{file}"
        """
        return f"summary_status:{collection_name}:{file_name}"

    def set_status(
        self, collection_name: str, file_name: str, status_data: dict[str, Any]
    ) -> bool:
        """
        Set summary status for a file.

        Args:
            collection_name: Name of the document collection
            file_name: Name of the file (basename only)
            status_data: Dictionary containing status information with fields:
                - status: One of PENDING, IN_PROGRESS, SUCCESS, FAILED
                - started_at: ISO format timestamp
                - error: Error message (if FAILED)
                - progress: Progress information (optional)

        Returns:
            bool: True if stored successfully, False if Redis unavailable or error
        """
        if not self._redis_available:
            logger.debug(
                f"Redis unavailable - skipping status update for {file_name} "
                f"in collection {collection_name}"
            )
            return False

        try:
            key = self._get_key(collection_name, file_name)
            self._redis_client.json().set(key, "$", status_data)
            # Set expiration for auto-cleanup of old status entries
            self._redis_client.expire(key, REDIS_STATUS_TTL_SECONDS)
            logger.debug(
                f"Set summary status for {file_name}: {status_data.get('status')}"
            )
            return True
        except (ConnectionError, RedisError, AttributeError) as e:
            logger.error(
                f"Failed to set summary status for {file_name}: {e}",
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            self._redis_available = False  # Mark as unavailable on error
            return False

    def get_status(self, collection_name: str, file_name: str) -> dict[str, Any] | None:
        """
        Get summary status for a file.

        Args:
            collection_name: Name of the document collection
            file_name: Name of the file (basename only)

        Returns:
            dict: Status data if found, None if not found or Redis unavailable
        """
        if not self._redis_available:
            logger.debug(
                f"Redis unavailable - cannot get status for {file_name} "
                f"in collection {collection_name}"
            )
            return None

        try:
            key = self._get_key(collection_name, file_name)
            status = self._redis_client.json().get(key)
            if status:
                logger.debug(
                    f"Retrieved summary status for {file_name}: "
                    f"{status.get('status') if isinstance(status, dict) else status}"
                )
            return status
        except (ConnectionError, RedisError, AttributeError) as e:
            logger.error(
                f"Failed to get summary status for {file_name}: {e}",
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            self._redis_available = False  # Mark as unavailable on error
            return None

    def update_progress(
        self,
        collection_name: str,
        file_name: str,
        status: str,
        error: str | None = None,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update summary status with optional error and progress information.

        Args:
            collection_name: Name of the document collection
            file_name: Name of the file (basename only)
            status: One of PENDING, IN_PROGRESS, SUCCESS, FAILED
            error: Error message (included if status is FAILED)
            progress: Progress information dictionary (optional)

        Returns:
            bool: True if updated successfully, False otherwise
        """
        if not self._redis_available:
            return False

        # Get current status or create new
        current = self.get_status(collection_name, file_name) or {}

        # Update fields
        current["status"] = status
        current["updated_at"] = datetime.now(UTC).isoformat()

        # Set started_at on first transition to IN_PROGRESS
        if status == "IN_PROGRESS" and "started_at" not in current:
            current["started_at"] = datetime.now(UTC).isoformat()

        if error:
            current["error"] = error

        if progress:
            current["progress"] = progress

        # Set completion timestamp for terminal states
        if status in ["SUCCESS", "FAILED"]:
            current["completed_at"] = datetime.now(UTC).isoformat()

        return self.set_status(collection_name, file_name, current)

    def get_redis_info(self) -> dict[str, Any]:
        """
        Get Redis connection information for debugging.

        Returns:
            dict: Connection details including host, port, db, and availability
        """
        return {
            "host": self._redis_host,
            "port": self._redis_port,
            "db": self._redis_db,
            "available": self._redis_available,
        }


# Singleton instance shared across the application
SUMMARY_STATUS_HANDLER = SummaryStatusHandler()

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
Unit tests for SummaryStatusHandler.

This module tests Redis-based status tracking for document summarization:
- Redis connection and availability detection
- Status CRUD operations (set, get, update)
- Progress tracking with chunk-level updates
- Error handling and graceful degradation
- TTL and expiration behavior
"""

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from nvidia_rag.utils.summary_status_handler import (
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_TIMEOUT_SECONDS,
    REDIS_STATUS_TTL_SECONDS,
    SummaryStatusHandler,
)


class TestSummaryStatusHandlerInitialization:
    """Tests for SummaryStatusHandler initialization and connection"""

    def test_successful_redis_connection(self):
        """Test successful Redis connection sets available flag to True"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            assert handler.is_available() is True
            mock_redis.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                decode_responses=False,
            )
            mock_client.ping.assert_called_once()

    def test_failed_redis_connection_sets_unavailable(self):
        """Test failed Redis connection sets available flag to False"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = RedisConnectionError("Connection refused")

            handler = SummaryStatusHandler()

            assert handler.is_available() is False
            assert handler._redis_client is None

    def test_redis_error_sets_unavailable(self):
        """Test Redis error during ping sets available flag to False"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.side_effect = RedisError("Redis error")
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            assert handler.is_available() is False

    def test_redis_oserror_sets_unavailable(self):
        """Test OSError during Redis connection sets available flag to False"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = OSError("Connection failed")

            handler = SummaryStatusHandler()

            assert handler.is_available() is False
            assert handler._redis_client is None


class TestSummaryStatusHandlerKeyGeneration:
    """Tests for Redis key generation"""

    def test_get_key_format(self):
        """Test Redis key format is correct"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis"):
            handler = SummaryStatusHandler()

            key = handler._get_key("my_collection", "document.pdf")

            assert key == "summary_status:my_collection:document.pdf"

    def test_get_key_with_special_characters(self):
        """Test key generation with special characters"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis"):
            handler = SummaryStatusHandler()

            key = handler._get_key("collection-name_123", "file name (1).pdf")

            assert key == "summary_status:collection-name_123:file name (1).pdf"


class TestSummaryStatusHandlerSetStatus:
    """Tests for set_status method"""

    def test_set_status_success(self):
        """Test successful status set operation"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()
            status_data = {
                "status": "PENDING",
                "file_name": "test.pdf",
                "collection_name": "test_col",
            }

            result = handler.set_status("test_col", "test.pdf", status_data)

            assert result is True
            mock_json.set.assert_called_once_with(
                "summary_status:test_col:test.pdf", "$", status_data
            )
            mock_client.expire.assert_called_once_with(
                "summary_status:test_col:test.pdf", REDIS_STATUS_TTL_SECONDS
            )

    def test_set_status_when_redis_unavailable(self):
        """Test set_status returns False when Redis is unavailable"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = RedisConnectionError("Connection refused")

            handler = SummaryStatusHandler()
            status_data = {"status": "PENDING"}

            result = handler.set_status("test_col", "test.pdf", status_data)

            assert result is False

    def test_set_status_handles_redis_error(self):
        """Test set_status handles Redis errors gracefully"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.set.side_effect = RedisError("Set failed")
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()
            status_data = {"status": "PENDING"}

            result = handler.set_status("test_col", "test.pdf", status_data)

            assert result is False
            assert handler.is_available() is False

    def test_set_status_with_complex_data(self):
        """Test set_status with complex status data including timestamps and progress"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()
            status_data = {
                "status": "IN_PROGRESS",
                "started_at": "2025-01-24T10:30:00.000Z",
                "updated_at": "2025-01-24T10:30:15.000Z",
                "progress": {
                    "current": 3,
                    "total": 5,
                    "message": "Processing chunk 3/5",
                },
            }

            result = handler.set_status("test_col", "test.pdf", status_data)

            assert result is True
            mock_json.set.assert_called_once()


class TestSummaryStatusHandlerGetStatus:
    """Tests for get_status method"""

    def test_get_status_success(self):
        """Test successful status retrieval"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            status_data = {
                "status": "SUCCESS",
                "completed_at": "2025-01-24T10:35:00.000Z",
            }
            mock_json.get.return_value = status_data
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.get_status("test_col", "test.pdf")

            assert result == status_data
            mock_json.get.assert_called_once_with("summary_status:test_col:test.pdf")

    def test_get_status_not_found(self):
        """Test get_status returns None when key not found"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = None
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.get_status("test_col", "test.pdf")

            assert result is None

    def test_get_status_when_redis_unavailable(self):
        """Test get_status returns None when Redis is unavailable"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = RedisConnectionError("Connection refused")

            handler = SummaryStatusHandler()

            result = handler.get_status("test_col", "test.pdf")

            assert result is None

    def test_get_status_handles_redis_error(self):
        """Test get_status handles Redis errors gracefully"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.side_effect = RedisError("Get failed")
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.get_status("test_col", "test.pdf")

            assert result is None
            assert handler.is_available() is False


class TestSummaryStatusHandlerUpdateProgress:
    """Tests for update_progress method"""

    def test_update_progress_creates_new_status(self):
        """Test update_progress creates new status if none exists"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = None  # No existing status
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="IN_PROGRESS",
                progress={"current": 1, "total": 5},
            )

            assert result is True
            # Verify set was called with new status data
            call_args = mock_json.set.call_args
            assert call_args[0][0] == "summary_status:test_col:test.pdf"
            status_data = call_args[0][2]
            assert status_data["status"] == "IN_PROGRESS"
            assert status_data["progress"] == {"current": 1, "total": 5}
            assert "updated_at" in status_data

    def test_update_progress_updates_existing_status(self):
        """Test update_progress updates existing status"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            existing_status = {
                "status": "IN_PROGRESS",
                "started_at": "2025-01-24T10:30:00.000Z",
                "progress": {"current": 1, "total": 5},
            }
            mock_json.get.return_value = existing_status
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="IN_PROGRESS",
                progress={"current": 2, "total": 5},
            )

            assert result is True
            call_args = mock_json.set.call_args
            status_data = call_args[0][2]
            assert status_data["progress"]["current"] == 2
            assert status_data["started_at"] == "2025-01-24T10:30:00.000Z"  # Preserved

    def test_update_progress_adds_started_at_for_in_progress(self):
        """Test update_progress adds started_at when transitioning to IN_PROGRESS"""
        with (
            patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis,
            patch("nvidia_rag.utils.summary_status_handler.datetime") as mock_datetime,
        ):
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = {"status": "PENDING"}
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            # Mock datetime
            mock_now = Mock()
            mock_now.isoformat.return_value = "2025-01-24T10:30:00.000Z"
            mock_datetime.now.return_value = mock_now

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="IN_PROGRESS",
            )

            assert result is True
            call_args = mock_json.set.call_args
            status_data = call_args[0][2]
            assert "started_at" in status_data

    def test_update_progress_adds_completed_at_for_success(self):
        """Test update_progress adds completed_at for SUCCESS status"""
        with (
            patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis,
            patch("nvidia_rag.utils.summary_status_handler.datetime") as mock_datetime,
        ):
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = {"status": "IN_PROGRESS"}
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            # Mock datetime
            mock_now = Mock()
            mock_now.isoformat.return_value = "2025-01-24T10:35:00.000Z"
            mock_datetime.now.return_value = mock_now

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="SUCCESS",
            )

            assert result is True
            call_args = mock_json.set.call_args
            status_data = call_args[0][2]
            assert "completed_at" in status_data

    def test_update_progress_adds_completed_at_for_failed(self):
        """Test update_progress adds completed_at for FAILED status"""
        with (
            patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis,
            patch("nvidia_rag.utils.summary_status_handler.datetime") as mock_datetime,
        ):
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = {"status": "IN_PROGRESS"}
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            # Mock datetime
            mock_now = Mock()
            mock_now.isoformat.return_value = "2025-01-24T10:35:00.000Z"
            mock_datetime.now.return_value = mock_now

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="FAILED",
                error="LLM connection timeout",
            )

            assert result is True
            call_args = mock_json.set.call_args
            status_data = call_args[0][2]
            assert "completed_at" in status_data
            assert status_data["error"] == "LLM connection timeout"

    def test_update_progress_when_redis_unavailable(self):
        """Test update_progress returns False when Redis is unavailable"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = RedisConnectionError("Connection refused")

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="IN_PROGRESS",
            )

            assert result is False


class TestSummaryStatusHandlerGetRedisInfo:
    """Tests for get_redis_info method"""

    def test_get_redis_info_returns_correct_structure(self):
        """Test get_redis_info returns correct information structure"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            info = handler.get_redis_info()

            assert "host" in info
            assert "port" in info
            assert "db" in info
            assert "available" in info
            assert info["host"] == "localhost"
            assert info["port"] == 6379
            assert info["db"] == 0
            assert info["available"] is True

    def test_get_redis_info_when_unavailable(self):
        """Test get_redis_info shows unavailable status correctly"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_redis.side_effect = RedisConnectionError("Connection refused")

            handler = SummaryStatusHandler()

            info = handler.get_redis_info()

            assert info["available"] is False


class TestSummaryStatusHandlerEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_handler_with_empty_collection_name(self):
        """Test handler operations with empty collection name"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            key = handler._get_key("", "test.pdf")
            assert key == "summary_status::test.pdf"

    def test_handler_with_empty_file_name(self):
        """Test handler operations with empty file name"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            key = handler._get_key("test_col", "")
            assert key == "summary_status:test_col:"

    def test_set_status_with_none_values(self):
        """Test set_status handles None values in status data"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()
            status_data = {
                "status": "PENDING",
                "error": None,
                "progress": None,
            }

            result = handler.set_status("test_col", "test.pdf", status_data)

            assert result is True

    def test_update_progress_with_special_characters_in_error(self):
        """Test update_progress handles special characters in error message"""
        with patch("nvidia_rag.utils.summary_status_handler.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_json = MagicMock()
            mock_json.get.return_value = None
            mock_client.json.return_value = mock_json
            mock_redis.return_value = mock_client

            handler = SummaryStatusHandler()

            result = handler.update_progress(
                collection_name="test_col",
                file_name="test.pdf",
                status="FAILED",
                error="Error: 'Connection' to \"server\" failed\n\t(timeout)",
            )

            assert result is True

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

"""Unit tests for task_handler.py."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nvidia_rag.ingestor_server.task_handler import (
    INGESTION_TASK_HANDLER,
    IngestionTaskHandler,
)


class TestIngestionTaskHandler:
    """Test cases for IngestionTaskHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a fresh IngestionTaskHandler instance."""
        return IngestionTaskHandler()

    @pytest.mark.asyncio
    async def test_submit_task_success(self, handler):
        """Test successful task submission."""

        started = asyncio.Event()
        block = asyncio.Event()

        async def mock_task():
            started.set()
            await block.wait()
            return {"result": "success"}

        task_id = await handler.submit_task(mock_task)

        assert task_id is not None
        assert task_id in handler.task_map
        assert handler.get_task_state(task_id) == "PENDING"

        # Wait for task to start and block, then unblock it
        await started.wait()
        block.set()

    @pytest.mark.asyncio
    async def test_submit_task_with_custom_id(self, handler):
        """Test task submission with custom task ID."""

        async def mock_task():
            return {"result": "success"}

        custom_id = "custom-task-id-123"
        task_id = await handler.submit_task(mock_task, task_id=custom_id)

        assert task_id == custom_id
        assert task_id in handler.task_map

    @pytest.mark.asyncio
    async def test_execute_ingestion_task_success(self, handler):
        """Test successful task execution."""

        async def mock_task():
            return {"result": "success"}

        task_id = "test-task-123"
        result = await handler._execute_ingestion_task(task_id, mock_task)

        assert result == {"result": "success"}
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FINISHED"
        assert status_result["result"] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_execute_ingestion_task_failure(self, handler):
        """Test task execution failure handling."""

        async def mock_task():
            raise ValueError("Task failed")

        task_id = "test-task-123"

        with pytest.raises(ValueError, match="Task failed"):
            await handler._execute_ingestion_task(task_id, mock_task)

        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FAILED"
        assert "message" in status_result["result"]

    @pytest.mark.asyncio
    async def test_set_task_status_and_result(self, handler):
        """Test setting task status and result."""
        task_id = "test-task-123"
        status = "RUNNING"
        result = {"progress": 50}

        await handler.set_task_status_and_result(task_id, status, result)

        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == status
        assert status_result["result"] == result

    def test_get_task_state(self, handler):
        """Test getting task state."""
        task_id = "test-task-123"
        handler.task_status_result_map[task_id] = {"state": "RUNNING"}

        state = handler.get_task_state(task_id)

        assert state == "RUNNING"

    def test_get_task_status_and_result(self, handler):
        """Test getting task status and result."""
        task_id = "test-task-123"
        expected = {"state": "FINISHED", "result": {"data": "test"}}
        handler.task_status_result_map[task_id] = expected

        result = handler.get_task_status_and_result(task_id)

        assert result == expected

    def test_get_task_result(self, handler):
        """Test getting task result."""
        task_id = "test-task-123"
        handler.task_status_result_map[task_id] = {
            "state": "FINISHED",
            "result": {"data": "test"},
        }

        result = handler.get_task_result(task_id)

        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_set_task_state_dict(self, handler):
        """Test setting task state dictionary."""
        task_id = "test-task-123"
        state_dict = {"key1": "value1", "key2": "value2"}

        await handler.set_task_state_dict(task_id, state_dict)

        result = handler.get_task_state_dict(task_id)
        assert result == state_dict

    def test_get_task_state_dict(self, handler):
        """Test getting task state dictionary."""
        task_id = "test-task-123"
        state_dict = {"key1": "value1"}
        handler.task_state_map[task_id] = state_dict

        result = handler.get_task_state_dict(task_id)

        assert result == state_dict

    def test_get_task_state_dict_not_found(self, handler):
        """Test getting task state dictionary when not found."""
        result = handler.get_task_state_dict("nonexistent-task")

        assert result == {}


class TestIngestionTaskHandlerRedisBackend:
    """Test cases for IngestionTaskHandler with Redis backend enabled."""

    @pytest.mark.asyncio
    async def test_submit_task_with_redis(self):
        """Test task submission with Redis backend enabled."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        async def mock_task():
            return {"result": "success"}

        task_id = await handler.submit_task(mock_task)

        assert task_id is not None
        mock_json.set.assert_called_once()

    def test_get_task_state_with_redis(self):
        """Test getting task state with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_json.get.return_value = {"state": "RUNNING"}
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        state = handler.get_task_state("test-task-123")

        assert state == "RUNNING"
        mock_json.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_task_status_and_result_with_redis(self):
        """Test setting task status and result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        await handler.set_task_status_and_result(
            "test-task-123", "FINISHED", {"result": "success"}
        )

        mock_json.set.assert_called_once()

    def test_get_task_status_and_result_with_redis(self):
        """Test getting task status and result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_json.get.return_value = {"state": "FINISHED", "result": {"data": "test"}}
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_status_and_result("test-task-123")

        assert result == {"state": "FINISHED", "result": {"data": "test"}}
        mock_json.get.assert_called_once()

    def test_get_task_result_with_redis(self):
        """Test getting task result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_json.get.return_value = {"state": "FINISHED", "result": {"data": "test"}}
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_result("test-task-123")

        assert result == {"data": "test"}
        assert mock_json.get.call_count == 1

    @pytest.mark.asyncio
    async def test_set_task_state_dict_with_redis(self):
        """Test setting task state dictionary with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        await handler.set_task_state_dict("test-task-123", {"key": "value"})

        mock_json.set.assert_called_once()

    def test_get_task_state_dict_with_redis(self):
        """Test getting task state dictionary with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_json.get.return_value = {"state_dict": {"key": "value"}}
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_state_dict("test-task-123")

        assert result == {"key": "value"}

    def test_get_task_state_dict_with_redis_not_found(self):
        """Test getting task state dictionary with Redis when not found."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_json = MagicMock()
        mock_json.get.return_value = None
        mock_client.json.return_value = mock_json
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_state_dict("nonexistent-task")

        assert result == {}


class TestIngestionTaskHandlerSingleton:
    """Test cases for INGESTION_TASK_HANDLER singleton."""

    def test_singleton_instance_exists(self):
        """Test that INGESTION_TASK_HANDLER singleton exists."""
        assert INGESTION_TASK_HANDLER is not None
        assert isinstance(INGESTION_TASK_HANDLER, IngestionTaskHandler)

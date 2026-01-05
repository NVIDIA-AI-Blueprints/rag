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
import json
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nvidia_rag.ingestor_server.task_handler import (
    INGESTION_TASK_HANDLER,
    IngestionTaskHandler,
    IngestionTaskStateSchema,
    TaskStateDictSchema,
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

    @pytest.mark.asyncio
    async def test_multiple_concurrent_task_submissions(self, handler):
        """Test submitting multiple tasks concurrently."""
        
        async def mock_task(task_num):
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return {"task": task_num, "result": "success"}
        
        # Submit 5 tasks concurrently
        tasks = []
        for i in range(5):
            task_id = await handler.submit_task(lambda i=i: mock_task(i))
            tasks.append(task_id)
        
        # Verify all tasks are in the task map
        assert len(tasks) == 5
        for task_id in tasks:
            assert task_id in handler.task_map
            assert handler.get_task_state(task_id) == "PENDING"

    @pytest.mark.asyncio
    async def test_task_lifecycle_from_pending_to_finished(self, handler):
        """Test complete task lifecycle from PENDING to FINISHED."""
        
        async def mock_task():
            await asyncio.sleep(0.01)
            return {"result": "completed"}
        
        task_id = await handler.submit_task(mock_task)
        
        # Initially should be PENDING
        assert handler.get_task_state(task_id) == "PENDING"
        
        # Wait for task to complete
        await handler.task_map[task_id]
        
        # Should now be FINISHED
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FINISHED"
        assert status_result["result"] == {"result": "completed"}

    @pytest.mark.asyncio
    async def test_task_lifecycle_from_pending_to_failed(self, handler):
        """Test complete task lifecycle from PENDING to FAILED."""
        
        async def failing_task():
            await asyncio.sleep(0.01)
            raise RuntimeError("Task failed intentionally")
        
        task_id = await handler.submit_task(failing_task)
        
        # Initially should be PENDING
        assert handler.get_task_state(task_id) == "PENDING"
        
        # Wait for task to complete (it will fail)
        with pytest.raises(RuntimeError):
            await handler.task_map[task_id]
        
        # Should now be FAILED
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FAILED"
        assert "message" in status_result["result"]
        assert "Task failed intentionally" in status_result["result"]["message"]

    @pytest.mark.asyncio
    async def test_multiple_tasks_with_different_outcomes(self, handler):
        """Test multiple tasks with some succeeding and some failing."""
        
        async def success_task(num):
            await asyncio.sleep(0.01)
            return {"task": num, "status": "success"}
        
        async def failure_task(num):
            await asyncio.sleep(0.01)
            raise ValueError(f"Task {num} failed")
        
        # Submit 3 successful and 2 failing tasks
        success_ids = []
        failure_ids = []
        
        for i in range(3):
            task_id = await handler.submit_task(lambda i=i: success_task(i))
            success_ids.append(task_id)
        
        for i in range(2):
            task_id = await handler.submit_task(lambda i=i: failure_task(i))
            failure_ids.append(task_id)
        
        # Wait for all tasks to complete
        for task_id in success_ids:
            await handler.task_map[task_id]
        
        for task_id in failure_ids:
            with pytest.raises(ValueError):
                await handler.task_map[task_id]
        
        # Verify successful tasks
        for task_id in success_ids:
            status_result = handler.get_task_status_and_result(task_id)
            assert status_result["state"] == "FINISHED"
            assert status_result["result"]["status"] == "success"
        
        # Verify failed tasks
        for task_id in failure_ids:
            status_result = handler.get_task_status_and_result(task_id)
            assert status_result["state"] == "FAILED"
            assert "message" in status_result["result"]

    @pytest.mark.asyncio
    async def test_concurrent_status_updates(self, handler):
        """Test concurrent status and result updates to the same task."""
        
        task_id = "concurrent-test-task"
        
        async def update_status(status_num):
            await handler.set_task_status_and_result(
                task_id, f"STATUS_{status_num}", {"update": status_num}
            )
        
        # Perform multiple concurrent updates
        await asyncio.gather(*[update_status(i) for i in range(10)])
        
        # Verify final state exists (one of the updates succeeded)
        status_result = handler.get_task_status_and_result(task_id)
        assert "state" in status_result
        assert status_result["state"].startswith("STATUS_")
        assert "result" in status_result

    @pytest.mark.asyncio
    async def test_concurrent_state_dict_updates(self, handler):
        """Test concurrent state dictionary updates."""
        
        task_id = "state-dict-test-task"
        
        async def update_state_dict(key, value):
            current = handler.get_task_state_dict(task_id)
            current[key] = value
            await handler.set_task_state_dict(task_id, current)
        
        # Perform multiple concurrent updates with different keys
        await asyncio.gather(*[
            update_state_dict(f"key_{i}", f"value_{i}") for i in range(5)
        ])
        
        # Verify state dict has entries
        state_dict = handler.get_task_state_dict(task_id)
        assert isinstance(state_dict, dict)

    @pytest.mark.asyncio
    async def test_update_task_status_multiple_times(self, handler):
        """Test updating task status multiple times sequentially."""
        
        task_id = "multi-update-task"
        
        # Update through different states
        await handler.set_task_status_and_result(task_id, "PENDING", {})
        assert handler.get_task_state(task_id) == "PENDING"
        
        await handler.set_task_status_and_result(task_id, "RUNNING", {"progress": 25})
        assert handler.get_task_state(task_id) == "RUNNING"
        assert handler.get_task_result(task_id)["progress"] == 25
        
        await handler.set_task_status_and_result(task_id, "RUNNING", {"progress": 75})
        assert handler.get_task_state(task_id) == "RUNNING"
        assert handler.get_task_result(task_id)["progress"] == 75
        
        await handler.set_task_status_and_result(task_id, "FINISHED", {"progress": 100})
        assert handler.get_task_state(task_id) == "FINISHED"
        assert handler.get_task_result(task_id)["progress"] == 100

    @pytest.mark.asyncio
    async def test_task_with_complex_result_data(self, handler):
        """Test task with complex nested result data."""
        
        async def complex_task():
            return {
                "data": {
                    "nested": {
                        "deeply": {
                            "value": 123
                        }
                    },
                    "list": [1, 2, 3, 4, 5],
                    "mixed": {
                        "strings": ["a", "b", "c"],
                        "numbers": [1.1, 2.2, 3.3]
                    }
                },
                "metadata": {
                    "processed": True,
                    "count": 42
                }
            }
        
        task_id = await handler.submit_task(complex_task)
        await handler.task_map[task_id]
        
        result = handler.get_task_result(task_id)
        assert result["data"]["nested"]["deeply"]["value"] == 123
        assert result["data"]["list"] == [1, 2, 3, 4, 5]
        assert result["metadata"]["count"] == 42

    @pytest.mark.asyncio
    async def test_state_dict_with_nested_data(self, handler):
        """Test state dictionary with nested complex data."""
        
        task_id = "nested-state-task"
        state_dict = {
            "config": {
                "params": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                }
            },
            "progress": {
                "current_step": 150,
                "total_steps": 1000,
                "metrics": {
                    "loss": 0.45,
                    "accuracy": 0.92
                }
            }
        }
        
        await handler.set_task_state_dict(task_id, state_dict)
        
        retrieved = handler.get_task_state_dict(task_id)
        assert retrieved["config"]["params"]["batch_size"] == 32
        assert retrieved["progress"]["metrics"]["accuracy"] == 0.92

    @pytest.mark.asyncio
    async def test_execute_ingestion_task_with_exception_details(self, handler):
        """Test that exception details are properly captured."""
        
        async def task_with_detailed_error():
            # Simulate a complex error scenario
            try:
                x = 10 / 0
            except ZeroDivisionError as e:
                raise RuntimeError(f"Math error occurred: {e}")
        
        task_id = "error-detail-task"
        
        with pytest.raises(RuntimeError, match="Math error occurred"):
            await handler._execute_ingestion_task(task_id, task_with_detailed_error)
        
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FAILED"
        assert "Math error occurred" in status_result["result"]["message"]


class TestIngestionTaskHandlerRedisBackend:
    """Test cases for IngestionTaskHandler with Redis backend enabled."""

    @pytest.mark.asyncio
    async def test_submit_task_with_redis(self):
        """Test task submission with Redis backend enabled."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        async def mock_task():
            return {"result": "success"}

        task_id = await handler.submit_task(mock_task)

        assert task_id is not None
        mock_client.set.assert_called_once()

    def test_get_task_state_with_redis(self):
        """Test getting task state with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"state": "RUNNING"})
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        state = handler.get_task_state("test-task-123")

        assert state == "RUNNING"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_task_status_and_result_with_redis(self):
        """Test setting task status and result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        await handler.set_task_status_and_result(
            "test-task-123", "FINISHED", {"result": "success"}
        )

        mock_client.set.assert_called_once()

    def test_get_task_status_and_result_with_redis(self):
        """Test getting task status and result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"state": "FINISHED", "result": {"data": "test"}})
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_status_and_result("test-task-123")

        assert result == {"state": "FINISHED", "result": {"data": "test"}}
        mock_client.get.assert_called_once()

    def test_get_task_result_with_redis(self):
        """Test getting task result with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"state": "FINISHED", "result": {"data": "test"}})
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_result("test-task-123")

        assert result == {"data": "test"}
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_set_task_state_dict_with_redis(self):
        """Test setting task state dictionary with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        await handler.set_task_state_dict("test-task-123", {"key": "value"})

        mock_client.set.assert_called_once()

    def test_get_task_state_dict_with_redis(self):
        """Test getting task state dictionary with Redis backend."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"state_dict": {"key": "value"}})
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_state_dict("test-task-123")

        assert result == {"key": "value"}

    def test_get_task_state_dict_with_redis_not_found(self):
        """Test getting task state dictionary with Redis when not found."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = None
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        result = handler.get_task_state_dict("nonexistent-task")

        assert result == {}

    @pytest.mark.asyncio
    async def test_redis_state_dict_key_format(self):
        """Test that Redis state dict uses correct key format."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        task_id = "test-task-123"
        await handler.set_task_state_dict(task_id, {"key": "value"})

        # Verify the key format includes :state_dict suffix
        calls = mock_client.set.call_args_list
        assert len(calls) == 1
        call_args = calls[0][0]
        assert call_args[0] == f"{task_id}:state_dict"

    def test_redis_get_task_state_dict_key_format(self):
        """Test that Redis get state dict uses correct key format."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"state_dict": {"key": "value"}})
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        task_id = "test-task-123"
        result = handler.get_task_state_dict(task_id)

        # Verify the correct key format was used
        mock_client.get.assert_called_once_with(f"{task_id}:state_dict")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_redis_concurrent_operations(self):
        """Test concurrent Redis operations."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        async def mock_task():
            return {"result": "success"}

        # Submit multiple tasks concurrently
        tasks = await asyncio.gather(*[
            handler.submit_task(mock_task) for _ in range(5)
        ])

        assert len(tasks) == 5
        # Verify Redis set was called for each task (2 times per task: PENDING + FINISHED)
        assert mock_client.set.call_count == 10  # 5 tasks * 2 states each

    @pytest.mark.asyncio
    async def test_execute_task_with_redis_updates_state(self):
        """Test that task execution updates Redis with final state."""
        handler = IngestionTaskHandler()

        mock_client = MagicMock()
        handler._enable_redis_backend = True
        handler._redis_client = mock_client

        async def mock_task():
            return {"data": "completed"}

        task_id = "test-task-redis"
        await handler._execute_ingestion_task(task_id, mock_task)

        # Verify Redis was updated with FINISHED state
        calls = mock_client.set.call_args_list
        # Last call should be the FINISHED state
        last_call = calls[-1]
        assert last_call[0][0] == task_id
        state_data = json.loads(last_call[0][1])
        assert state_data["state"] == "FINISHED"
        assert state_data["result"] == {"data": "completed"}


class TestIngestionTaskHandlerSingleton:
    """Test cases for INGESTION_TASK_HANDLER singleton."""

    def test_singleton_instance_exists(self):
        """Test that INGESTION_TASK_HANDLER singleton exists."""
        assert INGESTION_TASK_HANDLER is not None
        assert isinstance(INGESTION_TASK_HANDLER, IngestionTaskHandler)


class TestSchemaModels:
    """Test cases for schema models."""

    def test_ingestion_task_state_schema_creation(self):
        """Test IngestionTaskStateSchema model creation."""
        schema = IngestionTaskStateSchema(
            task_id="test-123",
            state="PENDING",
            result={"data": "test"}
        )

        assert schema.task_id == "test-123"
        assert schema.state == "PENDING"
        assert schema.result == {"data": "test"}

    def test_ingestion_task_state_schema_default_result(self):
        """Test IngestionTaskStateSchema with default empty result."""
        schema = IngestionTaskStateSchema(
            task_id="test-456",
            state="RUNNING"
        )

        assert schema.task_id == "test-456"
        assert schema.state == "RUNNING"
        assert schema.result == {}

    def test_ingestion_task_state_schema_model_dump(self):
        """Test IngestionTaskStateSchema model_dump method."""
        schema = IngestionTaskStateSchema(
            task_id="test-789",
            state="FINISHED",
            result={"status": "success"}
        )

        dumped = schema.model_dump()
        assert dumped["task_id"] == "test-789"
        assert dumped["state"] == "FINISHED"
        assert dumped["result"] == {"status": "success"}

    def test_task_state_dict_schema_creation(self):
        """Test TaskStateDictSchema model creation."""
        schema = TaskStateDictSchema(
            task_id="test-dict-123",
            state_dict={"key1": "value1", "key2": "value2"}
        )

        assert schema.task_id == "test-dict-123"
        assert schema.state_dict == {"key1": "value1", "key2": "value2"}

    def test_task_state_dict_schema_default_state_dict(self):
        """Test TaskStateDictSchema with default empty state_dict."""
        schema = TaskStateDictSchema(task_id="test-dict-456")

        assert schema.task_id == "test-dict-456"
        assert schema.state_dict == {}

    def test_task_state_dict_schema_nested_data(self):
        """Test TaskStateDictSchema with nested data."""
        schema = TaskStateDictSchema(
            task_id="test-nested",
            state_dict={
                "level1": {
                    "level2": {
                        "level3": "deep_value"
                    }
                },
                "arrays": [1, 2, 3]
            }
        )

        assert schema.state_dict["level1"]["level2"]["level3"] == "deep_value"
        assert schema.state_dict["arrays"] == [1, 2, 3]

    def test_task_state_dict_schema_model_dump(self):
        """Test TaskStateDictSchema model_dump method."""
        schema = TaskStateDictSchema(
            task_id="test-dump",
            state_dict={"config": {"timeout": 30}}
        )

        dumped = schema.model_dump()
        assert dumped["task_id"] == "test-dump"
        assert dumped["state_dict"]["config"]["timeout"] == 30


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    @pytest.fixture
    def handler(self):
        """Create a fresh IngestionTaskHandler instance."""
        return IngestionTaskHandler()

    @pytest.mark.asyncio
    async def test_task_with_empty_result(self, handler):
        """Test task that returns empty result."""
        
        async def empty_task():
            return {}
        
        task_id = await handler.submit_task(empty_task)
        await handler.task_map[task_id]
        
        result = handler.get_task_result(task_id)
        assert result == {}

    @pytest.mark.asyncio
    async def test_task_with_none_result(self, handler):
        """Test task that returns None causes validation error."""
        
        async def none_task():
            return None
        
        task_id = await handler.submit_task(none_task)
        
        # Task should fail because None is not a valid dict result
        with pytest.raises(Exception):  # ValidationError
            await handler.task_map[task_id]
        
        # Task should be marked as FAILED
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FAILED"
        assert "validation error" in status_result["result"]["message"].lower()

    @pytest.mark.asyncio
    async def test_rapid_task_submissions(self, handler):
        """Test rapid successive task submissions."""
        
        async def quick_task(num):
            return {"num": num}
        
        # Submit 100 tasks rapidly
        task_ids = []
        for i in range(100):
            task_id = await handler.submit_task(lambda i=i: quick_task(i))
            task_ids.append(task_id)
        
        # All tasks should be tracked
        assert len(task_ids) == 100
        assert len(set(task_ids)) == 100  # All unique

    @pytest.mark.asyncio
    async def test_task_with_very_long_execution_time(self, handler):
        """Test task with longer execution time."""
        
        async def slow_task():
            await asyncio.sleep(0.1)  # Simulate longer operation
            return {"status": "completed after delay"}
        
        task_id = await handler.submit_task(slow_task)
        
        # Task should still be PENDING initially
        assert handler.get_task_state(task_id) == "PENDING"
        
        # Wait for completion
        await handler.task_map[task_id]
        
        # Should be FINISHED now
        assert handler.get_task_state(task_id) == "FINISHED"

    @pytest.mark.asyncio
    async def test_concurrent_access_to_same_task_state(self, handler):
        """Test concurrent reads of the same task state."""
        
        task_id = "concurrent-read-task"
        await handler.set_task_status_and_result(
            task_id, "RUNNING", {"progress": 50}
        )
        
        # Perform concurrent reads
        results = await asyncio.gather(*[
            asyncio.to_thread(handler.get_task_state, task_id) for _ in range(10)
        ])
        
        # All reads should return the same state
        assert all(state == "RUNNING" for state in results)

    @pytest.mark.asyncio
    async def test_state_dict_overwrite(self, handler):
        """Test that state dict overwrites previous value."""
        
        task_id = "overwrite-test"
        
        # Set initial state
        await handler.set_task_state_dict(task_id, {"version": 1, "data": "old"})
        assert handler.get_task_state_dict(task_id) == {"version": 1, "data": "old"}
        
        # Overwrite with new state
        await handler.set_task_state_dict(task_id, {"version": 2, "data": "new"})
        result = handler.get_task_state_dict(task_id)
        assert result["version"] == 2
        assert result["data"] == "new"

    @pytest.mark.asyncio
    async def test_task_exception_with_custom_exception_type(self, handler):
        """Test task failure with custom exception type."""
        
        class CustomTaskError(Exception):
            pass
        
        async def custom_error_task():
            raise CustomTaskError("Custom error occurred")
        
        task_id = "custom-error-task"
        
        with pytest.raises(CustomTaskError, match="Custom error occurred"):
            await handler._execute_ingestion_task(task_id, custom_error_task)
        
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["state"] == "FAILED"
        assert "Custom error occurred" in status_result["result"]["message"]

    @pytest.mark.asyncio
    async def test_multiple_state_updates_preserve_task_id(self, handler):
        """Test that multiple state updates preserve task ID."""
        
        task_id = "preserve-id-test"
        
        # Multiple updates
        for i in range(5):
            await handler.set_task_status_and_result(
                task_id, f"STATE_{i}", {"update": i}
            )
        
        # Task ID should still be accessible
        status_result = handler.get_task_status_and_result(task_id)
        assert status_result["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_empty_state_dict_update(self, handler):
        """Test updating state dict with empty dictionary."""
        
        task_id = "empty-dict-test"
        
        # Set initial state
        await handler.set_task_state_dict(task_id, {"has": "data"})
        assert handler.get_task_state_dict(task_id) == {"has": "data"}
        
        # Update with empty dict
        await handler.set_task_state_dict(task_id, {})
        result = handler.get_task_state_dict(task_id)
        assert result == {}

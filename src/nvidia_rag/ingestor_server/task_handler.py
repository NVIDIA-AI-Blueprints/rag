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
Module for handling ingestion tasks.

This module is responsible for handling ingestion tasks.
It is used to submit tasks to the task handler and get the status and result of tasks.
1. IngestionTaskHandler: A class that handles ingestion tasks.
2. IngestionTaskStateSchema: A class that defines the schema of the Redis database.
3. TaskStateDictSchema: A class that defines the schema for task state dictionary operations.
4. submit_task: A method that submits a task to the task handler.
5. get_task_status: A method that gets the status of a task.
6. update_task_status: A method that updates the status of a task.
7. get_task_result: A method that gets the result of a task.
8. set_task_state_dict: A method that sets a custom state dictionary for a task.
9. get_task_state_dict: A method that gets the custom state dictionary for a task.
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from redis import Redis

logger = logging.getLogger(__name__)


class IngestionTaskStateSchema(BaseModel):
    """
    A class that defines the schema of the ingestion task state.
    """

    task_id: str
    state: str
    result: dict[str, Any] = {}


class TaskStateDictSchema(BaseModel):
    """
    A class that defines the schema for task state dictionary operations.
    Used for setting and getting custom task state data as a dictionary.
    """

    task_id: str
    state_dict: dict[str, Any] = {}


class IngestionTaskHandler:
    """
    A class that handles ingestion tasks.
    Responsible for submitting tasks to the task handler and getting the status and result of tasks.
    """

    # Redis configuration
    _redis_host: str = os.getenv("REDIS_HOST", "localhost")
    _redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    _redis_db: int = int(os.getenv("REDIS_DB", 0))
    _enable_redis_backend: bool = os.getenv(
        "ENABLE_REDIS_BACKEND", "False"
    ).lower() in ["true", "True"]

    # Asyncio lock to synchronize access to the task map
    _asyncio_lock = asyncio.Lock()

    if _enable_redis_backend:
        logger.info(
            f"Initializing Redis client with host {_redis_host}, port {_redis_port}, db {_redis_db}"
        )
        # Initialize the Redis client
        _redis_client: Redis = Redis(host=_redis_host, port=_redis_port, db=_redis_db)
    else:
        logger.info("Redis backend is disabled")
        _redis_client = None

    def __init__(self):
        # Local task map to store tasks
        self.task_map = {}  # {task_id: asyncio_task}
        self.task_status_result_map = {}  # {task_id: (status, result)}
        self.task_state_map = {}  # {task_id: state}

    async def _execute_ingestion_task(self, task_id: str, function: Callable):
        """
        Execute the ingestion task and update Redis with the result.
        Args:
            task_id: The id of the task.
            function: The function to execute.
        """
        try:
            result = await function()
            logger.info(f"Task {task_id} completed using IngestionTaskHandler")
            await self.set_task_status_and_result(task_id, "FINISHED", result)
            return result
        except Exception as e:
            await self.set_task_status_and_result(
                task_id, "FAILED", {"message": str(e)}
            )
            logger.error(
                f"Task {task_id} failed using IngestionTaskHandler with error: {e}",
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
            raise e

    async def submit_task(self, function: Callable, task_id: str = None):
        """
        Submit a task to the task handler.
        Args:
            function: The async function to submit to the task handler.
        Returns:
            task_id: The id of the task.
        """
        if task_id is None:
            task_id = str(uuid4())
        asyncio_task = asyncio.create_task(
            self._execute_ingestion_task(task_id, function)
        )
        self.task_map[task_id] = asyncio_task
        if self._enable_redis_backend:
            self._redis_client.set(
                task_id,
                json.dumps(IngestionTaskStateSchema(task_id=task_id, state="PENDING").model_dump()),
            )
        else:
            async with self._asyncio_lock:
                self.task_status_result_map[task_id] = IngestionTaskStateSchema(
                    task_id=task_id, state="PENDING"
                ).model_dump()
        return task_id

    def get_task_state(self, task_id: str):
        """
        Get the state of a task.
        Args:
            task_id: The id of the task.
        Returns:
            state: The state of the task.
        """
        if self._enable_redis_backend:
            data = self._redis_client.get(task_id)
            if data:
                return json.loads(data).get("state")
            return None
        return self.task_status_result_map[task_id].get("state")

    async def set_task_status_and_result(
        self, task_id: str, status: str, result: dict[str, Any]
    ) -> None:
        """
        Set the status and result of a task in Redis and the task map.
        Args:
            task_id: The id of the task.
            status: The status of the task.
            result: The result of the task.
        """
        if self._enable_redis_backend:
            self._redis_client.set(
                task_id,
                json.dumps(IngestionTaskStateSchema(
                    task_id=task_id, state=status, result=result
                ).model_dump()),
            )
        else:
            async with self._asyncio_lock:
                self.task_status_result_map[task_id] = IngestionTaskStateSchema(
                    task_id=task_id, state=status, result=result
                ).model_dump()
                logger.debug(f"Task status result map: {self.task_status_result_map}")
        logger.debug(f"Task {task_id} status set to {status} and result: {result}")

    def get_task_status_and_result(self, task_id: str):
        """
        Get the status and result of a task from Redis and the task map.
        Args:
            task_id: The id of the task.
        Returns:
            status: The status of the task.
            result: The result of the task.
        """
        logger.debug(
            f"Getting result of task {task_id}, enable_redis_backend: {self._enable_redis_backend}"
        )
        if self._enable_redis_backend:
            data = self._redis_client.get(task_id)
            if data:
                return json.loads(data)
            return None
        return self.task_status_result_map[task_id]

    def get_task_result(self, task_id: str):
        """
        Get the result of a task from Redis and the task map.
        Args:
            task_id: The id of the task.
        Returns:
            result: The result of the task.
        """
        logger.debug(
            f"Getting result of task {task_id}, enable_redis_backend: {self._enable_redis_backend}"
        )
        if self._enable_redis_backend:
            data = self._redis_client.get(task_id)
            if data:
                return json.loads(data).get("result")
            return None
        logger.debug(
            f"Task result: {self.task_status_result_map[task_id].get('result')}"
        )
        return self.task_status_result_map[task_id].get("result")

    async def set_task_state_dict(
        self, task_id: str, state_dict: dict[str, Any]
    ) -> None:
        """
        Set custom state dictionary for a task in Redis or the local task state map.
        This allows storing arbitrary state information as a dictionary.
        Args:
            task_id: The id of the task.
            state_dict: A dictionary containing custom state information for the task.
        """
        if self._enable_redis_backend:
            # Store the state_dict in Redis with a specific key pattern
            state_key = f"{task_id}:state_dict"
            self._redis_client.set(
                state_key,
                json.dumps(TaskStateDictSchema(
                    task_id=task_id, state_dict=state_dict
                ).model_dump()),
            )
        else:
            async with self._asyncio_lock:
                # Update task_state_map for custom state tracking
                self.task_state_map[task_id] = state_dict
                logger.debug(f"Task state map: {self.task_state_map}")
        logger.debug(f"Task {task_id} state_dict set to {state_dict}")

    def get_task_state_dict(self, task_id: str) -> dict[str, Any]:
        """
        Get custom state dictionary for a task from Redis or the local task state map.
        Args:
            task_id: The id of the task.
        Returns:
            state_dict: A dictionary containing custom state information for the task.
        """
        logger.debug(
            f"Getting state_dict of task {task_id}, enable_redis_backend: {self._enable_redis_backend}"
        )
        if self._enable_redis_backend:
            state_key = f"{task_id}:state_dict"
            data = self._redis_client.get(state_key)
            if data:
                result = json.loads(data)
                return result.get("state_dict", {})
            return {}
        return self.task_state_map.get(task_id, {})


# Create a singleton instance of the IngestionTaskHandler
# (Shared across asyncio coroutines in a single uvicorn worker)
INGESTION_TASK_HANDLER = IngestionTaskHandler()

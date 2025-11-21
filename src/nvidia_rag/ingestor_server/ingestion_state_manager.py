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
This module manages the state of the ingestion process.
"""

import asyncio
from typing import Any
from uuid import uuid4


class IngestionStateManager:
    def __init__(
        self,
        filepaths: list[str],
        collection_name: str,
        custom_metadata: list[dict[str, Any]],
        documents_catalog_metadata: list[dict[str, Any]] | None = None,
    ):
        self.task_id = str(uuid4())
        self._is_background = False  # Whether the ingestion is running in background

        # State variables
        self.filepaths = filepaths
        self.collection_name = collection_name
        self.custom_metadata = custom_metadata
        self.documents_catalog_metadata = documents_catalog_metadata or []

        self.validation_errors = []
        self.failed_validation_documents = []

        # Batch progress variables
        self.total_documents_completed = 0
        self.total_batches_completed = 0
        self.documents_completed_list = []  # list[dict[str, Any]]

        self.asyncio_lock = asyncio.Lock()

    @property
    def is_background(self) -> bool:
        return self._is_background

    @is_background.setter
    def is_background(self, is_background: bool):
        self._is_background = is_background

    def get_task_id(self):
        return self.task_id

    async def update_batch_progress(
        self,
        batch_progress_response: dict[str, Any],
        is_batch_zero: bool = False,
    ):
        async with self.asyncio_lock:
            self.total_documents_completed += len(
                batch_progress_response.get("documents", [])
            )
            if not is_batch_zero:
                self.total_batches_completed += 1
            self.documents_completed_list.extend(
                batch_progress_response.get("documents", [])
            )
        batch_progress_response.update(
            {
                "documents": self.documents_completed_list,
                "documents_completed": self.total_documents_completed,
                "batches_completed": self.total_batches_completed,
            }
        )
        return batch_progress_response

    async def update_total_progress(
        self,
        total_progress_response: dict[str, Any],
    ):
        total_progress_response.update(
            {
                "batches_completed": self.total_batches_completed,
                "documents_completed": len(
                    total_progress_response.get("documents", [])
                ),
            }
        )
        return total_progress_response

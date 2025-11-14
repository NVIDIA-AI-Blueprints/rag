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

"""Unit tests for IngestionStateManager"""

import asyncio
import pytest
from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager


class TestIngestionStateManager:
    """Test cases for IngestionStateManager class"""

    def test_init_creates_unique_task_id(self):
        """Test that initialization creates a unique task ID"""
        manager1 = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )
        manager2 = IngestionStateManager(
            filepaths=["file2.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        assert manager1.task_id != manager2.task_id
        assert len(manager1.task_id) == 36  # UUID4 format
        assert len(manager2.task_id) == 36

    def test_init_sets_initial_state(self):
        """Test that initialization sets correct initial state"""
        filepaths = ["file1.pdf", "file2.pdf"]
        collection_name = "test_collection"
        custom_metadata = [{"key1": "value1"}, {"key2": "value2"}]

        manager = IngestionStateManager(
            filepaths=filepaths,
            collection_name=collection_name,
            custom_metadata=custom_metadata,
        )

        assert manager.filepaths == filepaths
        assert manager.collection_name == collection_name
        assert manager.custom_metadata == custom_metadata
        assert manager.validation_errors == []
        assert manager.failed_validation_documents == []
        assert manager.total_documents_completed == 0
        assert manager.total_batches_completed == 0
        assert manager.documents_completed_list == []
        assert manager.is_background is False
        assert manager.asyncio_lock is not None

    def test_is_background_property_getter(self):
        """Test is_background property getter"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        assert manager.is_background is False
        assert manager._is_background is False

    def test_is_background_property_setter(self):
        """Test is_background property setter"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        manager.is_background = True
        assert manager.is_background is True
        assert manager._is_background is True

        manager.is_background = False
        assert manager.is_background is False

    def test_get_task_id(self):
        """Test get_task_id returns the task ID"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        task_id = manager.get_task_id()
        assert task_id == manager.task_id
        assert isinstance(task_id, str)

    @pytest.mark.asyncio
    async def test_update_batch_progress_single_batch(self):
        """Test updating batch progress with a single batch"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}],
        )

        batch_response = {
            "documents": [
                {"document_name": "file1.pdf", "id": "doc1"},
                {"document_name": "file2.pdf", "id": "doc2"},
            ],
            "status": "success",
        }

        result = await manager.update_batch_progress(batch_response)

        assert manager.total_documents_completed == 2
        assert manager.total_batches_completed == 1
        assert len(manager.documents_completed_list) == 2
        assert result["documents_completed"] == 2
        assert result["batches_completed"] == 1
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_update_batch_progress_multiple_batches(self):
        """Test updating batch progress with multiple batches"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf", "file3.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}, {}],
        )

        # First batch
        batch1_response = {
            "documents": [{"document_name": "file1.pdf", "id": "doc1"}],
            "status": "success",
        }
        result1 = await manager.update_batch_progress(batch1_response)

        assert manager.total_documents_completed == 1
        assert manager.total_batches_completed == 1
        assert result1["documents_completed"] == 1
        assert result1["batches_completed"] == 1

        # Second batch
        batch2_response = {
            "documents": [
                {"document_name": "file2.pdf", "id": "doc2"},
                {"document_name": "file3.pdf", "id": "doc3"},
            ],
            "status": "success",
        }
        result2 = await manager.update_batch_progress(batch2_response)

        assert manager.total_documents_completed == 3
        assert manager.total_batches_completed == 2
        assert result2["documents_completed"] == 3
        assert result2["batches_completed"] == 2
        assert len(manager.documents_completed_list) == 3

    @pytest.mark.asyncio
    async def test_update_batch_progress_empty_documents(self):
        """Test updating batch progress with empty documents list"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        batch_response = {"documents": [], "status": "no_documents"}

        result = await manager.update_batch_progress(batch_response)

        assert manager.total_documents_completed == 0
        assert manager.total_batches_completed == 1
        assert result["documents_completed"] == 0
        assert result["batches_completed"] == 1
        assert len(result["documents"]) == 0

    @pytest.mark.asyncio
    async def test_update_batch_progress_concurrent_updates(self):
        """Test that concurrent batch updates are properly synchronized"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}, {}, {}],
        )

        batch1_response = {
            "documents": [
                {"document_name": "file1.pdf", "id": "doc1"},
                {"document_name": "file2.pdf", "id": "doc2"},
            ]
        }

        batch2_response = {
            "documents": [
                {"document_name": "file3.pdf", "id": "doc3"},
                {"document_name": "file4.pdf", "id": "doc4"},
            ]
        }

        # Run both updates concurrently
        results = await asyncio.gather(
            manager.update_batch_progress(batch1_response),
            manager.update_batch_progress(batch2_response),
        )

        # Verify final state
        assert manager.total_documents_completed == 4
        assert manager.total_batches_completed == 2
        assert len(manager.documents_completed_list) == 4

        # Both results should reflect cumulative progress
        for result in results:
            assert "documents_completed" in result
            assert "batches_completed" in result
            assert result["batches_completed"] <= 2
            assert result["documents_completed"] <= 4

    @pytest.mark.asyncio
    async def test_update_total_progress(self):
        """Test updating total progress"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf", "file3.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}, {}],
        )

        # Simulate some batches completed
        manager.total_batches_completed = 2
        manager.total_documents_completed = 3

        total_response = {
            "documents": [
                {"document_name": "file1.pdf", "id": "doc1"},
                {"document_name": "file2.pdf", "id": "doc2"},
                {"document_name": "file3.pdf", "id": "doc3"},
            ],
            "status": "completed",
        }

        result = await manager.update_total_progress(total_response)

        assert result["batches_completed"] == 2
        assert result["documents_completed"] == 3
        assert result["status"] == "completed"
        assert len(result["documents"]) == 3

    @pytest.mark.asyncio
    async def test_update_total_progress_empty_documents(self):
        """Test updating total progress with empty documents"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        manager.total_batches_completed = 1

        total_response = {"documents": [], "status": "no_documents"}

        result = await manager.update_total_progress(total_response)

        assert result["batches_completed"] == 1
        assert result["documents_completed"] == 0

    @pytest.mark.asyncio
    async def test_lock_prevents_race_conditions(self):
        """Test that the asyncio lock prevents race conditions"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"] * 100,
            collection_name="test_collection",
            custom_metadata=[{}] * 100,
        )

        # Create many concurrent updates
        tasks = []
        for i in range(10):
            batch_response = {
                "documents": [{"document_name": f"file{i}.pdf", "id": f"doc{i}"}]
            }
            tasks.append(manager.update_batch_progress(batch_response))

        await asyncio.gather(*tasks)

        # Verify counts are correct (no race condition)
        assert manager.total_documents_completed == 10
        assert manager.total_batches_completed == 10
        assert len(manager.documents_completed_list) == 10

    def test_task_id_immutable_after_creation(self):
        """Test that task_id remains the same throughout lifecycle"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}],
        )

        initial_task_id = manager.task_id
        task_id_via_method = manager.get_task_id()

        # Task ID should not change
        assert manager.task_id == initial_task_id
        assert task_id_via_method == initial_task_id

    @pytest.mark.asyncio
    async def test_state_persistence_across_updates(self):
        """Test that state persists correctly across multiple updates"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}],
        )

        # Add some validation errors
        manager.validation_errors = ["Error 1", "Error 2"]
        manager.failed_validation_documents = ["doc1"]

        # Update batch progress
        batch_response = {
            "documents": [{"document_name": "file1.pdf", "id": "doc2"}]
        }
        await manager.update_batch_progress(batch_response)

        # Verify validation errors persist
        assert manager.validation_errors == ["Error 1", "Error 2"]
        assert manager.failed_validation_documents == ["doc1"]
        # But batch progress is updated
        assert manager.total_documents_completed == 1

    @pytest.mark.asyncio
    async def test_documents_completed_list_accumulation(self):
        """Test that documents_completed_list accumulates correctly"""
        manager = IngestionStateManager(
            filepaths=["file1.pdf", "file2.pdf", "file3.pdf"],
            collection_name="test_collection",
            custom_metadata=[{}, {}, {}],
        )

        # Batch 1
        batch1 = {"documents": [{"id": "doc1", "name": "file1.pdf"}]}
        await manager.update_batch_progress(batch1)
        assert len(manager.documents_completed_list) == 1

        # Batch 2
        batch2 = {
            "documents": [
                {"id": "doc2", "name": "file2.pdf"},
                {"id": "doc3", "name": "file3.pdf"},
            ]
        }
        await manager.update_batch_progress(batch2)
        assert len(manager.documents_completed_list) == 3

        # Verify order is preserved
        assert manager.documents_completed_list[0]["id"] == "doc1"
        assert manager.documents_completed_list[1]["id"] == "doc2"
        assert manager.documents_completed_list[2]["id"] == "doc3"


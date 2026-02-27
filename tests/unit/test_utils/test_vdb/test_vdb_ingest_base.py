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

"""Unit tests for SerializedVDBWrapper from vdb_ingest_base module."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from nvidia_rag.utils.vdb.vdb_ingest_base import SerializedVDBWrapper


@pytest.fixture
def mock_vdb_op():
    """Create a mock VDB operation object."""
    op = MagicMock()
    op.run_async.return_value = "run_async_result"
    op.run.return_value = "run_result"
    op.write_to_index.return_value = "write_result"
    op.create_index.return_value = "index_result"
    op.some_read_method.return_value = "read_result"
    return op


@pytest.fixture
def wrapper(mock_vdb_op):
    """Create a SerializedVDBWrapper around a mock VDB op."""
    return SerializedVDBWrapper(mock_vdb_op)


@pytest.mark.skipif(
    SerializedVDBWrapper is None,
    reason="nv_ingest_client not installed",
)
class TestSerializedVDBWrapper:
    """Test cases for SerializedVDBWrapper."""

    def test_run_async_delegates_to_wrapped_op(self, wrapper, mock_vdb_op):
        """Test that run_async delegates to the wrapped VDB op."""
        records = [{"data": "test"}]
        result = wrapper.run_async(records)

        mock_vdb_op.run_async.assert_called_once_with(records)
        assert result == "run_async_result"

    def test_run_delegates_to_wrapped_op(self, wrapper, mock_vdb_op):
        """Test that run delegates to the wrapped VDB op."""
        records = [{"data": "test"}]
        result = wrapper.run(records)

        mock_vdb_op.run.assert_called_once_with(records)
        assert result == "run_result"

    def test_write_to_index_delegates_with_kwargs(self, wrapper, mock_vdb_op):
        """Test that write_to_index passes kwargs to the wrapped VDB op."""
        records = [{"data": "test"}]
        result = wrapper.write_to_index(records, collection_name="test")

        mock_vdb_op.write_to_index.assert_called_once_with(
            records, collection_name="test"
        )
        assert result == "write_result"

    def test_create_index_delegates_with_kwargs(self, wrapper, mock_vdb_op):
        """Test that create_index passes kwargs to the wrapped VDB op."""
        result = wrapper.create_index(collection_name="test")

        mock_vdb_op.create_index.assert_called_once_with(collection_name="test")
        assert result == "index_result"

    def test_getattr_delegates_non_overridden_methods(self, wrapper, mock_vdb_op):
        """Test that non-write methods pass through to the wrapped VDB op."""
        result = wrapper.some_read_method()
        mock_vdb_op.some_read_method.assert_called_once()
        assert result == "read_result"

    def test_isinstance_check_with_vdb(self, wrapper):
        """Test that wrapper passes isinstance check for VDB (was a real bug)."""
        from nv_ingest_client.util.vdb.adt_vdb import VDB

        assert isinstance(wrapper, VDB)

    def test_write_methods_are_serialized(self, mock_vdb_op):
        """Test that concurrent write calls are serialized by the lock."""
        execution_log = []
        lock_held = threading.Event()

        def locked_write(records):
            batch = records[0]
            execution_log.append(f"acquired_{batch}")
            if batch == "batch_1":
                lock_held.set()
                threading.Event().wait(0.1)
            execution_log.append(f"released_{batch}")
            return "done"

        mock_vdb_op.run.side_effect = locked_write
        wrapper = SerializedVDBWrapper(mock_vdb_op)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(wrapper.run, ["batch_1"])
            lock_held.wait(timeout=2)
            f2 = executor.submit(wrapper.run, ["batch_2"])
            f1.result(timeout=5)
            f2.result(timeout=5)

        assert execution_log.index("released_batch_1") < execution_log.index(
            "acquired_batch_2"
        )

    def test_wrapper_propagates_exceptions(self, wrapper, mock_vdb_op):
        """Test that exceptions from the wrapped op propagate through the lock."""
        mock_vdb_op.run.side_effect = ValueError("indexing failed")

        with pytest.raises(ValueError, match="indexing failed"):
            wrapper.run([{"data": "test"}])

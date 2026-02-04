# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common test helpers and fixtures for unit tests."""

from unittest.mock import MagicMock, PropertyMock


def create_mock_config(vector_store_name="milvus"):
    """Create a properly configured mock NvidiaRAGConfig.

    Args:
        vector_store_name: Name of the vector store (default: "milvus")

    Returns:
        Mock config object with proper vector_store configuration
    """
    mock_config = MagicMock()

    # Configure vector store with actual values instead of MagicMock
    mock_config.vector_store.name = vector_store_name
    mock_config.vector_store.endpoint = "http://localhost:19530"

    # Configure other common attributes
    mock_config.enable_citations = True
    mock_config.nv_ingest.enable_batch_mode = False
    mock_config.nv_ingest.enable_parallel_batch_mode = False

    return mock_config


def create_mock_vdb_op(collection_name="test_collection"):
    """Create a properly configured mock VDB operator.

    Args:
        collection_name: Name of the collection (default: "test_collection")

    Returns:
        Mock VDB operator with common methods configured
    """
    mock_vdb = MagicMock()
    mock_vdb.collection_name = collection_name
    mock_vdb.get_collection.return_value = []
    mock_vdb.create_collection.return_value = None
    mock_vdb.list_collections.return_value = []
    mock_vdb.get_metadata_schema.return_value = {}

    return mock_vdb

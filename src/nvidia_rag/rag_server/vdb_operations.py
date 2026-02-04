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

"""Vector Database Operations Module.

This module provides helper functions for vector database operations including:
- VDB operation initialization and preparation
- Collection validation and existence checks

These functions handle the setup and validation of vector database connections
and ensure proper configuration before executing RAG operations.
"""

import logging

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.vdb import _get_vdb_op
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


def prepare_vdb_op(
    config: NvidiaRAGConfig,
    vdb_op: VDBRag | None = None,
    vdb_endpoint: str | None = None,
    embedding_model: str | None = None,
    embedding_endpoint: str | None = None,
    vdb_auth_token: str = "",
) -> VDBRag:
    """
    Prepare the VDBRag object for generation.

    Args:
        config: The NVIDIA RAG configuration object
        vdb_op: Pre-initialized VDB operation instance (optional)
        vdb_endpoint: Vector database endpoint URL (optional)
        embedding_model: Embedding model name (optional)
        embedding_endpoint: Embedding endpoint URL (optional)
        vdb_auth_token: Authentication token for VDB (optional)

    Returns:
        VDBRag: Initialized vector database operation instance

    Raises:
        ValueError: If runtime parameters are provided when vdb_op is already initialized
    """
    if vdb_op is not None:
        if vdb_endpoint is not None:
            raise ValueError(
                "vdb_endpoint is not supported when vdb_op is provided during initialization."
            )
        if embedding_model is not None:
            raise ValueError(
                "embedding_model is not supported when vdb_op is provided during initialization."
            )
        if embedding_endpoint is not None:
            raise ValueError(
                "embedding_endpoint is not supported when vdb_op is provided during initialization."
            )

        return vdb_op

    document_embedder = get_embedding_model(
        model=embedding_model or config.embeddings.model_name,
        url=embedding_endpoint or config.embeddings.server_url,
        config=config,
    )

    return _get_vdb_op(
        vdb_endpoint=vdb_endpoint or config.vector_store.url,
        embedding_model=document_embedder,
        config=config,
        vdb_auth_token=vdb_auth_token,
    )


def validate_collections_exist(
    collection_names: list[str], vdb_op: VDBRag
) -> None:
    """Validate that all specified collections exist in the vector database.

    Args:
        collection_names: List of collection names to validate
        vdb_op: Vector database operation instance

    Raises:
        APIError: If any collection does not exist
    """
    for collection_name in collection_names:
        if not vdb_op.check_collection_exists(collection_name):
            raise APIError(
                f"Collection {collection_name} does not exist. Ensure a collection is created using POST /collection endpoint first "
                f"and documents are uploaded using POST /document endpoint",
                ErrorCodeMapping.BAD_REQUEST,
            )

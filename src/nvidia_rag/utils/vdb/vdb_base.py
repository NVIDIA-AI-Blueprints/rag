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
This module contains the implementation of the VDBRag class,
which provides an abstract base class for vector database operations in RAG applications.

Collection Management:
1. create_collection: Create a new collection with specified dimensions and type
2. check_collection_exists: Check if the specified collection exists
3. get_collection: Retrieve all collections with their metadata schemas
4. delete_collections: Delete multiple collections and their associated metadata

Document Management:
5. get_documents: Retrieve all unique documents from the specified collection
6. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
7. create_metadata_schema_collection: Initialize the metadata schema storage collection
8. add_metadata_schema: Store metadata schema configuration for the collection
9. get_metadata_schema: Retrieve the metadata schema for the specified collection

Retrieval Operations:
10. retrieval_langchain: Perform semantic search and return top-k relevant documents
"""

from abc import abstractmethod
from typing import Any

from langchain_core.vectorstores import VectorStore
from nv_ingest_client.util.vdb.adt_vdb import VDB


class VDBRag(VDB):
    """
    VDBRag is a class for the RAG mode of the VDB class.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the collection name."""
        pass

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Check the health of the VDB."""
        pass

    # ----------------------------------------------------------------------------------------------
    # Abstract methods for the VDBRag class for ingestion
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """Create a new collection with specified dimensions and type."""
        pass

    @abstractmethod
    def check_collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Check if the specified collection exists."""
        pass

    @abstractmethod
    def get_collection(self) -> list[dict[str, Any]]:
        """Retrieve all collections with their metadata schemas."""
        pass

    @abstractmethod
    def delete_collections(
        self,
        collection_names: list[str],
    ) -> None:
        """Delete multiple collections and their associated metadata."""
        pass

    @abstractmethod
    def get_documents(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve all unique documents from the specified collection."""
        pass

    @abstractmethod
    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
    ) -> bool:
        """Remove documents matching the specified source values."""
        pass

    @abstractmethod
    def create_metadata_schema_collection(
        self,
    ) -> None:
        """Initialize the metadata schema storage collection."""
        pass

    @abstractmethod
    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """Store metadata schema configuration for the collection."""
        pass

    @abstractmethod
    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve the metadata schema for the specified collection."""
        pass

    # ----------------------------------------------------------------------------------------------
    # Abstract methods for the VDBRag class for retrieval
    @abstractmethod
    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> VectorStore:
        """Get the vectorstore for a collection."""
        pass

    @abstractmethod
    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
    ) -> list[dict[str, Any]]:
        """Perform semantic search and return top-k relevant documents."""
        pass

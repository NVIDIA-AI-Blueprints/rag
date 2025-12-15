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

"""Unit tests for VDBRag abstract base class."""

from typing import Any

import pytest
from langchain_core.vectorstores import VectorStore

from nvidia_rag.utils.vdb.vdb_base import VDBRag


class ConcreteVDBRag(VDBRag):
    """Concrete implementation of VDBRag for testing."""

    def __init__(self):
        """Initialize concrete VDBRag."""
        self._collection_name = "test_collection"

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    async def check_health(self) -> dict[str, Any]:
        """Check the health of the VDB."""
        return {"status": "healthy"}

    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """Create a new collection."""
        pass

    def check_collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return True

    def get_collection(self) -> list[dict[str, Any]]:
        """Get all collections."""
        return []

    def delete_collections(self, collection_names: list[str]) -> None:
        """Delete collections."""
        pass

    def get_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """Get documents."""
        return []

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """Delete documents."""
        return True

    def create_metadata_schema_collection(self) -> None:
        """Create metadata schema collection."""
        pass

    def add_metadata_schema(
        self, collection_name: str, metadata_schema: list[dict[str, Any]]
    ) -> None:
        """Add metadata schema."""
        pass

    def get_metadata_schema(self, collection_name: str) -> list[dict[str, Any]]:
        """Get metadata schema."""
        return []

    def get_catalog_metadata(self, collection_name: str) -> dict[str, Any]:
        """Get catalog metadata."""
        return {}

    def update_catalog_metadata(
        self, collection_name: str, updates: dict[str, Any]
    ) -> None:
        """Update catalog metadata."""
        pass

    def get_document_catalog_metadata(
        self, collection_name: str, document_name: str
    ) -> dict[str, Any]:
        """Get document catalog metadata."""
        return {}

    def update_document_catalog_metadata(
        self, collection_name: str, document_name: str, updates: dict[str, Any]
    ) -> None:
        """Update document catalog metadata."""
        pass

    def get_langchain_vectorstore(self, collection_name: str) -> VectorStore:
        """Get langchain vectorstore."""
        raise NotImplementedError(
            "get_langchain_vectorstore must be implemented by concrete VDBRag subclass"
        )

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
    ) -> list[dict[str, Any]]:
        """Retrieval langchain."""
        return []

    def create_index(self) -> None:
        """Create index (VDB abstract method)."""
        pass

    def retrieval(self, queries: list, **kwargs) -> list[dict[str, Any]]:
        """Retrieval (VDB abstract method)."""
        return []

    def run(self, records: list) -> None:
        """Run (VDB abstract method)."""
        pass

    def write_to_index(self, records: list, **kwargs) -> None:
        """Write to index (VDB abstract method)."""
        pass


class TestVDBRagConcreteMethods:
    """Test cases for VDBRag concrete methods."""

    def test_create_document_info_collection(self):
        """Test create_document_info_collection method."""
        vdb = ConcreteVDBRag()
        result = vdb.create_document_info_collection()
        assert result is None

    def test_add_document_info(self):
        """Test add_document_info method."""
        vdb = ConcreteVDBRag()
        result = vdb.add_document_info(
            info_type="summary",
            collection_name="test_collection",
            document_name="test_doc.pdf",
            info_value={"summary": "test summary"},
        )
        assert result is None

    def test_get_document_info(self):
        """Test get_document_info method returns empty dict."""
        vdb = ConcreteVDBRag()
        result = vdb.get_document_info(
            info_type="summary",
            collection_name="test_collection",
            document_name="test_doc.pdf",
        )
        assert result == {}

    def test_get_document_info_with_different_params(self):
        """Test get_document_info with different parameters."""
        vdb = ConcreteVDBRag()
        result = vdb.get_document_info(
            info_type="metadata",
            collection_name="other_collection",
            document_name="other_doc.pdf",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test check_health method."""
        vdb = ConcreteVDBRag()
        result = await vdb.check_health()
        assert result == {"status": "healthy"}

    def test_collection_name_property(self):
        """Test collection_name property."""
        vdb = ConcreteVDBRag()
        assert vdb.collection_name == "test_collection"

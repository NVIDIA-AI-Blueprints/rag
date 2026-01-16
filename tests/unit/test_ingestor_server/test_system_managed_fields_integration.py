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
Unit tests for system-managed fields integration in NvidiaRAGIngestor.
Tests the complete flow of system field auto-addition and filtering.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor
from nvidia_rag.utils.metadata_validation import SYSTEM_MANAGED_FIELDS

logger = logging.getLogger(__name__)


class TestSystemManagedFieldsAutoAddition:
    """Test automatic addition of system-managed fields during collection creation"""

    @pytest.fixture
    def ingestor(self):
        """Create NvidiaRAGIngestor instance"""
        return NvidiaRAGIngestor()

    @pytest.fixture
    def mock_vdb_op(self):
        """Create mock VDB operations"""
        mock = MagicMock()
        mock.create_metadata_schema_collection = MagicMock()
        mock.get_collection = MagicMock(return_value=[])
        mock.create_collection = MagicMock()
        mock.add_metadata_schema = MagicMock()
        return mock

    def test_auto_add_all_system_fields_when_schema_empty(self, ingestor):
        """Test that all system fields are auto-added when schema is empty"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()
            mock_vdb.create_metadata_schema_collection = MagicMock()
            mock_vdb.get_collection = MagicMock(return_value=[])
            mock_vdb.create_collection = MagicMock()
            mock_vdb.add_metadata_schema = MagicMock()
            mock_prepare.return_value = (mock_vdb, "test_collection")

            ingestor.create_collection(
                collection_name="test_collection",
                metadata_schema=[],  # Empty schema
            )

            # Check that add_metadata_schema was called
            assert mock_vdb.add_metadata_schema.called
            call_args = mock_vdb.add_metadata_schema.call_args[0]
            schema = call_args[1]

            # All system fields should be present
            field_names = {field["name"] for field in schema}
            assert "filename" in field_names
            assert "page_number" in field_names
            assert "start_time" in field_names
            assert "end_time" in field_names

    def test_user_provided_fields_take_priority(self, ingestor):
        """Test that user-provided field definitions override system defaults"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()
            mock_vdb.create_metadata_schema_collection = MagicMock()
            mock_vdb.get_collection = MagicMock(return_value=[])
            mock_vdb.create_collection = MagicMock()
            mock_vdb.add_metadata_schema = MagicMock()
            mock_prepare.return_value = (mock_vdb, "test_collection")

            # User provides custom filename field
            user_schema = [
                {
                    "name": "filename",
                    "type": "string",
                    "description": "Custom filename description",
                    "required": True,
                    "max_length": 500,
                }
            ]

            ingestor.create_collection(
                collection_name="test_collection",
                metadata_schema=user_schema,
            )

            # Check the schema that was added
            assert mock_vdb.add_metadata_schema.called
            call_args = mock_vdb.add_metadata_schema.call_args[0]
            schema = call_args[1]

            # Find the filename field
            filename_field = next((f for f in schema if f["name"] == "filename"), None)
            assert filename_field is not None
            # Should use user's definition, not system default
            assert filename_field["description"] == "Custom filename description"
            assert filename_field["required"] is True
            assert filename_field["max_length"] == 500

            # Other system fields should still be auto-added
            field_names = {field["name"] for field in schema}
            assert "page_number" in field_names
            assert "start_time" in field_names
            assert "end_time" in field_names

    def test_system_fields_have_correct_flags(self, ingestor):
        """Test that auto-added system fields have correct user_defined and support_dynamic_filtering flags"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()
            mock_vdb.create_metadata_schema_collection = MagicMock()
            mock_vdb.get_collection = MagicMock(return_value=[])
            mock_vdb.create_collection = MagicMock()
            mock_vdb.add_metadata_schema = MagicMock()
            mock_prepare.return_value = (mock_vdb, "test_collection")

            ingestor.create_collection(
                collection_name="test_collection",
                metadata_schema=[],
            )

            assert mock_vdb.add_metadata_schema.called
            call_args = mock_vdb.add_metadata_schema.call_args[0]
            schema = call_args[1]

            # Check filename (RAG-managed, user_defined=True)
            filename_field = next((f for f in schema if f["name"] == "filename"), None)
            assert filename_field["user_defined"] is True
            assert filename_field["support_dynamic_filtering"] is True

            # Check page_number (auto-extracted, user_defined=False, but filterable)
            page_field = next((f for f in schema if f["name"] == "page_number"), None)
            assert page_field["user_defined"] is False
            assert page_field["support_dynamic_filtering"] is True

            # Check start_time (auto-extracted, not filterable)
            start_field = next((f for f in schema if f["name"] == "start_time"), None)
            assert start_field["user_defined"] is False
            assert start_field["support_dynamic_filtering"] is False

            # Check end_time (auto-extracted, not filterable)
            end_field = next((f for f in schema if f["name"] == "end_time"), None)
            assert end_field["user_defined"] is False
            assert end_field["support_dynamic_filtering"] is False


class TestGetCollectionsFiltering:
    """Test that get_collections filters out user_defined=False fields from UI responses"""

    @pytest.fixture
    def ingestor(self):
        """Create NvidiaRAGIngestor instance"""
        return NvidiaRAGIngestor()

    def test_get_collections_filters_auto_extracted_fields(self, ingestor):
        """Test that auto-extracted fields are filtered from collection list response"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()

            # Mock collection with both user-defined and auto-extracted fields
            mock_collection = {
                "collection_name": "test_collection",
                "metadata_schema": [
                    {
                        "name": "filename",
                        "type": "string",
                        "user_defined": True,
                        "support_dynamic_filtering": True,
                    },
                    {
                        "name": "category",
                        "type": "string",
                        "user_defined": True,
                        "support_dynamic_filtering": True,
                    },
                    {
                        "name": "page_number",
                        "type": "integer",
                        "user_defined": False,
                        "support_dynamic_filtering": True,
                    },
                    {
                        "name": "start_time",
                        "type": "integer",
                        "user_defined": False,
                        "support_dynamic_filtering": False,
                    },
                ],
            }
            mock_vdb.get_collection = MagicMock(return_value=[mock_collection])
            mock_prepare.return_value = (mock_vdb, None)

            result = ingestor.get_collections()

            # Check that auto-extracted fields are filtered out
            collections = result["collections"]
            assert len(collections) == 1

            schema = collections[0]["metadata_schema"]
            field_names = [f["name"] for f in schema]

            # User-defined fields should be present
            assert "filename" in field_names
            assert "category" in field_names

            # Auto-extracted fields should be hidden
            assert "page_number" not in field_names
            assert "start_time" not in field_names

    def test_get_collections_removes_internal_keys(self, ingestor):
        """Test that internal keys (user_defined, support_dynamic_filtering) are removed from response"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()

            mock_collection = {
                "collection_name": "test_collection",
                "metadata_schema": [
                    {
                        "name": "filename",
                        "type": "string",
                        "description": "File name",
                        "user_defined": True,
                        "support_dynamic_filtering": True,
                    }
                ],
            }
            mock_vdb.get_collection = MagicMock(return_value=[mock_collection])
            mock_prepare.return_value = (mock_vdb, None)

            collections_result = ingestor.get_collections()

            schema = collections_result["collections"][0]["metadata_schema"]
            filename_field = schema[0]

            # Internal keys should be removed
            assert "user_defined" not in filename_field
            assert "support_dynamic_filtering" not in filename_field

            # Other keys should remain
            assert "name" in filename_field
            assert "type" in filename_field
            assert "description" in filename_field


class TestGetDocumentsFiltering:
    """Test that get_documents filters out user_defined=False fields from metadata"""

    @pytest.fixture
    def ingestor(self):
        """Create NvidiaRAGIngestor instance"""
        return NvidiaRAGIngestor()

    def test_get_documents_filters_auto_extracted_metadata(self, ingestor):
        """Test that auto-extracted metadata fields are filtered from document list"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()

            # Mock metadata schema
            mock_schema = [
                {"name": "filename", "type": "string", "user_defined": True},
                {"name": "category", "type": "string", "user_defined": True},
                {"name": "page_number", "type": "integer", "user_defined": False},
                {"name": "start_time", "type": "integer", "user_defined": False},
            ]
            mock_vdb.get_metadata_schema = MagicMock(return_value=mock_schema)

            # Mock documents with both types of metadata
            mock_documents = [
                {
                    "document_name": "path/to/doc1.pdf",
                    "metadata": {
                        "filename": "doc1.pdf",
                        "category": "technical",
                        "page_number": 5,
                        "start_time": 1000,
                    },
                }
            ]
            mock_vdb.get_documents = MagicMock(return_value=mock_documents)
            mock_prepare.return_value = (mock_vdb, "test_collection")

            result = ingestor.get_documents(collection_name="test_collection")

            # Check filtered metadata
            documents = result["documents"]
            assert len(documents) == 1

            metadata = documents[0]["metadata"]

            # User-defined fields should be present
            assert "filename" in metadata
            assert "category" in metadata

            # Auto-extracted fields should be hidden
            assert "page_number" not in metadata
            assert "start_time" not in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

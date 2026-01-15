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


class TestReservedFieldsFiltering:
    """Test that reserved fields are not auto-added to collection schema"""

    @pytest.fixture
    def ingestor(self):
        """Create NvidiaRAGIngestor instance"""
        return NvidiaRAGIngestor()

    def test_reserved_fields_not_added_to_schema(self, ingestor):
        """Test that reserved fields (type, subtype, location) are not auto-added to schema"""
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

            # Extract field names
            field_names = {field["name"] for field in schema}

            # Reserved fields should NOT be present
            assert "type" not in field_names
            assert "subtype" not in field_names
            assert "location" not in field_names

            # Non-reserved system fields SHOULD be present
            assert "filename" in field_names
            assert "page_number" in field_names

    def test_reserved_fields_are_marked_in_system_managed_fields(self):
        """Test that reserved fields are properly marked in SYSTEM_MANAGED_FIELDS"""
        # Reserved fields should have reserved=True
        assert SYSTEM_MANAGED_FIELDS["type"]["reserved"] is True
        assert SYSTEM_MANAGED_FIELDS["subtype"]["reserved"] is True
        assert SYSTEM_MANAGED_FIELDS["location"]["reserved"] is True

        # Non-reserved fields should not have the flag or have reserved=False
        assert SYSTEM_MANAGED_FIELDS["filename"].get("reserved", False) is False
        assert SYSTEM_MANAGED_FIELDS["page_number"].get("reserved", False) is False
        assert SYSTEM_MANAGED_FIELDS["start_time"].get("reserved", False) is False
        assert SYSTEM_MANAGED_FIELDS["end_time"].get("reserved", False) is False

    def test_reserved_fields_properties(self):
        """Test the properties of reserved fields in SYSTEM_MANAGED_FIELDS"""
        # Type field
        assert SYSTEM_MANAGED_FIELDS["type"]["type"] == "string"
        assert SYSTEM_MANAGED_FIELDS["type"]["rag_managed"] is False
        assert SYSTEM_MANAGED_FIELDS["type"]["support_dynamic_filtering"] is False
        assert (
            "Content type extracted by NV-Ingest"
            in SYSTEM_MANAGED_FIELDS["type"]["description"]
        )

        # Subtype field
        assert SYSTEM_MANAGED_FIELDS["subtype"]["type"] == "string"
        assert SYSTEM_MANAGED_FIELDS["subtype"]["rag_managed"] is False
        assert SYSTEM_MANAGED_FIELDS["subtype"]["support_dynamic_filtering"] is False
        assert (
            "Content subtype extracted by NV-Ingest"
            in SYSTEM_MANAGED_FIELDS["subtype"]["description"]
        )

        # Location field
        assert SYSTEM_MANAGED_FIELDS["location"]["type"] == "array"
        assert SYSTEM_MANAGED_FIELDS["location"]["rag_managed"] is False
        assert SYSTEM_MANAGED_FIELDS["location"]["support_dynamic_filtering"] is False
        assert (
            "Bounding box coordinates extracted by NV-Ingest"
            in SYSTEM_MANAGED_FIELDS["location"]["description"]
        )

    def test_only_non_reserved_system_fields_auto_added(self, ingestor):
        """Test that only non-reserved system fields are auto-added when filtering is applied"""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_vdb = MagicMock()
            mock_vdb.create_metadata_schema_collection = MagicMock()
            mock_vdb.get_collection = MagicMock(return_value=[])
            mock_vdb.create_collection = MagicMock()
            mock_vdb.add_metadata_schema = MagicMock()
            mock_prepare.return_value = (mock_vdb, "test_collection")

            # Create collection with custom fields
            user_schema = [
                {"name": "title", "type": "string"},
                {"name": "author", "type": "string"},
            ]

            ingestor.create_collection(
                collection_name="test_collection",
                metadata_schema=user_schema,
            )

            # Check that add_metadata_schema was called
            assert mock_vdb.add_metadata_schema.called
            call_args = mock_vdb.add_metadata_schema.call_args[0]
            schema = call_args[1]

            field_names = {field["name"] for field in schema}

            # User fields should be present
            assert "title" in field_names
            assert "author" in field_names

            # Non-reserved system fields should be auto-added
            assert "filename" in field_names
            assert "page_number" in field_names
            assert "start_time" in field_names
            assert "end_time" in field_names

            # Reserved system fields should NOT be auto-added
            assert "type" not in field_names
            assert "subtype" not in field_names
            assert "location" not in field_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

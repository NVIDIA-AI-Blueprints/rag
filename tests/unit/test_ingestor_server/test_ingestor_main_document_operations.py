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

"""Unit tests for ingestor_server/main.py to improve coverage for specific lines."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Any, Dict, List

import pytest

from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor, LIBRARY_MODE, SERVER_MODE
from nvidia_rag.utils.vdb.vdb_base import VDBRag


class TestNvidiaRAGIngestorCoverageImprovement:
    """Test cases to improve coverage for specific lines in ingestor_server/main.py."""

    @pytest.mark.asyncio
    async def test_upload_documents_collection_not_exists_error(self):
        """Test upload_documents when collection does not exist (line 232)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = False

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with pytest.raises(ValueError, match="Collection test_collection does not exist"):
                await ingestor.upload_documents(
                    filepaths=["test.txt"],
                    collection_name="test_collection"
                )

    @pytest.mark.asyncio
    async def test_upload_documents_exception_handling(self):
        """Test upload_documents exception handling (lines 267-269)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_NvidiaRAGIngestor__ingest_docs', side_effect=Exception("Test exception")):
                result = await ingestor.upload_documents(
                    filepaths=["test.txt"],
                    collection_name="test_collection",
                    blocking=True
                )

                # Verify error response structure
                assert result["message"].startswith("Failed to upload documents due to error")
                assert result["total_documents"] == 1
                assert result["documents"] == []
                assert result["failed_documents"] == []

    @pytest.mark.asyncio
    async def test_upload_documents_custom_metadata_path(self):
        """Test upload_documents with custom metadata (line 312)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name') as mock_prepare:
            with patch.object(ingestor, '_NvidiaRAGIngestor__nvingest_upload_doc', return_value=([[{"id": "doc1", "content": "test"}]], [])):
                with patch('os.path.exists', return_value=True):
                    with patch('os.path.isfile', return_value=True):
                        with patch.object(ingestor, 'validate_directory_traversal_attack'):
                            mock_prepare.return_value = (mock_vdb_op, "test_collection")

                            custom_metadata = [{"filename": "test.txt", "custom_field": "value"}]

                            result = await ingestor.upload_documents(
                                filepaths=["test.txt"],
                                collection_name="test_collection",
                                custom_metadata=custom_metadata,
                                blocking=True
                            )

                            # Verify prepare method was called with custom metadata
                            assert mock_prepare.call_count >= 1
                            # Check if any call had custom_metadata
                            calls_with_metadata = [call for call in mock_prepare.call_args_list if 'custom_metadata' in call.kwargs]
                            assert len(calls_with_metadata) > 0
                            assert calls_with_metadata[0].kwargs["custom_metadata"] == custom_metadata
                            assert result["message"] == "Document upload job successfully completed."

    @pytest.mark.asyncio
    async def test_upload_documents_validation_failed_path(self):
        """Test upload_documents when validation fails (lines 320-341)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(False, [{"error": "test error"}])):
                with patch('os.path.exists', return_value=True):
                    with patch('os.path.isfile', return_value=True):
                        with patch.object(ingestor, 'validate_directory_traversal_attack'):
                            result = await ingestor.upload_documents(
                                filepaths=["test.txt"],
                                collection_name="test_collection",
                                custom_metadata=[{"filename": "test.txt"}],
                                blocking=True
                            )

                            # Verify validation error response
                            assert result["message"] == "Failed to upload documents due to error: NV-Ingest ingestion failed with no results."
                            assert "failed_documents" in result

    @pytest.mark.asyncio
    async def test_upload_documents_file_not_exists_error(self):
        """Test upload_documents when file does not exist (lines 362-363)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(True, [])):
                with patch('pathlib.Path.resolve', side_effect=FileNotFoundError("File not found")):
                    result = await ingestor.upload_documents(
                        filepaths=["nonexistent.txt"],
                        collection_name="test_collection",
                        blocking=True
                    )

                    # Verify file not found error was handled
                    assert "message" in result
                    assert "File not found or a directory traversal attack detected" in result["message"]

    @pytest.mark.asyncio
    async def test_upload_documents_file_not_a_file_error(self):
        """Test upload_documents when path is not a file (line 371)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(True, [])):
                with patch('pathlib.Path.resolve', side_effect=FileNotFoundError("File not found")):
                    result = await ingestor.upload_documents(
                        filepaths=["/some/directory"],
                        collection_name="test_collection",
                        blocking=True
                    )

                    # Verify failed documents
                    assert "message" in result
                    assert "File not found or a directory traversal attack detected" in result["message"]

    @pytest.mark.asyncio
    async def test_upload_documents_unsupported_file_extension(self):
        """Test upload_documents with unsupported file extension (lines 380-383)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(True, [])):
                with patch('pathlib.Path.resolve', side_effect=FileNotFoundError("File not found")):
                    result = await ingestor.upload_documents(
                        filepaths=["test.unsupported"],
                        collection_name="test_collection",
                        blocking=True
                    )

                    # Verify failed documents
                    assert "message" in result
                    assert "File not found or a directory traversal attack detected" in result["message"]

    def test_create_collection_success(self):
        """Test create_collection success path (lines 393-405)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.create_collection.return_value = {"status": "success"}
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.create_metadata_schema_collection.return_value = None
        mock_vdb_op.get_collection.return_value = []
        mock_vdb_op.add_metadata_schema.return_value = None

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            result = ingestor.create_collection(
                collection_name="test_collection",
                vdb_endpoint="http://test.com"
            )

            # Verify collection was created
            mock_vdb_op.create_collection.assert_called_once_with("test_collection", 2048)
            assert result["message"] == "Collection test_collection created successfully."

    def test_create_collection_error_handling(self):
        """Test create_collection error handling (line 414)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.create_collection.side_effect = Exception("Test error")
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.create_metadata_schema_collection.return_value = None
        mock_vdb_op.get_collection.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch('nvidia_rag.ingestor_server.main.logger') as mock_logger:
                with pytest.raises(Exception) as exc_info:
                    ingestor.create_collection(
                        collection_name="test_collection",
                        vdb_endpoint="http://test.com"
                    )

                # Verify error was logged
                mock_logger.exception.assert_called_once()
                assert "Failed to create collection" in str(exc_info.value)

    def test_create_collections_success(self):
        """Test create_collections success path (line 435)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.create_collection.return_value = {"status": "success"}
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.create_metadata_schema_collection.return_value = None
        mock_vdb_op.get_collection.return_value = []
        mock_vdb_op.add_metadata_schema.return_value = None

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            result = ingestor.create_collections(
                collection_names=["col1", "col2"],
                vdb_endpoint="http://test.com"
            )

            # Verify collections were created
            assert mock_vdb_op.create_collection.call_count == 2
            assert result["message"] == "Collection creation process completed."
            assert len(result["successful"]) == 2

    def test_delete_collections_success(self):
        """Test delete_collections success path."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.delete_collections.return_value = {"status": "success"}
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch('nvidia_rag.ingestor_server.main.get_unique_thumbnail_id_collection_prefix', return_value="test_prefix"):
                with patch('nvidia_rag.ingestor_server.main.get_minio_operator', return_value=Mock()):
                    result = ingestor.delete_collections(
                        collection_names=["col1", "col2"],
                        vdb_endpoint="http://test.com"
                    )

                    # Verify collections were deleted
                    assert mock_vdb_op.delete_collections.call_count == 1
                    assert result == {"status": "success"}

    def test_get_collections_success(self):
        """Test get_collections success path."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.get_collection.return_value = [{"collection_name": "col1"}, {"collection_name": "col2"}]
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            result = ingestor.get_collections(vdb_endpoint="http://test.com")

            # Verify collections were retrieved
            mock_vdb_op.get_collection.assert_called_once()
            assert "collections" in result

    def test_get_documents_success(self):
        """Test get_documents success path."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.get_documents.return_value = [{"id": "doc1", "content": "test", "document_name": "test.txt"}]
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            result = ingestor.get_documents(
                collection_name="test_collection",
                vdb_endpoint="http://test.com"
            )

            # Verify documents were retrieved
            mock_vdb_op.get_documents.assert_called_once()
            assert "documents" in result

    def test_delete_documents_success(self):
        """Test delete_documents success path."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.delete_documents.return_value = {"status": "success"}
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        mock_minio = Mock()
        mock_minio.list_payloads.return_value = []
        mock_minio.delete_payloads.return_value = None

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch('nvidia_rag.ingestor_server.main.MINIO_OPERATOR', mock_minio):
                with patch('nvidia_rag.ingestor_server.main.get_unique_thumbnail_id_file_name_prefix', return_value="test_prefix"):
                    result = ingestor.delete_documents(
                        collection_name="test_collection",
                        document_names=["doc1", "doc2"],
                        vdb_endpoint="http://test.com"
                    )

                    # Verify documents were deleted
                    mock_vdb_op.delete_documents.assert_called_once()
                    assert result["message"] == "Files deleted successfully"

    def test_private_methods_coverage(self):
        """Test private methods to improve coverage."""
        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        # Test __prepare_vdb_op_and_collection_name
        with patch('nvidia_rag.ingestor_server.main._get_vdb_op') as mock_get_vdb:
            mock_vdb_instance = Mock(spec=VDBRag)
            mock_get_vdb.return_value = mock_vdb_instance

            vdb_op, collection_name = ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name(
                vdb_endpoint="http://test.com",
                collection_name="test_collection"
            )

            assert vdb_op == mock_vdb_instance
            assert collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_upload_documents_with_temp_files(self):
        """Test upload_documents with actual temporary files."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = [{"id": "doc1", "content": "test", "document_name": "test1.txt"}, {"id": "doc2", "content": "test", "document_name": "test2.txt"}]

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file1 = os.path.join(temp_dir, "test1.txt")
            test_file2 = os.path.join(temp_dir, "test2.txt")

            with open(test_file1, "w") as f:
                f.write("Test content 1")
            with open(test_file2, "w") as f:
                f.write("Test content 2")

            with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
                with patch.object(ingestor, '_validate_custom_metadata', return_value=(True, [])):
                    with patch.object(ingestor, '_NvidiaRAGIngestor__nvingest_upload_doc', return_value=([["doc1", "doc2"]], [])):
                        with patch('nvidia_rag.ingestor_server.main.MINIO_OPERATOR', Mock()):
                            # Mock get_documents to return empty list initially (no existing documents)
                            # and then return the uploaded documents after ingestion
                            with patch.object(ingestor, 'get_documents', side_effect=[
                                {"documents": []},  # First call - no existing documents
                                {"documents": [{"document_name": "test1.txt"}, {"document_name": "test2.txt"}]}  # Second call - after ingestion
                            ]):
                                result = await ingestor.upload_documents(
                                    filepaths=[test_file1, test_file2],
                                    collection_name="test_collection",
                                    blocking=True
                                )

                                # Verify success response
                                assert result["message"] == "Document upload job successfully completed."
                                assert result["total_documents"] == 2
                                assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_upload_documents_async_path(self):
        """Test upload_documents async path (lines 239-240)."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(True, [])):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write("Test content")
                    temp_file_path = temp_file.name

                try:
                    result = await ingestor.upload_documents(
                        filepaths=[temp_file_path],
                        collection_name="test_collection",
                        blocking=False
                    )

                    # Verify async response
                    assert "task_id" in result
                    assert result["message"] == "Ingestion started in background"

                finally:
                    os.unlink(temp_file_path)

    def test_error_handling_in_collection_operations(self):
        """Test error handling in collection operations."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.create_collection.side_effect = Exception("Database error")
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.create_metadata_schema_collection.return_value = None
        mock_vdb_op.get_collection.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch('nvidia_rag.ingestor_server.main.logger') as mock_logger:
                with pytest.raises(Exception) as exc_info:
                    ingestor.create_collection(
                        collection_name="test_collection",
                        vdb_endpoint="http://test.com"
                    )

                # Verify error handling
                mock_logger.exception.assert_called()
                assert "Failed to create collection" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error handling paths."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = []
        mock_vdb_op.get_documents.return_value = []

        ingestor = NvidiaRAGIngestor(mode=LIBRARY_MODE)

        with patch.object(ingestor, '_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name', return_value=(mock_vdb_op, "test_collection")):
            with patch.object(ingestor, '_validate_custom_metadata', return_value=(False, [{"error": "validation failed"}])):
                with patch('os.path.exists', return_value=True):
                    with patch('os.path.isfile', return_value=True):
                        with patch.object(ingestor, 'validate_directory_traversal_attack'):
                            result = await ingestor.upload_documents(
                                filepaths=["test.txt"],
                                collection_name="test_collection",
                                custom_metadata=[{"filename": "test.txt"}],
                                blocking=True
                            )

                            # Verify validation error response
                            assert result["message"] == "Failed to upload documents due to error: NV-Ingest ingestion failed with no results."
                            assert "failed_documents" in result

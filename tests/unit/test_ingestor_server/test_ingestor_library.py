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
This is the test module for the RAG ingestor server.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor


class TestNvidiaRAGIngestor:
    """Test cases for NvidiaRAGIngestor class focusing on upload_documents method."""

    # Helper static methods
    @staticmethod
    def create_mock_files(filepaths: list[str]):
        """Create mock files for testing."""
        for filepath in filepaths:
            with open(filepath, "w") as f:
                f.write("test content")

    @staticmethod
    def cleanup_mock_files(filepaths: list[str]):
        """Remove mock files after testing."""
        for filepath in filepaths:
            os.remove(filepath)

    # Pytest fixtures
    @pytest.fixture
    def mock_config(self):
        """Mock configuration object."""
        config = MagicMock()
        config.vector_store.url = "http://localhost:19530"
        config.nv_ingest.chunk_size = 1024
        config.nv_ingest.chunk_overlap = 200
        return config

    @pytest.fixture
    def mock_nv_ingest_client(self):
        """Mock NV-Ingest client."""
        return MagicMock()

    @pytest.fixture
    def mock_minio_operator(self):
        """Mock MinIO operator."""
        return MagicMock()

    @pytest.fixture
    def mock_task_handler(self):
        """Mock ingestion task handler."""
        handler = MagicMock()
        handler.submit_task.return_value = "test-task-id-123"
        return handler

    @pytest.fixture
    def mock_vdb_op(self):
        """Mock VDB operator."""
        from nvidia_rag.utils.vdb.vdb_base import VDBRag

        vdb_op = MagicMock(spec=VDBRag)
        vdb_op.check_collection_exists.return_value = True
        vdb_op.get_documents.return_value = []
        vdb_op.get_metadata_schema.return_value = []
        vdb_op.delete_documents.return_value = True
        vdb_op.csv_file_path = None
        vdb_op.vdb_endpoint = "http://localhost:19530"
        return vdb_op

    @pytest.fixture
    def ingestor(
        self,
        mock_config,
        mock_nv_ingest_client,
        mock_minio_operator,
        mock_task_handler,
        mock_vdb_op,
    ):
        """Create NvidiaRAGIngestor instance with mocked dependencies."""
        with (
            patch(
                "nvidia_rag.ingestor_server.main.get_config",
                return_value=mock_config,
            ),
            patch(
                "nvidia_rag.ingestor_server.main.get_nv_ingest_client",
                return_value=mock_nv_ingest_client,
            ),
            patch(
                "nvidia_rag.ingestor_server.main.INGESTION_TASK_HANDLER",
                mock_task_handler,
            ),
        ):
            return NvidiaRAGIngestor(vdb_op=mock_vdb_op, mode="library")

    @pytest.fixture
    def mock_nvingest_upload_doc(self, ingestor):
        """Mock the __nvingest_upload_doc method."""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__nvingest_upload_doc", new_callable=AsyncMock
        ) as mock_upload_doc:
            mock_upload_doc.return_value = ([], [])
            yield

    @pytest.fixture
    def mock_get_documents(self, ingestor):
        """Mock the get_documents method."""
        with patch.object(ingestor, "get_documents", return_value={"documents": []}):
            yield

    @pytest.fixture
    def mock_get_failed_documents(self, ingestor):
        """Mock the get_failed_documents method."""
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__get_failed_documents", return_value=[]
        ):
            yield

    # Test cases
    def test_ingestor_initialization(
        self,
    ):
        """Test NvidiaRAGIngestor initialization with different modes."""
        # Import VDBRag for proper mocking
        from nvidia_rag.utils.vdb.vdb_base import VDBRag

        # Mock VDB operator for initialization tests - use spec to make isinstance work
        mock_vdb = MagicMock(spec=VDBRag)
        mock_vdb.vdb_endpoint = "http://localhost:19530"

        # Test library mode
        with (
            patch("nvidia_rag.ingestor_server.main.get_config"),
            patch("nvidia_rag.ingestor_server.main.get_nv_ingest_client"),
        ):
            ingestor_lib = NvidiaRAGIngestor(vdb_op=mock_vdb, mode="library")
            assert ingestor_lib.mode == "library"

            ingestor_server = NvidiaRAGIngestor(vdb_op=mock_vdb, mode="server")
            assert ingestor_server.mode == "server"

            # Test invalid mode
            with pytest.raises(ValueError, match="Invalid mode"):
                NvidiaRAGIngestor(vdb_op=mock_vdb, mode="invalid")

    @pytest.mark.asyncio
    async def test_upload_documents_non_blocking(
        self,
        ingestor,
        # Mock dependencies
        mock_nvingest_upload_doc,
        mock_get_documents,
        mock_get_failed_documents,
    ):
        """Test upload_documents with default parameters."""
        # Arrange
        filepaths = ["test_file1.pdf"]
        self.create_mock_files(filepaths)

        # Act - using default parameters
        result = await ingestor.upload_documents(filepaths=filepaths, blocking=False)
        task_id = result.get("task_id")

        # Assert
        assert result.get("message") == "Ingestion started in background"
        assert task_id

        # Wait for the task to complete and get status
        for i in range(10):
            status = await ingestor.status(task_id)
            if status.get("state") == "FINISHED" or status.get("state") == "FAILED":
                break
            await asyncio.sleep(0.1)  # Poll the task status every 0.1 seconds

        assert status.get("state") == "FINISHED"
        assert (
            status.get("result").get("message")
            == "Document upload job successfully completed."
        )
        assert status.get("result").get("total_documents") == 1
        assert len(status.get("result").get("documents")) == 1

        # Clean up the mock files
        self.cleanup_mock_files(filepaths)

    @pytest.mark.asyncio
    async def test_upload_documents_blocking(
        self,
        ingestor,
        # Mock dependencies
        mock_nvingest_upload_doc,
        mock_get_documents,
        mock_get_failed_documents,
    ):
        """Test upload_documents with default parameters."""
        # Arrange
        filepaths = ["test_file1.pdf"]
        self.create_mock_files(filepaths)

        # Act - using default parameters
        result = await ingestor.upload_documents(filepaths=filepaths, blocking=True)

        # Assert
        assert result.get("message") == "Document upload job successfully completed."
        assert result.get("total_documents") == 1
        assert len(result.get("documents")) == 1
        assert len(result.get("failed_documents")) == 0

        # Clean up the mock files
        self.cleanup_mock_files(filepaths)

    @pytest.mark.asyncio
    async def test_health_without_dependencies(self, ingestor):
        """Test health check without dependency checks."""
        result = await ingestor.health(check_dependencies=False)

        assert result["message"] == "Service is up."
        assert "message" in result

    def test_create_collection_success(self, ingestor):
        """Test successful collection creation."""
        # Mock collection operations
        ingestor.vdb_op.collection_name = "test_collection"
        ingestor.vdb_op.get_collection.return_value = []
        ingestor.vdb_op.create_collection.return_value = None
        ingestor.vdb_op.create_metadata_schema_collection.return_value = None
        ingestor.vdb_op.add_metadata_schema.return_value = None

        metadata_schema = [
            {
                "name": "category",
                "type": "string",
                "description": "Document category",
                "required": True,
            }
        ]

        result = ingestor.create_collection(
            embedding_dimension=1024, metadata_schema=metadata_schema
        )

        assert result["message"] == "Collection test_collection created successfully."
        assert result["collection_name"] == "test_collection"
        ingestor.vdb_op.create_collection.assert_called_once_with(
            "test_collection", 1024
        )

    def test_create_collection_already_exists(self, ingestor):
        """Test collection creation when collection already exists."""
        ingestor.vdb_op.collection_name = "test_collection"
        ingestor.vdb_op.get_collection.return_value = [
            {"collection_name": "test_collection"}
        ]

        result = ingestor.create_collection()

        assert result["message"] == "Collection test_collection already exists."
        assert result["collection_name"] == "test_collection"

    def test_create_collection_invalid_metadata_schema(self, ingestor):
        """Test collection creation with invalid metadata schema."""
        ingestor.vdb_op.collection_name = "test_collection"
        ingestor.vdb_op.get_collection.return_value = []

        invalid_schema = [
            {"name": "category", "type": "invalid_type"}  # Missing required fields
        ]

        with pytest.raises(Exception, match="Invalid metadata field 'category'"):
            ingestor.create_collection(metadata_schema=invalid_schema)

    def test_get_collections_success(self, ingestor):
        """Test successful retrieval of collections."""
        mock_collections = [
            {"collection_name": "collection1"},
            {"collection_name": "collection2"},
        ]
        ingestor.vdb_op.get_collection.return_value = mock_collections

        result = ingestor.get_collections()

        assert result["message"] == "Collections listed successfully."
        assert result["collections"] == mock_collections
        assert result["total_collections"] == 2

    def test_get_collections_error(self, ingestor):
        """Test collection retrieval with error."""
        ingestor.vdb_op.get_collection.side_effect = Exception("Database error")

        result = ingestor.get_collections()

        assert "Failed to retrieve collections" in result["message"]
        assert result["collections"] == []
        assert result["total_collections"] == 0

    def test_get_documents_success(self, ingestor):
        """Test successful document retrieval."""
        mock_documents = [
            {"document_name": "/path/to/doc1.pdf", "metadata": {"category": "test"}},
            {"document_name": "/path/to/doc2.pdf", "metadata": {"category": "test2"}},
        ]
        ingestor.vdb_op.collection_name = "test_collection"
        ingestor.vdb_op.get_documents.return_value = mock_documents

        result = ingestor.get_documents()

        assert result["message"] == "Document listing successfully completed."
        assert len(result["documents"]) == 2
        assert result["documents"][0]["document_name"] == "doc1.pdf"  # basename
        assert result["total_documents"] == 2

    def test_get_documents_error(self, ingestor):
        """Test document retrieval with error."""
        ingestor.vdb_op.collection_name = "test_collection"
        ingestor.vdb_op.get_documents.side_effect = Exception("Database error")

        result = ingestor.get_documents()

        assert "Document listing failed" in result["message"]
        assert result["documents"] == []
        assert result["total_documents"] == 0

    def test_delete_documents_failure(self, ingestor):
        """Test document deletion failure."""
        ingestor.vdb_op.delete_documents.side_effect = Exception("Delete failed")

        result = ingestor.delete_documents(
            document_names=["doc1.pdf"], collection_name="test_collection"
        )

        assert "Failed to delete files" in result["message"]
        assert result["total_documents"] == 0
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_update_documents_success(self, ingestor):
        """Test successful document update."""
        filepaths = ["test_file1.pdf"]
        self.create_mock_files(filepaths)

        # Mock delete operation
        ingestor.delete_documents = MagicMock(return_value={"total_documents": 1})

        # Mock upload operation
        with patch.object(ingestor, "upload_documents") as mock_upload:
            mock_upload.return_value = {"message": "success", "total_documents": 1}

            result = await ingestor.update_documents(
                filepaths=filepaths, blocking=True, collection_name="test_collection"
            )

            assert result["message"] == "success"
            ingestor.delete_documents.assert_called_once()
            mock_upload.assert_called_once()

        self.cleanup_mock_files(filepaths)

    @pytest.mark.asyncio
    async def test_validate_custom_metadata_success(self, ingestor):
        """Test successful metadata validation."""
        filepaths = ["test_file1.pdf"]
        self.create_mock_files(filepaths)

        custom_metadata = [
            {
                "filename": "test_file1.pdf",
                "metadata": {"category": "test", "author": "test_author"},
            }
        ]

        # Mock metadata schema
        ingestor.vdb_op.get_metadata_schema.return_value = [
            {"name": "category", "type": "string", "required": False},
            {"name": "author", "type": "string", "required": False},
        ]

        with patch(
            "nvidia_rag.ingestor_server.main.MetadataValidator"
        ) as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate_and_normalize_metadata_values.return_value = (
                True,
                [],
                {"category": "test", "author": "test_author"},
            )
            mock_validator_class.return_value = mock_validator

            validation_status, errors = await ingestor._validate_custom_metadata(
                custom_metadata, "test_collection", ingestor.vdb_op, filepaths
            )

            assert validation_status is True
            assert len(errors) == 0

        self.cleanup_mock_files(filepaths)

    @pytest.mark.asyncio
    async def test_validate_directory_traversal_attack_error(self, ingestor):
        error_detected = False
        file = (
            "http://127.0.0.1:8000/createfile/?param=../../../../../../../../etc/passwd"
        )
        try:
            await ingestor.validate_directory_traversal_attack(file)
        except ValueError as e:
            error_detected = True
            assert "File not found or a directory traversal attack detected" in str(e)
        except Exception as e:
            raise AssertionError(f"Unexpected exception: {e}") from e
        finally:
            assert error_detected is True

    @pytest.mark.asyncio
    async def test_validate_directory_traversal_attack_success(self, ingestor):
        file = "../rag/data/multimodal/woods_frost.docx"
        await ingestor.validate_directory_traversal_attack(file)

    @pytest.mark.asyncio
    async def test_validate_custom_metadata_missing_file(self, ingestor):
        """Test metadata validation with missing file."""
        filepaths = ["test_file1.pdf"]
        custom_metadata = [
            {
                "filename": "missing_file.pdf",  # File not in filepaths
                "metadata": {"category": "test"},
            }
        ]

        ingestor.vdb_op.get_metadata_schema.return_value = []

        validation_status, errors = await ingestor._validate_custom_metadata(
            custom_metadata, "test_collection", ingestor.vdb_op, filepaths
        )

        assert validation_status is False
        assert len(errors) == 1
        assert "not provided in the ingestion request" in errors[0]["error"]

    @pytest.mark.asyncio
    async def test_get_failed_documents(self, ingestor):
        """Test getting failed documents from failures list."""
        failures = [
            ("test_file1.pdf", "Processing error"),
            ("test_file2.pdf", "Unsupported format"),
        ]
        filepaths = ["test_file1.pdf", "test_file2.pdf", "test_file3.pdf"]

        # Mock get_documents to return existing documents
        ingestor.get_documents = MagicMock(
            return_value={"documents": [{"document_name": "test_file3.pdf"}]}
        )

        # Mock get_non_supported_files
        with patch.object(
            ingestor, "_NvidiaRAGIngestor__get_non_supported_files"
        ) as mock_unsupported:
            mock_unsupported.return_value = []

            result = await ingestor._NvidiaRAGIngestor__get_failed_documents(
                failures, filepaths, "test_collection"
            )

            assert len(result) == 2
            assert result[0]["document_name"] == "test_file1.pdf"
            assert result[1]["document_name"] == "test_file2.pdf"

    @pytest.mark.asyncio
    async def test_get_non_supported_files(self, ingestor):
        """Test identification of non-supported file extensions."""
        filepaths = [
            "test.pdf",  # supported
            "test.txt",  # supported
            "test.xyz",  # not supported
            "test.abc",  # not supported
        ]

        with patch(
            "nvidia_rag.ingestor_server.main._DEFAULT_EXTRACTOR_MAP",
            {"pdf": "pdf", "txt": "text"},
        ):
            result = await ingestor._NvidiaRAGIngestor__get_non_supported_files(
                filepaths
            )

            assert len(result) == 2
            assert "test.xyz" in result
            assert "test.abc" in result

    def test_parse_documents_text_content(self, ingestor):
        """Test parsing of text documents from nv-ingest results."""
        results = [
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "content": "Sample text content",
                        "source_metadata": {"source_id": "/path/to/test.pdf"},
                        "content_metadata": {},
                    },
                }
            ]
        ]

        documents = ingestor._NvidiaRAGIngestor__parse_documents(results)

        assert len(documents) == 1
        assert documents[0].page_content == "Sample text content"
        assert documents[0].metadata["source"] == "/path/to/test.pdf"
        assert documents[0].metadata["chunk_type"] == "text"
        assert documents[0].metadata["source_name"] == "test.pdf"

    def test_prepare_metadata(self, ingestor):
        """Test metadata preparation for a single chunk."""
        result_element = {
            "document_type": "text",
            "metadata": {
                "source_metadata": {"source_id": "/path/to/document.pdf"},
                "content_metadata": {},
            },
        }

        metadata = ingestor._NvidiaRAGIngestor__prepare_metadata(result_element)

        assert metadata["source"] == "/path/to/document.pdf"
        assert metadata["chunk_type"] == "text"
        assert metadata["source_name"] == "document.pdf"

    @pytest.mark.asyncio
    async def test_status_pending_task(self):
        """Test status check for pending task."""
        with patch(
            "nvidia_rag.ingestor_server.main.INGESTION_TASK_HANDLER"
        ) as mock_handler:
            mock_handler.get_task_status.return_value = "PENDING"

            result = await NvidiaRAGIngestor.status("test-task-id")

            assert result["state"] == "PENDING"
            assert result["result"]["message"] == "Task is pending"

    @pytest.mark.asyncio
    async def test_status_finished_task(self):
        """Test status check for finished task."""
        with patch(
            "nvidia_rag.ingestor_server.main.INGESTION_TASK_HANDLER"
        ) as mock_handler:
            mock_handler.get_task_status.return_value = "FINISHED"
            mock_handler.get_task_result.return_value = {"message": "success"}

            result = await NvidiaRAGIngestor.status("test-task-id")

            assert result["state"] == "FINISHED"
            assert result["result"]["message"] == "success"

    @pytest.mark.asyncio
    async def test_status_failed_task(self):
        """Test status check for failed task."""
        with patch(
            "nvidia_rag.ingestor_server.main.INGESTION_TASK_HANDLER"
        ) as mock_handler:
            mock_handler.get_task_status.return_value = "FINISHED"
            mock_handler.get_task_result.return_value = {
                "state": "FAILED",
                "message": "error",
            }

            result = await NvidiaRAGIngestor.status("test-task-id")

            assert result["state"] == "FAILED"
            assert result["result"]["message"] == "error"

    @pytest.mark.asyncio
    async def test_status_unknown_task(self):
        """Test status check for unknown task."""
        with patch(
            "nvidia_rag.ingestor_server.main.INGESTION_TASK_HANDLER"
        ) as mock_handler:
            mock_handler.get_task_status.side_effect = KeyError("Task not found")

            result = await NvidiaRAGIngestor.status("unknown-task-id")

            assert result["state"] == "UNKNOWN"
            assert result["result"]["message"] == "Unknown task state"

    def test_create_collections_success(self, ingestor):
        """Test successful creation of multiple collections."""
        collection_names = ["collection1", "collection2", "collection3"]

        result = ingestor.create_collections(
            collection_names=collection_names,
            embedding_dimension=1024,
            collection_type="text",
        )

        assert result["message"] == "Collection creation process completed."
        assert result["total_success"] == 3
        assert result["total_failed"] == 0
        assert len(result["successful"]) == 3
        assert len(result["failed"]) == 0

    def test_create_collections_partial_failure(self, ingestor):
        """Test collection creation with some failures."""
        collection_names = ["collection1", "collection2"]

        def side_effect(collection_name, **kwargs):
            if collection_name == "collection2":
                raise Exception("Creation failed")

        ingestor.vdb_op.create_collection.side_effect = side_effect

        result = ingestor.create_collections(collection_names=collection_names)

        assert result["total_success"] == 1
        assert result["total_failed"] == 1
        assert "collection1" in result["successful"]
        assert len(result["failed"]) == 1
        assert result["failed"][0]["collection_name"] == "collection2"

    def test_delete_collections_error(self, ingestor):
        """Test collection deletion with error."""
        collection_names = ["collection1"]
        ingestor.vdb_op.delete_collections.side_effect = Exception("Delete failed")

        result = ingestor.delete_collections(collection_names)

        assert "Failed to delete collections" in result["message"]
        assert result["total_collections"] == 0

    @pytest.mark.asyncio
    async def test_csv_deletion_timing_sequential_batches(self, ingestor):
        """Test that CSV file is deleted AFTER all sequential batches complete, not during individual batches."""
        filepaths = ["test_file1.pdf", "test_file2.pdf", "test_file3.pdf"]

        ingestor.vdb_op.csv_file_path = "/tmp/test_metadata.csv"

        with patch("os.remove") as mock_remove:
            with patch.dict(
                os.environ,
                {
                    "ENABLE_NV_INGEST_BATCH_MODE": "true",
                    "NV_INGEST_FILES_PER_BATCH": "1",
                    "ENABLE_NV_INGEST_PARALLEL_BATCH_MODE": "false",
                },
            ):
                with patch.object(
                    ingestor,
                    "_NvidiaRAGIngestor__nv_ingest_ingestion",
                    new_callable=AsyncMock,
                ) as mock_ingestion:
                    mock_ingestion.return_value = ([["result"]], [])

                    await ingestor._NvidiaRAGIngestor__nvingest_upload_doc(
                        filepaths=filepaths,
                        collection_name="test_collection",
                        vdb_op=ingestor.vdb_op,
                        split_options={"chunk_size": 1024, "chunk_overlap": 200},
                        generate_summary=False,
                    )

                    mock_remove.assert_called_once_with("/tmp/test_metadata.csv")

    @pytest.mark.asyncio
    async def test_csv_deletion_timing_parallel_batches(self, ingestor):
        """Test that CSV file is deleted AFTER all parallel batches complete, not during individual batches."""
        # Arrange
        filepaths = [
            "test_file1.pdf",
            "test_file2.pdf",
            "test_file3.pdf",
            "test_file4.pdf",
        ]

        ingestor.vdb_op.csv_file_path = "/tmp/test_metadata.csv"

        with patch("os.remove") as mock_remove:
            with patch.dict(
                os.environ,
                {
                    "ENABLE_NV_INGEST_BATCH_MODE": "true",
                    "NV_INGEST_FILES_PER_BATCH": "1",
                    "ENABLE_NV_INGEST_PARALLEL_BATCH_MODE": "true",
                    "NV_INGEST_CONCURRENT_BATCHES": "2",
                },
            ):
                with patch.object(
                    ingestor,
                    "_NvidiaRAGIngestor__nv_ingest_ingestion",
                    new_callable=AsyncMock,
                ) as mock_ingestion:
                    mock_ingestion.return_value = ([["result"]], [])

                    await ingestor._NvidiaRAGIngestor__nvingest_upload_doc(
                        filepaths=filepaths,
                        collection_name="test_collection",
                        vdb_op=ingestor.vdb_op,
                        split_options={"chunk_size": 1024, "chunk_overlap": 200},
                        generate_summary=False,
                    )

                    mock_remove.assert_called_once_with("/tmp/test_metadata.csv")

    @pytest.mark.asyncio
    async def test_csv_deletion_single_batch_mode(self, ingestor):
        """Test that CSV file is deleted during single batch processing (no batch mode)."""
        # Arrange
        filepaths = ["test_file1.pdf"]

        ingestor.vdb_op.csv_file_path = "/tmp/test_metadata.csv"

        with patch("os.remove") as mock_remove:
            with patch.dict(os.environ, {"ENABLE_NV_INGEST_BATCH_MODE": "false"}):
                with patch.object(
                    ingestor,
                    "_NvidiaRAGIngestor__nv_ingest_ingestion",
                    new_callable=AsyncMock,
                ) as mock_ingestion:
                    mock_ingestion.return_value = ([["result"]], [])

                    await ingestor._NvidiaRAGIngestor__nvingest_upload_doc(
                        filepaths=filepaths,
                        collection_name="test_collection",
                        vdb_op=ingestor.vdb_op,
                        split_options={"chunk_size": 1024, "chunk_overlap": 200},
                        generate_summary=False,
                    )

                    mock_remove.assert_called_once_with("/tmp/test_metadata.csv")

    @pytest.mark.asyncio
    async def test_csv_deletion_with_no_csv_file(self, ingestor):
        """Test that no error occurs when CSV file path is None."""
        filepaths = ["test_file1.pdf", "test_file2.pdf"]

        ingestor.vdb_op.csv_file_path = None

        with patch("os.remove") as mock_remove:
            with patch.dict(
                os.environ,
                {
                    "ENABLE_NV_INGEST_BATCH_MODE": "true",
                    "NV_INGEST_FILES_PER_BATCH": "1",
                    "ENABLE_NV_INGEST_PARALLEL_BATCH_MODE": "false",
                },
            ):
                with patch.object(
                    ingestor,
                    "_NvidiaRAGIngestor__nv_ingest_ingestion",
                    new_callable=AsyncMock,
                ) as mock_ingestion:
                    mock_ingestion.return_value = ([["result"]], [])

                    await ingestor._NvidiaRAGIngestor__nvingest_upload_doc(
                        filepaths=filepaths,
                        collection_name="test_collection",
                        vdb_op=ingestor.vdb_op,
                        split_options={"chunk_size": 1024, "chunk_overlap": 200},
                        generate_summary=False,
                    )

                    mock_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_csv_deletion_with_missing_csv_file(self, ingestor):
        """Test that FileNotFoundError is raised when CSV file doesn't exist on filesystem."""
        filepaths = ["test_file1.pdf", "test_file2.pdf"]

        ingestor.vdb_op.csv_file_path = "/tmp/non_existent_metadata.csv"

        with patch(
            "os.remove", side_effect=FileNotFoundError("File not found")
        ) as mock_remove:
            with patch.dict(
                os.environ,
                {
                    "ENABLE_NV_INGEST_BATCH_MODE": "true",
                    "NV_INGEST_FILES_PER_BATCH": "1",
                    "ENABLE_NV_INGEST_PARALLEL_BATCH_MODE": "false",
                },
            ):
                with patch.object(
                    ingestor,
                    "_NvidiaRAGIngestor__nv_ingest_ingestion",
                    new_callable=AsyncMock,
                ) as mock_ingestion:
                    mock_ingestion.return_value = ([["result"]], [])

                    with pytest.raises(FileNotFoundError, match="File not found"):
                        await ingestor._NvidiaRAGIngestor__nvingest_upload_doc(
                            filepaths=filepaths,
                            collection_name="test_collection",
                            vdb_op=ingestor.vdb_op,
                            split_options={"chunk_size": 1024, "chunk_overlap": 200},
                            generate_summary=False,
                        )

                    mock_remove.assert_called_once_with(
                        "/tmp/non_existent_metadata.csv"
                    )

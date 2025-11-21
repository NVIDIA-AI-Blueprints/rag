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

"""Comprehensive unit tests for ingestor_server/main.py to improve coverage."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from nvidia_rag.ingestor_server.main import Mode, NvidiaRAGIngestor
from nvidia_rag.utils.vdb.vdb_base import VDBRag


class TestNvidiaRAGIngestorInit:
    """Test cases for NvidiaRAGIngestor initialization."""

    def test_init_with_library_mode(self):
        """Test initialization with library mode."""
        ingestor = NvidiaRAGIngestor(mode=Mode.LIBRARY)
        assert ingestor.mode == Mode.LIBRARY
        assert ingestor.vdb_op is None

    def test_init_with_server_mode(self):
        """Test initialization with server mode."""
        ingestor = NvidiaRAGIngestor(mode=Mode.SERVER)
        assert ingestor.mode == Mode.SERVER
        assert ingestor.vdb_op is None

    def test_init_with_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(
            ValueError, match="Invalid mode: invalid_mode. Supported modes are:"
        ):
            NvidiaRAGIngestor(mode="invalid_mode")

    def test_init_with_valid_vdb_op(self):
        """Test initialization with valid VDBRag instance."""
        mock_vdb_op = Mock(spec=VDBRag)
        ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op, mode=Mode.LIBRARY)
        assert ingestor.vdb_op == mock_vdb_op

    def test_init_with_invalid_vdb_op(self):
        """Test initialization with invalid vdb_op type."""
        with pytest.raises(
            ValueError,
            match="vdb_op must be an instance of nvidia_rag.utils.vdb.vdb_base.VDBRag",
        ):
            NvidiaRAGIngestor(vdb_op="invalid_type", mode=Mode.LIBRARY)

    def test_init_with_vdb_class(self):
        """Test initialization with VDB class instance."""

        # Mock VDB class
        class MockVDB:
            pass

        with patch("nvidia_rag.ingestor_server.main.VDB", MockVDB):
            mock_vdb_op = MockVDB()
            ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op, mode=Mode.LIBRARY)
            assert ingestor.vdb_op == mock_vdb_op


class TestNvidiaRAGIngestorHealth:
    """Test cases for NvidiaRAGIngestor health method."""

    @pytest.mark.asyncio
    async def test_health_basic(self):
        """Test basic health check without dependencies."""
        ingestor = NvidiaRAGIngestor()

        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            mock_prepare.return_value = (Mock(), "test_collection")

            result = await ingestor.health(check_dependencies=False)

            from nvidia_rag.utils.health_models import IngestorHealthResponse
            assert isinstance(result, IngestorHealthResponse)
            assert result.message == "Service is up."
            # Verify VDB preparation is NOT called for simple health checks
            mock_prepare.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_with_dependencies(self):
        """Test health check with dependencies."""
        ingestor = NvidiaRAGIngestor()

        from nvidia_rag.utils.health_models import IngestorHealthResponse

        mock_vdb_op = Mock()
        mock_dependencies = IngestorHealthResponse(message="Service is up.")

        with patch.object(
            ingestor, "_NvidiaRAGIngestor__prepare_vdb_op_and_collection_name"
        ) as mock_prepare:
            with patch(
                "nvidia_rag.ingestor_server.health.check_all_services_health"
            ) as mock_check:
                mock_prepare.return_value = (mock_vdb_op, "test_collection")
                mock_check.return_value = mock_dependencies

                result = await ingestor.health(check_dependencies=True)

                assert isinstance(result, IngestorHealthResponse)
                assert result.message == "Service is up."
                # Verify VDB preparation IS called when checking dependencies
                mock_prepare.assert_called_once_with(bypass_validation=True)
                # Verify check_all_services_health was called with vdb_op and ANY config
                mock_check.assert_called_once()
                call_args = mock_check.call_args
                assert (
                    call_args[0][0] == mock_vdb_op
                )  # First argument should be mock_vdb_op
                # Second argument should be a NvidiaRAGConfig instance (or None)
                assert len(call_args[0]) == 2  # Should have 2 positional arguments


class TestNvidiaRAGIngestorValidateDirectoryTraversal:
    """Test cases for NvidiaRAGIngestor validate_directory_traversal_attack method."""

    @pytest.mark.asyncio
    async def test_validate_directory_traversal_attack_success(self):
        """Test successful directory traversal attack validation."""
        ingestor = NvidiaRAGIngestor()

        mock_file = "test.pdf"

        with patch("nvidia_rag.ingestor_server.main.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.resolve.return_value = Mock()
            mock_path.return_value = mock_path_instance

            # Should not raise any exception
            await ingestor.validate_directory_traversal_attack(mock_file)

    @pytest.mark.asyncio
    async def test_validate_directory_traversal_attack_os_error(self):
        """Test directory traversal attack validation with OSError."""
        ingestor = NvidiaRAGIngestor()

        mock_file = "test.pdf"

        with patch("nvidia_rag.ingestor_server.main.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.resolve.side_effect = OSError("Path error")
            mock_path.return_value = mock_path_instance

            with pytest.raises(
                ValueError,
                match="File not found or a directory traversal attack detected",
            ):
                await ingestor.validate_directory_traversal_attack(mock_file)

    @pytest.mark.asyncio
    async def test_validate_directory_traversal_attack_value_error(self):
        """Test directory traversal attack validation with ValueError."""
        ingestor = NvidiaRAGIngestor()

        mock_file = "test.pdf"

        with patch("nvidia_rag.ingestor_server.main.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.resolve.side_effect = ValueError("Value error")
            mock_path.return_value = mock_path_instance

            with pytest.raises(
                ValueError,
                match="File not found or a directory traversal attack detected",
            ):
                await ingestor.validate_directory_traversal_attack(mock_file)


class TestNvidiaRAGIngestorPrepareVDBOp:
    """Test cases for NvidiaRAGIngestor __prepare_vdb_op_and_collection_name method."""

    def test_prepare_vdb_op_with_existing_vdb_op(self):
        """Test __prepare_vdb_op when vdb_op is already set."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.collection_name = "test_collection"
        ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op)

        result = ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name()
        assert result == (mock_vdb_op, "test_collection")

    def test_prepare_vdb_op_with_collection_name_error(self):
        """Test __prepare_vdb_op with collection_name when vdb_op is set."""
        mock_vdb_op = Mock(spec=VDBRag)
        ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op)

        with pytest.raises(
            ValueError,
            match="`collection_name` and `custom_metadata` arguments are not supported when `vdb_op` is provided during initialization",
        ):
            ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name(
                collection_name="test"
            )

    def test_prepare_vdb_op_with_custom_metadata_error(self):
        """Test __prepare_vdb_op with custom_metadata when vdb_op is set."""
        mock_vdb_op = Mock(spec=VDBRag)
        ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op)

        with pytest.raises(
            ValueError,
            match="`collection_name` and `custom_metadata` arguments are not supported when `vdb_op` is provided during initialization",
        ):
            ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name(
                custom_metadata=[{"key": "value"}]
            )

    def test_prepare_vdb_op_without_vdb_op_missing_collection_name(self):
        """Test __prepare_vdb_op without vdb_op and missing collection_name."""
        ingestor = NvidiaRAGIngestor()

        with pytest.raises(
            ValueError,
            match="`collection_name` argument is required when `vdb_op` is not provided during initialization",
        ):
            ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name()

    @patch("nvidia_rag.ingestor_server.main._get_vdb_op")
    def test_prepare_vdb_op_without_vdb_op_with_collection_name(self, mock_get_vdb):
        """Test __prepare_vdb_op without vdb_op but with collection_name."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_get_vdb.return_value = mock_vdb_op

        ingestor = NvidiaRAGIngestor()

        result = ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name(
            collection_name="test_collection"
        )

        assert result == (mock_vdb_op, "test_collection")
        mock_get_vdb.assert_called_once()

    @patch("nvidia_rag.ingestor_server.main._get_vdb_op")
    def test_prepare_vdb_op_bypass_validation(self, mock_get_vdb):
        """Test __prepare_vdb_op with bypass_validation=True."""
        mock_vdb_op = Mock(spec=VDBRag)
        mock_get_vdb.return_value = mock_vdb_op

        ingestor = NvidiaRAGIngestor()

        result = ingestor._NvidiaRAGIngestor__prepare_vdb_op_and_collection_name(
            bypass_validation=True
        )

        assert result == (mock_vdb_op, None)
        mock_get_vdb.assert_called_once()


class TestNvidiaRAGIngestorLogResultInfo:
    """Test cases for NvidiaRAGIngestor _log_result_info method."""

    def test_log_result_info_success(self):
        """Test logging result info for successful operation."""
        ingestor = NvidiaRAGIngestor()

        batch_number = 1
        results = [
            [{"content": "test content", "metadata": {"source": "file1.pdf"}}],
            [{"content": "test content 2", "metadata": {"source": "file2.pdf"}}],
        ]
        failures = []
        total_ingestion_time = 1.5

        with patch("nvidia_rag.ingestor_server.main.logger") as mock_logger:
            ingestor._log_result_info(
                batch_number, results, failures, total_ingestion_time
            )

            # Verify info log was called
            mock_logger.info.assert_called()

    def test_log_result_info_with_failures(self):
        """Test logging result info with failures."""
        ingestor = NvidiaRAGIngestor()

        batch_number = 1
        results = [[{"content": "test content", "metadata": {"source": "file1.pdf"}}]]
        failures = [("file2.pdf", "Test error")]
        total_ingestion_time = 2.0

        with patch("nvidia_rag.ingestor_server.main.logger") as mock_logger:
            ingestor._log_result_info(
                batch_number, results, failures, total_ingestion_time
            )

            # Verify info log was called
            mock_logger.info.assert_called()

    def test_log_result_info_empty_results(self):
        """Test logging result info with empty results."""
        ingestor = NvidiaRAGIngestor()

        batch_number = 1
        results = []
        failures = []
        total_ingestion_time = 0.0

        with patch("nvidia_rag.ingestor_server.main.logger") as mock_logger:
            ingestor._log_result_info(
                batch_number, results, failures, total_ingestion_time
            )

            # Should not raise any exception
            mock_logger.info.assert_called()


class TestNvidiaRAGIngestorPutContentToMinio:
    """Test cases for NvidiaRAGIngestor __put_content_to_minio method."""

    def test_put_content_to_minio_success(self):
        """Test putting content to MinIO successfully."""
        ingestor = NvidiaRAGIngestor()

        results = [[{"content": "test content", "metadata": {"source": "file.pdf"}}]]
        collection_name = "test_collection"

        # Set config and mock minio_operator on instance
        ingestor.config.enable_citations = True
        mock_minio = Mock()
        mock_minio.put_content.return_value = "minio_url"
        ingestor.minio_operator = mock_minio

        # Should not raise any exception
        ingestor._NvidiaRAGIngestor__put_content_to_minio(results, collection_name)

    def test_put_content_to_minio_citations_disabled(self):
        """Test putting content to MinIO when citations are disabled."""
        ingestor = NvidiaRAGIngestor()

        results = [[{"content": "test content", "metadata": {"source": "file.pdf"}}]]
        collection_name = "test_collection"

        # Set config on instance
        ingestor.config.enable_citations = False

        with patch("nvidia_rag.ingestor_server.main.logger") as mock_logger:
            # Should not raise any exception and should log skip message
            ingestor._NvidiaRAGIngestor__put_content_to_minio(results, collection_name)
            mock_logger.info.assert_called()

    def test_put_content_to_minio_empty_results(self):
        """Test putting content to MinIO with empty results."""
        ingestor = NvidiaRAGIngestor()

        results = []
        collection_name = "test_collection"

        # Set config on instance
        ingestor.config.enable_citations = True
        mock_minio = Mock()
        ingestor.minio_operator = mock_minio

        # Should not raise any exception with empty results
        ingestor._NvidiaRAGIngestor__put_content_to_minio(results, collection_name)

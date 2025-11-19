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

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
)


class TestGetNvIngestClient:
    """Test get_nv_ingest_client function"""

    @patch("nvidia_rag.ingestor_server.nvingest.NvIngestClient")
    def test_get_nv_ingest_client_success(self, mock_nv_ingest_client):
        """Test get_nv_ingest_client creates client with correct parameters"""
        mock_config = Mock()
        mock_config.nv_ingest = Mock()
        mock_config.nv_ingest.message_client_hostname = "test-host"
        mock_config.nv_ingest.message_client_port = 7670

        mock_client = Mock()
        mock_nv_ingest_client.return_value = mock_client

        result = get_nv_ingest_client(mock_config)

        assert result == mock_client
        mock_nv_ingest_client.assert_called_once_with(
            message_client_hostname="test-host",
            message_client_port=7670,
            message_client_kwargs={"api_version": "v2"}
        )

    def test_get_nv_ingest_client_config_error(self):
        """Test get_nv_ingest_client handles config errors"""
        with patch(
            "nvidia_rag.ingestor_server.nvingest.NvidiaRAGConfig"
        ) as mock_config_class:
            mock_config_class.side_effect = Exception("Config error")

            with pytest.raises(Exception, match="Config error"):
                get_nv_ingest_client()


class TestGetNvIngestIngestor:
    """Test get_nv_ingest_ingestor function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.filepaths = ["/path/to/file1.pdf", "/path/to/file2.txt"]
        self.mock_vdb_op = Mock()
        self.mock_vdb_op.collection_name = "test_collection"

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_get_nv_ingest_ingestor_basic_config(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test get_nv_ingest_ingestor with basic configuration"""
        # Setup mocks
        mock_config = self._create_mock_config()
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=self.mock_vdb_op,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        mock_ingestor_class.assert_called_once_with(client=self.mock_client)
        mock_ingestor_instance.files.assert_called_once_with(self.filepaths)

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_get_nv_ingest_ingestor_with_custom_split_options(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test get_nv_ingest_ingestor with custom split options"""
        mock_config = self._create_mock_config()
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        custom_split_options = {"chunk_size": 1000, "chunk_overlap": 200}

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            split_options=custom_split_options,
            vdb_op=self.mock_vdb_op,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify split was called with custom options
        mock_ingestor_instance.split.assert_called_once()
        call_args = mock_ingestor_instance.split.call_args
        assert call_args[1]["chunk_size"] == 1000
        assert call_args[1]["chunk_overlap"] == 200

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_get_nv_ingest_ingestor_with_images_enabled(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test get_nv_ingest_ingestor with image extraction enabled"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.extract_images = True
        mock_config.nv_ingest.caption_endpoint_url = "http://test-caption-url"
        mock_config.nv_ingest.caption_model_name = "test-caption-model"
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.caption.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=self.mock_vdb_op,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify caption was called when extract_images is True
        mock_ingestor_instance.caption.assert_called_once()

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_get_nv_ingest_ingestor_with_structured_elements_modality(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test get_nv_ingest_ingestor with structured elements modality"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.structured_elements_modality = "test_modality"
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=self.mock_vdb_op,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify embed was called with structured_elements_modality
        mock_ingestor_instance.embed.assert_called_once()
        call_args = mock_ingestor_instance.embed.call_args
        assert call_args[1]["structured_elements_modality"] == "test_modality"
        assert call_args[1]["endpoint_url"] == "http://test-embedding-url"
        assert call_args[1]["model_name"] == "test-embedding-model"
        assert call_args[1]["dimensions"] == 768

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_get_nv_ingest_ingestor_with_image_elements_modality(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test get_nv_ingest_ingestor with image elements modality"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.image_elements_modality = "test_image_modality"
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=self.mock_vdb_op,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify embed was called with image_elements_modality
        mock_ingestor_instance.embed.assert_called_once()
        call_args = mock_ingestor_instance.embed.call_args
        assert call_args[1]["image_elements_modality"] == "test_image_modality"
        assert call_args[1]["endpoint_url"] == "http://test-embedding-url"
        assert call_args[1]["model_name"] == "test-embedding-model"
        assert call_args[1]["dimensions"] == 768

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    @patch("nvidia_rag.ingestor_server.nvingest.os.makedirs")
    def test_get_nv_ingest_ingestor_with_save_to_disk(
        self,
        mock_makedirs,
        mock_ingestor_class,
        mock_sanitize_url,
    ):
        """Test get_nv_ingest_ingestor with save_to_disk enabled"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.save_to_disk = True
        mock_sanitize_url.return_value = "http://test-embedding-url"

        with patch.dict(os.environ, {"INGESTOR_SERVER_DATA_DIR": "/test/data"}):
            mock_ingestor_instance = Mock()
            mock_ingestor_class.return_value = mock_ingestor_instance
            mock_ingestor_instance.files.return_value = mock_ingestor_instance
            mock_ingestor_instance.extract.return_value = mock_ingestor_instance
            mock_ingestor_instance.split.return_value = mock_ingestor_instance
            mock_ingestor_instance.embed.return_value = mock_ingestor_instance
            mock_ingestor_instance.save_to_disk.return_value = mock_ingestor_instance
            mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

            result = get_nv_ingest_ingestor(
                self.mock_client,
                self.filepaths,
                vdb_op=self.mock_vdb_op,
                config=mock_config,
            )

            assert result == mock_ingestor_instance
            # Verify save_to_disk was called
            mock_ingestor_instance.save_to_disk.assert_called_once()
            call_args = mock_ingestor_instance.save_to_disk.call_args
            assert (
                call_args[1]["output_directory"]
                == "/test/data/nv-ingest-results/test_collection"
            )
            assert call_args[1]["cleanup"] is False
            # Verify os.makedirs was called with expected path
            mock_makedirs.assert_called_once_with(
                "/test/data/nv-ingest-results/test_collection", exist_ok=True
            )
            # Verify embed was called with correct parameters
            mock_ingestor_instance.embed.assert_called_once()
            embed_call_args = mock_ingestor_instance.embed.call_args
            assert embed_call_args[1]["endpoint_url"] == "http://test-embedding-url"
            assert embed_call_args[1]["model_name"] == "test-embedding-model"
            assert embed_call_args[1]["dimensions"] == 768

    def test_get_nv_ingest_ingestor_config_error(self):
        """Test get_nv_ingest_ingestor handles config errors"""
        with patch(
            "nvidia_rag.ingestor_server.nvingest.NvidiaRAGConfig"
        ) as mock_config_class:
            mock_config_class.side_effect = Exception("Config error")

            with pytest.raises(Exception, match="Config error"):
                get_nv_ingest_ingestor(
                    self.mock_client, self.filepaths, vdb_op=self.mock_vdb_op
                )

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_extract_override_and_vdb_op_none(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test extract_override parameter and vdb_op=None behavior for shallow summaries"""
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        # Test 1: extract_override with text-only settings
        extract_override = {
            "extract_text": True,
            "extract_infographics": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_images": False,
            "extract_method": "pdfium",
            "text_depth": "document",
            "table_output_format": "pseudo_markdown",
            "extract_audio_params": {"segment_audio": False},
            "extract_page_as_image": False,
        }

        # Create a mock config for this test
        mock_config = self._create_mock_config()

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=None,
            extract_override=extract_override,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify extract was called with override parameters
        mock_ingestor_instance.extract.assert_called_once()
        call_kwargs = mock_ingestor_instance.extract.call_args[1]

        assert call_kwargs["extract_text"] is True
        assert call_kwargs["extract_infographics"] is False
        assert call_kwargs["extract_tables"] is False
        assert call_kwargs["extract_charts"] is False
        assert call_kwargs["extract_images"] is False
        assert call_kwargs["extract_page_as_image"] is False

        # Test 2: Verify VDB-related methods were NOT called when vdb_op=None
        mock_ingestor_instance.embed.assert_not_called()
        mock_ingestor_instance.vdb_upload.assert_not_called()

        # Test 3: extract_override with different custom parameters
        mock_ingestor_instance.extract.reset_mock()
        extract_override_custom = {
            "extract_text": True,
            "extract_infographics": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_images": False,
            "extract_method": "unstructured_python",
            "text_depth": "block",
            "table_output_format": "markdown",
            "extract_audio_params": {"segment_audio": True},
            "extract_page_as_image": True,
        }

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            vdb_op=None,
            extract_override=extract_override_custom,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        call_kwargs = mock_ingestor_instance.extract.call_args[1]

        assert call_kwargs["extract_method"] == "unstructured_python"
        assert call_kwargs["text_depth"] == "block"
        assert call_kwargs["table_output_format"] == "markdown"
        assert call_kwargs["extract_audio_params"] == {"segment_audio": True}
        assert call_kwargs["extract_page_as_image"] is True

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_split_skipped_when_split_options_none(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test that splitting is skipped when split_options is None (shallow extraction)"""
        mock_sanitize_url.return_value = "http://test-embedding-url"

        # Create a mock config for this test
        mock_config = self._create_mock_config()

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance

        # Call with split_options=None (shallow extraction use case)
        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            split_options=None,
            vdb_op=None,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify split was NOT called when split_options is None
        mock_ingestor_instance.split.assert_not_called()
        # Verify extract was still called
        mock_ingestor_instance.extract.assert_called_once()

    @patch("nvidia_rag.ingestor_server.nvingest.sanitize_nim_url")
    @patch("nvidia_rag.ingestor_server.nvingest.Ingestor")
    def test_split_called_with_default_options_when_dict_provided(
        self, mock_ingestor_class, mock_sanitize_url
    ):
        """Test that splitting is called with default options when empty dict provided"""
        mock_sanitize_url.return_value = "http://test-embedding-url"

        # Create a mock config for this test
        mock_config = self._create_mock_config()

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance

        # Call with empty dict (should use default values from config)
        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            split_options={},
            vdb_op=None,
            config=mock_config,
        )

        assert result == mock_ingestor_instance
        # Verify split WAS called with default config values
        mock_ingestor_instance.split.assert_called_once()
        call_kwargs = mock_ingestor_instance.split.call_args[1]
        assert call_kwargs["chunk_size"] == 1000  # from mock config
        assert call_kwargs["chunk_overlap"] == 200  # from mock config

    def _create_mock_config(self):
        """Create a mock config object with default values"""
        mock_config = Mock()
        mock_config.nv_ingest = Mock()
        mock_config.nv_ingest.enable_pdf_split = False
        mock_config.nv_ingest.pages_per_chunk = 10
        mock_config.nv_ingest.extract_text = True
        mock_config.nv_ingest.extract_infographics = True
        mock_config.nv_ingest.extract_tables = True
        mock_config.nv_ingest.extract_charts = True
        mock_config.nv_ingest.extract_images = False
        mock_config.nv_ingest.pdf_extract_method = "test_method"
        mock_config.nv_ingest.text_depth = 1
        mock_config.nv_ingest.segment_audio = True
        mock_config.nv_ingest.extract_page_as_image = True
        mock_config.nv_ingest.enable_pdf_splitter = True
        mock_config.nv_ingest.tokenizer = "test_tokenizer"
        mock_config.nv_ingest.chunk_size = 1000
        mock_config.nv_ingest.chunk_overlap = 200
        mock_config.nv_ingest.caption_endpoint_url = "http://test-caption-url"
        mock_config.nv_ingest.caption_model_name = "test-caption-model"
        mock_config.nv_ingest.structured_elements_modality = None
        mock_config.nv_ingest.image_elements_modality = None
        mock_config.nv_ingest.save_to_disk = False

        mock_config.embeddings = Mock()
        mock_config.embeddings.server_url = "http://test-embedding-server"
        mock_config.embeddings.model_name = "test-embedding-model"
        mock_config.embeddings.dimensions = 768

        return mock_config

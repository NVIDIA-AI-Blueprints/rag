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
from unittest.mock import Mock, patch, MagicMock

import pytest

from nvidia_rag.ingestor_server.nvingest import (
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
    ENABLE_NV_INGEST_VDB_UPLOAD
)


class TestGetNvIngestClient:
    """Test get_nv_ingest_client function"""

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.NvIngestClient')
    def test_get_nv_ingest_client_success(self, mock_nv_ingest_client, mock_get_config):
        """Test get_nv_ingest_client creates client with correct parameters"""
        mock_config = Mock()
        mock_config.nv_ingest = Mock()
        mock_config.nv_ingest.message_client_hostname = "test-host"
        mock_config.nv_ingest.message_client_port = 7670
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_nv_ingest_client.return_value = mock_client

        result = get_nv_ingest_client()

        assert result == mock_client
        mock_nv_ingest_client.assert_called_once_with(
            message_client_hostname="test-host",
            message_client_port=7670
        )

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    def test_get_nv_ingest_client_config_error(self, mock_get_config):
        """Test get_nv_ingest_client handles config errors"""
        mock_get_config.side_effect = Exception("Config error")

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

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    def test_get_nv_ingest_ingestor_basic_config(self, mock_ingestor_class, mock_sanitize_url,
                                                mock_get_env, mock_get_config):
        """Test get_nv_ingest_ingestor with basic configuration"""
        # Setup mocks
        mock_config = self._create_mock_config()
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
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
            vdb_op=self.mock_vdb_op
        )

        assert result == mock_ingestor_instance
        mock_ingestor_class.assert_called_once_with(client=self.mock_client)
        mock_ingestor_instance.files.assert_called_once_with(self.filepaths)

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    def test_get_nv_ingest_ingestor_with_custom_split_options(self, mock_ingestor_class,
                                                            mock_sanitize_url, mock_get_env,
                                                            mock_get_config):
        """Test get_nv_ingest_ingestor with custom split options"""
        mock_config = self._create_mock_config()
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
        mock_sanitize_url.return_value = "http://test-embedding-url"

        mock_ingestor_instance = Mock()
        mock_ingestor_class.return_value = mock_ingestor_instance
        mock_ingestor_instance.files.return_value = mock_ingestor_instance
        mock_ingestor_instance.extract.return_value = mock_ingestor_instance
        mock_ingestor_instance.split.return_value = mock_ingestor_instance
        mock_ingestor_instance.embed.return_value = mock_ingestor_instance
        mock_ingestor_instance.vdb_upload.return_value = mock_ingestor_instance

        custom_split_options = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }

        result = get_nv_ingest_ingestor(
            self.mock_client,
            self.filepaths,
            split_options=custom_split_options,
            vdb_op=self.mock_vdb_op
        )

        assert result == mock_ingestor_instance
        # Verify split was called with custom options
        mock_ingestor_instance.split.assert_called_once()
        call_args = mock_ingestor_instance.split.call_args
        assert call_args[1]["chunk_size"] == 1000
        assert call_args[1]["chunk_overlap"] == 200

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    def test_get_nv_ingest_ingestor_with_images_enabled(self, mock_ingestor_class,
                                                      mock_sanitize_url, mock_get_env,
                                                      mock_get_config):
        """Test get_nv_ingest_ingestor with image extraction enabled"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.extract_images = True
        mock_config.nv_ingest.caption_endpoint_url = "http://test-caption-url"
        mock_config.nv_ingest.caption_model_name = "test-caption-model"
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
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
            vdb_op=self.mock_vdb_op
        )

        assert result == mock_ingestor_instance
        # Verify caption was called when extract_images is True
        mock_ingestor_instance.caption.assert_called_once_with(
            api_key="test_api_key",
            endpoint_url="http://test-caption-url",
            model_name="test-caption-model"
        )

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    def test_get_nv_ingest_ingestor_with_structured_elements_modality(self, mock_ingestor_class,
                                                                    mock_sanitize_url, mock_get_env,
                                                                    mock_get_config):
        """Test get_nv_ingest_ingestor with structured elements modality"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.structured_elements_modality = "test_modality"
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
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
            vdb_op=self.mock_vdb_op
        )

        assert result == mock_ingestor_instance
        # Verify embed was called with structured_elements_modality
        mock_ingestor_instance.embed.assert_called_once()
        call_args = mock_ingestor_instance.embed.call_args
        assert call_args[1]["structured_elements_modality"] == "test_modality"

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    def test_get_nv_ingest_ingestor_with_image_elements_modality(self, mock_ingestor_class,
                                                               mock_sanitize_url, mock_get_env,
                                                               mock_get_config):
        """Test get_nv_ingest_ingestor with image elements modality"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.image_elements_modality = "test_image_modality"
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
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
            vdb_op=self.mock_vdb_op
        )

        assert result == mock_ingestor_instance
        # Verify embed was called with image_elements_modality
        mock_ingestor_instance.embed.assert_called_once()
        call_args = mock_ingestor_instance.embed.call_args
        assert call_args[1]["image_elements_modality"] == "test_image_modality"

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    @patch('nvidia_rag.ingestor_server.nvingest.get_env_variable')
    @patch('nvidia_rag.ingestor_server.nvingest.sanitize_nim_url')
    @patch('nvidia_rag.ingestor_server.nvingest.Ingestor')
    @patch('nvidia_rag.ingestor_server.nvingest.os.makedirs')
    def test_get_nv_ingest_ingestor_with_save_to_disk(self, mock_makedirs, mock_ingestor_class,
                                                    mock_sanitize_url, mock_get_env,
                                                    mock_get_config):
        """Test get_nv_ingest_ingestor with save_to_disk enabled"""
        mock_config = self._create_mock_config()
        mock_config.nv_ingest.save_to_disk = True
        mock_get_config.return_value = mock_config
        mock_get_env.return_value = "test_api_key"
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
                vdb_op=self.mock_vdb_op
            )

            assert result == mock_ingestor_instance
            # Verify save_to_disk was called
            mock_ingestor_instance.save_to_disk.assert_called_once()
            call_args = mock_ingestor_instance.save_to_disk.call_args
            assert call_args[1]["output_directory"] == "/test/data/nv-ingest-results/test_collection"
            assert call_args[1]["cleanup"] is False

    @patch('nvidia_rag.ingestor_server.nvingest.get_config')
    def test_get_nv_ingest_ingestor_config_error(self, mock_get_config):
        """Test get_nv_ingest_ingestor handles config errors"""
        mock_get_config.side_effect = Exception("Config error")

        with pytest.raises(Exception, match="Config error"):
            get_nv_ingest_ingestor(self.mock_client, self.filepaths, vdb_op=self.mock_vdb_op)

    def test_enable_nv_ingest_vdb_upload_constant(self):
        """Test ENABLE_NV_INGEST_VDB_UPLOAD constant"""
        assert ENABLE_NV_INGEST_VDB_UPLOAD is True

    def _create_mock_config(self):
        """Create a mock config object with default values"""
        mock_config = Mock()
        mock_config.nv_ingest = Mock()
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

        return mock_config

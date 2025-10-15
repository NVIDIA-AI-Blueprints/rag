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

from nvidia_rag.utils.vdb import _get_vdb_op, DEFAULT_METADATA_SCHEMA_COLLECTION


class TestVDBInit:
    """Test vdb/__init__.py module"""

    def test_default_metadata_schema_collection_constant(self):
        """Test DEFAULT_METADATA_SCHEMA_COLLECTION constant"""
        assert DEFAULT_METADATA_SCHEMA_COLLECTION == "metadata_schema"

    def test_config_constant(self):
        """Test CONFIG constant is set from get_config"""
        # The CONFIG is set at module import time, so we just test it exists
        from nvidia_rag.utils.vdb import CONFIG
        assert CONFIG is not None

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_milvus_without_custom_metadata(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Milvus without custom metadata"""
        # Setup mocks
        mock_config.vector_store.name = "milvus"
        mock_config.vector_store.url = "http://milvus:19530"
        mock_config.vector_store.search_type = "dense"
        mock_config.vector_store.enable_gpu_index = True
        mock_config.vector_store.enable_gpu_search = True
        mock_config.nv_ingest.extract_images = False
        mock_config.nv_ingest.extract_page_as_image = False
        mock_config.embeddings.dimensions = 768

        mock_get_metadata_config.return_value = (None, None, None)

        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESSKEY": "minioadmin",
            "MINIO_SECRETKEY": "minioadmin",
            "NVINGEST_MINIO_BUCKET": "test-bucket"
        }):
            with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusVDB') as mock_milvus_vdb:
                mock_vdb_instance = Mock()
                mock_milvus_vdb.return_value = mock_vdb_instance

                result = _get_vdb_op(
                    vdb_endpoint="http://test-milvus:19530",
                    collection_name="test_collection",
                    embedding_model="test-embedding"
                )

                assert result == mock_vdb_instance
                mock_milvus_vdb.assert_called_once()
                call_args = mock_milvus_vdb.call_args[1]
                assert call_args["collection_name"] == "test_collection"
                assert call_args["milvus_uri"] == "http://test-milvus:19530"
                assert call_args["minio_endpoint"] == "http://minio:9000"
                assert call_args["access_key"] == "minioadmin"
                assert call_args["secret_key"] == "minioadmin"
                assert call_args["bucket_name"] == "test-bucket"
                assert call_args["sparse"] is False
                assert call_args["enable_images"] is False
                assert call_args["recreate"] is False
                assert call_args["dense_dim"] == 768
                assert call_args["gpu_index"] is True
                assert call_args["gpu_search"] is True
                assert call_args["embedding_model"] == "test-embedding"

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_milvus_with_custom_metadata(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Milvus with custom metadata"""
        # Setup mocks
        mock_config.vector_store.name = "milvus"
        mock_config.vector_store.url = "http://milvus:19530"
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.enable_gpu_index = False
        mock_config.vector_store.enable_gpu_search = False
        mock_config.nv_ingest.extract_images = True
        mock_config.nv_ingest.extract_page_as_image = True
        mock_config.embeddings.dimensions = 1024

        mock_get_metadata_config.return_value = ("/path/to/metadata.csv", "source_field", ["field1", "field2"])

        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESSKEY": "minioadmin",
            "MINIO_SECRETKEY": "minioadmin",
            "NVINGEST_MINIO_BUCKET": "test-bucket"
        }):
            with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusVDB') as mock_milvus_vdb:
                mock_vdb_instance = Mock()
                mock_milvus_vdb.return_value = mock_vdb_instance

                result = _get_vdb_op(
                    vdb_endpoint="http://test-milvus:19530",
                    collection_name="test_collection",
                    custom_metadata=[{"field": "value"}],
                    all_file_paths=["/path/to/file1.pdf"],
                    embedding_model="test-embedding"
                )

                assert result == mock_vdb_instance
                mock_milvus_vdb.assert_called_once()
                call_args = mock_milvus_vdb.call_args[1]
                assert call_args["collection_name"] == "test_collection"
                assert call_args["sparse"] is True  # hybrid search
                assert call_args["enable_images"] is True
                assert call_args["meta_dataframe"] == "/path/to/metadata.csv"
                assert call_args["meta_source_field"] == "source_field"
                assert call_args["meta_fields"] == ["field1", "field2"]

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_milvus_uses_config_url_when_endpoint_none(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Milvus uses config URL when endpoint is None"""
        # Setup mocks
        mock_config.vector_store.name = "milvus"
        mock_config.vector_store.url = "http://config-milvus:19530"
        mock_config.vector_store.search_type = "dense"
        mock_config.vector_store.enable_gpu_index = False
        mock_config.vector_store.enable_gpu_search = False
        mock_config.nv_ingest.extract_images = False
        mock_config.nv_ingest.extract_page_as_image = False
        mock_config.embeddings.dimensions = 768

        mock_get_metadata_config.return_value = (None, None, None)

        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESSKEY": "minioadmin",
            "MINIO_SECRETKEY": "minioadmin",
            "NVINGEST_MINIO_BUCKET": "test-bucket"
        }):
            with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusVDB') as mock_milvus_vdb:
                mock_vdb_instance = Mock()
                mock_milvus_vdb.return_value = mock_vdb_instance

                result = _get_vdb_op(
                    vdb_endpoint=None,
                    collection_name="test_collection"
                )

                assert result == mock_vdb_instance
                call_args = mock_milvus_vdb.call_args[1]
                assert call_args["milvus_uri"] == "http://config-milvus:19530"

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_elasticsearch_without_custom_metadata(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Elasticsearch without custom metadata"""
        # Setup mocks
        mock_config.vector_store.name = "elasticsearch"
        mock_config.vector_store.url = "http://elasticsearch:9200"
        mock_config.vector_store.search_type = "dense"

        mock_get_metadata_config.return_value = (None, None, None)

        with patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticVDB') as mock_elastic_vdb:
            mock_vdb_instance = Mock()
            mock_elastic_vdb.return_value = mock_vdb_instance

            result = _get_vdb_op(
                vdb_endpoint="http://test-elasticsearch:9200",
                collection_name="test_index",
                embedding_model="test-embedding"
            )

            assert result == mock_vdb_instance
            mock_elastic_vdb.assert_called_once()
            call_args = mock_elastic_vdb.call_args[1]
            assert call_args["index_name"] == "test_index"
            assert call_args["es_url"] == "http://test-elasticsearch:9200"
            assert call_args["hybrid"] is False
            assert call_args["meta_dataframe"] is None
            assert call_args["meta_source_field"] is None
            assert call_args["meta_fields"] is None
            assert call_args["embedding_model"] == "test-embedding"
            assert call_args["csv_file_path"] is None

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_elasticsearch_with_custom_metadata(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Elasticsearch with custom metadata"""
        # Setup mocks
        mock_config.vector_store.name = "elasticsearch"
        mock_config.vector_store.url = "http://elasticsearch:9200"
        mock_config.vector_store.search_type = "hybrid"

        mock_get_metadata_config.return_value = ("/path/to/metadata.csv", "source_field", ["field1", "field2"])

        with patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticVDB') as mock_elastic_vdb, \
             patch('nvidia_rag.utils.vdb.pandas_file_reader') as mock_pandas_reader:

            mock_vdb_instance = Mock()
            mock_elastic_vdb.return_value = mock_vdb_instance
            mock_dataframe = Mock()
            mock_pandas_reader.return_value = mock_dataframe

            result = _get_vdb_op(
                vdb_endpoint="http://test-elasticsearch:9200",
                collection_name="test_index",
                custom_metadata=[{"field": "value"}],
                all_file_paths=["/path/to/file1.pdf"],
                embedding_model="test-embedding"
            )

            assert result == mock_vdb_instance
            mock_elastic_vdb.assert_called_once()
            call_args = mock_elastic_vdb.call_args[1]
            assert call_args["index_name"] == "test_index"
            assert call_args["es_url"] == "http://test-elasticsearch:9200"
            assert call_args["hybrid"] is True  # hybrid search
            assert call_args["meta_dataframe"] == mock_dataframe
            assert call_args["meta_source_field"] == "source_field"
            assert call_args["meta_fields"] == ["field1", "field2"]
            assert call_args["embedding_model"] == "test-embedding"
            assert call_args["csv_file_path"] == "/path/to/metadata.csv"
            mock_pandas_reader.assert_called_once_with("/path/to/metadata.csv")

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_elasticsearch_uses_config_url_when_endpoint_none(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with Elasticsearch uses config URL when endpoint is None"""
        # Setup mocks
        mock_config.vector_store.name = "elasticsearch"
        mock_config.vector_store.url = "http://config-elasticsearch:9200"
        mock_config.vector_store.search_type = "dense"

        mock_get_metadata_config.return_value = (None, None, None)

        with patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticVDB') as mock_elastic_vdb:
            mock_vdb_instance = Mock()
            mock_elastic_vdb.return_value = mock_vdb_instance

            result = _get_vdb_op(
                vdb_endpoint=None,
                collection_name="test_index"
            )

            assert result == mock_vdb_instance
            call_args = mock_elastic_vdb.call_args[1]
            assert call_args["es_url"] == "http://config-elasticsearch:9200"

    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_invalid_vector_store(self, mock_config):
        """Test _get_vdb_op with invalid vector store name raises ValueError"""
        mock_config.vector_store.name = "invalid_store"

        with pytest.raises(ValueError, match="Invalid vector store name: invalid_store"):
            _get_vdb_op(
                vdb_endpoint="http://test:1234",
                collection_name="test_collection"
            )

    @patch('nvidia_rag.utils.vdb.get_metadata_configuration')
    @patch('nvidia_rag.utils.vdb.CONFIG')
    def test_get_vdb_op_default_parameters(self, mock_config, mock_get_metadata_config):
        """Test _get_vdb_op with default parameters"""
        # Setup mocks
        mock_config.vector_store.name = "milvus"
        mock_config.vector_store.url = "http://milvus:19530"
        mock_config.vector_store.search_type = "dense"
        mock_config.vector_store.enable_gpu_index = False
        mock_config.vector_store.enable_gpu_search = False
        mock_config.nv_ingest.extract_images = False
        mock_config.nv_ingest.extract_page_as_image = False
        mock_config.embeddings.dimensions = 768

        mock_get_metadata_config.return_value = (None, None, None)

        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESSKEY": "minioadmin",
            "MINIO_SECRETKEY": "minioadmin",
            "NVINGEST_MINIO_BUCKET": "test-bucket"
        }):
            with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusVDB') as mock_milvus_vdb:
                mock_vdb_instance = Mock()
                mock_milvus_vdb.return_value = mock_vdb_instance

                result = _get_vdb_op(vdb_endpoint="http://test-milvus:19530")

                assert result == mock_vdb_instance
                call_args = mock_milvus_vdb.call_args[1]
                assert call_args["collection_name"] == ""
                assert call_args["embedding_model"] is None

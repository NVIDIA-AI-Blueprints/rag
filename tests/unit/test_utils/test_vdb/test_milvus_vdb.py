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

"""Unit tests for Milvus VDB functionality."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from opentelemetry import context as otel_context

from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB
from nvidia_rag.utils.vdb import DEFAULT_METADATA_SCHEMA_COLLECTION


class TestMilvusVDB:
    """Test the MilvusVDB class."""

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse')
    def test_init(self, mock_urlparse, mock_get_config, mock_connections):
        """Test MilvusVDB initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_url = Mock()
        mock_url.hostname = "localhost"
        mock_url.port = 19530
        mock_urlparse.return_value = mock_url

        embedding_model = Mock()
        kwargs = {
            "embedding_model": embedding_model,
            "milvus_uri": "http://localhost:19530",
            "collection_name": "test_collection",
            "meta_dataframe": "/path/to/csv"
        }

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__') as mock_super_init:
            mock_super_init.return_value = None
            vdb = MilvusVDB(**kwargs)

            assert vdb.embedding_model == embedding_model
            assert vdb.vdb_endpoint == "http://localhost:19530"
            assert vdb.collection_name == "test_collection"
            assert vdb.connection_alias.startswith("milvus_localhost_19530_")
            assert vdb.csv_file_path == "/path/to/csv"

            mock_connections.connect.assert_called_once_with(
                vdb.connection_alias, uri="http://localhost:19530"
            )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.create_nvingest_collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.CONFIG')
    def test_create_collection(self, mock_config, mock_connections, mock_create_nvingest):
        """Test create_collection method."""
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.enable_gpu_index = True
        mock_config.vector_store.enable_gpu_search = True

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb.create_collection("test_collection", dimension=1024, collection_type="text")

            mock_create_nvingest.assert_called_once_with(
                collection_name="test_collection",
                milvus_uri="http://localhost:19530",
                sparse=True,
                recreate=False,
                gpu_index=True,
                gpu_search=True,
                dense_dim=1024,
            )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_check_collection_exists_true(self, mock_get_config, mock_connections, mock_utility):
        """Test check_collection_exists when collection exists."""
        mock_utility.has_collection.return_value = True

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.check_collection_exists("test_collection")

            assert result is True
            mock_utility.has_collection.assert_called_once_with(
                "test_collection", using=vdb.connection_alias
            )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_check_collection_exists_false(self, mock_get_config, mock_connections, mock_utility):
        """Test check_collection_exists when collection doesn't exist."""
        mock_utility.has_collection.return_value = False

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.check_collection_exists("test_collection")

            assert result is False

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_milvus_entities(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test _get_milvus_entities method."""
        mock_client = Mock()
        mock_entities = [{"id": 1, "data": "test"}]
        mock_client.query.return_value = mock_entities
        mock_milvus_client.return_value = mock_client

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_milvus_entities("test_collection", "filter_expr")

            assert result == mock_entities
            mock_client.query.assert_called_once_with(
                collection_name="test_collection", filter="filter_expr", limit=1000
            )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_milvus_entities_empty(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test _get_milvus_entities method with empty result."""
        mock_client = Mock()
        mock_client.query.return_value = []
        mock_milvus_client.return_value = mock_client

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_milvus_entities("test_collection", "filter_expr")

            assert result == []

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_collection_info(self, mock_get_config, mock_connections, mock_utility, mock_collection):
        """Test _get_collection_info method."""
        mock_utility.list_collections.return_value = ["collection1", "collection2"]
        
        mock_collection_obj1 = Mock()
        mock_collection_obj1.num_entities = 100
        mock_collection_obj2 = Mock()
        mock_collection_obj2.num_entities = 200
        
        mock_collection.side_effect = [mock_collection_obj1, mock_collection_obj2]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_collection_info()

            expected = [
                {"collection_name": "collection1", "num_entities": 100},
                {"collection_name": "collection2", "num_entities": 200}
            ]
            assert result == expected

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_collection(self, mock_get_config, mock_connections):
        """Test get_collection method."""
        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            mock_collection_info = [
                {"collection_name": "collection1", "num_entities": 100},
                {"collection_name": "collection2", "num_entities": 200}
            ]
            mock_entities = [
                {"collection_name": "collection1", "metadata_schema": [{"name": "field1"}]},
                {"collection_name": "collection2", "metadata_schema": [{"name": "field2"}]}
            ]

            with patch.object(vdb, 'create_metadata_schema_collection') as mock_create, \
                 patch.object(vdb, '_get_collection_info', return_value=mock_collection_info) as mock_get_info, \
                 patch.object(vdb, '_get_milvus_entities', return_value=mock_entities) as mock_get_entities:

                result = vdb.get_collection()

                expected = [
                    {
                        "collection_name": "collection1", 
                        "num_entities": 100,
                        "metadata_schema": [{"name": "field1"}]
                    },
                    {
                        "collection_name": "collection2", 
                        "num_entities": 200,
                        "metadata_schema": [{"name": "field2"}]
                    }
                ]
                assert result == expected
                mock_create.assert_called_once()
                mock_get_info.assert_called_once()
                mock_get_entities.assert_called_once_with(DEFAULT_METADATA_SCHEMA_COLLECTION, filter="")

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_collections_success(self, mock_get_config, mock_connections, mock_utility):
        """Test _delete_collections method with successful deletion."""
        mock_utility.has_collection.return_value = True

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            deleted, failed = vdb._delete_collections(["collection1", "collection2"])

            assert deleted == ["collection1", "collection2"]
            assert failed == []
            assert mock_utility.drop_collection.call_count == 2

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_collections_not_found(self, mock_get_config, mock_connections, mock_utility):
        """Test _delete_collections method with collection not found."""
        mock_utility.has_collection.return_value = False

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            deleted, failed = vdb._delete_collections(["collection1"])

            assert deleted == []
            assert len(failed) == 1
            assert failed[0]["collection_name"] == "collection1"
            assert "not found" in failed[0]["error_message"]

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.utility')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_collections_exception(self, mock_get_config, mock_connections, mock_utility):
        """Test _delete_collections method with exception."""
        mock_utility.has_collection.return_value = True
        mock_utility.drop_collection.side_effect = Exception("Drop error")

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            deleted, failed = vdb._delete_collections(["collection1"])

            assert deleted == []
            assert len(failed) == 1
            assert failed[0]["collection_name"] == "collection1"
            assert "Drop error" in failed[0]["error_message"]

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_entities(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test _delete_entities method."""
        mock_client = Mock()
        mock_client.has_collection.return_value = True
        mock_milvus_client.return_value = mock_client

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb._delete_entities("test_collection", "filter_expr")

            mock_client.delete.assert_called_once_with(
                collection_name="test_collection", filter="filter_expr"
            )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_entities_collection_not_exists(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test _delete_entities method when collection doesn't exist."""
        mock_client = Mock()
        mock_client.has_collection.return_value = False
        mock_milvus_client.return_value = mock_client

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb._delete_entities("test_collection", "filter_expr")

            mock_client.delete.assert_not_called()

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_collections_complete(self, mock_get_config, mock_connections):
        """Test delete_collections method (complete flow)."""
        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            deleted_collections = ["collection1", "collection2"]
            failed_collections = []

            with patch.object(vdb, '_delete_collections', return_value=(deleted_collections, failed_collections)) as mock_delete, \
                 patch.object(vdb, '_delete_entities') as mock_delete_entities:

                result = vdb.delete_collections(["collection1", "collection2"])

                mock_delete.assert_called_once_with(["collection1", "collection2"])
                assert mock_delete_entities.call_count == 2
                
                expected = {
                    "message": "Collection deletion process completed.",
                    "successful": deleted_collections,
                    "failed": failed_collections,
                    "total_success": 2,
                    "total_failed": 0,
                }
                assert result == expected

    def test_extract_filename_string_source(self):
        """Test _extract_filename with string source."""
        metadata = {"source": "/path/to/file.txt"}
        result = MilvusVDB._extract_filename(metadata)
        assert result == "file.txt"

    def test_extract_filename_dict_source(self):
        """Test _extract_filename with dict source."""
        metadata = {"source": {"source_name": "/path/to/file.txt"}}
        result = MilvusVDB._extract_filename(metadata)
        assert result == "file.txt"

    def test_extract_filename_invalid_source(self):
        """Test _extract_filename with invalid source."""
        metadata = {"source": {"other_field": "/path/to/file.txt"}}
        result = MilvusVDB._extract_filename(metadata)
        assert result is None

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents_list(self, mock_get_config, mock_connections, mock_collection):
        """Test _get_documents_list method."""
        mock_collection_obj = Mock()
        mock_query_iterator = Mock()
        
        # Mock iterator behavior
        mock_data_batch1 = [
            {
                "source": "/path/to/file1.txt",
                "content_metadata": {"field1": "value1", "field2": "value2"}
            },
            {
                "source": "/path/to/file2.txt",
                "content_metadata": {"field1": "value3", "field2": "value4"}
            }
        ]
        mock_data_batch2 = [
            {
                "source": "/path/to/file1.txt",  # Duplicate - should be skipped
                "content_metadata": {"field1": "value5", "field2": "value6"}
            }
        ]
        
        mock_query_iterator.next.side_effect = [mock_data_batch1, mock_data_batch2, StopIteration()]
        mock_collection_obj.query_iterator.return_value = mock_query_iterator
        mock_collection.return_value = mock_collection_obj

        metadata_schema = [
            {"name": "field1"},
            {"name": "field2"}
        ]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_documents_list("test_collection", metadata_schema)

            expected = [
                {
                    "document_name": "file1.txt",
                    "metadata": {"field1": "value1", "field2": "value2"}
                },
                {
                    "document_name": "file2.txt",
                    "metadata": {"field1": "value3", "field2": "value4"}
                }
            ]
            assert result == expected

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents_list_no_collection(self, mock_get_config, mock_connections, mock_collection):
        """Test _get_documents_list method when collection doesn't exist."""
        mock_collection.return_value = None

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_documents_list("test_collection", [])

            assert result == []

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents_list_exception(self, mock_get_config, mock_connections, mock_collection):
        """Test _get_documents_list method with exception."""
        mock_collection_obj = Mock()
        mock_query_iterator = Mock()
        mock_query_iterator.next.side_effect = Exception("Query error")
        mock_collection_obj.query_iterator.return_value = mock_query_iterator
        mock_collection.return_value = mock_collection_obj

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_documents_list("test_collection", [])

            assert result == []

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents_list_iterator_none(self, mock_get_config, mock_connections, mock_collection):
        """Test _get_documents_list method when iterator returns None."""
        mock_collection_obj = Mock()
        mock_query_iterator = Mock()
        
        # Mock iterator behavior - first call returns data, second returns None
        mock_data_batch = [
            {
                "source": "/path/to/file1.txt",
                "content_metadata": {"field1": "value1"}
            }
        ]
        
        mock_query_iterator.next.side_effect = [mock_data_batch, None]
        mock_collection_obj.query_iterator.return_value = mock_query_iterator
        mock_collection.return_value = mock_collection_obj

        metadata_schema = [{"name": "field1"}]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_documents_list("test_collection", metadata_schema)

            expected = [
                {
                    "document_name": "file1.txt",
                    "metadata": {"field1": "value1"}
                }
            ]
            assert result == expected

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents_list_iterator_attribute_error(self, mock_get_config, mock_connections, mock_collection):
        """Test _get_documents_list method when iterator raises AttributeError."""
        mock_collection_obj = Mock()
        mock_query_iterator = Mock()
        
        # Mock iterator behavior - first call returns data, second raises AttributeError
        mock_data_batch = [
            {
                "source": "/path/to/file1.txt",
                "content_metadata": {"field1": "value1"}
            }
        ]
        
        mock_query_iterator.next.side_effect = [mock_data_batch, AttributeError("No next method")]
        mock_collection_obj.query_iterator.return_value = mock_query_iterator
        mock_collection.return_value = mock_collection_obj

        metadata_schema = [{"name": "field1"}]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb._get_documents_list("test_collection", metadata_schema)

            expected = [
                {
                    "document_name": "file1.txt",
                    "metadata": {"field1": "value1"}
                }
            ]
            assert result == expected

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_documents(self, mock_get_config, mock_connections):
        """Test get_documents method."""
        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            mock_metadata_schema = [{"name": "field1"}]
            mock_documents_list = [{"document_name": "file1.txt"}]

            with patch.object(vdb, 'get_metadata_schema', return_value=mock_metadata_schema) as mock_get_metadata, \
                 patch.object(vdb, '_get_documents_list', return_value=mock_documents_list) as mock_get_docs:

                result = vdb.get_documents("test_collection")

                assert result == mock_documents_list
                mock_get_metadata.assert_called_once_with("test_collection")
                mock_get_docs.assert_called_once_with(
                    collection_name="test_collection", metadata_schema=mock_metadata_schema
                )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_documents_success(self, mock_get_config, mock_connections, mock_collection):
        """Test delete_documents method with successful deletion."""
        mock_collection_obj = Mock()
        mock_resp = Mock()
        mock_resp.delete_count = 5
        mock_collection_obj.delete.return_value = mock_resp
        mock_collection.return_value = mock_collection_obj

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.delete_documents("test_collection", ["file1.txt", "file2.txt"])

            assert result is True
            mock_collection_obj.flush.assert_called_once()

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_documents_not_found(self, mock_get_config, mock_connections, mock_collection):
        """Test delete_documents method when document not found."""
        mock_collection_obj = Mock()
        mock_resp = Mock()
        mock_resp.delete_count = 0
        mock_collection_obj.delete.return_value = mock_resp
        mock_collection.return_value = mock_collection_obj

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.delete_documents("test_collection", ["file1.txt"])

            assert result is False

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Collection')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_delete_documents_milvus_exception(self, mock_get_config, mock_connections, mock_collection):
        """Test delete_documents method with MilvusException fallback."""
        from pymilvus import MilvusException
        
        mock_collection_obj = Mock()
        mock_resp = Mock()
        mock_resp.delete_count = 1
        
        # First call raises MilvusException, second call succeeds
        mock_collection_obj.delete.side_effect = [MilvusException("Error"), mock_resp]
        mock_collection.return_value = mock_collection_obj

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.delete_documents("test_collection", ["file1.txt"])

            assert result is True
            assert mock_collection_obj.delete.call_count == 2

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_create_metadata_schema_collection_new(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test create_metadata_schema_collection when collection doesn't exist."""
        mock_client = Mock()
        mock_client.has_collection.return_value = False
        mock_milvus_client.return_value = mock_client
        
        # Mock schema creation
        mock_schema = Mock()
        mock_milvus_client.create_schema.return_value = mock_schema
        
        # Mock index params
        mock_index_params = Mock()
        mock_milvus_client.prepare_index_params.return_value = mock_index_params

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb.create_metadata_schema_collection()

            mock_client.create_collection.assert_called_once()
            mock_schema.add_field.assert_called()

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_create_metadata_schema_collection_exists(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test create_metadata_schema_collection when collection already exists."""
        mock_client = Mock()
        mock_client.has_collection.return_value = True
        mock_milvus_client.return_value = mock_client

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb.create_metadata_schema_collection()

            mock_client.create_collection.assert_not_called()

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.MilvusClient')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_add_metadata_schema(self, mock_get_config, mock_connections, mock_milvus_client):
        """Test add_metadata_schema method."""
        mock_client = Mock()
        mock_milvus_client.return_value = mock_client

        metadata_schema = [{"name": "field1", "type": "string"}]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            vdb.add_metadata_schema("test_collection", metadata_schema)

            # Should delete existing schema first
            mock_client.delete.assert_called_once_with(
                collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
                filter="collection_name == 'test_collection'"
            )
            
            # Should insert new schema
            mock_client.insert.assert_called_once()

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_metadata_schema_found(self, mock_get_config, mock_connections):
        """Test get_metadata_schema method when schema exists."""
        mock_entities = [{"metadata_schema": [{"name": "field1"}]}]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            with patch.object(vdb, '_get_milvus_entities', return_value=mock_entities) as mock_get_entities:
                result = vdb.get_metadata_schema("test_collection")

                assert result == [{"name": "field1"}]
                mock_get_entities.assert_called_once_with(
                    DEFAULT_METADATA_SCHEMA_COLLECTION, 
                    "collection_name == 'test_collection'"
                )

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.get_config')
    def test_get_metadata_schema_not_found(self, mock_get_config, mock_connections):
        """Test get_metadata_schema method when schema doesn't exist."""
        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            with patch.object(vdb, '_get_milvus_entities', return_value=[]) as mock_get_entities:
                result = vdb.get_metadata_schema("test_collection")

                assert result == []

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.CONFIG')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.time')
    def test_retrieval_langchain(self, mock_time, mock_config, mock_connections):
        """Test retrieval_langchain method."""
        mock_time.time.side_effect = [0.0, 1.5]  # start and end times
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_vectorstore.collection_name = "test_collection"
        # The actual code accesses retriever.vectorstore.collection_name, so we need to set that
        mock_retriever.vectorstore.collection_name = "test_collection"

        mock_docs = [
            Document(page_content="doc1", metadata={"source": "file1.txt"}),
            Document(page_content="doc2", metadata={"source": "file2.txt"})
        ]

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):
            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            with patch.object(vdb, 'get_langchain_vectorstore', return_value=mock_vectorstore) as mock_get_vs, \
                 patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.RunnableLambda') as mock_runnable_lambda, \
                 patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.RunnableAssign') as mock_runnable_assign, \
                 patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.otel_context') as mock_otel:

                # Create a mock chain that will be returned from the | operation
                mock_chain = Mock()
                mock_chain.invoke.return_value = {"context": mock_docs}
                
                # Mock RunnableAssign to properly handle the __ror__ method (when dict | RunnableAssign)
                mock_assign_instance = Mock()
                mock_assign_instance.__ror__ = Mock(return_value=mock_chain)
                mock_runnable_assign.return_value = mock_assign_instance
                
                # Mock RunnableLambda
                mock_lambda_instance = Mock()
                mock_runnable_lambda.return_value = mock_lambda_instance
                
                # Mock otel context
                mock_token = Mock()
                mock_otel.attach.return_value = mock_token
                mock_ctx = Mock()

                result = vdb.retrieval_langchain(
                    query="test query",
                    collection_name="test_collection", 
                    top_k=5,
                    filter_expr="filter",
                    otel_ctx=mock_ctx
                )

                # Verify the results have collection_name added to metadata
                assert len(result) == 2
                for doc in result:
                    assert doc.metadata["collection_name"] == "test_collection"

                mock_get_vs.assert_called_once_with("test_collection")
                mock_otel.attach.assert_called_once_with(mock_ctx)
                mock_otel.detach.assert_called_once_with(mock_token)

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.CONFIG')
    def test_get_langchain_vectorstore_hybrid(self, mock_config, mock_connections):
        """Test get_langchain_vectorstore method for hybrid search."""
        mock_config.vector_store.search_type = "hybrid"

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.LangchainMilvus') as mock_langchain_milvus, \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.BM25BuiltInFunction') as mock_bm25:

            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.get_langchain_vectorstore("test_collection")

            mock_langchain_milvus.assert_called_once()
            call_args = mock_langchain_milvus.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            # Check for vector_field parameter (line 527-530 in source)
            assert call_args[1]["vector_field"] == ["vector", "sparse"]

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.CONFIG')
    def test_get_langchain_vectorstore_dense(self, mock_config, mock_connections):
        """Test get_langchain_vectorstore method for dense search."""
        mock_config.vector_store.search_type = "dense"
        mock_config.vector_store.index_type = "IVF_FLAT"
        mock_config.vector_store.nlist = 1024
        mock_config.vector_store.nprobe = 10

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.LangchainMilvus') as mock_langchain_milvus:

            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            result = vdb.get_langchain_vectorstore("test_collection")

            mock_langchain_milvus.assert_called_once()
            call_args = mock_langchain_milvus.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            # The CONFIG.vector_store.index_type should be used in index_params
            assert call_args[1]["index_params"]["index_type"] == "IVF_FLAT"

    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.connections')
    @patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.CONFIG')
    def test_get_langchain_vectorstore_invalid_search_type(self, mock_config, mock_connections):
        """Test get_langchain_vectorstore method with invalid search type."""
        mock_config.vector_store.search_type = "invalid"

        with patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.urlparse'), \
             patch('nvidia_rag.utils.vdb.milvus.milvus_vdb.Milvus.__init__'):

            vdb = MilvusVDB(
                embedding_model=Mock(),
                milvus_uri="http://localhost:19530",
                collection_name="test_collection"
            )

            with pytest.raises(ValueError, match="invalid search type is not supported"):
                vdb.get_langchain_vectorstore("test_collection")

    

    def test_add_collection_name_to_retrieved_docs(self):
        """Test _add_collection_name_to_retreived_docs static method."""
        docs = [
            Document(page_content="doc1", metadata={"source": "file1.txt"}),
            Document(page_content="doc2", metadata={"source": "file2.txt"})
        ]

        result = MilvusVDB._add_collection_name_to_retreived_docs(docs, "test_collection")

        assert len(result) == 2
        for doc in result:
            assert doc.metadata["collection_name"] == "test_collection"
            assert "source" in doc.metadata  # Original metadata preserved

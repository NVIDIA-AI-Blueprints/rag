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

"""Unit tests for elastic VDB functionality."""

import os
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest
from langchain_core.documents import Document
from opentelemetry import context as otel_context
from pydantic import SecretStr

from nvidia_rag.utils.vdb.elasticsearch import es_queries
from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB


class TestElasticVDB(unittest.TestCase):
    """Test cases for ElasticVDB class."""

    def setUp(self):
        """Set up test fixtures."""
        self.index_name = "test_index"
        self.es_url = "http://localhost:9200"
        self.meta_dataframe = pd.DataFrame({"source": ["doc1"], "field1": ["value1"]})
        self.meta_source_field = "source"
        self.meta_fields = ["field1"]
        self.embedding_model = "test_embedding_model"
        self.csv_file_path = "/path/to/test.csv"

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_init(
        self,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test ElasticVDB initialization."""
        # Mock config
        mock_config = Mock()
        # Ensure embeddings.dimensions is set
        mock_config.embeddings.dimensions = 768
        # Ensure vector_store auth fields are empty (no auth scenario)
        mock_config.vector_store.api_key = None
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = ""
        mock_config.vector_store.password = None
        mock_config = mock_config

        # Mock Elasticsearch connection
        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        # Mock VectorStore
        mock_es_store = Mock()
        mock_vector_store.return_value = mock_es_store

        # Create ElasticVDB instance
        elastic_vdb = ElasticVDB(
            index_name=self.index_name,
            es_url=self.es_url,
            hybrid=True,
            meta_dataframe=self.meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
            embedding_model=self.embedding_model,
            csv_file_path=self.csv_file_path,
            config=mock_config,
        )

        # Assertions
        self.assertEqual(elastic_vdb.index_name, self.index_name)
        self.assertEqual(elastic_vdb.es_url, self.es_url)
        self.assertEqual(elastic_vdb._embedding_model, self.embedding_model)
        self.assertEqual(elastic_vdb.meta_dataframe.equals(self.meta_dataframe), True)
        self.assertEqual(elastic_vdb.meta_source_field, self.meta_source_field)
        self.assertEqual(elastic_vdb.meta_fields, self.meta_fields)
        self.assertEqual(elastic_vdb.csv_file_path, self.csv_file_path)

        # Expect client constructed with only hosts (no auth in this test)
        self.assertTrue(mock_elasticsearch.called)
        _, kwargs = mock_elasticsearch.call_args
        self.assertEqual(kwargs.get("hosts"), [self.es_url])
        # No auth params should be present when no auth is configured
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("basic_auth", kwargs)
        self.assertNotIn("bearer_auth", kwargs)
        mock_vector_store.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_check_index_exists(self, mock_vector_store, mock_elasticsearch):
        """Test _check_index_exists method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.indices.exists.return_value = True

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        result = elastic_vdb._check_index_exists("test_index")

        self.assertTrue(result)
        mock_es_connection.indices.exists.assert_called_once_with(index="test_index")

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_create_index(self, mock_logger, mock_vector_store, mock_elasticsearch):
        """Test create_index method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_es_store = Mock()
        mock_vector_store.return_value = mock_es_store

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        elastic_vdb.create_index()

        mock_logger.info.assert_called_once_with(
            f"Creating Elasticsearch index if not exists: {self.index_name}"
        )
        mock_es_store._create_index_if_not_exists.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.cleanup_records")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_write_to_index(
        self,
        mock_logger,
        mock_cleanup_records,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test write_to_index method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_es_store = Mock()
        mock_vector_store.return_value = mock_es_store

        # Mock cleaned records
        cleaned_records = [
            {
                "text": "test text 1",
                "vector": [0.1, 0.2, 0.3],
                "source": "doc1.pdf",
                "content_metadata": {"title": "Test Doc 1"},
            },
            {
                "text": "test text 2",
                "vector": [0.4, 0.5, 0.6],
                "source": "doc2.pdf",
                "content_metadata": {"title": "Test Doc 2"},
            },
        ]
        mock_cleanup_records.return_value = cleaned_records

        # Test data
        records = [{"raw": "record1"}, {"raw": "record2"}]

        # Create instance and test
        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            meta_dataframe=self.meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
        )
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        elastic_vdb.write_to_index(records)

        # Assertions
        mock_cleanup_records.assert_called_once_with(
            records=records,
            meta_dataframe=self.meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
        )

        expected_texts = ["test text 1", "test text 2"]
        expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        expected_metadatas = [
            {"source": "doc1.pdf", "content_metadata": {"title": "Test Doc 1"}},
            {"source": "doc2.pdf", "content_metadata": {"title": "Test Doc 2"}},
        ]

        mock_es_store.add_texts.assert_called_once_with(
            texts=expected_texts, vectors=expected_vectors, metadatas=expected_metadatas
        )

        mock_logger.info.assert_called_with(
            "Elasticsearch ingestion completed. Total records processed: 2"
        )
        mock_es_connection.indices.refresh.assert_called_once_with(
            index=self.index_name
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_retrieval_not_implemented(self, mock_vector_store, mock_elasticsearch):
        """Test retrieval method raises NotImplementedError."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)

        with self.assertRaises(NotImplementedError) as context:
            elastic_vdb.retrieval(["query1", "query2"])

        self.assertEqual(
            str(context.exception), "retrieval must be implemented for ElasticVDB"
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_reindex_not_implemented(self, mock_vector_store, mock_elasticsearch):
        """Test reindex method raises NotImplementedError."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)

        with self.assertRaises(NotImplementedError) as context:
            elastic_vdb.reindex([{"record": "data"}])

        self.assertEqual(
            str(context.exception), "reindex must be implemented for ElasticVDB"
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_run(self, mock_vector_store, mock_elasticsearch):
        """Test run method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_es_store = Mock()
        mock_vector_store.return_value = mock_es_store

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)

        # Mock the methods that run() calls
        elastic_vdb.create_index = Mock()
        elastic_vdb.write_to_index = Mock()

        records = [{"test": "data"}]
        elastic_vdb.run(records)

        elastic_vdb.create_index.assert_called_once()
        elastic_vdb.write_to_index.assert_called_once_with(records)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_create_collection(self, mock_vector_store, mock_elasticsearch):
        """Test create_collection method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_es_store = Mock()
        mock_vector_store.return_value = mock_es_store

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection

        elastic_vdb.create_collection(
            "test_collection", dimension=1024, collection_type="text"
        )

        mock_es_store._create_index_if_not_exists.assert_called_once()
        mock_es_connection.cluster.health.assert_called_once_with(
            index="test_collection", wait_for_status="yellow", timeout="5s"
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_check_collection_exists(self, mock_vector_store, mock_elasticsearch):
        """Test check_collection_exists method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.indices.exists.return_value = True

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        result = elastic_vdb.check_collection_exists("test_collection")

        self.assertTrue(result)
        mock_es_connection.indices.exists.assert_called_once_with(
            index="test_collection"
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_get_collection(self, mock_vector_store, mock_elasticsearch):
        """Test get_collection method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        # Mock cat.indices response
        mock_indices_response = [
            {"index": "test_index_1", "docs.count": "100"},
            {"index": ".hidden_index", "docs.count": "50"},  # Should be ignored
            {"index": "test_index_2", "docs.count": "200"},
        ]
        mock_es_connection.cat.indices.return_value = mock_indices_response

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        elastic_vdb.create_metadata_schema_collection = Mock()
        elastic_vdb.create_document_info_collection = Mock()
        elastic_vdb.get_metadata_schema = Mock(
            side_effect=[
                [{"name": "field1", "type": "string"}],
                [{"name": "field2", "type": "integer"}],
            ]
        )
        elastic_vdb.get_document_info = Mock(
            side_effect=[
                # First collection: catalog data (info_type='catalog')
                {"description": "Test collection 1", "tags": ["test"]},
                # First collection: metrics data (info_type='collection')
                {"total_pages": 10, "total_chunks": 100},
                # Second collection: catalog data (info_type='catalog')
                {"description": "Test collection 2", "tags": ["prod"]},
                # Second collection: metrics data (info_type='collection')
                {"total_pages": 20, "total_chunks": 200},
            ]
        )

        result = elastic_vdb.get_collection()

        expected_result = [
            {
                "collection_name": "test_index_1",
                "num_entities": "100",
                "metadata_schema": [{"name": "field1", "type": "string"}],
                "collection_info": {
                    "description": "Test collection 1",
                    "tags": ["test"],
                    "total_pages": 10,
                    "total_chunks": 100,
                },
            },
            {
                "collection_name": "test_index_2",
                "num_entities": "200",
                "metadata_schema": [{"name": "field2", "type": "integer"}],
                "collection_info": {
                    "description": "Test collection 2",
                    "tags": ["prod"],
                    "total_pages": 20,
                    "total_chunks": 200,
                },
            },
        ]

        self.assertEqual(result, expected_result)
        elastic_vdb.create_metadata_schema_collection.assert_called_once()
        elastic_vdb.create_document_info_collection.assert_called_once()
        mock_es_connection.cat.indices.assert_called_once_with(format="json")

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_delete_metadata_schema_query"
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_delete_document_info_query_by_collection_name"
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_DOCUMENT_INFO_COLLECTION",
        "document_info",
    )
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_delete_collections(
        self,
        mock_logger,
        mock_delete_doc_info_query,
        mock_delete_query,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test delete_collections method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.indices.delete.return_value = {"acknowledged": True}
        mock_es_connection.delete_by_query.return_value = {"deleted": 1}

        mock_delete_query.return_value = {"query": "test_query"}
        mock_delete_doc_info_query.return_value = {"query": "test_doc_info_query"}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        collection_names = ["collection1", "collection2"]

        result = elastic_vdb.delete_collections(collection_names)

        expected_result = {
            "message": "Collection deletion process completed.",
            "successful": collection_names,
            "failed": [],
            "total_success": 2,
            "total_failed": 0,
        }

        self.assertEqual(result, expected_result)
        mock_es_connection.indices.delete.assert_called_once_with(
            index="collection1,collection2", ignore_unavailable=True
        )
        # Now expects 4 calls: 2 for metadata schema and 2 for document info
        self.assertEqual(mock_es_connection.delete_by_query.call_count, 4)
        mock_logger.info.assert_called_with(f"Collections deleted: {collection_names}")

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_unique_sources_query")
    def test_get_documents(
        self, mock_sources_query, mock_vector_store, mock_elasticsearch
    ):
        """Test get_documents method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        # Mock search response
        mock_search_response = {
            "aggregations": {
                "unique_sources": {
                    "buckets": [
                        {
                            "key": {"source_name": "/path/to/doc1.pdf"},
                            "top_hit": {
                                "hits": {
                                    "hits": [
                                        {
                                            "_source": {
                                                "metadata": {
                                                    "content_metadata": {
                                                        "title": "Document 1",
                                                        "author": "Author 1",
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                        },
                        {
                            "key": {"source_name": "/path/to/doc2.pdf"},
                            "top_hit": {
                                "hits": {
                                    "hits": [
                                        {
                                            "_source": {
                                                "metadata": {
                                                    "content_metadata": {
                                                        "title": "Document 2"
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                        },
                    ]
                }
            }
        }
        mock_es_connection.search.return_value = mock_search_response
        mock_sources_query.return_value = {"query": "test_query"}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        elastic_vdb.get_metadata_schema = Mock(
            return_value=[{"name": "title"}, {"name": "author"}]
        )
        elastic_vdb.get_document_info = Mock(
            side_effect=[
                {"total_pages": 5, "total_chunks": 50},
                {"total_pages": 10, "total_chunks": 100},
            ]
        )

        result = elastic_vdb.get_documents("test_collection")

        expected_result = [
            {
                "document_name": "doc1.pdf",
                "metadata": {"title": "Document 1", "author": "Author 1"},
                "document_info": {"total_pages": 5, "total_chunks": 50},
            },
            {
                "document_name": "doc2.pdf",
                "metadata": {"title": "Document 2", "author": None},
                "document_info": {"total_pages": 10, "total_chunks": 100},
            },
        ]

        self.assertEqual(result, expected_result)
        mock_es_connection.search.assert_called_once_with(
            index="test_collection", body={"query": "test_query"}
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_delete_docs_query")
    def test_delete_documents(
        self, mock_delete_query, mock_vector_store, mock_elasticsearch
    ):
        """Test delete_documents method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.delete_by_query.return_value = {"deleted": 1}

        mock_delete_query.side_effect = [
            {"query": {"term": {"source": "doc1.pdf"}}},
            {"query": {"term": {"source": "doc2.pdf"}}},
        ]

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        source_values = ["doc1.pdf", "doc2.pdf"]

        result = elastic_vdb.delete_documents("test_collection", source_values)

        self.assertTrue(result)
        self.assertEqual(mock_es_connection.delete_by_query.call_count, 2)
        mock_es_connection.indices.refresh.assert_called_once_with(
            index="test_collection"
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.create_metadata_collection_mapping"
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_create_metadata_schema_collection_new(
        self,
        mock_logger,
        mock_mapping,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test create_metadata_schema_collection method when collection doesn't exist."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.indices.exists.return_value = False
        mock_es_connection.indices.create.return_value = {"acknowledged": True}

        mock_mapping.return_value = {"mappings": {"properties": {}}}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        elastic_vdb.create_metadata_schema_collection()

        mock_es_connection.indices.exists.assert_called_once_with(
            index="metadata_schema"
        )
        mock_es_connection.indices.create.assert_called_once_with(
            index="metadata_schema", body={"mappings": {"properties": {}}}
        )
        mock_logger.info.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.create_metadata_collection_mapping"
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_create_metadata_schema_collection_exists(
        self,
        mock_logger,
        mock_mapping,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test create_metadata_schema_collection method when collection exists."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.indices.exists.return_value = True

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        elastic_vdb.create_metadata_schema_collection()

        mock_es_connection.indices.exists.assert_called_once_with(
            index="metadata_schema"
        )
        mock_es_connection.indices.create.assert_not_called()
        mock_logger.info.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_delete_metadata_schema_query"
    )
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_add_metadata_schema(
        self,
        mock_logger,
        mock_delete_query,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test add_metadata_schema method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_es_connection.delete_by_query.return_value = {"deleted": 1}
        mock_es_connection.index.return_value = {"_id": "test_id"}

        mock_delete_query.return_value = {"query": "delete_query"}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        metadata_schema = [{"name": "title", "type": "string"}]

        elastic_vdb.add_metadata_schema("test_collection", metadata_schema)

        expected_data = {
            "collection_name": "test_collection",
            "metadata_schema": metadata_schema,
        }

        mock_es_connection.delete_by_query.assert_called_once_with(
            index="metadata_schema", body={"query": "delete_query"}
        )
        mock_es_connection.index.assert_called_once_with(
            index="metadata_schema", body=expected_data
        )
        mock_logger.info.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_metadata_schema_query")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    def test_get_metadata_schema_found(
        self, mock_schema_query, mock_vector_store, mock_elasticsearch
    ):
        """Test get_metadata_schema method when schema is found."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_search_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "metadata_schema": [{"name": "title", "type": "string"}]
                        }
                    }
                ]
            }
        }
        mock_es_connection.search.return_value = mock_search_response
        mock_schema_query.return_value = {"query": "schema_query"}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        result = elastic_vdb.get_metadata_schema("test_collection")

        expected_result = [{"name": "title", "type": "string"}]
        self.assertEqual(result, expected_result)

        mock_es_connection.search.assert_called_once_with(
            index="metadata_schema", body={"query": "schema_query"}
        )

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_metadata_schema_query")
    @patch(
        "nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DEFAULT_METADATA_SCHEMA_COLLECTION",
        "metadata_schema",
    )
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_get_metadata_schema_not_found(
        self,
        mock_logger,
        mock_schema_query,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test get_metadata_schema method when schema is not found."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_search_response = {"hits": {"hits": []}}
        mock_es_connection.search.return_value = mock_search_response
        mock_schema_query.return_value = {"query": "schema_query"}

        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url)
        # Replace the actual connection with our mock
        elastic_vdb._es_connection = mock_es_connection
        result = elastic_vdb.get_metadata_schema("test_collection")

        self.assertEqual(result, [])
        mock_logger.info.assert_called_once()

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.time")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.otel_context")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.logger")
    def test_retrieval_langchain(
        self,
        mock_logger,
        mock_otel_context,
        mock_time,
        mock_es_store_class,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test retrieval_langchain method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        mock_config = mock_config

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        # Mock time
        mock_time.time.side_effect = [1000.0, 1002.5]  # 2.5 second latency

        # Mock otel context
        mock_token = Mock()
        mock_otel_context.attach.return_value = mock_token

        # Mock ElasticsearchStore
        mock_vectorstore = Mock()
        mock_es_store_class.return_value = mock_vectorstore

        # Mock retriever
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        # Mock documents
        mock_docs = [
            Document(page_content="doc1", metadata={"source": "file1.pdf"}),
            Document(page_content="doc2", metadata={"source": "file2.pdf"}),
        ]
        mock_retriever.invoke.return_value = mock_docs

        # Create instance and test
        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
        )

        result = elastic_vdb.retrieval_langchain(
            query="test query",
            collection_name="test_collection",
            top_k=5,
            filter_expr={"field": "value"},
            otel_ctx=Mock(),
        )

        # Verify results have collection_name added
        for doc in result:
            self.assertEqual(doc.metadata["collection_name"], "test_collection")

        mock_logger.info.assert_called_with(
            " Elasticsearch Retrieval latency: 2.5000 seconds"
        )
        mock_otel_context.attach.assert_called_once()
        mock_otel_context.detach.assert_called_once_with(mock_token)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy")
    def test_get_langchain_vectorstore_no_auth(
        self,
        mock_dense_strategy,
        mock_es_store_class,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test get_langchain_vectorstore method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        # Ensure no auth present in CONFIG to reflect "no auth" scenario
        mock_config.vector_store.api_key = None
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = ""
        mock_config.vector_store.password = None

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection

        mock_vectorstore = Mock()
        mock_es_store_class.return_value = mock_vectorstore

        mock_strategy = Mock()
        mock_dense_strategy.return_value = mock_strategy

        # Create instance and test
        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
        )

        # Reset mock to only track calls from the method being tested
        mock_dense_strategy.reset_mock()

        result = elastic_vdb.get_langchain_vectorstore("test_collection")

        self.assertEqual(result, mock_vectorstore)

        # Expect no auth args passed to vectorstore
        self.assertTrue(mock_es_store_class.called)
        _, vs_kwargs = mock_es_store_class.call_args
        self.assertEqual(vs_kwargs.get("index_name"), "test_collection")
        self.assertEqual(vs_kwargs.get("es_url"), self.es_url)
        self.assertNotIn("es_api_key", vs_kwargs)
        self.assertNotIn("es_user", vs_kwargs)
        self.assertNotIn("es_password", vs_kwargs)
        self.assertNotIn("es_params", vs_kwargs)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy")
    def test_get_langchain_vectorstore_basic_auth(
        self,
        mock_dense_strategy,
        mock_es_store_class,
    ):
        """Vectorstore uses basic auth when username/password are set and no API key."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.api_key = None
        mock_config.vector_store.api_key_id = None
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = "elastic"
        mock_config.vector_store.password = SecretStr("password")
        mock_es_store_class.return_value = Mock()
        mock_dense_strategy.return_value = Mock()
        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
        )
        _ = elastic_vdb.get_langchain_vectorstore("test_collection")
        _, vs_kwargs = mock_es_store_class.call_args
        self.assertEqual(vs_kwargs.get("es_user"), "elastic")
        self.assertEqual(vs_kwargs.get("es_password"), "password")
        self.assertNotIn("es_api_key", vs_kwargs)
        self.assertNotIn("es_params", vs_kwargs)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy")
    def test_get_langchain_vectorstore_api_key_precedence(
        self,
        mock_dense_strategy,
        mock_es_store_class,
    ):
        """API key should be preferred over basic auth when both are present."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.api_key = SecretStr("base64-id-secret")
        mock_config.vector_store.api_key_id = None
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = "elastic"
        mock_config.vector_store.password = SecretStr("password")
        mock_es_store_class.return_value = Mock()
        mock_dense_strategy.return_value = Mock()
        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
        )
        _ = elastic_vdb.get_langchain_vectorstore("test_collection")
        _, vs_kwargs = mock_es_store_class.call_args
        self.assertEqual(vs_kwargs.get("es_api_key"), "base64-id-secret")
        self.assertNotIn("es_user", vs_kwargs)
        self.assertNotIn("es_password", vs_kwargs)
        self.assertNotIn("es_params", vs_kwargs)

        # Ensure called with hybrid=True (may be invoked elsewhere too)
        mock_dense_strategy.assert_any_call(hybrid=True)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_init_with_bearer_auth(
        self,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Test ElasticVDB initialization with bearer auth token."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.api_key = None
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = ""
        mock_config.vector_store.password = None

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_vector_store.return_value = Mock()

        # Create with auth_token
        elastic_vdb = ElasticVDB(
            index_name=self.index_name,
            es_url=self.es_url,
            config=mock_config,
            auth_token="test_bearer_token",
        )

        # Verify bearer_auth was passed to Elasticsearch client
        self.assertTrue(mock_elasticsearch.called)
        _, kwargs = mock_elasticsearch.call_args
        self.assertEqual(kwargs.get("bearer_auth"), "test_bearer_token")
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("basic_auth", kwargs)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore")
    def test_bearer_auth_priority_over_api_key(
        self,
        mock_vector_store,
        mock_elasticsearch,
    ):
        """Bearer auth should have priority over API key and basic auth."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.api_key = SecretStr("some_api_key")
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = "elastic"
        mock_config.vector_store.password = SecretStr("password")

        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        mock_vector_store.return_value = Mock()

        # Create with auth_token (should override api_key and basic_auth)
        elastic_vdb = ElasticVDB(
            index_name=self.index_name,
            es_url=self.es_url,
            config=mock_config,
            auth_token="test_bearer_token",
        )

        # Verify only bearer_auth was passed
        self.assertTrue(mock_elasticsearch.called)
        _, kwargs = mock_elasticsearch.call_args
        self.assertEqual(kwargs.get("bearer_auth"), "test_bearer_token")
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("basic_auth", kwargs)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy")
    def test_get_langchain_vectorstore_bearer_auth(
        self,
        mock_dense_strategy,
        mock_es_store_class,
    ):
        """Test get_langchain_vectorstore uses bearer_auth via es_params."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.api_key = None
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = ""
        mock_config.vector_store.password = None

        mock_es_store_class.return_value = Mock()
        mock_dense_strategy.return_value = Mock()

        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
            auth_token="test_bearer_token",
        )

        _ = elastic_vdb.get_langchain_vectorstore("test_collection")
        _, vs_kwargs = mock_es_store_class.call_args

        # Bearer auth should be passed via es_params
        self.assertIn("es_params", vs_kwargs)
        self.assertEqual(vs_kwargs["es_params"], {"bearer_auth": "test_bearer_token"})
        self.assertNotIn("es_api_key", vs_kwargs)
        self.assertNotIn("es_user", vs_kwargs)
        self.assertNotIn("es_password", vs_kwargs)

    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore")
    @patch("nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy")
    def test_get_langchain_vectorstore_bearer_auth_priority(
        self,
        mock_dense_strategy,
        mock_es_store_class,
    ):
        """Bearer auth should take precedence over API key and basic auth in vectorstore."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_config.vector_store.search_type = "hybrid"
        mock_config.vector_store.api_key = SecretStr("some_api_key")
        mock_config.vector_store.api_key_id = ""
        mock_config.vector_store.api_key_secret = None
        mock_config.vector_store.username = "elastic"
        mock_config.vector_store.password = SecretStr("password")

        mock_es_store_class.return_value = Mock()
        mock_dense_strategy.return_value = Mock()

        elastic_vdb = ElasticVDB(
            self.index_name,
            self.es_url,
            embedding_model="test_model",
            config=mock_config,
            auth_token="test_bearer_token",
        )

        _ = elastic_vdb.get_langchain_vectorstore("test_collection")
        _, vs_kwargs = mock_es_store_class.call_args

        # Only bearer auth via es_params should be present
        self.assertIn("es_params", vs_kwargs)
        self.assertEqual(vs_kwargs["es_params"], {"bearer_auth": "test_bearer_token"})
        self.assertNotIn("es_api_key", vs_kwargs)
        self.assertNotIn("es_user", vs_kwargs)
        self.assertNotIn("es_password", vs_kwargs)

    def test_add_collection_name_to_retreived_docs(self):
        """Test _add_collection_name_to_retreived_docs static method."""
        # Create test documents
        docs = [
            Document(page_content="doc1", metadata={"source": "file1.pdf"}),
            Document(page_content="doc2", metadata={"source": "file2.pdf"}),
        ]

        # Test the static method
        result = ElasticVDB._add_collection_name_to_retreived_docs(
            docs, "test_collection"
        )

        # Verify collection_name is added to metadata
        for doc in result:
            self.assertEqual(doc.metadata["collection_name"], "test_collection")

        # Verify original metadata is preserved
        self.assertEqual(result[0].metadata["source"], "file1.pdf")
        self.assertEqual(result[1].metadata["source"], "file2.pdf")


class TestEsQueries(unittest.TestCase):
    """Test cases for es_queries module functions."""

    def test_get_unique_sources_query(self):
        """Test get_unique_sources_query function returns correct aggregation query."""
        result = es_queries.get_unique_sources_query()

        # Verify the basic structure
        self.assertIn("size", result)
        self.assertEqual(result["size"], 0)
        self.assertIn("aggs", result)

        # Verify aggregation structure
        unique_sources = result["aggs"]["unique_sources"]
        self.assertIn("composite", unique_sources)
        self.assertIn("aggs", unique_sources)

        # Verify composite aggregation
        composite = unique_sources["composite"]
        self.assertEqual(composite["size"], 1000)
        self.assertIn("sources", composite)

        # Verify source field configuration
        sources = composite["sources"][0]
        self.assertIn("source_name", sources)
        terms = sources["source_name"]["terms"]
        self.assertEqual(terms["field"], "metadata.source.source_name.keyword")

        # Verify top_hits aggregation
        top_hit = unique_sources["aggs"]["top_hit"]
        self.assertIn("top_hits", top_hit)
        self.assertEqual(top_hit["top_hits"]["size"], 1)

    def test_get_delete_metadata_schema_query(self):
        """Test get_delete_metadata_schema_query function with collection name."""
        collection_name = "test_collection"
        result = es_queries.get_delete_metadata_schema_query(collection_name)

        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])

        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("collection_name.keyword", term_query)
        self.assertEqual(term_query["collection_name.keyword"], collection_name)

    def test_get_metadata_schema_query(self):
        """Test get_metadata_schema_query function with collection name."""
        collection_name = "test_collection"
        result = es_queries.get_metadata_schema_query(collection_name)

        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])

        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("collection_name", term_query)
        self.assertEqual(term_query["collection_name"], collection_name)

    def test_get_delete_docs_query(self):
        """Test get_delete_docs_query function with source value."""
        source_value = "test_document.pdf"
        result = es_queries.get_delete_docs_query(source_value)

        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])

        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("metadata.source.source_name.keyword", term_query)
        self.assertEqual(
            term_query["metadata.source.source_name.keyword"], source_value
        )

    def test_create_metadata_collection_mapping(self):
        """Test create_metadata_collection_mapping function returns correct mapping."""
        result = es_queries.create_metadata_collection_mapping()

        # Verify top-level structure
        self.assertIn("mappings", result)
        self.assertIn("properties", result["mappings"])

        # Verify properties structure
        properties = result["mappings"]["properties"]
        self.assertIn("collection_name", properties)
        self.assertIn("metadata_schema", properties)

        # Verify collection_name field
        collection_name_field = properties["collection_name"]
        self.assertEqual(collection_name_field["type"], "keyword")

        # Verify metadata_schema field
        metadata_schema_field = properties["metadata_schema"]
        self.assertEqual(metadata_schema_field["type"], "object")
        self.assertTrue(metadata_schema_field["enabled"])

    def test_get_delete_metadata_schema_query_empty_collection(self):
        """Test get_delete_metadata_schema_query with empty collection name."""
        collection_name = ""
        result = es_queries.get_delete_metadata_schema_query(collection_name)

        # Should still return valid structure with empty string
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(term_query["collection_name.keyword"], "")

    def test_get_metadata_schema_query_special_characters(self):
        """Test get_metadata_schema_query with special characters in collection name."""
        collection_name = "test-collection_with.special@chars"
        result = es_queries.get_metadata_schema_query(collection_name)

        # Should handle special characters properly
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(term_query["collection_name"], collection_name)

    def test_get_delete_docs_query_with_spaces(self):
        """Test get_delete_docs_query with source value containing spaces."""
        source_value = "document with spaces.pdf"
        result = es_queries.get_delete_docs_query(source_value)

        # Should handle spaces in source value
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(
            term_query["metadata.source.source_name.keyword"], source_value
        )


if __name__ == "__main__":
    unittest.main()

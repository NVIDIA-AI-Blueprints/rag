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

"""Unit tests for Elasticsearch query utilities."""

import pytest

from nvidia_rag.utils.vdb.elasticsearch import es_queries


class TestGetUniqueSourcesQuery:
    """Test cases for get_unique_sources_query function."""

    def test_get_unique_sources_query_structure(self):
        """Test that get_unique_sources_query returns correct query structure."""
        query = es_queries.get_unique_sources_query()

        assert query["size"] == 0
        assert "aggs" in query
        assert "unique_sources" in query["aggs"]
        assert query["aggs"]["unique_sources"]["composite"]["size"] == 1000
        assert "top_hit" in query["aggs"]["unique_sources"]["aggs"]
        assert (
            query["aggs"]["unique_sources"]["composite"]["sources"][0]["source_name"][
                "terms"
            ]["field"]
            == "metadata.source.source_name.keyword"
        )


class TestGetDeleteMetadataSchemaQuery:
    """Test cases for get_delete_metadata_schema_query function."""

    def test_get_delete_metadata_schema_query(self):
        """Test that get_delete_metadata_schema_query returns correct query."""
        collection_name = "test_collection"
        query = es_queries.get_delete_metadata_schema_query(collection_name)

        assert "query" in query
        assert query["query"]["term"]["collection_name.keyword"] == collection_name

    def test_get_delete_metadata_schema_query_empty_name(self):
        """Test get_delete_metadata_schema_query with empty collection name."""
        query = es_queries.get_delete_metadata_schema_query("")

        assert query["query"]["term"]["collection_name.keyword"] == ""


class TestGetMetadataSchemaQuery:
    """Test cases for get_metadata_schema_query function."""

    def test_get_metadata_schema_query(self):
        """Test that get_metadata_schema_query returns correct query."""
        collection_name = "test_collection"
        query = es_queries.get_metadata_schema_query(collection_name)

        assert "query" in query
        assert query["query"]["term"]["collection_name"] == collection_name

    def test_get_metadata_schema_query_empty_name(self):
        """Test get_metadata_schema_query with empty collection name."""
        query = es_queries.get_metadata_schema_query("")

        assert query["query"]["term"]["collection_name"] == ""


class TestGetDeleteDocsQuery:
    """Test cases for get_delete_docs_query function."""

    def test_get_delete_docs_query(self):
        """Test that get_delete_docs_query returns correct query."""
        source_value = "test_document.pdf"
        query = es_queries.get_delete_docs_query(source_value)

        assert "query" in query
        assert (
            query["query"]["term"]["metadata.source.source_name.keyword"]
            == source_value
        )

    def test_get_delete_docs_query_empty_source(self):
        """Test get_delete_docs_query with empty source value."""
        query = es_queries.get_delete_docs_query("")

        assert query["query"]["term"]["metadata.source.source_name.keyword"] == ""


class TestCreateMetadataCollectionMapping:
    """Test cases for create_metadata_collection_mapping function."""

    def test_create_metadata_collection_mapping(self):
        """Test that create_metadata_collection_mapping returns correct mapping."""
        mapping = es_queries.create_metadata_collection_mapping()

        assert "mappings" in mapping
        assert "properties" in mapping["mappings"]
        assert mapping["mappings"]["properties"]["collection_name"]["type"] == "keyword"
        assert mapping["mappings"]["properties"]["metadata_schema"]["type"] == "object"
        assert mapping["mappings"]["properties"]["metadata_schema"]["enabled"] is True


class TestCreateDocumentInfoCollectionMapping:
    """Test cases for create_document_info_collection_mapping function."""

    def test_create_document_info_collection_mapping(self):
        """Test that create_document_info_collection_mapping returns correct mapping."""
        mapping = es_queries.create_document_info_collection_mapping()

        assert "mappings" in mapping
        assert "properties" in mapping["mappings"]
        assert mapping["mappings"]["properties"]["collection_name"]["type"] == "keyword"
        assert mapping["mappings"]["properties"]["info_type"]["type"] == "keyword"
        assert mapping["mappings"]["properties"]["document_name"]["type"] == "keyword"
        assert mapping["mappings"]["properties"]["info_value"]["type"] == "object"
        assert mapping["mappings"]["properties"]["info_value"]["enabled"] is True


class TestGetDeleteDocumentInfoQuery:
    """Test cases for get_delete_document_info_query function."""

    def test_get_delete_document_info_query(self):
        """Test that get_delete_document_info_query returns correct query."""
        collection_name = "test_collection"
        document_name = "test_doc.pdf"
        info_type = "summary"

        query = es_queries.get_delete_document_info_query(
            collection_name, document_name, info_type
        )

        assert "query" in query
        assert "bool" in query["query"]
        assert "must" in query["query"]["bool"]
        assert len(query["query"]["bool"]["must"]) == 3
        assert {"term": {"collection_name": collection_name}} in query["query"]["bool"][
            "must"
        ]
        assert {"term": {"document_name": document_name}} in query["query"]["bool"][
            "must"
        ]
        assert {"term": {"info_type": info_type}} in query["query"]["bool"]["must"]

    def test_get_delete_document_info_query_empty_values(self):
        """Test get_delete_document_info_query with empty values."""
        query = es_queries.get_delete_document_info_query("", "", "")

        assert len(query["query"]["bool"]["must"]) == 3


class TestGetCollectionDocumentInfoQuery:
    """Test cases for get_collection_document_info_query function."""

    def test_get_collection_document_info_query(self):
        """Test that get_collection_document_info_query returns correct query."""
        info_type = "summary"
        collection_name = "test_collection"

        query = es_queries.get_collection_document_info_query(
            info_type, collection_name
        )

        assert "query" in query
        assert "bool" in query["query"]
        assert "must" in query["query"]["bool"]
        assert len(query["query"]["bool"]["must"]) == 2
        assert {"term": {"collection_name": collection_name}} in query["query"]["bool"][
            "must"
        ]
        assert {"term": {"info_type": info_type}} in query["query"]["bool"]["must"]

    def test_get_collection_document_info_query_empty_values(self):
        """Test get_collection_document_info_query with empty values."""
        query = es_queries.get_collection_document_info_query("", "")

        assert len(query["query"]["bool"]["must"]) == 2


class TestGetDocumentInfoQuery:
    """Test cases for get_document_info_query function."""

    def test_get_document_info_query(self):
        """Test that get_document_info_query returns correct query."""
        collection_name = "test_collection"
        document_name = "test_doc.pdf"
        info_type = "summary"

        query = es_queries.get_document_info_query(
            collection_name, document_name, info_type
        )

        assert "query" in query
        assert "bool" in query["query"]
        assert "must" in query["query"]["bool"]
        assert len(query["query"]["bool"]["must"]) == 3
        assert {"term": {"collection_name": collection_name}} in query["query"]["bool"][
            "must"
        ]
        assert {"term": {"document_name": document_name}} in query["query"]["bool"][
            "must"
        ]
        assert {"term": {"info_type": info_type}} in query["query"]["bool"]["must"]

    def test_get_document_info_query_empty_values(self):
        """Test get_document_info_query with empty values."""
        query = es_queries.get_document_info_query("", "", "")

        assert len(query["query"]["bool"]["must"]) == 3


class TestGetDeleteDocumentInfoQueryByCollectionName:
    """Test cases for get_delete_document_info_query_by_collection_name function."""

    def test_get_delete_document_info_query_by_collection_name(self):
        """Test that get_delete_document_info_query_by_collection_name returns correct query."""
        collection_name = "test_collection"
        query = es_queries.get_delete_document_info_query_by_collection_name(
            collection_name
        )

        assert "query" in query
        assert query["query"]["term"]["collection_name"] == collection_name

    def test_get_delete_document_info_query_by_collection_name_empty(self):
        """Test get_delete_document_info_query_by_collection_name with empty name."""
        query = es_queries.get_delete_document_info_query_by_collection_name("")

        assert query["query"]["term"]["collection_name"] == ""

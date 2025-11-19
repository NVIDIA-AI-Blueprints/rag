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
This module contains Elasticsearch query utilities for vector database operations.
Provides pre-built query functions for document and metadata management in Elasticsearch.

1. get_unique_sources_query: Generate aggregation query to retrieve all unique document sources
2. get_delete_docs_query: Construct deletion query for documents matching the source value
3. get_metadata_schema_query: Build search query to retrieve metadata schema for specified collection
4. get_delete_metadata_schema_query: Create deletion query for removing metadata schema by collection name
5. create_metadata_collection_mapping: Generate Elasticsearch index mapping for metadata schema collections
"""


def get_unique_sources_query():
    """
    Generate aggregation query to retrieve all unique document sources.
    """
    query_unique_sources = {
        "size": 0,
        "aggs": {
            "unique_sources": {
                "composite": {
                    "size": 1000,  # Adjust size depending on number of unique values
                    "sources": [
                        {
                            "source_name": {
                                "terms": {
                                    "field": "metadata.source.source_name.keyword"
                                }
                            }
                        }
                    ],
                },
                "aggs": {
                    "top_hit": {
                        "top_hits": {
                            "size": 1  # Just one document per source_name
                        }
                    }
                },
            }
        },
    }
    return query_unique_sources


def get_delete_metadata_schema_query(collection_name: str):
    """
    Create deletion query for removing metadata schema by collection name.
    """
    query_delete_metadata_schema = {
        "query": {"term": {"collection_name.keyword": collection_name}}
    }
    return query_delete_metadata_schema


def get_metadata_schema_query(collection_name: str):
    """
    Build search query to retrieve metadata schema for specified collection.
    """
    query_metadata_schema = {"query": {"term": {"collection_name": collection_name}}}
    return query_metadata_schema


def get_delete_docs_query(source_value: str):
    """
    Construct deletion query for documents matching the source value.
    """
    query_delete_documents = {
        "query": {"term": {"metadata.source.source_name.keyword": source_value}}
    }
    return query_delete_documents


def create_metadata_collection_mapping():
    """Generate Elasticsearch index mapping for metadata schema collections."""
    return {
        "mappings": {
            "properties": {
                "collection_name": {
                    "type": "keyword"  # or "text" depending on your search needs
                },
                "metadata_schema": {
                    "type": "object",  # For JSON-like structure
                    "enabled": True,  # Set to False if you don't want to index its fields
                },
            }
        }
    }


def create_document_info_collection_mapping():
    """Generate Elasticsearch index mapping for document info collections."""
    return {
        "mappings": {
            "properties": {
                "collection_name": {
                    "type": "keyword"  # or "text" depending on your search needs
                },
                "info_type": {
                    "type": "keyword"  # or "text" depending on your search needs
                },
                "document_name": {
                    "type": "keyword"  # or "text" depending on your search needs
                },
                "info_value": {
                    "type": "object",  # For JSON-like structure
                    "enabled": True,  # Set to False if you don't want to index its fields
                },
            }
        }
    }


def get_delete_document_info_query(
    collection_name: str, document_name: str, info_type: str
):
    """
    Create deletion query for removing document info by collection name, document name, and info type.
    """
    query_delete_document_info = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"collection_name": collection_name}},
                    {"term": {"document_name": document_name}},
                    {"term": {"info_type": info_type}},
                ]
            }
        }
    }
    return query_delete_document_info


def get_collection_document_info_query(info_type: str, collection_name: str):
    """
    Create search query for retrieving document info by collection name and info type.
    """
    query_collection_document_info = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"collection_name": collection_name}},
                    {"term": {"info_type": info_type}},
                ]
            }
        }
    }
    return query_collection_document_info


def get_document_info_query(collection_name: str, document_name: str, info_type: str):
    """
    Create search query for retrieving document info by collection name, document name, and info type.
    """
    query_document_info = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"collection_name": collection_name}},
                    {"term": {"document_name": document_name}},
                    {"term": {"info_type": info_type}},
                ]
            }
        }
    }
    return query_document_info


def get_delete_document_info_query_by_collection_name(collection_name: str):
    """
    Create deletion query for removing document info by collection name.
    """
    query_delete_document_info = {
        "query": {"term": {"collection_name": collection_name}}
    }
    return query_delete_document_info

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
Oracle 26ai SQL query utilities for vector database operations.
Provides DDL and query functions for document and metadata management.

Functions:
1. create_vector_table_ddl: Generate DDL for vector collection tables
2. create_vector_index_ddl: Generate DDL for IVF vector index
3. create_metadata_schema_table_ddl: DDL for metadata schema storage
4. create_document_info_table_ddl: DDL for document info storage
5. get_unique_sources_query: Retrieve all unique document sources
6. get_delete_docs_query: Delete documents by source value
7. get_similarity_search_query: Vector similarity search query
8. get_hybrid_search_query: Combined vector + text search query
"""

from typing import Literal

DistanceMetric = Literal["COSINE", "L2", "DOT", "MANHATTAN"]
IndexType = Literal["IVF", "HNSW"]


def create_vector_table_ddl(
    table_name: str,
    dimension: int = 2048,
) -> str:
    """
    Generate DDL for creating a vector collection table.

    Args:
        table_name: Name of the table to create
        dimension: Vector dimension size

    Returns:
        DDL statement string
    """
    return f"""
    CREATE TABLE {table_name} (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        text CLOB,
        vector VECTOR({dimension}, FLOAT32),
        source VARCHAR2(4000),
        content_metadata CLOB CHECK (content_metadata IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """


def create_vector_index_ddl(
    table_name: str,
    index_type: IndexType = "IVF",
    distance_metric: DistanceMetric = "COSINE",
    ivf_neighbor_partitions: int = 100,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 200,
) -> str:
    """
    Generate DDL for creating a vector index (IVF or HNSW).

    Args:
        table_name: Name of the table
        index_type: IVF or HNSW
        distance_metric: COSINE, L2, DOT, or MANHATTAN
        ivf_neighbor_partitions: Number of partitions for IVF index
        hnsw_m: Max connections per node for HNSW
        hnsw_ef_construction: Size of dynamic candidate list for HNSW

    Returns:
        DDL statement string
    """
    index_name = f"{table_name}_vec_idx"

    if index_type == "IVF":
        return f"""
        CREATE VECTOR INDEX {index_name} ON {table_name}(vector)
        ORGANIZATION NEIGHBOR PARTITIONS
        WITH DISTANCE {distance_metric}
        WITH TARGET ACCURACY 95
        PARAMETERS (
            type IVF,
            neighbor_partitions {ivf_neighbor_partitions}
        )
        """
    else:  # HNSW
        return f"""
        CREATE VECTOR INDEX {index_name} ON {table_name}(vector)
        ORGANIZATION INMEMORY NEIGHBOR GRAPH
        WITH DISTANCE {distance_metric}
        WITH TARGET ACCURACY 95
        PARAMETERS (
            type HNSW,
            m {hnsw_m},
            efConstruction {hnsw_ef_construction}
        )
        """


def create_text_index_ddl(table_name: str) -> str:
    """
    Generate DDL for creating Oracle Text index for hybrid search.

    Args:
        table_name: Name of the table

    Returns:
        DDL statement string
    """
    index_name = f"{table_name}_text_idx"
    return f"""
    CREATE INDEX {index_name} ON {table_name}(text)
    INDEXTYPE IS CTXSYS.CONTEXT
    PARAMETERS ('SYNC (ON COMMIT)')
    """


def create_metadata_schema_table_ddl() -> str:
    """Generate DDL for metadata schema storage table."""
    return """
    CREATE TABLE metadata_schema (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        collection_name VARCHAR2(255) NOT NULL UNIQUE,
        metadata_schema CLOB CHECK (metadata_schema IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """


def create_document_info_table_ddl() -> str:
    """Generate DDL for document info storage table."""
    return """
    CREATE TABLE document_info (
        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        collection_name VARCHAR2(255) NOT NULL,
        info_type VARCHAR2(50) NOT NULL,
        document_name VARCHAR2(4000) NOT NULL,
        info_value CLOB CHECK (info_value IS JSON),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT doc_info_unique UNIQUE (collection_name, info_type, document_name)
    )
    """


def get_unique_sources_query(table_name: str) -> str:
    """
    Generate query to retrieve all unique document sources.

    Args:
        table_name: Name of the collection table

    Returns:
        SQL query string
    """
    return f"""
    SELECT DISTINCT source,
           MAX(content_metadata) KEEP (DENSE_RANK FIRST ORDER BY created_at DESC) as content_metadata
    FROM {table_name}
    WHERE source IS NOT NULL
    GROUP BY source
    ORDER BY source
    """


def get_delete_docs_query(table_name: str) -> str:
    """
    Generate parameterized delete query for documents by source.

    Args:
        table_name: Name of the collection table

    Returns:
        SQL query string with :source_value bind variable
    """
    return f"""
    DELETE FROM {table_name}
    WHERE source = :source_value
    """


def get_delete_metadata_schema_query() -> str:
    """Generate parameterized delete query for metadata schema."""
    return """
    DELETE FROM metadata_schema
    WHERE collection_name = :collection_name
    """


def get_metadata_schema_query() -> str:
    """Generate parameterized query to retrieve metadata schema."""
    return """
    SELECT metadata_schema
    FROM metadata_schema
    WHERE collection_name = :collection_name
    """


def get_delete_document_info_query() -> str:
    """Generate parameterized delete query for document info."""
    return """
    DELETE FROM document_info
    WHERE collection_name = :collection_name
    AND document_name = :document_name
    AND info_type = :info_type
    """


def get_delete_document_info_by_collection_query() -> str:
    """Generate parameterized delete query for all document info in a collection."""
    return """
    DELETE FROM document_info
    WHERE collection_name = :collection_name
    """


def get_document_info_query() -> str:
    """Generate parameterized query to retrieve document info."""
    return """
    SELECT info_value
    FROM document_info
    WHERE collection_name = :collection_name
    AND document_name = :document_name
    AND info_type = :info_type
    """


def get_collection_document_info_query() -> str:
    """Generate parameterized query to retrieve collection-level document info."""
    return """
    SELECT info_value
    FROM document_info
    WHERE collection_name = :collection_name
    AND info_type = :info_type
    """


def get_similarity_search_query(
    table_name: str,
    distance_metric: DistanceMetric = "COSINE",
) -> str:
    """
    Generate vector similarity search query.

    Args:
        table_name: Name of the collection table
        distance_metric: Distance function to use

    Returns:
        SQL query string with :query_vector and :top_k bind variables
    """
    return f"""
    SELECT id, text, source, content_metadata,
           VECTOR_DISTANCE(vector, :query_vector, {distance_metric}) as distance
    FROM {table_name}
    ORDER BY distance
    FETCH FIRST :top_k ROWS ONLY
    """


def get_hybrid_search_query(
    table_name: str,
    distance_metric: DistanceMetric = "COSINE",
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> str:
    """
    Generate hybrid search query combining vector similarity and text search.

    Args:
        table_name: Name of the collection table
        distance_metric: Distance function for vector search
        vector_weight: Weight for vector similarity score
        text_weight: Weight for text search score

    Returns:
        SQL query string with :query_vector, :query_text, and :top_k bind variables
    """
    return f"""
    WITH vector_results AS (
        SELECT id, text, source, content_metadata,
               VECTOR_DISTANCE(vector, :query_vector, {distance_metric}) as vec_distance,
               ROW_NUMBER() OVER (ORDER BY VECTOR_DISTANCE(vector, :query_vector, {distance_metric})) as vec_rank
        FROM {table_name}
        FETCH FIRST :top_k * 2 ROWS ONLY
    ),
    text_results AS (
        SELECT id, SCORE(1) as text_score,
               ROW_NUMBER() OVER (ORDER BY SCORE(1) DESC) as text_rank
        FROM {table_name}
        WHERE CONTAINS(text, :query_text, 1) > 0
        FETCH FIRST :top_k * 2 ROWS ONLY
    )
    SELECT v.id, v.text, v.source, v.content_metadata,
           ({vector_weight} * (1 / (1 + v.vec_distance)) + 
            {text_weight} * COALESCE(t.text_score / 100, 0)) as combined_score
    FROM vector_results v
    LEFT JOIN text_results t ON v.id = t.id
    ORDER BY combined_score DESC
    FETCH FIRST :top_k ROWS ONLY
    """


def get_count_query(table_name: str) -> str:
    """Generate query to count documents in a collection."""
    return f"SELECT COUNT(*) as cnt FROM {table_name}"


def check_table_exists_query() -> str:
    """Generate query to check if a table exists."""
    return """
    SELECT COUNT(*) as cnt
    FROM user_tables
    WHERE table_name = UPPER(:table_name)
    """


def drop_table_ddl(table_name: str) -> str:
    """Generate DDL to drop a table."""
    return f"DROP TABLE {table_name} CASCADE CONSTRAINTS PURGE"


def get_all_collections_query() -> str:
    """Generate query to list all collection tables."""
    return """
    SELECT table_name
    FROM user_tables
    WHERE table_name NOT IN ('METADATA_SCHEMA', 'DOCUMENT_INFO')
    AND table_name NOT LIKE 'SYS%'
    AND table_name NOT LIKE 'DR$%'
    ORDER BY table_name
    """

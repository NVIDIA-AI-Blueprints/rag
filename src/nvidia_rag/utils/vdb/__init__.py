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

from typing import Any

from nvidia_rag.utils.common import get_metadata_configuration
from nvidia_rag.utils.configuration import NvidiaRAGConfig, SearchType

DEFAULT_METADATA_SCHEMA_COLLECTION = "metadata_schema"
DEFAULT_DOCUMENT_INFO_COLLECTION = "document_info"
SYSTEM_COLLECTIONS = [
    DEFAULT_METADATA_SCHEMA_COLLECTION,
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    "meta",
]


def _get_vdb_op(
    vdb_endpoint: str,
    collection_name: str = "",
    custom_metadata: list[dict[str, Any]] | None = None,
    all_file_paths: list[str] | None = None,
    embedding_model: str | None = None,  # Needed in case of retrieval
    metadata_schema: list[dict[str, Any]] | None = None,
    config: NvidiaRAGConfig | None = None,
    vdb_auth_token: str | None = None,
):
    """
    Get VDBRag class object based on configuration.

    Args:
        vdb_endpoint: Vector database endpoint URL
        collection_name: Name of the collection
        custom_metadata: Custom metadata configuration
        all_file_paths: List of file paths for metadata
        embedding_model: Embedding model instance for retrieval
        metadata_schema: Metadata schema definition
        config: NvidiaRAGConfig instance. If None, creates a new one.
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Get metadata configuration
    csv_file_path, meta_source_field, meta_fields = get_metadata_configuration(
        collection_name=collection_name,
        custom_metadata=custom_metadata,
        all_file_paths=all_file_paths,
        metadata_schema=metadata_schema,
        config=config,
    )

    # Get VDBRag class object based on the configuration
    if config.vector_store.name == "milvus":
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB

        return MilvusVDB(
            # Milvus configurations
            collection_name=collection_name,
            milvus_uri=vdb_endpoint or config.vector_store.url,
            embedding_model=embedding_model,
            config=config,
            # Milvus bulk import passes this to the MinIO SDK, which expects
            # host:port without an http(s) scheme.
            object_store_endpoint=config.object_store.endpoint,
            access_key=config.object_store.access_key.get_secret_value(),
            secret_key=config.object_store.secret_key.get_secret_value(),
            bucket_name=config.nv_ingest.object_store_bucket,
            # Hybrid search configurations
            sparse=(config.vector_store.search_type == SearchType.HYBRID),
            # Additional configurations
            enable_images=(
                config.nv_ingest.extract_images
                or config.nv_ingest.extract_page_as_image
            ),
            recreate=False,  # Don't re-create milvus collection
            dense_dim=config.embeddings.dimensions,
            # GPU configurations
            gpu_index=config.vector_store.enable_gpu_index,
            gpu_search=config.vector_store.enable_gpu_search,
            # Authentication for Milvus
            username=config.vector_store.username,
            password=(
                config.vector_store.password.get_secret_value()
                if config.vector_store.password is not None
                else ""
            ),
            # Custom metadata configurations (optional)
            meta_dataframe=csv_file_path,
            meta_source_field=meta_source_field,
            meta_fields=meta_fields,
            auth_token=vdb_auth_token,
        )

    elif config.vector_store.name == "elasticsearch":
        from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB

        # Note: meta_dataframe is loaded lazily inside ElasticVDB.write_to_index()
        # when actually needed for ingestion. This allows search to work without nv_ingest.
        return ElasticVDB(
            index_name=collection_name,
            es_url=vdb_endpoint or config.vector_store.url,
            hybrid=config.vector_store.search_type == SearchType.HYBRID,
            auth_token=vdb_auth_token,
            meta_source_field=meta_source_field,
            meta_fields=meta_fields,
            embedding_model=embedding_model,
            csv_file_path=csv_file_path,
            config=config,
        )

    elif config.vector_store.name == "lancedb":
        from nvidia_rag.utils.vdb.lancedb.lancedb_vdb import LanceDBVDB

        # LanceDB is a local/embedded VDB used exclusively with the NRL ingestion
        # pipeline.  Speed and scalability are not requirements for this backend.
        # All lancedb imports are kept inside methods (not fork-safe).
        return LanceDBVDB(
            table_name=collection_name,
            uri=vdb_endpoint or config.vector_store.url,
            embedding_model=embedding_model,
            config=config,
            hybrid=(config.vector_store.search_type == SearchType.HYBRID),
        )

    else:
        raise ValueError(f"Invalid vector store name: {config.vector_store.name}")

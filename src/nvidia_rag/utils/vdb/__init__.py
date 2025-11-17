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
from typing import Any

from nv_ingest_client.util.milvus import pandas_file_reader

from nvidia_rag.utils.common import get_metadata_configuration
from nvidia_rag.utils.configuration import NvidiaRAGConfig, SearchType

DEFAULT_METADATA_SCHEMA_COLLECTION = "metadata_schema"
DEFAULT_DOCUMENT_INFO_COLLECTION = "document_info"


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
            # Minio configurations
            minio_endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESSKEY"),
            secret_key=os.getenv("MINIO_SECRETKEY"),
            bucket_name=os.getenv("NVINGEST_MINIO_BUCKET", "nv-ingest"),
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

        if csv_file_path is not None:
            meta_dataframe = pandas_file_reader(csv_file_path)
        else:
            meta_dataframe = None

        return ElasticVDB(
            index_name=collection_name,
            es_url=vdb_endpoint or config.vector_store.url,
            hybrid=config.vector_store.search_type == SearchType.HYBRID,
            auth_token=vdb_auth_token,
            meta_dataframe=meta_dataframe,
            meta_source_field=meta_source_field,
            meta_fields=meta_fields,
            embedding_model=embedding_model,
            csv_file_path=csv_file_path,
            config=config,
        )

    else:
        raise ValueError(f"Invalid vector store name: {config.vector_store.name}")

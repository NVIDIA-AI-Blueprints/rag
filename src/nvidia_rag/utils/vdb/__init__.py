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

from nvidia_rag.utils.common import ConfigProxy, get_metadata_configuration

CONFIG = ConfigProxy()
DEFAULT_METADATA_SCHEMA_COLLECTION = "metadata_schema"
DEFAULT_DOCUMENT_INFO_COLLECTION = "document_info"


def _get_vdb_op(
    vdb_endpoint: str,
    collection_name: str = "",
    custom_metadata: list[dict[str, Any]] | None = None,
    all_file_paths: list[str] | None = None,
    embedding_model: str | None = None,  # Needed in case of retrieval
    metadata_schema: list[dict[str, Any]] | None = None,
):
    """
    Get VDBRag class object based on the environment variables.
    """
    # Get metadata configuration
    csv_file_path, meta_source_field, meta_fields = get_metadata_configuration(
        collection_name=collection_name,
        custom_metadata=custom_metadata,
        all_file_paths=all_file_paths,
        metadata_schema=metadata_schema,
    )

    # Get VDBRag class object based on the environment variables.
    if CONFIG.vector_store.name == "milvus":
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB

        vdb_upload_kwargs = {
            # Milvus configurations
            "collection_name": collection_name,
            "milvus_uri": vdb_endpoint or CONFIG.vector_store.url,
            # Minio configurations
            "minio_endpoint": os.getenv("MINIO_ENDPOINT"),
            "access_key": os.getenv("MINIO_ACCESSKEY"),
            "secret_key": os.getenv("MINIO_SECRETKEY"),
            "bucket_name": os.getenv("NVINGEST_MINIO_BUCKET", "nv-ingest"),
            # Hybrid search configurations
            "sparse": (CONFIG.vector_store.search_type == "hybrid"),
            # Additional configurations
            "enable_images": (
                CONFIG.nv_ingest.extract_images
                or CONFIG.nv_ingest.extract_page_as_image
            ),
            "recreate": False,  # Don't re-create milvus collection
            "dense_dim": CONFIG.embeddings.dimensions,
            # GPU configurations
            "gpu_index": CONFIG.vector_store.enable_gpu_index,
            "gpu_search": CONFIG.vector_store.enable_gpu_search,
            "embedding_model": embedding_model,
            # Authentication for Milvus
            "username": CONFIG.vector_store.username,
            "password": CONFIG.vector_store.password,
        }
        if csv_file_path is not None:
            # Add custom metadata configurations
            vdb_upload_kwargs.update(
                {
                    "meta_dataframe": csv_file_path,
                    "meta_source_field": meta_source_field,
                    "meta_fields": meta_fields,
                }
            )
        return MilvusVDB(**vdb_upload_kwargs)

    elif CONFIG.vector_store.name == "elasticsearch":
        from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB

        if csv_file_path is not None:
            meta_dataframe = pandas_file_reader(csv_file_path)
        else:
            meta_dataframe = None

        # Build auth kwargs similar to Milvus, supporting both API key and basic auth
        auth_kwargs: dict[str, Any] = {}
        # Username/password from config
        if CONFIG.vector_store.username and CONFIG.vector_store.password:
            auth_kwargs.update(
                {
                    "username": CONFIG.vector_store.username,
                    "password": CONFIG.vector_store.password,
                }
            )
        # API key from configuration (supports base64 'id:secret' or id+secret)
        apikey_str = CONFIG.vector_store.api_key
        apikey_id = CONFIG.vector_store.api_key_id
        apikey_secret = CONFIG.vector_store.api_key_secret
        if apikey_str:
            auth_kwargs["api_key"] = apikey_str
        elif apikey_id and apikey_secret:
            # Either pass a tuple as api_key or separate id/secret for clarity
            auth_kwargs["api_key"] = (apikey_id, apikey_secret)

        return ElasticVDB(
            index_name=collection_name,
            es_url=vdb_endpoint or CONFIG.vector_store.url,
            hybrid=CONFIG.vector_store.search_type == "hybrid",
            meta_dataframe=meta_dataframe,
            meta_source_field=meta_source_field,
            meta_fields=meta_fields,
            embedding_model=embedding_model,
            csv_file_path=csv_file_path,
            **auth_kwargs,
        )

    else:
        raise ValueError(f"Invalid vector store name: {CONFIG.vector_store.name}")

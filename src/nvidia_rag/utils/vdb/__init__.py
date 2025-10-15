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

from nvidia_rag.utils.common import get_config, get_metadata_configuration

CONFIG = get_config()
DEFAULT_METADATA_SCHEMA_COLLECTION = "metadata_schema"


def _get_vdb_op(
    vdb_endpoint: str,
    collection_name: str = "",
    custom_metadata: list[dict[str, Any]] = None,
    all_file_paths: list[str] = None,
    embedding_model: str = None,  # Needed in case of retrieval
):
    """
    Get VDBRag class object based on the environment variables.
    """
    # Get metadata configuration
    csv_file_path, meta_source_field, meta_fields = get_metadata_configuration(
        collection_name=collection_name,
        custom_metadata=custom_metadata,
        all_file_paths=all_file_paths,
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

        return ElasticVDB(
            index_name=collection_name,
            es_url=vdb_endpoint or CONFIG.vector_store.url,
            hybrid=CONFIG.vector_store.search_type == "hybrid",
            meta_dataframe=meta_dataframe,
            meta_source_field=meta_source_field,
            meta_fields=meta_fields,
            embedding_model=embedding_model,
            csv_file_path=csv_file_path,
        )

    else:
        raise ValueError(f"Invalid vector store name: {CONFIG.vector_store.name}")

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
DenseVectorStrategyWithIndexOptions — DenseVectorStrategy extended with
``index_options`` on the vector mapping field.

Required for GPU-accelerated Elasticsearch, where the server needs explicit
index quantization / HNSW parameters (e.g. ``int8_hnsw``) to activate GPU
indexing.  The standard ``DenseVectorStrategy`` does not expose
``index_options``; ``custom_index_settings`` on ``VectorStore`` only merges
index *settings*, not field *mappings*.
"""

from typing import Any

from elasticsearch.helpers.vectorstore import DenseVectorStrategy

# Default vector index_options for GPU-accelerated Elasticsearch.
# These match Elasticsearch's own defaults for int8_hnsw and work well as a
# starting point for GPU-indexed deployments.
_DEFAULT_GPU_INDEX_OPTIONS: dict[str, Any] = {
    "type": "int8_hnsw",
    "m": 16,
    "ef_construction": 100,
}


class DenseVectorStrategyWithIndexOptions(DenseVectorStrategy):
    """DenseVectorStrategy that injects ``index_options`` into the vector field mapping.

    Elasticsearch GPU indexing requires explicit ``index_options`` on the
    ``dense_vector`` field (e.g. ``int8_hnsw`` with ``m`` and
    ``ef_construction``).  This subclass intercepts the mapping produced by
    ``DenseVectorStrategy.es_mappings_settings`` and merges in the desired
    options so the index is created correctly on first use.

    Args:
        vector_index_options: Mapping injected verbatim as the
            ``index_options`` block on the vector field.  Defaults to
            ``{"type": "int8_hnsw", "m": 16, "ef_construction": 100}``.
        **kwargs: Forwarded to ``DenseVectorStrategy`` (e.g. ``hybrid``).

    Example::

        strategy = DenseVectorStrategyWithIndexOptions(
            hybrid=False,
            vector_index_options={"type": "int8_hnsw", "m": 16, "ef_construction": 100},
        )
        store = VectorStore(
            client=es,
            index="my_index",
            num_dimensions=2048,
            retrieval_strategy=strategy,
        )
    """

    def __init__(
        self,
        *,
        vector_index_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._vector_index_options: dict[str, Any] = (
            vector_index_options if vector_index_options is not None else dict(_DEFAULT_GPU_INDEX_OPTIONS)
        )

    def es_mappings_settings(
        self,
        *,
        text_field: str,
        vector_field: str,
        num_dimensions: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mappings, settings = super().es_mappings_settings(
            text_field=text_field,
            vector_field=vector_field,
            num_dimensions=num_dimensions,
        )
        mappings["properties"][vector_field]["index_options"] = self._vector_index_options
        return mappings, settings

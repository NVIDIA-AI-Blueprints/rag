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

"""Factory for creating graph store instances from configuration."""

from __future__ import annotations

import logging

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


def get_graph_store(config: NvidiaRAGConfig | None = None) -> GraphStore | None:
    """Create a GraphStore instance based on configuration.

    Returns None if GraphRAG is disabled.
    """
    if config is None:
        config = NvidiaRAGConfig()

    if not config.graph_rag.enable_graph_rag:
        return None

    store_type = config.graph_rag.graph_store_type.lower()

    if store_type == "networkx":
        from nvidia_rag.utils.graph.networkx_store import NetworkXGraphStore

        logger.info("Initializing NetworkX graph store at %s", config.graph_rag.graph_data_dir)
        return NetworkXGraphStore(data_dir=config.graph_rag.graph_data_dir)

    elif store_type == "neo4j":
        from nvidia_rag.utils.graph.neo4j_store import Neo4jGraphStore

        password = ""
        if config.graph_rag.graph_store_password:
            password = config.graph_rag.graph_store_password.get_secret_value()

        logger.info("Initializing Neo4j graph store at %s", config.graph_rag.graph_store_url)
        return Neo4jGraphStore(
            url=config.graph_rag.graph_store_url,
            username=config.graph_rag.graph_store_username,
            password=password,
        )

    else:
        raise ValueError(
            f"Unknown graph store type: '{store_type}'. Supported: 'networkx', 'neo4j'"
        )

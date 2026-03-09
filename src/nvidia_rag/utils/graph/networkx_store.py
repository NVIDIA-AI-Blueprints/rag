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

"""NetworkX-based in-memory graph store with pickle persistence."""

from __future__ import annotations

import logging
import os
import pickle
from collections import deque
from typing import Any

import networkx as nx

from nvidia_rag.utils.graph.graph_store import (
    CommunityInfo,
    Entity,
    GraphStore,
    Relationship,
)

logger = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """In-memory graph store backed by NetworkX with optional pickle persistence.

    Each collection gets its own DiGraph. Entities are nodes and relationships are
    directed edges. Community data is stored as a side structure.
    """

    def __init__(self, data_dir: str = "./graph-data"):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._graphs: dict[str, nx.DiGraph] = {}
        self._communities: dict[str, list[CommunityInfo]] = {}
        self._load_all()

    def _graph_path(self, collection_name: str) -> str:
        return os.path.join(self._data_dir, f"{collection_name}_graph.pkl")

    def _communities_path(self, collection_name: str) -> str:
        return os.path.join(self._data_dir, f"{collection_name}_communities.pkl")

    def _load_all(self) -> None:
        """Load all persisted graphs from the data directory."""
        if not os.path.isdir(self._data_dir):
            return
        for fname in os.listdir(self._data_dir):
            if fname.endswith("_graph.pkl"):
                collection = fname.replace("_graph.pkl", "")
                try:
                    with open(os.path.join(self._data_dir, fname), "rb") as f:
                        self._graphs[collection] = pickle.load(f)  # noqa: S301
                    logger.info("Loaded graph for collection '%s'", collection)
                except Exception:
                    logger.warning("Failed to load graph for '%s'", collection, exc_info=True)
            elif fname.endswith("_communities.pkl"):
                collection = fname.replace("_communities.pkl", "")
                try:
                    with open(os.path.join(self._data_dir, fname), "rb") as f:
                        self._communities[collection] = pickle.load(f)  # noqa: S301
                except Exception:
                    logger.warning("Failed to load communities for '%s'", collection, exc_info=True)

    def _get_graph(self, collection_name: str) -> nx.DiGraph:
        if collection_name not in self._graphs:
            self._graphs[collection_name] = nx.DiGraph()
        return self._graphs[collection_name]

    def add_entities(self, entities: list[Entity], collection_name: str) -> int:
        g = self._get_graph(collection_name)
        added = 0
        for entity in entities:
            key = entity.key
            if g.has_node(key):
                node_data = g.nodes[key]
                existing_chunks = set(node_data.get("source_chunk_ids", []))
                existing_chunks.update(entity.source_chunk_ids)
                node_data["source_chunk_ids"] = list(existing_chunks)
                if entity.description and len(entity.description) > len(node_data.get("description", "")):
                    node_data["description"] = entity.description
            else:
                g.add_node(
                    key,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    source_chunk_ids=list(entity.source_chunk_ids),
                    metadata=entity.metadata,
                )
                added += 1
        return added

    def add_relationships(self, relationships: list[Relationship], collection_name: str) -> int:
        g = self._get_graph(collection_name)
        added = 0
        for rel in relationships:
            src_key = rel.source.strip().lower()
            tgt_key = rel.target.strip().lower()
            if not g.has_node(src_key):
                g.add_node(src_key, name=rel.source, entity_type="unknown", description="", source_chunk_ids=[], metadata={})
            if not g.has_node(tgt_key):
                g.add_node(tgt_key, name=rel.target, entity_type="unknown", description="", source_chunk_ids=[], metadata={})

            edge_key = (src_key, tgt_key, rel.relation_type)
            if g.has_edge(src_key, tgt_key) and g[src_key][tgt_key].get("relation_type") == rel.relation_type:
                edge_data = g[src_key][tgt_key]
                existing_chunks = set(edge_data.get("source_chunk_ids", []))
                existing_chunks.update(rel.source_chunk_ids)
                edge_data["source_chunk_ids"] = list(existing_chunks)
                edge_data["weight"] = max(edge_data.get("weight", 1.0), rel.weight)
            else:
                g.add_edge(
                    src_key,
                    tgt_key,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=list(rel.source_chunk_ids),
                    metadata=rel.metadata,
                )
                added += 1
        return added

    def _node_to_entity(self, key: str, data: dict[str, Any]) -> Entity:
        return Entity(
            name=data.get("name", key),
            entity_type=data.get("entity_type", "unknown"),
            description=data.get("description", ""),
            source_chunk_ids=data.get("source_chunk_ids", []),
            metadata=data.get("metadata", {}),
        )

    def _edge_to_relationship(self, src: str, tgt: str, data: dict[str, Any]) -> Relationship:
        g_src_name = self._graphs.get(src, {})
        return Relationship(
            source=data.get("source_name", src),
            target=data.get("target_name", tgt),
            relation_type=data.get("relation_type", "related_to"),
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            source_chunk_ids=data.get("source_chunk_ids", []),
            metadata=data.get("metadata", {}),
        )

    def _fuzzy_match(self, name: str, collection_name: str, max_matches: int = 3) -> list[str]:
        """Find graph nodes that fuzzy-match the given name.

        Tries, in order: exact match, substring containment, and token overlap.
        Returns up to ``max_matches`` node keys, best matches first.
        """
        g = self._get_graph(collection_name)
        key = name.strip().lower()

        if g.has_node(key):
            return [key]

        candidates: list[tuple[float, str]] = []
        key_tokens = set(key.split())

        for node in g.nodes():
            if key in node or node in key:
                candidates.append((0.9, node))
                continue
            node_tokens = set(node.split())
            overlap = key_tokens & node_tokens
            if overlap:
                score = len(overlap) / max(len(key_tokens), len(node_tokens))
                if score >= 0.4:
                    candidates.append((score, node))

        candidates.sort(key=lambda x: x[0], reverse=True)
        matched = [node for _, node in candidates[:max_matches]]
        if matched:
            logger.debug("Fuzzy match for '%s': %s", key, matched)
        return matched

    def get_entity(self, name: str, collection_name: str) -> Entity | None:
        g = self._get_graph(collection_name)
        key = name.strip().lower()
        if g.has_node(key):
            return self._node_to_entity(key, g.nodes[key])
        matches = self._fuzzy_match(name, collection_name, max_matches=1)
        if matches:
            return self._node_to_entity(matches[0], g.nodes[matches[0]])
        return None

    def get_neighbors(
        self, entity_name: str, collection_name: str, depth: int = 1
    ) -> tuple[list[Entity], list[Relationship]]:
        g = self._get_graph(collection_name)
        start_key = entity_name.strip().lower()
        if not g.has_node(start_key):
            matches = self._fuzzy_match(entity_name, collection_name, max_matches=3)
            if not matches:
                return [], []
            all_entities: list[Entity] = []
            all_relationships: list[Relationship] = []
            for matched_key in matches:
                ents, rels = self._get_neighbors_exact(matched_key, g, depth)
                all_entities.extend(ents)
                all_relationships.extend(rels)
            return all_entities, all_relationships

        return self._get_neighbors_exact(start_key, g, depth)

    def _get_neighbors_exact(
        self, start_key: str, g: nx.DiGraph, depth: int
    ) -> tuple[list[Entity], list[Relationship]]:
        """BFS traversal from a known node key."""
        if not g.has_node(start_key):
            return [], []

        visited_nodes: set[str] = set()
        visited_edges: set[tuple[str, str]] = set()
        queue: deque[tuple[str, int]] = deque([(start_key, 0)])
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        while queue:
            current, current_depth = queue.popleft()
            if current in visited_nodes:
                continue
            visited_nodes.add(current)
            entities.append(self._node_to_entity(current, g.nodes[current]))

            if current_depth >= depth:
                continue

            for neighbor in g.successors(current):
                edge_data = g[current][neighbor]
                edge_pair = (current, neighbor)
                if edge_pair not in visited_edges:
                    visited_edges.add(edge_pair)
                    rel = Relationship(
                        source=g.nodes[current].get("name", current),
                        target=g.nodes[neighbor].get("name", neighbor),
                        relation_type=edge_data.get("relation_type", "related_to"),
                        description=edge_data.get("description", ""),
                        weight=edge_data.get("weight", 1.0),
                        source_chunk_ids=edge_data.get("source_chunk_ids", []),
                    )
                    relationships.append(rel)
                if neighbor not in visited_nodes:
                    queue.append((neighbor, current_depth + 1))

            for predecessor in g.predecessors(current):
                edge_data = g[predecessor][current]
                edge_pair = (predecessor, current)
                if edge_pair not in visited_edges:
                    visited_edges.add(edge_pair)
                    rel = Relationship(
                        source=g.nodes[predecessor].get("name", predecessor),
                        target=g.nodes[current].get("name", current),
                        relation_type=edge_data.get("relation_type", "related_to"),
                        description=edge_data.get("description", ""),
                        weight=edge_data.get("weight", 1.0),
                        source_chunk_ids=edge_data.get("source_chunk_ids", []),
                    )
                    relationships.append(rel)
                if predecessor not in visited_nodes:
                    queue.append((predecessor, current_depth + 1))

        return entities, relationships

    def get_entities_by_names(self, names: list[str], collection_name: str) -> list[Entity]:
        g = self._get_graph(collection_name)
        results = []
        for name in names:
            key = name.strip().lower()
            if g.has_node(key):
                results.append(self._node_to_entity(key, g.nodes[key]))
            else:
                matched = self._fuzzy_match(name, collection_name, max_matches=1)
                if matched:
                    results.append(self._node_to_entity(matched[0], g.nodes[matched[0]]))
        return results

    def get_all_entities(self, collection_name: str) -> list[Entity]:
        g = self._get_graph(collection_name)
        return [self._node_to_entity(k, d) for k, d in g.nodes(data=True)]

    def get_all_relationships(self, collection_name: str) -> list[Relationship]:
        g = self._get_graph(collection_name)
        results = []
        for src, tgt, data in g.edges(data=True):
            results.append(
                Relationship(
                    source=g.nodes[src].get("name", src),
                    target=g.nodes[tgt].get("name", tgt),
                    relation_type=data.get("relation_type", "related_to"),
                    description=data.get("description", ""),
                    weight=data.get("weight", 1.0),
                    source_chunk_ids=data.get("source_chunk_ids", []),
                    metadata=data.get("metadata", {}),
                )
            )
        return results

    def set_communities(self, communities: list[CommunityInfo], collection_name: str) -> None:
        self._communities[collection_name] = communities

    def get_communities(self, collection_name: str) -> list[CommunityInfo]:
        return self._communities.get(collection_name, [])

    def get_community_for_entity(
        self, entity_name: str, collection_name: str
    ) -> CommunityInfo | None:
        key = entity_name.strip().lower()
        for community in self._communities.get(collection_name, []):
            if key in [n.strip().lower() for n in community.entity_names]:
                return community
        matched_keys = self._fuzzy_match(entity_name, collection_name, max_matches=1)
        if matched_keys:
            for community in self._communities.get(collection_name, []):
                if matched_keys[0] in [n.strip().lower() for n in community.entity_names]:
                    return community
        return None

    def delete_collection(self, collection_name: str) -> None:
        self._graphs.pop(collection_name, None)
        self._communities.pop(collection_name, None)
        for path in [self._graph_path(collection_name), self._communities_path(collection_name)]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("Deleted graph data for collection '%s'", collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._graphs and len(self._graphs[collection_name]) > 0

    def get_stats(self, collection_name: str) -> dict[str, int]:
        g = self._get_graph(collection_name)
        communities = self._communities.get(collection_name, [])
        return {
            "entities": g.number_of_nodes(),
            "relationships": g.number_of_edges(),
            "communities": len(communities),
        }

    def persist(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        for collection_name, graph in self._graphs.items():
            with open(self._graph_path(collection_name), "wb") as f:
                pickle.dump(graph, f)
        for collection_name, communities in self._communities.items():
            with open(self._communities_path(collection_name), "wb") as f:
                pickle.dump(communities, f)
        logger.info("Persisted graph data for %d collections", len(self._graphs))

    def get_networkx_graph(self, collection_name: str) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph (for community detection)."""
        return self._get_graph(collection_name)

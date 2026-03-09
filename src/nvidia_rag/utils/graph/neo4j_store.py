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

"""Neo4j-backed graph store for production GraphRAG deployments."""

from __future__ import annotations

import logging
from typing import Any

from nvidia_rag.utils.graph.graph_store import (
    CommunityInfo,
    Entity,
    GraphStore,
    Relationship,
)

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Graph store backed by Neo4j for production use.

    Uses a label-per-collection scheme: entities get label ``Entity_{collection}``
    and relationships include a ``collection`` property so multiple collections can
    coexist in one Neo4j instance.
    """

    def __init__(self, url: str, username: str = "neo4j", password: str = ""):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "neo4j package is required for Neo4jGraphStore. "
                "Install it with: pip install neo4j"
            ) from e

        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except Exception:
            logger.error("Failed to connect to Neo4j", exc_info=True)
            raise

    def _label(self, collection_name: str) -> str:
        safe = collection_name.replace("-", "_").replace(" ", "_")
        return f"Entity_{safe}"

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def add_entities(self, entities: list[Entity], collection_name: str) -> int:
        label = self._label(collection_name)
        query = f"""
        UNWIND $entities AS e
        MERGE (n:`{label}` {{key: e.key}})
        ON CREATE SET
            n.name = e.name,
            n.entity_type = e.entity_type,
            n.description = e.description,
            n.source_chunk_ids = e.source_chunk_ids,
            n.created = true
        ON MATCH SET
            n.description = CASE
                WHEN size(e.description) > size(coalesce(n.description, ''))
                THEN e.description ELSE n.description END,
            n.source_chunk_ids = apoc.coll.toSet(n.source_chunk_ids + e.source_chunk_ids),
            n.created = false
        RETURN count(CASE WHEN n.created THEN 1 END) AS added
        """
        entity_dicts = [
            {
                "key": e.key,
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "source_chunk_ids": e.source_chunk_ids,
            }
            for e in entities
        ]
        result = self._run(query, entities=entity_dicts)
        return result[0]["added"] if result else 0

    def add_relationships(self, relationships: list[Relationship], collection_name: str) -> int:
        label = self._label(collection_name)
        query = f"""
        UNWIND $rels AS r
        MERGE (s:`{label}` {{key: r.src_key}})
        ON CREATE SET s.name = r.source, s.entity_type = 'unknown', s.description = ''
        MERGE (t:`{label}` {{key: r.tgt_key}})
        ON CREATE SET t.name = r.target, t.entity_type = 'unknown', t.description = ''
        MERGE (s)-[rel:RELATES_TO {{relation_type: r.relation_type}}]->(t)
        ON CREATE SET
            rel.description = r.description,
            rel.weight = r.weight,
            rel.source_chunk_ids = r.source_chunk_ids,
            rel.collection = $collection
        ON MATCH SET
            rel.weight = CASE WHEN r.weight > rel.weight THEN r.weight ELSE rel.weight END,
            rel.source_chunk_ids = apoc.coll.toSet(rel.source_chunk_ids + r.source_chunk_ids)
        RETURN count(rel) AS added
        """
        rel_dicts = [
            {
                "src_key": r.source.strip().lower(),
                "tgt_key": r.target.strip().lower(),
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "description": r.description,
                "weight": r.weight,
                "source_chunk_ids": r.source_chunk_ids,
            }
            for r in relationships
        ]
        result = self._run(query, rels=rel_dicts, collection=collection_name)
        return result[0]["added"] if result else 0

    def get_entity(self, name: str, collection_name: str) -> Entity | None:
        label = self._label(collection_name)
        key = name.strip().lower()
        query = f"MATCH (n:`{label}` {{key: $key}}) RETURN n"
        result = self._run(query, key=key)
        if result:
            n = result[0]["n"]
            return Entity(
                name=n.get("name", key),
                entity_type=n.get("entity_type", "unknown"),
                description=n.get("description", ""),
                source_chunk_ids=n.get("source_chunk_ids", []),
            )
        return None

    def get_neighbors(
        self, entity_name: str, collection_name: str, depth: int = 1
    ) -> tuple[list[Entity], list[Relationship]]:
        label = self._label(collection_name)
        key = entity_name.strip().lower()
        query = f"""
        MATCH path = (start:`{label}` {{key: $key}})-[*1..{depth}]-(neighbor:`{label}`)
        WITH nodes(path) AS ns, relationships(path) AS rs
        UNWIND ns AS n
        WITH collect(DISTINCT n) AS entities, rs
        UNWIND rs AS r
        WITH entities, collect(DISTINCT r) AS rels
        RETURN entities, rels
        """
        result = self._run(query, key=key)
        entities = []
        relationships = []
        if result:
            for n in result[0].get("entities", []):
                entities.append(Entity(
                    name=n.get("name", ""),
                    entity_type=n.get("entity_type", "unknown"),
                    description=n.get("description", ""),
                    source_chunk_ids=n.get("source_chunk_ids", []),
                ))
            for r in result[0].get("rels", []):
                relationships.append(Relationship(
                    source=r.start_node.get("name", ""),
                    target=r.end_node.get("name", ""),
                    relation_type=r.get("relation_type", "related_to"),
                    description=r.get("description", ""),
                    weight=r.get("weight", 1.0),
                    source_chunk_ids=r.get("source_chunk_ids", []),
                ))
        return entities, relationships

    def get_entities_by_names(self, names: list[str], collection_name: str) -> list[Entity]:
        label = self._label(collection_name)
        keys = [n.strip().lower() for n in names]
        query = f"MATCH (n:`{label}`) WHERE n.key IN $keys RETURN n"
        result = self._run(query, keys=keys)
        return [
            Entity(
                name=r["n"].get("name", ""),
                entity_type=r["n"].get("entity_type", "unknown"),
                description=r["n"].get("description", ""),
                source_chunk_ids=r["n"].get("source_chunk_ids", []),
            )
            for r in result
        ]

    def get_all_entities(self, collection_name: str) -> list[Entity]:
        label = self._label(collection_name)
        query = f"MATCH (n:`{label}`) RETURN n"
        result = self._run(query)
        return [
            Entity(
                name=r["n"].get("name", ""),
                entity_type=r["n"].get("entity_type", "unknown"),
                description=r["n"].get("description", ""),
                source_chunk_ids=r["n"].get("source_chunk_ids", []),
            )
            for r in result
        ]

    def get_all_relationships(self, collection_name: str) -> list[Relationship]:
        label = self._label(collection_name)
        query = f"""
        MATCH (s:`{label}`)-[r:RELATES_TO]->(t:`{label}`)
        RETURN s.name AS source, t.name AS target,
               r.relation_type AS relation_type, r.description AS description,
               r.weight AS weight, r.source_chunk_ids AS source_chunk_ids
        """
        result = self._run(query)
        return [
            Relationship(
                source=r["source"],
                target=r["target"],
                relation_type=r.get("relation_type", "related_to"),
                description=r.get("description", ""),
                weight=r.get("weight", 1.0),
                source_chunk_ids=r.get("source_chunk_ids", []),
            )
            for r in result
        ]

    def set_communities(self, communities: list[CommunityInfo], collection_name: str) -> None:
        label = self._label(collection_name)
        self._run(
            f"MATCH (c:Community_{label.replace('Entity_', '')}) DETACH DELETE c"
        )
        community_label = f"Community_{label.replace('Entity_', '')}"
        for comm in communities:
            self._run(
                f"""
                CREATE (c:`{community_label}` {{
                    community_id: $cid,
                    summary: $summary,
                    level: $level,
                    entity_names: $entity_names
                }})
                """,
                cid=comm.community_id,
                summary=comm.summary,
                level=comm.level,
                entity_names=comm.entity_names,
            )

    def get_communities(self, collection_name: str) -> list[CommunityInfo]:
        label = self._label(collection_name)
        community_label = f"Community_{label.replace('Entity_', '')}"
        result = self._run(f"MATCH (c:`{community_label}`) RETURN c")
        return [
            CommunityInfo(
                community_id=r["c"].get("community_id", 0),
                entity_names=r["c"].get("entity_names", []),
                summary=r["c"].get("summary", ""),
                level=r["c"].get("level", 0),
            )
            for r in result
        ]

    def get_community_for_entity(
        self, entity_name: str, collection_name: str
    ) -> CommunityInfo | None:
        key = entity_name.strip().lower()
        for comm in self.get_communities(collection_name):
            if key in [n.strip().lower() for n in comm.entity_names]:
                return comm
        return None

    def delete_collection(self, collection_name: str) -> None:
        label = self._label(collection_name)
        community_label = f"Community_{label.replace('Entity_', '')}"
        self._run(f"MATCH (n:`{label}`) DETACH DELETE n")
        self._run(f"MATCH (c:`{community_label}`) DETACH DELETE c")
        logger.info("Deleted Neo4j graph data for collection '%s'", collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        label = self._label(collection_name)
        result = self._run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt LIMIT 1")
        return result[0]["cnt"] > 0 if result else False

    def get_stats(self, collection_name: str) -> dict[str, int]:
        label = self._label(collection_name)
        entity_count = self._run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        rel_count = self._run(
            f"MATCH (:`{label}`)-[r:RELATES_TO]->(:`{label}`) RETURN count(r) AS cnt"
        )
        communities = self.get_communities(collection_name)
        return {
            "entities": entity_count[0]["cnt"] if entity_count else 0,
            "relationships": rel_count[0]["cnt"] if rel_count else 0,
            "communities": len(communities),
        }

    def persist(self) -> None:
        pass

    def close(self) -> None:
        self._driver.close()

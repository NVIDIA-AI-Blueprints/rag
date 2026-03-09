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

"""Tests for the NetworkX graph store implementation."""

import os
import tempfile

import pytest

from nvidia_rag.utils.graph.graph_store import CommunityInfo, Entity, Relationship
from nvidia_rag.utils.graph.networkx_store import NetworkXGraphStore


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(temp_dir):
    return NetworkXGraphStore(data_dir=temp_dir)


COLLECTION = "test_collection"


class TestEntity:
    def test_entity_key_normalization(self):
        e = Entity(name="NVIDIA Corporation", entity_type="Organization")
        assert e.key == "nvidia corporation"

    def test_entity_with_metadata(self):
        e = Entity(
            name="ServiceA",
            entity_type="Service",
            description="A microservice",
            source_chunk_ids=["abc123"],
            metadata={"page": 1},
        )
        assert e.name == "ServiceA"
        assert e.metadata["page"] == 1


class TestNetworkXGraphStore:
    def test_add_entities(self, store):
        entities = [
            Entity(name="Alice", entity_type="Person", description="Engineer"),
            Entity(name="Bob", entity_type="Person", description="Manager"),
        ]
        added = store.add_entities(entities, COLLECTION)
        assert added == 2

    def test_add_duplicate_entity_merges(self, store):
        e1 = Entity(name="Alice", entity_type="Person", description="Short", source_chunk_ids=["c1"])
        e2 = Entity(name="Alice", entity_type="Person", description="A longer description", source_chunk_ids=["c2"])
        store.add_entities([e1], COLLECTION)
        added = store.add_entities([e2], COLLECTION)
        assert added == 0  # duplicate, not added
        entity = store.get_entity("Alice", COLLECTION)
        assert entity is not None
        assert entity.description == "A longer description"
        assert set(entity.source_chunk_ids) == {"c1", "c2"}

    def test_add_relationships(self, store):
        entities = [
            Entity(name="Alice", entity_type="Person"),
            Entity(name="Bob", entity_type="Person"),
        ]
        store.add_entities(entities, COLLECTION)
        rels = [
            Relationship(
                source="Alice",
                target="Bob",
                relation_type="manages",
                description="Alice manages Bob",
            )
        ]
        added = store.add_relationships(rels, COLLECTION)
        assert added == 1

    def test_add_relationship_creates_missing_nodes(self, store):
        rels = [
            Relationship(source="X", target="Y", relation_type="links_to")
        ]
        store.add_relationships(rels, COLLECTION)
        assert store.get_entity("X", COLLECTION) is not None
        assert store.get_entity("Y", COLLECTION) is not None

    def test_get_entity(self, store):
        store.add_entities(
            [Entity(name="Alice", entity_type="Person", description="Engineer")],
            COLLECTION,
        )
        entity = store.get_entity("alice", COLLECTION)
        assert entity is not None
        assert entity.name == "Alice"
        assert entity.entity_type == "Person"

    def test_get_entity_not_found(self, store):
        assert store.get_entity("nonexistent", COLLECTION) is None

    def test_get_neighbors_depth_1(self, store):
        store.add_entities(
            [
                Entity(name="A", entity_type="Node"),
                Entity(name="B", entity_type="Node"),
                Entity(name="C", entity_type="Node"),
            ],
            COLLECTION,
        )
        store.add_relationships(
            [
                Relationship(source="A", target="B", relation_type="connects"),
                Relationship(source="B", target="C", relation_type="connects"),
            ],
            COLLECTION,
        )
        entities, rels = store.get_neighbors("A", COLLECTION, depth=1)
        entity_names = {e.name for e in entities}
        assert "A" in entity_names
        assert "B" in entity_names
        assert "C" not in entity_names
        assert len(rels) == 1

    def test_get_neighbors_depth_2(self, store):
        store.add_entities(
            [
                Entity(name="A", entity_type="Node"),
                Entity(name="B", entity_type="Node"),
                Entity(name="C", entity_type="Node"),
            ],
            COLLECTION,
        )
        store.add_relationships(
            [
                Relationship(source="A", target="B", relation_type="connects"),
                Relationship(source="B", target="C", relation_type="connects"),
            ],
            COLLECTION,
        )
        entities, rels = store.get_neighbors("A", COLLECTION, depth=2)
        entity_names = {e.name for e in entities}
        assert "C" in entity_names
        assert len(rels) == 2

    def test_get_neighbors_includes_predecessors(self, store):
        store.add_entities(
            [
                Entity(name="X", entity_type="Node"),
                Entity(name="Y", entity_type="Node"),
            ],
            COLLECTION,
        )
        store.add_relationships(
            [Relationship(source="X", target="Y", relation_type="points_to")],
            COLLECTION,
        )
        entities, rels = store.get_neighbors("Y", COLLECTION, depth=1)
        entity_names = {e.name for e in entities}
        assert "X" in entity_names

    def test_get_all_entities(self, store):
        store.add_entities(
            [
                Entity(name="A", entity_type="Node"),
                Entity(name="B", entity_type="Node"),
            ],
            COLLECTION,
        )
        all_entities = store.get_all_entities(COLLECTION)
        assert len(all_entities) == 2

    def test_get_all_relationships(self, store):
        store.add_entities(
            [Entity(name="A", entity_type="Node"), Entity(name="B", entity_type="Node")],
            COLLECTION,
        )
        store.add_relationships(
            [Relationship(source="A", target="B", relation_type="links")],
            COLLECTION,
        )
        all_rels = store.get_all_relationships(COLLECTION)
        assert len(all_rels) == 1
        assert all_rels[0].relation_type == "links"

    def test_communities(self, store):
        communities = [
            CommunityInfo(
                community_id=0,
                entity_names=["A", "B"],
                summary="A and B are related",
                level=0,
            ),
            CommunityInfo(
                community_id=1,
                entity_names=["C", "D"],
                summary="C and D form a group",
                level=0,
            ),
        ]
        store.set_communities(communities, COLLECTION)
        result = store.get_communities(COLLECTION)
        assert len(result) == 2

    def test_get_community_for_entity(self, store):
        communities = [
            CommunityInfo(
                community_id=0,
                entity_names=["Alice", "Bob"],
                summary="Team members",
            ),
        ]
        store.set_communities(communities, COLLECTION)
        comm = store.get_community_for_entity("alice", COLLECTION)
        assert comm is not None
        assert comm.community_id == 0

    def test_get_community_for_entity_not_found(self, store):
        assert store.get_community_for_entity("nobody", COLLECTION) is None

    def test_collection_exists(self, store):
        assert store.collection_exists(COLLECTION) is False
        store.add_entities([Entity(name="X", entity_type="Node")], COLLECTION)
        assert store.collection_exists(COLLECTION) is True

    def test_delete_collection(self, store):
        store.add_entities([Entity(name="X", entity_type="Node")], COLLECTION)
        store.delete_collection(COLLECTION)
        assert store.collection_exists(COLLECTION) is False

    def test_get_stats(self, store):
        store.add_entities(
            [Entity(name="A", entity_type="Node"), Entity(name="B", entity_type="Node")],
            COLLECTION,
        )
        store.add_relationships(
            [Relationship(source="A", target="B", relation_type="links")],
            COLLECTION,
        )
        store.set_communities(
            [CommunityInfo(community_id=0, entity_names=["A", "B"], summary="test")],
            COLLECTION,
        )
        stats = store.get_stats(COLLECTION)
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
        assert stats["communities"] == 1

    def test_persist_and_reload(self, temp_dir):
        store1 = NetworkXGraphStore(data_dir=temp_dir)
        store1.add_entities(
            [Entity(name="Persist", entity_type="Test", description="Should persist")],
            COLLECTION,
        )
        store1.add_relationships(
            [Relationship(source="Persist", target="Persist", relation_type="self_ref")],
            COLLECTION,
        )
        store1.set_communities(
            [CommunityInfo(community_id=0, entity_names=["Persist"], summary="Self")],
            COLLECTION,
        )
        store1.persist()

        store2 = NetworkXGraphStore(data_dir=temp_dir)
        assert store2.collection_exists(COLLECTION)
        entity = store2.get_entity("Persist", COLLECTION)
        assert entity is not None
        assert entity.description == "Should persist"
        assert len(store2.get_all_relationships(COLLECTION)) == 1
        assert len(store2.get_communities(COLLECTION)) == 1

    def test_get_entities_by_names(self, store):
        store.add_entities(
            [
                Entity(name="A", entity_type="Node"),
                Entity(name="B", entity_type="Node"),
                Entity(name="C", entity_type="Node"),
            ],
            COLLECTION,
        )
        found = store.get_entities_by_names(["A", "C", "D"], COLLECTION)
        names = {e.name for e in found}
        assert names == {"A", "C"}

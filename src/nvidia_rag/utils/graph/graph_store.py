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

"""Abstract base class and data models for knowledge graph storage."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """A node in the knowledge graph."""

    name: str = Field(description="Canonical name of the entity")
    entity_type: str = Field(description="Type/category (e.g. Person, Service, Concept)")
    description: str = Field(default="", description="Brief description of the entity")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source chunks this entity was extracted from",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Normalized lookup key for entity resolution."""
        return self.name.strip().lower()


class Relationship(BaseModel):
    """An edge in the knowledge graph."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Type of relationship (e.g. depends_on, authored_by)")
    description: str = Field(default="", description="Description of the relationship")
    weight: float = Field(default=1.0, description="Strength/confidence of the relationship")
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Triple(BaseModel):
    """A Subject-Predicate-Object triple for knowledge graph construction."""

    subject: str = Field(description="Subject entity name")
    subject_type: str = Field(description="Type of the subject entity")
    predicate: str = Field(description="Relationship type between subject and object")
    object: str = Field(description="Object entity name")
    object_type: str = Field(description="Type of the object entity")
    description: str = Field(default="", description="Description of this relationship")


class CommunityInfo(BaseModel):
    """A community (cluster) of related entities with a summary."""

    community_id: int
    entity_names: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    level: int = Field(default=0, description="Hierarchy level in community detection")


class GraphStore(ABC):
    """Abstract base class for knowledge graph storage backends."""

    @abstractmethod
    def add_entities(self, entities: list[Entity], collection_name: str) -> int:
        """Add entities (nodes) to the graph. Returns count of new entities added."""

    @abstractmethod
    def add_relationships(self, relationships: list[Relationship], collection_name: str) -> int:
        """Add relationships (edges) to the graph. Returns count of new edges added."""

    @abstractmethod
    def get_entity(self, name: str, collection_name: str) -> Entity | None:
        """Look up a single entity by name."""

    @abstractmethod
    def get_neighbors(
        self, entity_name: str, collection_name: str, depth: int = 1
    ) -> tuple[list[Entity], list[Relationship]]:
        """Get entities and relationships within N hops of the given entity."""

    @abstractmethod
    def get_entities_by_names(
        self, names: list[str], collection_name: str
    ) -> list[Entity]:
        """Look up multiple entities by name."""

    @abstractmethod
    def get_all_entities(self, collection_name: str) -> list[Entity]:
        """Return all entities in the given collection."""

    @abstractmethod
    def get_all_relationships(self, collection_name: str) -> list[Relationship]:
        """Return all relationships in the given collection."""

    @abstractmethod
    def set_communities(
        self, communities: list[CommunityInfo], collection_name: str
    ) -> None:
        """Store community detection results."""

    @abstractmethod
    def get_communities(self, collection_name: str) -> list[CommunityInfo]:
        """Retrieve all communities for a collection."""

    @abstractmethod
    def get_community_for_entity(
        self, entity_name: str, collection_name: str
    ) -> CommunityInfo | None:
        """Get the community that contains the given entity."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Remove all graph data for a collection."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if graph data exists for a collection."""

    @abstractmethod
    def get_stats(self, collection_name: str) -> dict[str, int]:
        """Return entity/relationship/community counts."""

    @abstractmethod
    def get_entity_degree(self, name: str, collection_name: str) -> int:
        """Return the total number of edges (in + out) for an entity. 0 if not found."""

    @abstractmethod
    def persist(self) -> None:
        """Persist graph data to durable storage (no-op for server-backed stores)."""

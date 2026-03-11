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

"""Tests for graph retriever structural ranking and GraphRetrievalResult."""

import pytest

from nvidia_rag.rag_server.graph_retriever import (
    GraphRetrievalResult,
    _structural_entity_score,
)
from nvidia_rag.utils.graph.graph_store import Entity


class TestGraphRetrievalResult:
    def test_defaults(self):
        r = GraphRetrievalResult()
        assert r.community_summary == ""
        assert r.boost_chunk_ids == set()
        assert r.entity_count == 0

    def test_with_values(self):
        r = GraphRetrievalResult(
            community_summary="summary",
            boost_chunk_ids={"abc123", "def456"},
            entity_count=5,
            relationship_count=10,
            community_count=2,
        )
        assert len(r.boost_chunk_ids) == 2
        assert r.community_summary == "summary"


class TestStructuralEntityScore:
    def _make_entity(self, hop=0, chunks=1, desc_len=100):
        return Entity(
            name="test",
            entity_type="test",
            description="x" * desc_len,
            source_chunk_ids=["c"] * chunks,
            metadata={"hop_distance": hop},
        )

    def test_hop0_scores_higher_than_hop2(self):
        e0 = self._make_entity(hop=0)
        e2 = self._make_entity(hop=2)
        assert _structural_entity_score(e0, 1) > _structural_entity_score(e2, 1)

    def test_more_connections_boost_score(self):
        e = self._make_entity(hop=1)
        score_1 = _structural_entity_score(e, 1)
        score_3 = _structural_entity_score(e, 3)
        assert score_3 > score_1

    def test_more_chunks_boost_score(self):
        e1 = self._make_entity(chunks=1)
        e5 = self._make_entity(chunks=5)
        assert _structural_entity_score(e5, 1) > _structural_entity_score(e1, 1)


class TestConfigurationIntegration:
    def test_graphrag_config_defaults(self):
        from nvidia_rag.utils.configuration import GraphRAGConfig

        cfg = GraphRAGConfig()
        assert cfg.enable_graph_rag is False
        assert cfg.graph_store_type == "networkx"
        assert cfg.traversal_depth == 2
        assert cfg.graph_boost_weight == 0.1
        assert cfg.graph_boost_top_entities == 20
        assert cfg.graph_chunk_replacement is False
        assert cfg.graph_replacement_max == 2
        assert cfg.graph_replacement_score_threshold == 0.35

    def test_graphrag_config_validation(self):
        from nvidia_rag.utils.configuration import GraphRAGConfig

        with pytest.raises(ValueError, match="traversal_depth"):
            GraphRAGConfig(traversal_depth=10)

    def test_nvidia_rag_config_includes_graph_rag(self):
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        cfg = NvidiaRAGConfig()
        assert hasattr(cfg, "graph_rag")
        assert cfg.graph_rag.enable_graph_rag is False

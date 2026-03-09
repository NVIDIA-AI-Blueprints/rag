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

"""Tests for graph retriever and RRF fusion."""

import pytest
from langchain_core.documents import Document

from nvidia_rag.rag_server.graph_retriever import (
    classify_query_complexity,
    hybrid_rank_fusion,
)


class TestClassifyQueryComplexity:
    def test_simple_factual_query(self):
        assert classify_query_complexity("What is CUDA?") == "simple"

    def test_simple_short_query(self):
        assert classify_query_complexity("Tell me about GPU") == "simple"

    def test_complex_relationship_query(self):
        result = classify_query_complexity(
            "What is the relationship between ServiceA and ServiceB and how do they interact?"
        )
        assert result == "complex"

    def test_complex_comparison_query(self):
        result = classify_query_complexity(
            "Compare the performance differences between CUDA and OpenCL"
        )
        assert result == "complex"

    def test_complex_multi_entity_query(self):
        result = classify_query_complexity(
            "How does the authentication flow connect to the database service and what impact does it have on the API gateway?"
        )
        assert result == "complex"

    def test_complex_overview_query(self):
        result = classify_query_complexity(
            "Summarize all the security-related decisions across multiple documents"
        )
        assert result == "complex"

    def test_complex_trace_query(self):
        result = classify_query_complexity(
            "Trace the flow of data between the ingestion pipeline and the retrieval service"
        )
        assert result == "complex"


class TestHybridRankFusion:
    def _make_docs(self, contents, source="vector"):
        return [
            Document(
                page_content=c,
                metadata={"source": source, "idx": i},
            )
            for i, c in enumerate(contents)
        ]

    def test_vector_only(self):
        vector_docs = self._make_docs(["a", "b", "c"])
        result = hybrid_rank_fusion(vector_docs, [], graph_weight=0.4, top_k=10)
        assert len(result) == 3
        assert all("rrf_score" in d.metadata for d in result)

    def test_graph_only(self):
        graph_docs = self._make_docs(["x", "y"], source="graph")
        result = hybrid_rank_fusion([], graph_docs, graph_weight=0.4, top_k=10)
        assert len(result) == 2

    def test_fusion_merges_results(self):
        vector_docs = self._make_docs(["a", "b", "c"])
        graph_docs = self._make_docs(["x", "y"], source="graph")
        result = hybrid_rank_fusion(vector_docs, graph_docs, graph_weight=0.4, top_k=10)
        assert len(result) == 5

    def test_fusion_respects_top_k(self):
        vector_docs = self._make_docs(["a", "b", "c"])
        graph_docs = self._make_docs(["x", "y", "z"], source="graph")
        result = hybrid_rank_fusion(vector_docs, graph_docs, graph_weight=0.4, top_k=3)
        assert len(result) == 3

    def test_fusion_scores_are_sorted_descending(self):
        vector_docs = self._make_docs(["a", "b", "c"])
        graph_docs = self._make_docs(["x", "y"], source="graph")
        result = hybrid_rank_fusion(vector_docs, graph_docs, graph_weight=0.5, top_k=10)
        scores = [d.metadata["rrf_score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs(self):
        result = hybrid_rank_fusion([], [], graph_weight=0.4, top_k=10)
        assert result == []

    def test_high_graph_weight_favors_graph(self):
        vector_docs = self._make_docs(["v1"])
        graph_docs = self._make_docs(["g1"], source="graph")
        result = hybrid_rank_fusion(vector_docs, graph_docs, graph_weight=0.9, top_k=2)
        assert len(result) == 2
        graph_score = next(d.metadata["rrf_score"] for d in result if d.metadata.get("source") == "graph")
        vector_score = next(d.metadata["rrf_score"] for d in result if d.metadata.get("source") == "vector")
        assert graph_score > vector_score


class TestConfigurationIntegration:
    def test_graphrag_config_defaults(self):
        from nvidia_rag.utils.configuration import GraphRAGConfig

        cfg = GraphRAGConfig()
        assert cfg.enable_graph_rag is False
        assert cfg.graph_store_type == "networkx"
        assert cfg.traversal_depth == 2
        assert cfg.graph_weight_in_fusion == 0.4

    def test_graphrag_config_validation(self):
        from nvidia_rag.utils.configuration import GraphRAGConfig

        with pytest.raises(ValueError, match="graph_weight_in_fusion"):
            GraphRAGConfig(graph_weight_in_fusion=1.5)

        with pytest.raises(ValueError, match="traversal_depth"):
            GraphRAGConfig(traversal_depth=10)

    def test_nvidia_rag_config_includes_graph_rag(self):
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        cfg = NvidiaRAGConfig()
        assert hasattr(cfg, "graph_rag")
        assert cfg.graph_rag.enable_graph_rag is False

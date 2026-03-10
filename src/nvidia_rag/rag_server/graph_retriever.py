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

"""Graph-augmented retrieval for GraphRAG.

Provides:
1. Query complexity classification to route simple queries to vector-only search
2. Graph traversal retrieval using extracted entities and their neighborhoods
3. Reciprocal Rank Fusion (RRF) to merge vector and graph retrieval results
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.graph.entity_extractor import extract_entities_from_query
from nvidia_rag.utils.graph.graph_store import CommunityInfo, Entity, GraphStore, Relationship

logger = logging.getLogger(__name__)

COMPLEXITY_KEYWORDS = {
    "compare",
    "relationship",
    "between",
    "depend",
    "connect",
    "relate",
    "interact",
    "collaborate",
    "impact",
    "affect",
    "influence",
    "cause",
    "lead to",
    "result in",
    "across",
    "multiple",
    "all",
    "every",
    "summarize",
    "overview",
    "how does",
    "what are the",
    "who are",
    "which",
    "trace",
    "flow",
    "chain",
    "sequence",
    "path",
    "history",
    "evolution",
    "timeline",
}


def classify_query_complexity(query: str) -> str:
    """Classify query as 'simple' or 'complex' using heuristics.

    Complex queries involve relationships, comparisons, multi-hop reasoning,
    or global/thematic questions. Simple queries are direct factual lookups.

    Returns:
        'simple' or 'complex'
    """
    query_lower = query.lower().strip()

    complexity_score = 0

    for keyword in COMPLEXITY_KEYWORDS:
        if keyword in query_lower:
            complexity_score += 1

    if query_lower.count(" and ") >= 2:
        complexity_score += 1
    if "?" in query and len(query.split()) > 12:
        complexity_score += 1
    if any(w in query_lower for w in ("vs", "versus", "difference", "differ")):
        complexity_score += 2

    result = "complex" if complexity_score >= 2 else "simple"
    logger.debug("Query complexity for '%s...': %s (score=%d)", query[:50], result, complexity_score)
    return result


def _format_entity_context(entity: Entity) -> str:
    """Format an entity into a context string."""
    parts = [f"{entity.name} ({entity.entity_type})"]
    if entity.description:
        parts.append(entity.description)
    return ": ".join(parts)


def _format_relationship_context(rel: Relationship) -> str:
    """Format a relationship into a context string."""
    line = f"{rel.source} --[{rel.relation_type}]--> {rel.target}"
    if rel.description:
        line += f": {rel.description}"
    return line


def _format_community_context(community: CommunityInfo) -> str:
    """Format a community summary into a context string."""
    return community.summary


async def graph_retrieval(
    query: str,
    graph_store: GraphStore,
    collection_name: str,
    config: NvidiaRAGConfig | None = None,
    prompts: dict | None = None,
) -> list[Document]:
    """Retrieve context from the knowledge graph for a given query.

    Steps:
    1. Extract entities from the query using an LLM.
    2. Look up those entities in the graph.
    3. Traverse N-hop neighborhoods to gather related entities and relationships.
    4. Fetch community summaries for matched entities.
    5. Assemble everything into LangChain Document objects.

    Args:
        query: User's natural language query.
        graph_store: GraphStore instance.
        collection_name: Collection to search.
        config: NvidiaRAGConfig instance.
        prompts: Prompts dict loaded from prompt.yaml via get_prompts().

    Returns:
        List of Documents with graph-derived context.
    """
    if config is None:
        config = NvidiaRAGConfig()

    graph_cfg = config.graph_rag
    top_k = graph_cfg.graph_top_k
    depth = graph_cfg.traversal_depth

    query_entities = await extract_entities_from_query(query, config=config, prompts=prompts)

    if not query_entities:
        logger.info("No entities extracted from query, returning empty graph results")
        return []

    all_entities: dict[str, Entity] = {}
    all_relationships: list[Relationship] = []
    matched_communities: dict[int, CommunityInfo] = {}

    for entity_name in query_entities:
        entities, relationships = graph_store.get_neighbors(
            entity_name, collection_name, depth=depth
        )
        for e in entities:
            all_entities[e.key] = e
        all_relationships.extend(relationships)

        community = graph_store.get_community_for_entity(entity_name, collection_name)
        if community and community.community_id not in matched_communities:
            matched_communities[community.community_id] = community

    if not all_entities and not matched_communities:
        logger.info("No graph matches found for query entities: %s", query_entities)
        return []

    seen_rels: set[tuple[str, str, str]] = set()
    unique_relationships = []
    for rel in all_relationships:
        key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relationships.append(rel)

    documents: list[Document] = []

    for community in matched_communities.values():
        if community.summary:
            documents.append(Document(
                page_content=community.summary,
                metadata={
                    "source": "graph_community",
                    "community_id": community.community_id,
                    "retrieval_type": "graph",
                },
            ))

    entity_lines = [
        _format_entity_context(e) for e in all_entities.values() if e.description
    ][:20]
    rel_lines = [_format_relationship_context(rel) for rel in unique_relationships[:30]]

    parts = []
    if entity_lines:
        parts.append("Entities: " + "; ".join(entity_lines))
    if rel_lines:
        parts.append("Relationships: " + "; ".join(rel_lines))
    if parts:
        documents.append(Document(
            page_content="\n".join(parts),
            metadata={
                "source": "graph_context",
                "retrieval_type": "graph",
            },
        ))

    documents = documents[:top_k]

    logger.info(
        "Graph retrieval for '%s': %d entities, %d relationships, %d communities -> %d documents",
        query[:50],
        len(all_entities),
        len(unique_relationships),
        len(matched_communities),
        len(documents),
    )

    return documents


def hybrid_rank_fusion(
    vector_docs: list[Document],
    graph_docs: list[Document],
    graph_weight: float = 0.4,
    top_k: int = 10,
) -> list[Document]:
    """Merge vector and graph retrieval results using Reciprocal Rank Fusion (RRF).

    RRF score for each document: sum(1 / (k + rank_i)) across all result lists where
    it appears. Graph results are weighted by ``graph_weight``.

    Args:
        vector_docs: Documents from vector similarity search.
        graph_docs: Documents from graph traversal.
        graph_weight: Weight multiplier for graph results (0.0-1.0).
        top_k: Maximum number of documents to return.

    Returns:
        Merged and re-ranked list of Documents.
    """
    k = 60  # RRF constant

    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    vector_weight = 1.0 - graph_weight

    for rank, doc in enumerate(vector_docs):
        doc_key = _doc_key(doc)
        score = vector_weight * (1.0 / (k + rank + 1))
        doc_scores[doc_key] = doc_scores.get(doc_key, 0.0) + score
        if doc_key not in doc_map:
            doc_map[doc_key] = doc

    for rank, doc in enumerate(graph_docs):
        doc_key = _doc_key(doc)
        score = graph_weight * (1.0 / (k + rank + 1))
        doc_scores[doc_key] = doc_scores.get(doc_key, 0.0) + score
        if doc_key not in doc_map:
            doc_map[doc_key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results = []
    for doc_key in sorted_keys[:top_k]:
        doc = doc_map[doc_key]
        doc.metadata["rrf_score"] = doc_scores[doc_key]
        results.append(doc)

    logger.info(
        "RRF fusion: %d vector + %d graph -> %d merged (weight=%.2f)",
        len(vector_docs),
        len(graph_docs),
        len(results),
        graph_weight,
    )

    return results


def _doc_key(doc: Document) -> str:
    """Generate a deduplication key for a document."""
    content = doc.page_content[:200] if doc.page_content else ""
    source = doc.metadata.get("source", "")
    return f"{source}:{hash(content)}"

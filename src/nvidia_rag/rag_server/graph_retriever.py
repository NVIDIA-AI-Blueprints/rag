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
3. Structural relevance ranking of entities, relationships, and communities
4. Reciprocal Rank Fusion (RRF) to merge vector and graph retrieval results
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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


def _structural_entity_score(
    entity: Entity,
    query_connection_count: int,
) -> float:
    """Score an entity by graph-structural relevance signals.

    Combines hop distance (from BFS metadata), how many query entities
    lead to this entity, source chunk frequency, and description richness.
    All data comes from the graph — no external API calls.
    """
    hop = entity.metadata.get("hop_distance", 2)
    chunk_count = len(entity.source_chunk_ids)
    desc_len = len(entity.description)

    hop_score = 1.0 / (1 + hop)
    multi_query_boost = 1.0 + 0.5 * max(0, query_connection_count - 1)
    chunk_boost = 1.0 + math.log1p(chunk_count) * 0.3
    desc_boost = 1.0 + min(desc_len / 200, 1.0) * 0.2

    return hop_score * multi_query_boost * chunk_boost * desc_boost


def _structural_rel_score(
    rel: Relationship,
    entity_scores: dict[str, float],
) -> float:
    """Score a relationship by its endpoint entity scores and edge strength."""
    src_score = entity_scores.get(rel.source.strip().lower(), 0.1)
    tgt_score = entity_scores.get(rel.target.strip().lower(), 0.1)

    base = (src_score + tgt_score) / 2
    weight_boost = 1.0 + math.log1p(max(0, rel.weight - 1)) * 0.3
    desc_boost = 1.0 + min(len(rel.description) / 200, 1.0) * 0.2

    return base * weight_boost * desc_boost


async def _rerank_by_query_embedding(
    query: str,
    texts: list[str],
    embedder: Embeddings,
) -> list[int]:
    """Re-rank texts by cosine similarity to the query embedding.

    Used as a second-stage ranker on a small, pre-filtered candidate set.
    Returns indices sorted by descending relevance.
    Falls back to original order on any error.
    """
    try:
        query_vec = await embedder.aembed_query(query)
        text_vecs = await embedder.aembed_documents(texts)

        q = np.array(query_vec, dtype=np.float32)
        t = np.array(text_vecs, dtype=np.float32)

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return list(range(len(texts)))
        q = q / q_norm

        t_norms = np.linalg.norm(t, axis=1, keepdims=True)
        t_norms[t_norms == 0] = 1.0
        t = t / t_norms

        similarities = t @ q
        return list(np.argsort(-similarities))
    except Exception:
        logger.warning("Embedding re-ranking failed, keeping structural order", exc_info=True)
        return list(range(len(texts)))


async def graph_retrieval(
    query: str,
    graph_store: GraphStore,
    collection_name: str,
    config: NvidiaRAGConfig | None = None,
    prompts: dict | None = None,
    embedder: Embeddings | None = None,
) -> list[Document]:
    """Retrieve context from the knowledge graph for a given query.

    Two-stage ranking:
    1. **Structural** (always): score all traversed entities by hop distance,
       query-entity connections, source chunk count, and description richness.
       This narrows thousands of entities to a manageable candidate set.
    2. **Embedding** (optional): if ``embedder`` is provided, re-rank the top
       structural candidates by cosine similarity to the query. This picks the
       most query-relevant entities from the structurally important set.

    Args:
        query: User's natural language query.
        graph_store: GraphStore instance.
        collection_name: Collection to search.
        config: NvidiaRAGConfig instance.
        prompts: Prompts dict loaded from prompt.yaml via get_prompts().
        embedder: Optional embedding model for second-stage re-ranking.

    Returns:
        List of Documents with graph-derived context.
    """
    if config is None:
        config = NvidiaRAGConfig()

    graph_cfg = config.graph_rag
    top_k = graph_cfg.graph_top_k
    depth = graph_cfg.traversal_depth
    hub_threshold = graph_cfg.hub_entity_threshold

    query_entities = await extract_entities_from_query(query, config=config, prompts=prompts)

    if not query_entities:
        logger.info("No entities extracted from query, returning empty graph results")
        return []

    traverse_entities: list[str] = []
    hub_entities: list[str] = []

    if hub_threshold > 0:
        for entity_name in query_entities:
            degree = graph_store.get_entity_degree(entity_name, collection_name)
            if degree > hub_threshold:
                hub_entities.append(entity_name)
                logger.info(
                    "Hub entity skipped for traversal: '%s' (degree=%d, threshold=%d)",
                    entity_name, degree, hub_threshold,
                )
            else:
                traverse_entities.append(entity_name)
        if not traverse_entities and hub_entities:
            logger.info(
                "All query entities are hubs (%s), using least-connected hub as fallback",
                hub_entities,
            )
            degrees = [(e, graph_store.get_entity_degree(e, collection_name)) for e in hub_entities]
            degrees.sort(key=lambda x: x[1])
            traverse_entities.append(degrees[0][0])
            hub_entities.remove(degrees[0][0])
    else:
        traverse_entities = list(query_entities)

    all_entities: dict[str, Entity] = {}
    all_relationships: list[Relationship] = []
    matched_communities: dict[int, CommunityInfo] = {}
    entity_query_sources: dict[str, set[str]] = defaultdict(set)

    for entity_name in traverse_entities:
        entities, relationships = graph_store.get_neighbors(
            entity_name, collection_name, depth=depth
        )
        for e in entities:
            key = e.key
            entity_query_sources[key].add(entity_name)
            if key not in all_entities:
                all_entities[key] = e
            else:
                existing_hop = all_entities[key].metadata.get("hop_distance", 99)
                new_hop = e.metadata.get("hop_distance", 99)
                if new_hop < existing_hop:
                    all_entities[key] = e
        all_relationships.extend(relationships)

    for entity_name in traverse_entities + hub_entities:
        community = graph_store.get_community_for_entity(entity_name, collection_name)
        if community and community.community_id not in matched_communities:
            matched_communities[community.community_id] = community

    if not all_entities and not matched_communities:
        logger.info("No graph matches found for query entities: %s", query_entities)
        return []

    logger.info(
        "Hub filtering: %d query entities -> %d traversed, %d hubs skipped",
        len(query_entities), len(traverse_entities), len(hub_entities),
    )

    seen_rels: set[tuple[str, str, str]] = set()
    unique_relationships = []
    for rel in all_relationships:
        key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relationships.append(rel)

    # --- Rank by structural relevance ---
    entity_list = [e for e in all_entities.values() if e.description]
    community_list = [c for c in matched_communities.values() if c.summary]

    # Score entities by graph-structural signals
    entity_scores: dict[str, float] = {}
    for e in entity_list:
        score = _structural_entity_score(e, len(entity_query_sources.get(e.key, set())))
        entity_scores[e.key] = score

    entity_list.sort(key=lambda e: entity_scores.get(e.key, 0), reverse=True)

    # Score relationships by endpoint relevance and edge strength
    for rel in unique_relationships:
        rel.metadata["_score"] = _structural_rel_score(rel, entity_scores)
    unique_relationships.sort(key=lambda r: r.metadata.get("_score", 0), reverse=True)

    # Rank communities by how many query entities they contain
    query_entity_keys = {name.strip().lower() for name in traverse_entities + hub_entities}
    community_list.sort(
        key=lambda c: sum(1 for n in c.entity_names if n.strip().lower() in query_entity_keys),
        reverse=True,
    )

    logger.info(
        "Structural ranking: %d entities (top score=%.3f), %d relationships, %d communities",
        len(entity_list),
        entity_scores[entity_list[0].key] if entity_list else 0,
        len(unique_relationships),
        len(community_list),
    )

    # --- Stage 2: Embedding tiebreak — only when structural scores are ambiguous ---
    ENTITY_FINAL = 20
    REL_FINAL = 30
    RERANK_POOL_MULTIPLIER = 4
    SCORE_TIE_RATIO = 0.85

    if embedder and len(entity_list) > ENTITY_FINAL:
        cutoff_score = entity_scores.get(entity_list[ENTITY_FINAL - 1].key, 0)
        boundary_idx = min(ENTITY_FINAL + 10, len(entity_list) - 1)
        boundary_score = entity_scores.get(entity_list[boundary_idx].key, 0)

        if cutoff_score > 0 and boundary_score / cutoff_score > SCORE_TIE_RATIO:
            pool_size = min(ENTITY_FINAL * RERANK_POOL_MULTIPLIER, len(entity_list))
            pool = entity_list[:pool_size]
            pool_texts = [_format_entity_context(e) for e in pool]
            rerank_order = await _rerank_by_query_embedding(query, pool_texts, embedder)
            entity_list = [pool[i] for i in rerank_order]
            logger.info(
                "Embedding tiebreak: re-ranked %d entities (score ratio=%.3f at cutoff=%d)",
                pool_size, boundary_score / cutoff_score, ENTITY_FINAL,
            )
        else:
            logger.info(
                "Structural ranking sufficient for entities (score ratio=%.3f)",
                boundary_score / cutoff_score if cutoff_score > 0 else 0,
            )

    if embedder and len(unique_relationships) > REL_FINAL:
        rel_scores_list = [r.metadata.get("_score", 0) for r in unique_relationships]
        cutoff_score = rel_scores_list[REL_FINAL - 1] if len(rel_scores_list) > REL_FINAL - 1 else 0
        boundary_idx = min(REL_FINAL + 10, len(rel_scores_list) - 1)
        boundary_score = rel_scores_list[boundary_idx]

        if cutoff_score > 0 and boundary_score / cutoff_score > SCORE_TIE_RATIO:
            pool_size = min(REL_FINAL * RERANK_POOL_MULTIPLIER, len(unique_relationships))
            pool = unique_relationships[:pool_size]
            pool_texts = [_format_relationship_context(r) for r in pool]
            rerank_order = await _rerank_by_query_embedding(query, pool_texts, embedder)
            unique_relationships = [pool[i] for i in rerank_order]
            logger.info(
                "Embedding tiebreak: re-ranked %d relationships (score ratio=%.3f)",
                pool_size, boundary_score / cutoff_score,
            )

    if embedder and len(community_list) > 1:
        comm_texts = [c.summary for c in community_list]
        rerank_order = await _rerank_by_query_embedding(query, comm_texts, embedder)
        community_list = [community_list[i] for i in rerank_order]

    # --- Build documents from ranked results ---
    documents: list[Document] = []

    for community in community_list:
        documents.append(Document(
            page_content=community.summary,
            metadata={
                "source": "graph_community",
                "community_id": community.community_id,
                "retrieval_type": "graph",
            },
        ))

    entity_lines = [_format_entity_context(e) for e in entity_list][:20]
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

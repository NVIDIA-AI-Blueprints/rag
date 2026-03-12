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
1. Graph traversal retrieval using extracted entities and their neighborhoods
2. Structural relevance ranking of entities and relationships
3. GraphRetrievalResult with intersection-ranked chunk hashes for pool
   expansion and guaranteed-slot injection into the reranker pipeline
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from langchain_core.embeddings import Embeddings

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.graph.entity_extractor import extract_entities_from_query
from nvidia_rag.utils.graph.graph_store import Entity, GraphStore, Relationship

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    """Structured output from graph retrieval — no LangChain Documents.

    ``intersection_chunk_hashes`` contains SHA-256[:16] hashes ranked by an
    aggregate structural score.  Chunks referenced by multiple query-related
    entities score highest (multi-hop bridging chunks).  These are used at
    query time for pool expansion (injected into the reranker input) and
    guaranteed-slot replacement.
    """

    intersection_chunk_hashes: list[str] = field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0


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
    lead to this entity, and source chunk frequency.
    All data comes from the graph — no external API calls.
    """
    hop = entity.metadata.get("hop_distance", 2)
    chunk_count = len(entity.source_chunk_ids)

    hop_score = 1.0 / (1 + hop)
    multi_query_boost = 1.0 + 0.5 * max(0, query_connection_count - 1)
    chunk_boost = 1.0 + math.log1p(chunk_count) * 0.3

    return hop_score * multi_query_boost * chunk_boost


def _structural_rel_score(
    rel: Relationship,
    entity_scores: dict[str, float],
) -> float:
    """Score a relationship by its endpoint entity scores and edge strength."""
    src_score = entity_scores.get(rel.source.strip().lower(), 0.1)
    tgt_score = entity_scores.get(rel.target.strip().lower(), 0.1)

    base = (src_score + tgt_score) / 2
    weight_boost = 1.0 + math.log1p(max(0, rel.weight - 1)) * 0.3

    return base * weight_boost


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
) -> GraphRetrievalResult | None:
    """Retrieve graph signals for parallel retrieval (pool expansion + guaranteed slots).

    Two-stage ranking:
    1. **Structural** (always): score all traversed entities by hop distance,
       query-entity connections, source chunk count, and description richness.
    2. **Embedding** (optional): if ``embedder`` is provided, re-rank the top
       structural candidates by cosine similarity to the query.

    Returns a :class:`GraphRetrievalResult` with:
    - ``intersection_chunk_hashes``: chunk hashes ranked by aggregate entity
      score — chunks referenced by more/higher-scored entities rank first.
    """
    if config is None:
        config = NvidiaRAGConfig()

    graph_cfg = config.graph_rag
    top_entities_for_chunks = graph_cfg.graph_boost_top_entities
    depth = graph_cfg.traversal_depth
    hub_threshold = graph_cfg.hub_entity_threshold

    query_entities = await extract_entities_from_query(query, config=config, prompts=prompts)

    if not query_entities:
        logger.info("No entities extracted from query, returning empty graph results")
        return None

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

    if not all_entities:
        logger.info("No graph matches found for query entities: %s", query_entities)
        return None

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

    logger.info(
        "Structural ranking: %d entities (top score=%.3f), %d relationships",
        len(entity_list),
        entity_scores[entity_list[0].key] if entity_list else 0,
        len(unique_relationships),
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

    # --- Collect intersection-ranked chunk hashes from top entities ---
    # For each chunk hash, accumulate the structural scores of all entities
    # that reference it.  Chunks referenced by multiple high-scoring entities
    # (multi-hop bridging chunks) naturally rank highest.
    chunk_agg_score: dict[str, float] = defaultdict(float)
    chunk_entity_count: dict[str, int] = defaultdict(int)
    for entity in entity_list[:top_entities_for_chunks]:
        e_score = entity_scores.get(entity.key, 0)
        for chunk_id in entity.source_chunk_ids:
            chunk_agg_score[chunk_id] += e_score
            chunk_entity_count[chunk_id] += 1

    min_refs = graph_cfg.graph_min_entity_refs
    ranked_hashes = sorted(
        chunk_agg_score.keys(),
        key=lambda h: chunk_agg_score[h],
        reverse=True,
    )
    intersection_hashes = [h for h in ranked_hashes if chunk_entity_count[h] >= min_refs]
    if not intersection_hashes:
        intersection_hashes = ranked_hashes

    result = GraphRetrievalResult(
        intersection_chunk_hashes=intersection_hashes,
        entity_count=len(all_entities),
        relationship_count=len(unique_relationships),
    )

    multi_ref = sum(1 for h in intersection_hashes if chunk_entity_count.get(h, 0) >= 2)
    logger.info(
        "Graph retrieval for '%s': %d entities, %d relationships "
        "-> %d intersection hashes (%d with 2+ entity refs)",
        query[:50],
        result.entity_count,
        result.relationship_count,
        len(intersection_hashes),
        multi_ref,
    )

    return result

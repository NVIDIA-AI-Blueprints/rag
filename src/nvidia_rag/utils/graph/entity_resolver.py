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

"""Fully dynamic entity resolution for knowledge graphs.

All detection is data-driven — no hardcoded domain-specific lists. Works on
financial filings, medical records, legal docs, or any other corpus.

Two modes:

1. **Embedding-based** (production): semantic dedup + embedding-variance
   generic detection.  Pass ``embeddings`` to ``resolve_entities()``.

2. **Rule-based** (fallback): token overlap dedup + statistical generic
   detection.  Used when no embedding service is available.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Callable

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic helpers — everything computed from the graph, nothing hardcoded
# ---------------------------------------------------------------------------

def _discover_common_suffixes(g: nx.DiGraph, min_frequency: int = 5) -> set[str]:
    """Auto-detect common trailing words from entity names in the graph.

    If "corporation" appears as the last word in >=5 entity names, it's a
    suffix worth stripping during normalization.  This adapts to any domain:
    financial ("corp", "inc"), medical ("hospital", "syndrome"), etc.
    """
    trailing_counts: dict[str, int] = defaultdict(int)
    for node in g.nodes():
        words = node.strip().lower().split()
        if len(words) >= 2:
            trailing_counts[words[-1]] += 1

    return {word for word, count in trailing_counts.items() if count >= min_frequency}


def _normalize_key(name: str, suffixes_to_strip: set[str] | None = None) -> str:
    """Normalize an entity name for dedup grouping.

    Only applies language-level transforms (case fold, whitespace, strip "the")
    plus dynamically discovered suffixes.
    """
    n = name.strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"^the\s+", "", n)
    if suffixes_to_strip:
        words = n.split()
        if len(words) >= 2 and words[-1] in suffixes_to_strip:
            n = " ".join(words[:-1])
    return n


def _types_compatible(type_a: str, type_b: str) -> bool:
    """Dynamically check if two entity types are compatible for merging.

    Compatible if: identical, either is unknown, or they share any token.
    E.g. "financial metric" and "metric" share "metric" → compatible.
    No hardcoded mapping needed.
    """
    a = type_a.strip().lower()
    b = type_b.strip().lower()
    if a == b or a == "unknown" or b == "unknown":
        return True
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    return bool(tokens_a & tokens_b)


def _merge_nodes(g: nx.DiGraph, keep: str, merge: str) -> None:
    """Merge node *merge* into node *keep*, redirecting all edges."""
    if not g.has_node(merge) or not g.has_node(keep) or keep == merge:
        return

    keep_data = g.nodes[keep]
    merge_data = g.nodes[merge]

    existing_chunks = set(keep_data.get("source_chunk_ids", []))
    existing_chunks.update(merge_data.get("source_chunk_ids", []))
    keep_data["source_chunk_ids"] = list(existing_chunks)

    if len(merge_data.get("description", "")) > len(keep_data.get("description", "")):
        keep_data["description"] = merge_data["description"]

    if keep_data.get("entity_type", "unknown") == "unknown" and merge_data.get("entity_type", "unknown") != "unknown":
        keep_data["entity_type"] = merge_data["entity_type"]

    for pred in list(g.predecessors(merge)):
        if pred == keep or pred == merge:
            continue
        edge_data = dict(g[pred][merge])
        if not g.has_edge(pred, keep):
            g.add_edge(pred, keep, **edge_data)
        else:
            existing = g[pred][keep]
            old_chunks = set(existing.get("source_chunk_ids", []))
            old_chunks.update(edge_data.get("source_chunk_ids", []))
            existing["source_chunk_ids"] = list(old_chunks)

    for succ in list(g.successors(merge)):
        if succ == keep or succ == merge:
            continue
        edge_data = dict(g[merge][succ])
        if not g.has_edge(keep, succ):
            g.add_edge(keep, succ, **edge_data)
        else:
            existing = g[keep][succ]
            old_chunks = set(existing.get("source_chunk_ids", []))
            old_chunks.update(edge_data.get("source_chunk_ids", []))
            existing["source_chunk_ids"] = list(old_chunks)

    g.remove_node(merge)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def compute_entity_embeddings(
    g: nx.DiGraph,
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 100,
) -> dict[str, list[float]]:
    """Generate embeddings for all entities in a graph.

    Args:
        g: The knowledge graph.
        embed_fn: Synchronous callable — ``list[str] -> list[list[float]]``.
            For LangChain: ``embedder.embed_documents``.
        batch_size: Texts per API call.

    Returns:
        ``{entity_name: embedding_vector}``
    """
    nodes = list(g.nodes())
    if not nodes:
        return {}

    texts = []
    for node in nodes:
        desc = g.nodes[node].get("description", "")
        texts.append(f"{node}: {desc}" if desc else node)

    embeddings: dict[str, list[float]] = {}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_nodes = nodes[i : i + batch_size]
        batch_vectors = embed_fn(batch_texts)
        for node, vec in zip(batch_nodes, batch_vectors):
            embeddings[node] = vec

    logger.info("Computed embeddings for %d entities", len(embeddings))
    return embeddings


async def compute_entity_embeddings_async(
    g: nx.DiGraph,
    embed_fn: Callable[[list[str]], Any],
    batch_size: int = 100,
) -> dict[str, list[float]]:
    """Async version of :func:`compute_entity_embeddings`."""
    nodes = list(g.nodes())
    if not nodes:
        return {}

    texts = []
    for node in nodes:
        desc = g.nodes[node].get("description", "")
        texts.append(f"{node}: {desc}" if desc else node)

    embeddings: dict[str, list[float]] = {}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_nodes = nodes[i : i + batch_size]
        batch_vectors = await embed_fn(batch_texts)
        for node, vec in zip(batch_nodes, batch_vectors):
            embeddings[node] = vec

    logger.info("Computed embeddings for %d entities", len(embeddings))
    return embeddings


# ---------------------------------------------------------------------------
# Resolution passes — all fully dynamic, computed from the graph data
# ---------------------------------------------------------------------------

def pass_remove_generic_statistical(
    g: nx.DiGraph,
    degree_percentile: float = 95,
    doc_freq_ratio: float = 0.15,
) -> int:
    """Dynamically detect and remove generic hub entities using graph statistics.

    An entity is flagged as generic when ALL conditions are met:
    1. Degree in the top ``degree_percentile`` (high connectivity)
    2. Appears in >= ``doc_freq_ratio`` of all source chunks (ubiquitous)
    3. Name does NOT look like a named entity (no digits, not a multi-word
       proper noun, not an all-caps acronym)

    Works on any domain without any hardcoded lists.
    """
    if g.number_of_nodes() < 10:
        return 0

    degrees = {n: g.in_degree(n) + g.out_degree(n) for n in g.nodes()}
    sorted_degs = sorted(degrees.values())
    degree_cutoff = sorted_degs[int(len(sorted_degs) * degree_percentile / 100)]

    all_chunks: set[str] = set()
    entity_chunks: dict[str, set[str]] = {}
    for node in g.nodes():
        chunks = set(g.nodes[node].get("source_chunk_ids", []))
        entity_chunks[node] = chunks
        all_chunks.update(chunks)
    total_chunks = max(len(all_chunks), 1)

    to_remove = []
    for node in g.nodes():
        if degrees[node] < degree_cutoff:
            continue
        if len(entity_chunks[node]) / total_chunks < doc_freq_ratio:
            continue
        words = node.strip().split()
        has_digits = any(c.isdigit() for c in node)
        is_acronym = node == node.upper() and len(node) <= 8
        if len(words) >= 3 or has_digits or is_acronym:
            continue
        to_remove.append(node)

    g.remove_nodes_from(to_remove)
    return len(to_remove)


def pass_remove_generic_by_embedding(
    g: nx.DiGraph,
    embeddings: dict[str, list[float]],
    degree_percentile: float = 90,
) -> int:
    """Remove generic hubs using embedding similarity variance.

    A generic entity like "company" is vaguely similar to everything (low
    variance in cosine similarities).  A specific entity like "Intel" is very
    similar to some things and very different from others (high variance).

    Only considers entities with degree above ``degree_percentile`` to avoid
    touching low-degree nodes.
    """
    if g.number_of_nodes() < 10:
        return 0

    degrees = {n: g.in_degree(n) + g.out_degree(n) for n in g.nodes()}
    sorted_degs = sorted(degrees.values())
    degree_cutoff = sorted_degs[int(len(sorted_degs) * degree_percentile / 100)]

    candidates = [n for n in g.nodes() if degrees[n] >= degree_cutoff and n in embeddings]
    if not candidates:
        return 0

    all_nodes = [n for n in g.nodes() if n in embeddings]
    all_matrix = np.array([embeddings[n] for n in all_nodes], dtype=np.float32)
    norms = np.linalg.norm(all_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_normed = all_matrix / norms

    variances = []
    candidate_names = []
    for node in candidates:
        if node not in embeddings:
            continue
        vec = np.array(embeddings[node], dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            continue
        sims = all_normed @ (vec / vec_norm)
        variances.append(np.var(sims).item())
        candidate_names.append(node)

    if not variances:
        return 0

    mean_var = float(np.mean(variances))
    std_var = float(np.std(variances))
    # Low variance = uniformly similar to everything = generic
    var_threshold = mean_var - std_var

    to_remove = [
        candidate_names[i]
        for i, v in enumerate(variances)
        if v <= var_threshold
    ]
    g.remove_nodes_from(to_remove)
    return len(to_remove)


def pass_normalize_and_merge(g: nx.DiGraph) -> int:
    """Merge entities that share the same dynamically-normalized name.

    Discovers common trailing words from the graph itself (e.g. "corporation",
    "inc" in financial data, "hospital" in medical data) and strips them
    before grouping.
    """
    suffixes = _discover_common_suffixes(g)
    if suffixes:
        logger.info("Auto-detected trailing suffixes to strip: %s", suffixes)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in list(g.nodes()):
        groups[_normalize_key(node, suffixes)].append(node)

    merged = 0
    for members in groups.values():
        if len(members) <= 1:
            continue

        compatible_groups: list[list[str]] = []
        for member in members:
            mtype = g.nodes[member].get("entity_type", "unknown")
            placed = False
            for group in compatible_groups:
                gtype = g.nodes[group[0]].get("entity_type", "unknown")
                if _types_compatible(mtype, gtype):
                    group.append(member)
                    placed = True
                    break
            if not placed:
                compatible_groups.append([member])

        for group in compatible_groups:
            if len(group) <= 1:
                continue
            group.sort(key=lambda n: len(g.nodes[n].get("description", "")), reverse=True)
            keep = group[0]
            for to_merge in group[1:]:
                _merge_nodes(g, keep, to_merge)
                merged += 1

    return merged


def pass_token_overlap_merge(g: nx.DiGraph, threshold: float = 0.75) -> int:
    """Merge entities with high token overlap and compatible types.

    Fallback dedup when embeddings are not available.
    """
    nodes = list(g.nodes())
    node_tokens = {n: set(n.split()) for n in nodes}

    merge_pairs: list[tuple[str, str]] = []
    processed: set[str] = set()

    for i, a in enumerate(nodes):
        if a not in g or a in processed:
            continue
        tokens_a = node_tokens[a]
        if len(tokens_a) < 2:
            continue
        type_a = g.nodes[a].get("entity_type", "unknown")

        for b in nodes[i + 1 :]:
            if b not in g or b in processed:
                continue
            tokens_b = node_tokens[b]
            if len(tokens_b) < 2:
                continue
            if not _types_compatible(type_a, g.nodes[b].get("entity_type", "unknown")):
                continue
            overlap = tokens_a & tokens_b
            score = len(overlap) / max(len(tokens_a), len(tokens_b))
            if score >= threshold:
                merge_pairs.append((a, b))
                processed.add(b)

    merged = 0
    for keep, to_merge in merge_pairs:
        if g.has_node(keep) and g.has_node(to_merge):
            _merge_nodes(g, keep, to_merge)
            merged += 1

    return merged


def pass_embedding_dedup(
    g: nx.DiGraph,
    embeddings: dict[str, list[float]],
    similarity_threshold: float = 0.85,
) -> int:
    """Merge entities with high embedding cosine similarity.

    Catches semantic duplicates like "R&D" / "Research and Development"
    that rule-based approaches miss.
    """
    nodes = [n for n in g.nodes() if n in embeddings]
    if len(nodes) < 2:
        return 0

    emb_matrix = np.array([embeddings[n] for n in nodes], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = emb_matrix / norms

    sim_matrix = emb_normed @ emb_normed.T
    np.fill_diagonal(sim_matrix, 0.0)

    rows, cols = np.where(np.triu(sim_matrix, k=1) >= similarity_threshold)
    if len(rows) == 0:
        return 0

    pair_sims = sim_matrix[rows, cols]
    sort_order = np.argsort(-pair_sims)
    rows = rows[sort_order]
    cols = cols[sort_order]

    consumed: set[int] = set()
    merge_pairs: list[tuple[str, str]] = []

    for idx in range(len(rows)):
        i, j = int(rows[idx]), int(cols[idx])
        if i in consumed or j in consumed:
            continue
        node_i, node_j = nodes[i], nodes[j]
        if not g.has_node(node_i) or not g.has_node(node_j):
            continue
        type_i = g.nodes[node_i].get("entity_type", "unknown")
        type_j = g.nodes[node_j].get("entity_type", "unknown")
        if not _types_compatible(type_i, type_j):
            continue
        merge_pairs.append((node_i, node_j))
        consumed.add(j)

    merged = 0
    for keep, to_merge in merge_pairs:
        if g.has_node(keep) and g.has_node(to_merge):
            _merge_nodes(g, keep, to_merge)
            merged += 1

    return merged


def pass_prune_stranded(g: nx.DiGraph, min_chunks: int = 2) -> int:
    """Remove nodes with zero edges and fewer than *min_chunks* source chunks."""
    to_remove = [
        node for node in g.nodes()
        if (g.in_degree(node) + g.out_degree(node)) == 0
        and len(g.nodes[node].get("source_chunk_ids", [])) < min_chunks
    ]
    g.remove_nodes_from(to_remove)
    return len(to_remove)


def pass_prune_weak(g: nx.DiGraph, max_degree: int = 1, min_desc_len: int = 10) -> int:
    """Remove low-value nodes: few connections and no meaningful description."""
    to_remove = [
        node for node in g.nodes()
        if (g.in_degree(node) + g.out_degree(node)) <= max_degree
        and len(g.nodes[node].get("description", "")) < min_desc_len
    ]
    g.remove_nodes_from(to_remove)
    return len(to_remove)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_entities(
    g: nx.DiGraph,
    *,
    embeddings: dict[str, list[float]] | None = None,
    statistical_degree_percentile: float = 95,
    statistical_doc_freq_ratio: float = 0.15,
    embedding_similarity_threshold: float = 0.85,
    embedding_generic_degree_percentile: float = 90,
    token_overlap_threshold: float = 0.75,
    prune_stranded_min_chunks: int = 2,
    prune_weak_max_degree: int = 1,
    prune_weak_min_desc: int = 10,
) -> dict[str, int]:
    """Run full entity resolution on a NetworkX graph **in-place**.

    Everything is data-driven.  When ``embeddings`` are provided, uses
    embedding similarity for dedup and variance analysis for generic
    detection.  Otherwise falls back to token overlap and degree+doc-frequency
    statistics.

    Args:
        g: Knowledge graph to resolve (modified in-place).
        embeddings: Pre-computed ``{node_name: vector}``.  Use
            :func:`compute_entity_embeddings` to generate.
        statistical_degree_percentile: Degree percentile for statistical generic pass.
        statistical_doc_freq_ratio: Min doc-frequency ratio to flag as generic.
        embedding_similarity_threshold: Cosine threshold for embedding dedup.
        embedding_generic_degree_percentile: Degree percentile for embedding generic pass.
        token_overlap_threshold: Token overlap score for rule-based dedup.
        prune_stranded_min_chunks: Min source chunks to keep a degree-0 entity.
        prune_weak_max_degree: Max degree for weak-node pruning.
        prune_weak_min_desc: Min description length to keep a weak node.

    Returns:
        Dict with per-pass counts and totals.
    """
    original_nodes = g.number_of_nodes()
    original_edges = g.number_of_edges()
    mode = "embedding" if embeddings else "rule-based"
    logger.info(
        "Entity resolution starting (%s mode): %d entities, %d relationships",
        mode, original_nodes, original_edges,
    )

    stats: dict[str, int] = {"mode": 1 if embeddings else 0}

    # --- Phase 1: Remove generic hubs ---
    if embeddings:
        stats["embedding_generic_removed"] = pass_remove_generic_by_embedding(
            g, embeddings, degree_percentile=embedding_generic_degree_percentile,
        )
        logger.info(
            "Pass (embedding generic): removed %d -> %d remaining",
            stats["embedding_generic_removed"], g.number_of_nodes(),
        )
    else:
        stats["statistical_generic_removed"] = pass_remove_generic_statistical(
            g,
            degree_percentile=statistical_degree_percentile,
            doc_freq_ratio=statistical_doc_freq_ratio,
        )
        logger.info(
            "Pass (statistical generic): removed %d -> %d remaining",
            stats["statistical_generic_removed"], g.number_of_nodes(),
        )

    # --- Phase 2: Merge duplicates ---
    stats["normalized_merged"] = pass_normalize_and_merge(g)
    logger.info(
        "Pass (normalization): merged %d -> %d remaining",
        stats["normalized_merged"], g.number_of_nodes(),
    )

    if embeddings:
        stats["embedding_merged"] = pass_embedding_dedup(
            g, embeddings, similarity_threshold=embedding_similarity_threshold,
        )
        logger.info(
            "Pass (embedding dedup): merged %d -> %d remaining",
            stats["embedding_merged"], g.number_of_nodes(),
        )
    else:
        stats["overlap_merged"] = pass_token_overlap_merge(
            g, threshold=token_overlap_threshold,
        )
        logger.info(
            "Pass (token overlap): merged %d -> %d remaining",
            stats["overlap_merged"], g.number_of_nodes(),
        )

    # --- Phase 3: Prune low-value nodes ---
    stats["stranded_pruned"] = pass_prune_stranded(g, min_chunks=prune_stranded_min_chunks)
    logger.info(
        "Pass (prune stranded): removed %d -> %d remaining",
        stats["stranded_pruned"], g.number_of_nodes(),
    )

    stats["weak_pruned"] = pass_prune_weak(
        g, max_degree=prune_weak_max_degree, min_desc_len=prune_weak_min_desc,
    )
    logger.info(
        "Pass (prune weak): removed %d -> %d remaining",
        stats["weak_pruned"], g.number_of_nodes(),
    )

    # --- Cleanup ---
    self_loops = list(nx.selfloop_edges(g))
    g.remove_edges_from(self_loops)
    stats["self_loops_removed"] = len(self_loops)
    if self_loops:
        logger.info("Removed %d self-loops", len(self_loops))

    final_nodes = g.number_of_nodes()
    final_edges = g.number_of_edges()
    reduction_pct = (1 - final_nodes / max(original_nodes, 1)) * 100

    stats["original_entities"] = original_nodes
    stats["original_relationships"] = original_edges
    stats["final_entities"] = final_nodes
    stats["final_relationships"] = final_edges

    logger.info(
        "Entity resolution complete: %d -> %d entities (%.0f%% reduction), %d -> %d relationships",
        original_nodes, final_nodes, reduction_pct, original_edges, final_edges,
    )

    return stats

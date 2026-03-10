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

"""Entity resolution (deduplication) for knowledge graphs.

Merges duplicate entities, removes generic/meaningless hub nodes, and prunes
low-value nodes from a NetworkX DiGraph. Designed to be called both from the
ingestion pipeline and from standalone scripts.

Usage from ingestor pipeline:
    from nvidia_rag.utils.graph.entity_resolver import resolve_entities
    stats = resolve_entities(nx_graph)

Usage from CLI script:
    python scripts/resolve_graph_entities.py --data-dir /path/to/graph-data
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)

COMMON_SUFFIXES = [
    " corporation", " corp", " corp.", " inc", " inc.", " llc", " ltd", " ltd.",
    " company", " co", " co.", " plc",
    " segment", " segments", " division",
    " expenses", " expense",
]

ABBREVIATION_MAP = {
    "r&d": "research and development",
    "r & d": "research and development",
    "ai": "artificial intelligence",
    "gpu": "graphics processing unit",
    "cpu": "central processing unit",
    "gaap": "generally accepted accounting principles",
    "sec": "securities and exchange commission",
    "ip": "intellectual property",
    "q1": "first quarter",
    "q2": "second quarter",
    "q3": "third quarter",
    "q4": "fourth quarter",
    "yoy": "year over year",
    "cogs": "cost of goods sold",
    "eps": "earnings per share",
    "rsu": "restricted stock unit",
    "rsus": "restricted stock units",
    "opex": "operating expenses",
    "capex": "capital expenditure",
}

GENERIC_ENTITIES = {
    "company", "the company", "companies", "corporation", "entity",
    "product", "products", "service", "services",
    "customer", "customers", "consumer", "consumers",
    "employee", "employees", "personnel", "staff", "workforce",
    "market", "markets", "industry", "industries",
    "period", "reporting period", "fiscal period",
    "year", "quarter", "month",
    "board", "board of directors", "management", "executive",
    "stockholder", "stockholders", "shareholder", "shareholders",
    "investor", "investors",
    "government", "governments", "regulator", "regulators",
    "country", "countries", "region", "regions",
    "risk", "risks", "risk factor", "risk factors",
    "agreement", "agreements", "contract", "contracts",
    "law", "laws", "regulation", "regulations",
    "technology", "technologies",
    "data", "information", "system", "systems",
    "asset", "assets", "liability", "liabilities",
    "cost", "costs", "price", "prices", "pricing",
    "tax", "taxes", "taxation",
    "debt", "equity", "capital", "cash",
    "share", "shares", "stock",
    "growth", "increase", "decrease", "change", "changes",
    "result", "results", "outcome", "outcomes",
    "operations", "operation", "business", "businesses",
    "segment", "segments",
    "note", "notes",
    "table", "tables", "figure", "figures",
    "item", "items",
    "form", "report", "reports", "filing", "filings",
    "date", "dates",
    "amount", "amounts", "value", "values", "rate", "rates",
    "factor", "factors",
    "party", "parties", "third party", "third parties",
    "supplier", "suppliers", "vendor", "vendors", "partner", "partners",
    "unit", "units",
    "people",
}

TYPE_COMPATIBILITY = {
    "organization": {"organization", "company", "corporation", "entity", "unknown"},
    "company": {"organization", "company", "corporation", "entity", "unknown"},
    "person": {"person", "people", "individual", "unknown"},
    "product": {"product", "technology", "hardware", "software", "platform", "unknown"},
    "technology": {"product", "technology", "hardware", "software", "platform", "unknown"},
    "financial metric": {"financial metric", "metric", "financial", "financial measure", "unknown"},
    "metric": {"financial metric", "metric", "financial", "financial measure", "unknown"},
    "market": {"market", "market segment", "segment", "industry", "unknown"},
    "segment": {"market", "market segment", "segment", "industry", "unknown"},
    "location": {"location", "region", "country", "geography", "unknown"},
    "region": {"location", "region", "country", "geography", "unknown"},
}


def _normalize_key(name: str) -> str:
    """Normalize an entity name for dedup grouping."""
    n = name.strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"^the\s+", "", n)
    for suffix in COMMON_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    if n in ABBREVIATION_MAP:
        n = ABBREVIATION_MAP[n]
    return n


def _types_compatible(type_a: str, type_b: str) -> bool:
    """Check if two entity types are compatible for merging."""
    a = type_a.strip().lower()
    b = type_b.strip().lower()
    if a == b or a == "unknown" or b == "unknown":
        return True
    return b in TYPE_COMPATIBILITY.get(a, {a})


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
# Resolution passes — each modifies the graph in-place and returns a count.
# ---------------------------------------------------------------------------

def pass_remove_generic(g: nx.DiGraph, generic_set: set[str] | None = None) -> int:
    """Remove generic/meaningless entities that act as noise hubs."""
    targets = generic_set if generic_set is not None else GENERIC_ENTITIES
    to_remove = [node for node in g.nodes() if node in targets]
    g.remove_nodes_from(to_remove)
    return len(to_remove)


def pass_normalize_and_merge(g: nx.DiGraph) -> int:
    """Group and merge entities that share the same normalized name."""
    groups: dict[str, list[str]] = defaultdict(list)
    for node in list(g.nodes()):
        groups[_normalize_key(node)].append(node)

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
    """Merge entities with high token overlap and compatible types."""
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
    remove_generic: bool = True,
    generic_set: set[str] | None = None,
    token_overlap_threshold: float = 0.75,
    prune_stranded_min_chunks: int = 2,
    prune_weak_max_degree: int = 1,
    prune_weak_min_desc: int = 10,
) -> dict[str, int]:
    """Run full entity resolution on a NetworkX graph **in-place**.

    This is the single entry point used by both the CLI script and the
    ingestion pipeline.  Every pass modifies *g* directly.

    Args:
        g: The NetworkX DiGraph to resolve.
        remove_generic: Whether to remove generic entities.
        generic_set: Custom set of generic entity keys to remove (uses default if None).
        token_overlap_threshold: Min token overlap score for merging.
        prune_stranded_min_chunks: Min source chunks to keep a degree-0 entity.
        prune_weak_max_degree: Max degree for a node to be considered "weak".
        prune_weak_min_desc: Min description length to keep a weak node.

    Returns:
        Dict with counts for each pass and totals.
    """
    original_nodes = g.number_of_nodes()
    original_edges = g.number_of_edges()
    logger.info("Entity resolution starting: %d entities, %d relationships", original_nodes, original_edges)

    stats: dict[str, int] = {}

    if remove_generic:
        stats["generic_removed"] = pass_remove_generic(g, generic_set)
        logger.info("Pass 1 (remove generic): removed %d -> %d remaining", stats["generic_removed"], g.number_of_nodes())

    stats["normalized_merged"] = pass_normalize_and_merge(g)
    logger.info("Pass 2 (normalization): merged %d -> %d remaining", stats["normalized_merged"], g.number_of_nodes())

    stats["overlap_merged"] = pass_token_overlap_merge(g, threshold=token_overlap_threshold)
    logger.info("Pass 3 (token overlap): merged %d -> %d remaining", stats["overlap_merged"], g.number_of_nodes())

    stats["stranded_pruned"] = pass_prune_stranded(g, min_chunks=prune_stranded_min_chunks)
    logger.info("Pass 4 (prune stranded): removed %d -> %d remaining", stats["stranded_pruned"], g.number_of_nodes())

    stats["weak_pruned"] = pass_prune_weak(g, max_degree=prune_weak_max_degree, min_desc_len=prune_weak_min_desc)
    logger.info("Pass 5 (prune weak): removed %d -> %d remaining", stats["weak_pruned"], g.number_of_nodes())

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

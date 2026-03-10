#!/usr/bin/env python3
"""CLI script to run entity resolution on existing knowledge graph pickle files.

Loads the NetworkX graph pickle, runs entity resolution (via the reusable
entity_resolver module), re-runs community detection, and saves back.
No LLM calls needed — takes minutes, not hours.

Usage:
    python scripts/resolve_graph_entities.py --data-dir /path/to/graph-data [--collection kg_rag]

For Docker volume data:
    docker cp <container>:/graph-data ./graph-data-backup
    python scripts/resolve_graph_entities.py --data-dir ./graph-data-backup
    docker cp ./graph-data-backup/. <container>:/graph-data/
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from collections import defaultdict

import networkx as nx

from nvidia_rag.utils.graph.entity_resolver import resolve_entities

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_community_detection(g: nx.DiGraph, resolution: float = 1.0) -> list:
    """Run Leiden/Louvain community detection and build CommunityInfo objects."""
    from nvidia_rag.utils.graph.graph_store import CommunityInfo

    if g.number_of_nodes() == 0:
        return []

    undirected = g.to_undirected()
    node_to_comm: dict[str, int] = {}

    try:
        from graspologic.partition import leiden
        partition = leiden(undirected, resolution=resolution)
        for node, comm_id in partition.items():
            node_to_comm[node] = comm_id
    except ImportError:
        logger.info("graspologic not available, using Louvain")
        communities = nx.community.louvain_communities(undirected, resolution=resolution, seed=42)
        for comm_id, members in enumerate(communities):
            for m in members:
                node_to_comm[m] = comm_id

    comm_members: dict[int, list[str]] = defaultdict(list)
    for node, cid in node_to_comm.items():
        comm_members[cid].append(node)

    results = []
    for comm_id, members in comm_members.items():
        entity_names = []
        desc_parts = []
        for key in members:
            if g.has_node(key):
                data = g.nodes[key]
                name = data.get("name", key)
                entity_names.append(name)
                etype = data.get("entity_type", "unknown")
                desc = data.get("description", "")
                if desc:
                    desc_parts.append(f"{name} ({etype}): {desc}")

        rel_parts = []
        member_set = set(members)
        for src, tgt, edata in g.edges(data=True):
            if src in member_set or tgt in member_set:
                src_name = g.nodes[src].get("name", src) if g.has_node(src) else src
                tgt_name = g.nodes[tgt].get("name", tgt) if g.has_node(tgt) else tgt
                rel_type = edata.get("relation_type", "related_to")
                rel_parts.append(f"{src_name} {rel_type} {tgt_name}")

        summary_parts = []
        if desc_parts:
            summary_parts.append("Key entities: " + "; ".join(desc_parts[:10]))
        if rel_parts:
            summary_parts.append("Key relationships: " + "; ".join(rel_parts[:10]))
        if not summary_parts:
            summary_parts.append(f"Community of {len(members)} entities: {', '.join(entity_names[:8])}")

        results.append(CommunityInfo(
            community_id=comm_id,
            entity_names=entity_names,
            summary=". ".join(summary_parts),
            level=0,
        ))

    return results


def resolve(data_dir: str, collection: str | None = None, resolution: float = 1.0) -> None:
    """Run entity resolution + community re-detection on persisted graph data."""

    graph_files = [f for f in os.listdir(data_dir) if f.endswith("_graph.pkl")]
    if not graph_files:
        logger.error("No graph pickle files found in %s", data_dir)
        return

    for graph_file in graph_files:
        coll_name = graph_file.replace("_graph.pkl", "")
        if collection and coll_name != collection:
            continue

        graph_path = os.path.join(data_dir, graph_file)
        communities_path = os.path.join(data_dir, f"{coll_name}_communities.pkl")

        logger.info("=" * 70)
        logger.info("Processing collection: %s", coll_name)
        logger.info("=" * 70)

        with open(graph_path, "rb") as f:
            g: nx.DiGraph = pickle.load(f)  # noqa: S301

        stats = resolve_entities(g)

        logger.info("Running community detection...")
        communities = run_community_detection(g, resolution=resolution)
        logger.info("Detected %d communities", len(communities))

        # Backup originals (only on first run)
        backup_graph = graph_path + ".bak"
        backup_comm = communities_path + ".bak"
        if not os.path.exists(backup_graph):
            os.rename(graph_path, backup_graph)
            logger.info("Backed up original graph to %s", backup_graph)
        if os.path.exists(communities_path) and not os.path.exists(backup_comm):
            os.rename(communities_path, backup_comm)
            logger.info("Backed up original communities to %s", backup_comm)

        with open(graph_path, "wb") as f:
            pickle.dump(g, f)
        logger.info("Saved resolved graph to %s", graph_path)

        with open(communities_path, "wb") as f:
            pickle.dump(communities, f)
        logger.info("Saved communities to %s", communities_path)

        # Print top entities by degree for verification
        degrees = [(n, g.in_degree(n) + g.out_degree(n)) for n in g.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        logger.info("Top 15 entities by degree after resolution:")
        for name, deg in degrees[:15]:
            node_data = g.nodes[name]
            logger.info("  %4d  %s (%s)", deg, node_data.get("name", name), node_data.get("entity_type", "?"))

        logger.info("Resolution stats: %s", stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve (deduplicate) entities in a knowledge graph pickle")
    parser.add_argument("--data-dir", required=True, help="Directory containing *_graph.pkl files")
    parser.add_argument("--collection", default=None, help="Process only this collection (default: all)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Community detection resolution (default: 1.0)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logger.error("Directory not found: %s", args.data_dir)
        return

    resolve(args.data_dir, collection=args.collection, resolution=args.resolution)


if __name__ == "__main__":
    main()

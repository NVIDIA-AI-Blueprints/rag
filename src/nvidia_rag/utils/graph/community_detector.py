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

"""Community detection and summarization for GraphRAG.

Uses the Leiden algorithm (via graspologic) to partition the knowledge graph into
hierarchical communities, then generates LLM summaries for each community.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.graph.graph_store import CommunityInfo, GraphStore
from nvidia_rag.utils.graph.networkx_store import NetworkXGraphStore
from nvidia_rag.utils.llm import get_llm

logger = logging.getLogger(__name__)

def _build_community_prompt(prompts: dict | None) -> ChatPromptTemplate:
    """Build community summary prompt from prompt.yaml, same as other components."""
    if prompts:
        prompt_config = prompts.get("graph_community_summary_prompt", {})
        system_prompt = prompt_config.get("system", "")
        human_prompt = prompt_config.get("human", prompt_config.get("user", ""))
        if system_prompt or human_prompt:
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt),
            ])

    return ChatPromptTemplate.from_messages([
        ("system", "/no_think\nYou are an expert at summarizing groups of related entities and their relationships from a knowledge graph."),
        ("human", "Summarize the following group of related entities and relationships in 2-3 concise sentences.\n\nEntities: {entities}\nRelationships: {relationships}\n\nSummary:"),
    ])


def _run_leiden(
    graph: nx.DiGraph, resolution: float = 1.0
) -> dict[str, int]:
    """Run the Leiden community detection algorithm on the graph.

    Falls back to Louvain if graspologic is not available.

    Returns:
        Dict mapping node key -> community_id
    """
    if graph.number_of_nodes() == 0:
        return {}

    undirected = graph.to_undirected()

    try:
        from graspologic.partition import leiden

        partition = leiden(undirected, resolution=resolution)
        node_to_community = {}
        for node, comm_id in partition.items():
            node_to_community[node] = comm_id
        return node_to_community
    except ImportError:
        logger.info("graspologic not available, falling back to Louvain via networkx")

    try:
        communities = nx.community.louvain_communities(
            undirected, resolution=resolution, seed=42
        )
        node_to_community = {}
        for comm_id, members in enumerate(communities):
            for member in members:
                node_to_community[member] = comm_id
        return node_to_community
    except Exception:
        logger.warning("Community detection failed", exc_info=True)
        return {node: 0 for node in graph.nodes}


async def detect_communities_and_summarize(
    graph_store: GraphStore,
    collection_name: str,
    config: NvidiaRAGConfig | None = None,
    prompts: dict | None = None,
) -> list[CommunityInfo]:
    """Run community detection on the graph and generate summaries.

    Args:
        graph_store: The graph store containing entities/relationships.
        collection_name: Collection to process.
        config: NvidiaRAGConfig instance.
        prompts: Prompts dict loaded from prompt.yaml via get_prompts().

    Returns:
        List of CommunityInfo objects with summaries.
    """
    if config is None:
        config = NvidiaRAGConfig()

    graph_cfg = config.graph_rag

    if isinstance(graph_store, NetworkXGraphStore):
        nx_graph = graph_store.get_networkx_graph(collection_name)
    else:
        nx_graph = _build_nx_from_store(graph_store, collection_name)

    if nx_graph.number_of_nodes() == 0:
        logger.info("Graph is empty for '%s', skipping community detection", collection_name)
        return []

    node_to_community = _run_leiden(nx_graph, resolution=graph_cfg.community_resolution)

    community_members: dict[int, list[str]] = {}
    for node_key, comm_id in node_to_community.items():
        community_members.setdefault(comm_id, []).append(node_key)

    logger.info(
        "Detected %d communities in '%s' (%d nodes)",
        len(community_members),
        collection_name,
        nx_graph.number_of_nodes(),
    )

    llm_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 1024,
    }
    if graph_cfg.entity_extraction_model:
        llm_kwargs["model"] = graph_cfg.entity_extraction_model
    if graph_cfg.entity_extraction_server_url:
        llm_kwargs["llm_endpoint"] = graph_cfg.entity_extraction_server_url
    api_key = graph_cfg.get_api_key()
    if api_key:
        llm_kwargs["api_key"] = api_key

    llm = get_llm(config=config, **llm_kwargs)

    prompt = _build_community_prompt(prompts)
    chain = prompt | llm | StrOutputParser()

    communities: list[CommunityInfo] = []

    for comm_id, member_keys in community_members.items():
        entity_names = []
        entities_text_parts = []
        for key in member_keys:
            if nx_graph.has_node(key):
                data = nx_graph.nodes[key]
                name = data.get("name", key)
                entity_names.append(name)
                etype = data.get("entity_type", "unknown")
                desc = data.get("description", "")
                line = f"- {name} ({etype})"
                if desc:
                    line += f": {desc}"
                entities_text_parts.append(line)

        relationships_text_parts = []
        for src, tgt, data in nx_graph.edges(data=True):
            if src in member_keys or tgt in member_keys:
                src_name = nx_graph.nodes[src].get("name", src)
                tgt_name = nx_graph.nodes[tgt].get("name", tgt)
                rel_type = data.get("relation_type", "related_to")
                desc = data.get("description", "")
                line = f"- {src_name} --[{rel_type}]--> {tgt_name}"
                if desc:
                    line += f": {desc}"
                relationships_text_parts.append(line)

        entities_text = "\n".join(entities_text_parts) or "No entities."
        relationships_text = "\n".join(relationships_text_parts[:50]) or "No relationships."

        summary = ""
        if len(member_keys) >= 2:
            try:
                summary = await chain.ainvoke(
                    {
                        "entities": entities_text,
                        "relationships": relationships_text,
                    },
                    config={"run_name": "community-summarization"},
                )
                summary = summary.strip()
            except Exception:
                logger.warning(
                    "Community summary generation failed for community %d",
                    comm_id,
                    exc_info=True,
                )
                summary = f"Community of {len(member_keys)} entities: {', '.join(entity_names[:5])}"
        else:
            summary = f"Single-entity community: {', '.join(entity_names)}"

        communities.append(CommunityInfo(
            community_id=comm_id,
            entity_names=entity_names,
            summary=summary,
            level=0,
        ))

    graph_store.set_communities(communities, collection_name)
    graph_store.persist()

    logger.info(
        "Community detection and summarization complete for '%s': %d communities",
        collection_name,
        len(communities),
    )
    return communities


def _build_nx_from_store(graph_store: GraphStore, collection_name: str) -> nx.DiGraph:
    """Build a NetworkX graph from any GraphStore backend (for non-NetworkX stores)."""
    g = nx.DiGraph()
    entities = graph_store.get_all_entities(collection_name)
    for entity in entities:
        g.add_node(
            entity.key,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            source_chunk_ids=entity.source_chunk_ids,
        )
    relationships = graph_store.get_all_relationships(collection_name)
    for rel in relationships:
        g.add_edge(
            rel.source.strip().lower(),
            rel.target.strip().lower(),
            relation_type=rel.relation_type,
            description=rel.description,
            weight=rel.weight,
            source_chunk_ids=rel.source_chunk_ids,
        )
    return g

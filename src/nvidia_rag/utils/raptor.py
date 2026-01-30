# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Module

This module provides hierarchical summary tree generation for RAG accuracy improvement.

Key Components:
1. RAPTORConfig: Configuration for tree building
2. RAPTORTreeBuilder: Core tree building logic
3. RAPTORRetriever: Tree-aware retrieval with traversal

Integration Points:
- Ingestor Server: Builds trees after NV-Ingest completes
- RAG Server: Enhanced retrieval with tree traversal
"""

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.summarization import (
    _get_tokenizer,
    _split_text_into_chunks,
    _token_length,
    acquire_global_summary_slot,
    release_global_summary_slot,
    get_summarization_semaphore,
    _create_llm_chains,
)

logger = logging.getLogger(__name__)


@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR tree building and retrieval."""
    min_cluster_size: int = 3
    batch_size: int = 50

    @classmethod
    def from_config(cls, config: Any) -> "RAPTORConfig":
        """Create RAPTOR config from NvidiaRAGConfig."""
        return cls()

    @staticmethod
    def calculate_cluster_size(num_chunks: int) -> int:
        """Calculate cluster size: sqrt(num_chunks) bounded to [5, 15]."""
        cluster_size = int(math.ceil(math.sqrt(num_chunks)))
        return max(5, min(15, cluster_size))

    @staticmethod
    def calculate_max_levels(num_chunks: int, cluster_size: int) -> int:
        """Calculate tree depth: log_cluster_size(num_chunks) bounded to [2, 5]."""
        if num_chunks <= cluster_size:
            return 1
        levels = int(math.ceil(math.log(num_chunks) / math.log(cluster_size)))
        return max(2, min(5, levels))


@dataclass
class RAPTORNode:
    """Represents a node in the RAPTOR tree."""
    content: str
    level: int
    document_id: str
    node_id: str
    embedding: Optional[List[float]] = None
    chunk_ids: List[str] = field(default_factory=list)
    summary_ids: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAPTORTreeBuilder:
    """Builds RAPTOR trees from ingested chunks."""

    def __init__(self, rag_config: Any, raptor_config: RAPTORConfig):
        self.rag_config = rag_config
        self.raptor_config = raptor_config
        self.llm = None
        self.embedding_model = None
        self.prompts = None
        self.initial_chain = None
        self.iterative_chain = None
        self.stats = {
            "documents_processed": 0,
            "summaries_created": 0,
            "time_summarization": 0.0,
            "time_embedding": 0.0,
        }

    async def initialize(self):
        """Initialize LLM and embedding models."""
        if self.llm is None:
            llm_params = {
                "model": self.rag_config.summarizer.model_name,
                "temperature": self.rag_config.summarizer.temperature,
                "top_p": self.rag_config.summarizer.top_p,
                "max_tokens": self.rag_config.summarizer.max_chunk_length,
                "api_key": self.rag_config.summarizer.get_api_key(),
            }

            if self.rag_config.summarizer.server_url:
                llm_params["llm_endpoint"] = self.rag_config.summarizer.server_url

            self.llm = get_llm(config=self.rag_config, **llm_params)
            self.prompts = get_prompts()
            
            # Reuse existing prompts: document_summary_prompt and iterative_summary_prompt
            self.initial_chain, self.iterative_chain = _create_llm_chains(
                self.llm, self.prompts, is_shallow=False
            )

        if self.embedding_model is None:
            self.embedding_model = get_embedding_model(
                model=self.rag_config.embeddings.model_name,
                url=self.rag_config.embeddings.server_url,
                config=self.rag_config,
            )

    async def build_trees_from_results(
        self, results: List[List[Dict[str, Any]]], collection_name: str, vdb_op: Any
    ) -> None:
        """Build RAPTOR trees from NV-Ingest results and upload to VDB."""
        if not results:
            return

        await self.initialize()
        semaphore = get_summarization_semaphore()
        documents_by_id = self._group_chunks_by_document(results)
        
        logger.info(f"RAPTOR: Processing {len(documents_by_id)} documents in collection '{collection_name}'")

        all_summary_nodes = []
        for doc_id, chunks in documents_by_id.items():
            slot_acquired = False
            try:
                if len(chunks) < self.raptor_config.min_cluster_size:
                    logger.info(f"RAPTOR: Skipping {doc_id} - only {len(chunks)} chunks")
                    continue

                async with semaphore:
                    while not await acquire_global_summary_slot(self.rag_config):
                        await asyncio.sleep(0.5)
                    
                    slot_acquired = True
                    summary_nodes = await self._build_tree_for_document(doc_id, chunks)

                    if summary_nodes:
                        summary_nodes = await self._embed_summaries(summary_nodes)
                        all_summary_nodes.extend(summary_nodes)
                        logger.info(f"RAPTOR: ✓ {doc_id} → {len(summary_nodes)} summary nodes")

                    self.stats["documents_processed"] += 1

            except Exception as e:
                logger.error(f"RAPTOR: ✗ {doc_id} failed: {e}")
                raise
            finally:
                if slot_acquired:
                    await release_global_summary_slot()

        if all_summary_nodes:
            try:
                records = self._format_as_vdb_records(all_summary_nodes, collection_name)
                vdb_op.run(records)
                logger.info(f"RAPTOR: Uploaded {len(records)} summary nodes to VDB")
            except Exception as e:
                logger.error(f"RAPTOR: VDB upload failed: {e}")
                raise

    def _group_chunks_by_document(
        self, results: List[List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by document ID"""
        documents_by_id = defaultdict(list)

        for result in results:
            if not result:
                continue

            # Get document ID from first chunk's metadata
            first_chunk = result[0]
            metadata = first_chunk.get("metadata", {})
            source_metadata = metadata.get("source_metadata", {})
            doc_id = source_metadata.get("source_id", "unknown")

            documents_by_id[doc_id].extend(result)

        return dict(documents_by_id)

    async def _build_tree_for_document(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> List[RAPTORNode]:
        """Build RAPTOR tree for a single document."""
        num_chunks = len(chunks)
        cluster_size = self.raptor_config.calculate_cluster_size(num_chunks)
        max_levels = self.raptor_config.calculate_max_levels(num_chunks, cluster_size)

        level_0_nodes = []
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("source_metadata", {}).get("chunk_id")
            if not chunk_id:
                chunk_id = f"{document_id}_chunk_{idx}"

            content = chunk.get("metadata", {}).get("content", "")
            if not content:
                continue

            node = RAPTORNode(
                content=content,
                level=0,
                document_id=document_id,
                node_id=chunk_id,
                chunk_ids=[chunk_id],
                metadata=chunk.get("metadata", {}).get("content_metadata", {}),
            )
            level_0_nodes.append(node)

        summary_nodes = await self._build_tree_recursive(
            nodes=level_0_nodes,
            document_id=document_id,
            current_level=1,
            cluster_size=cluster_size,
            max_levels=max_levels,
        )

        return summary_nodes

    async def _build_tree_recursive(
        self,
        nodes: List[RAPTORNode],
        document_id: str,
        current_level: int,
        cluster_size: int,
        max_levels: int,
    ) -> List[RAPTORNode]:
        """Recursively cluster and summarize nodes."""
        if current_level > max_levels:
            return []

        if len(nodes) <= cluster_size:
            summary_node = await self._generate_single_cluster_summary(
                cluster=nodes,
                cluster_idx=0,
                document_id=document_id,
                level=current_level
            )
            return [summary_node]

        clusters = self._cluster_nodes(nodes, cluster_size)
        if not clusters:
            return []

        start_time = time.time()
        summary_nodes = await self._generate_cluster_summaries(
            clusters, document_id, current_level
        )
        self.stats["time_summarization"] += time.time() - start_time
        self.stats["summaries_created"] += len(summary_nodes)

        higher_level_summaries = await self._build_tree_recursive(
            nodes=summary_nodes,
            document_id=document_id,
            current_level=current_level + 1,
            cluster_size=cluster_size,
            max_levels=max_levels,
        )

        return summary_nodes + higher_level_summaries

    def _cluster_nodes(
        self, nodes: List[RAPTORNode], cluster_size: int
    ) -> List[List[RAPTORNode]]:
        """Cluster nodes sequentially to preserve document order."""
        clusters = []
        for i in range(0, len(nodes), cluster_size):
            cluster = nodes[i : i + cluster_size]
            if len(cluster) >= self.raptor_config.min_cluster_size:
                clusters.append(cluster)
            elif clusters:
                clusters[-1].extend(cluster)
        return clusters

    async def _generate_cluster_summaries(
        self, clusters: List[List[RAPTORNode]], document_id: str, level: int
    ) -> List[RAPTORNode]:
        """Generate summaries for all clusters in parallel."""
        tasks = [
            self._generate_single_cluster_summary(cluster, cluster_idx, document_id, level)
            for cluster_idx, cluster in enumerate(clusters)
        ]
        
        summary_nodes = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for idx, result in enumerate(summary_nodes):
            if isinstance(result, Exception):
                logger.error(f"Failed to summarize cluster {idx} at level {level}: {result}")
                raise result
            results.append(result)
        
        return results

    async def _generate_single_cluster_summary(
        self,
        cluster: List[RAPTORNode],
        cluster_idx: int,
        document_id: str,
        level: int
    ) -> RAPTORNode:
        """Generate summary for a single cluster using iterative enrichment if needed."""
        tokenizer = _get_tokenizer(self.rag_config)
        texts = [node.content for node in cluster]
        combined_text = "\n\n".join(texts)
        
        PROMPT_OVERHEAD = 200
        max_safe_input_tokens = self.rag_config.summarizer.max_chunk_length - PROMPT_OVERHEAD
        current_tokens = _token_length(combined_text, self.rag_config)
        
        if current_tokens <= max_safe_input_tokens:
            summary = await self._generate_initial_summary(combined_text)
        else:
            text_chunks = _split_text_into_chunks(
                combined_text, tokenizer, max_safe_input_tokens,
                self.rag_config.summarizer.chunk_overlap
            )
            
            summary = await self._generate_initial_summary(text_chunks[0])
            for chunk in text_chunks[1:]:
                summary = await self._generate_enriched_summary(summary, chunk)
        
        if level == 1:
            chunk_ids = [node.node_id for node in cluster]
            summary_ids = []
        else:
            summary_ids = [node.node_id for node in cluster]
            chunk_ids = []
            for node in cluster:
                chunk_ids.extend(node.chunk_ids)
        
        node_id = f"{document_id}_L{level}_C{cluster_idx}"
        summary_node = RAPTORNode(
            content=summary,
            level=level,
            document_id=document_id,
            node_id=node_id,
            chunk_ids=chunk_ids,
            summary_ids=summary_ids,
            cluster_id=node_id,
            metadata={
                "cluster_size": len(cluster),
                "num_chunks_covered": len(chunk_ids),
            },
        )
        
        return summary_node
    
    async def _generate_initial_summary(self, text: str) -> str:
        """Generate initial summary using document_summary_prompt."""
        try:
            summary = await self.initial_chain.ainvoke(
                {"document_text": text},
                config={"run_name": "raptor-initial-summary"}
            )
            if not summary or not summary.strip():
                raise ValueError("LLM returned empty summary")
            return summary.strip()
        except asyncio.TimeoutError:
            logger.error("LLM timeout for initial summary")
            raise
        except Exception as e:
            logger.error(f"Initial summary failed: {type(e).__name__}: {e}")
            raise
    
    async def _generate_enriched_summary(self, previous_summary: str, new_chunk: str) -> str:
        """Generate enriched summary using iterative_summary_prompt."""
        try:
            summary = await self.iterative_chain.ainvoke(
                {"previous_summary": previous_summary, "new_chunk": new_chunk},
                config={"run_name": "raptor-enriched-summary"}
            )
            if not summary or not summary.strip():
                raise ValueError("LLM returned empty summary")
            return summary.strip()
        except asyncio.TimeoutError:
            logger.error("LLM timeout for enrichment")
            raise
        except Exception as e:
            logger.error(f"Enrichment failed: {type(e).__name__}: {e}")
            raise

    async def _embed_summaries(self, summary_nodes: List[RAPTORNode]) -> List[RAPTORNode]:
        """Embed all summary nodes in batches."""
        start_time = time.time()
        texts = [node.content for node in summary_nodes]

        all_embeddings = []
        batch_size = self.raptor_config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                batch_embeddings = await self.embedding_model.aembed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"RAPTOR: Embedding batch failed: {e}")
                all_embeddings.extend(
                    [[0.0] * self.rag_config.embeddings.dimensions] * len(batch_texts)
                )

        for node, embedding in zip(summary_nodes, all_embeddings):
            node.embedding = embedding

        self.stats["time_embedding"] += time.time() - start_time
        return summary_nodes

    def _format_as_vdb_records(
        self, summary_nodes: List[RAPTORNode], collection_name: str
    ) -> List[Dict[str, Any]]:
        """Format summary nodes as VDB records."""
        records = []
        for node in summary_nodes:
            record = {
                "text": node.content,
                "vector": node.embedding,
                "document_type": "text",
                "source": {
                    "source_id": node.document_id,
                    "source_name": node.document_id,
                    "collection_name": collection_name,
                    "source_type": "raptor_summary",
                },
                "content_metadata": {
                    "type": "summary",
                    "level": node.level,
                    "document_id": node.document_id,
                    "cluster_id": node.cluster_id,
                    "node_id": node.node_id,
                    "chunk_ids": node.chunk_ids,
                    "summary_ids": node.summary_ids,
                    "num_children": len(node.chunk_ids),
                    **node.metadata,
                },
            }
            records.append(record)
        return records


class RAPTORRetriever:
    """Enhanced retrieval with RAPTOR tree traversal."""

    def __init__(self, raptor_config: RAPTORConfig):
        self.raptor_config = raptor_config

    async def expand_with_tree_traversal(
        self, documents: List[Any], vdb_op: Any
    ) -> List[Any]:
        """Organize documents using RAPTOR tree structure by grouping summaries with their children."""
        doc_by_id = {}
        for doc in documents:
            doc_id = self._get_document_id(doc)
            if doc_id not in doc_by_id:
                doc_by_id[doc_id] = []
            doc_by_id[doc_id].append(doc)

        final_docs = []
        seen_docs = set()
        
        num_summaries = sum(1 for d in documents if self._is_summary_document(d))
        num_chunks = len(documents) - num_summaries
        total_children_added = 0

        for doc in documents:
            doc_id = self._get_document_id(doc)
            doc_obj_id = id(doc)

            if doc_obj_id in seen_docs:
                continue

            final_docs.append(doc)
            seen_docs.add(doc_obj_id)

            if self._is_summary_document(doc):
                chunk_ids = self._get_chunk_ids(doc)
                for chunk_id in chunk_ids:
                    if chunk_id in doc_by_id:
                        for child_doc in doc_by_id[chunk_id]:
                            child_obj_id = id(child_doc)
                            if child_obj_id not in seen_docs:
                                final_docs.append(child_doc)
                                seen_docs.add(child_obj_id)
                                total_children_added += 1

        added_docs = len(final_docs) - len(documents)
        logger.debug(
            f"RAPTOR tree traversal: {num_summaries} summaries, {num_chunks} chunks → "
            f"added {added_docs} children from tree"
        )
        return final_docs

    def _get_document_id(self, doc: Any) -> str:
        """Extract unique document/chunk ID from LangChain Document."""
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            source = doc.metadata.get("source", {})

            if content_metadata.get("type") == "summary":
                node_id = content_metadata.get("node_id")
                if node_id:
                    return str(node_id)
                cluster_id = content_metadata.get("cluster_id")
                if cluster_id:
                    return str(cluster_id)

            if isinstance(source, dict) and isinstance(content_metadata, dict):
                source_id = source.get("source_id")
                page_number = content_metadata.get("page_number")

                if source_id and page_number is not None:
                    chunk_index = page_number - 1
                    return f"{source_id}_chunk_{chunk_index}"

        return str(id(doc))

    def _is_summary_document(self, doc: Any) -> bool:
        """Check if document is a RAPTOR summary."""
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            return content_metadata.get("type") == "summary"
        return False

    def _get_chunk_ids(self, doc: Any) -> List[str]:
        """Extract chunk IDs from summary metadata."""
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            return content_metadata.get("chunk_ids", [])
        return []

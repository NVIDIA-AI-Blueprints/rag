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
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign

from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.summarization import (
    _extract_content_from_element,
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
    summary_top_k: int = 10  # Number of top summaries for Stage 1 document identification

    @classmethod
    def from_config(cls, config: Any) -> "RAPTORConfig":
        """Create RAPTOR config from NvidiaRAGConfig."""
        # Load summary_top_k from environment variable with default of 10
        summary_top_k = int(os.getenv("RAPTOR_SUMMARY_TOP_K", "10"))
        return cls(summary_top_k=summary_top_k)

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
        self.embedding_model = None
        self.prompts = None
        self.stats = {
            "documents_processed": 0,
            "summaries_created": 0,
            "time_summarization": 0.0,
            "time_embedding": 0.0,
        }

    async def initialize(self):
        """Initialize embedding model and load prompts."""
        if self.prompts is None:
            self.prompts = get_prompts()

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
                
                # Batch VDB inserts to avoid rate limiting
                batch_size = 1000  # Insert 100 records at a time
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]
                    vdb_op.run(batch)
                    logger.debug(f"RAPTOR: Uploaded batch {i//batch_size + 1} ({len(batch)} records)")
                
                logger.info(f"RAPTOR: ✓ Uploaded {len(records)} summary nodes to VDB")
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
        """Build RAPTOR tree for a single document with dedicated LLM instance."""
        # Create fresh LLM and chains for this document (like hierarchical strategy)
        llm_params = {
            "config": self.rag_config,
            "model": self.rag_config.summarizer.model_name,
            "temperature": self.rag_config.summarizer.temperature,
            "top_p": self.rag_config.summarizer.top_p,
            "api_key": self.rag_config.summarizer.get_api_key(),
        }
        if self.rag_config.summarizer.server_url:
            llm_params["llm_endpoint"] = self.rag_config.summarizer.server_url
        
        doc_llm = get_llm(**llm_params)
        
        # Create RAPTOR-specific chains using specialized prompts for cluster summarization
        raptor_cluster_prompt_config = self.prompts.get("raptor_cluster_summary_prompt")
        raptor_enrichment_prompt_config = self.prompts.get("raptor_enrichment_prompt")
        
        cluster_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", raptor_cluster_prompt_config["system"]),
            ("human", raptor_cluster_prompt_config["human"]),
        ])
        
        enrichment_prompt = ChatPromptTemplate.from_messages([
            ("system", raptor_enrichment_prompt_config["system"]),
            ("human", raptor_enrichment_prompt_config["human"]),
        ])
        
        doc_initial_chain = cluster_summary_prompt | doc_llm | StrOutputParser()
        doc_iterative_chain = enrichment_prompt | doc_llm | StrOutputParser()
        
        num_chunks = len(chunks)
        cluster_size = self.raptor_config.calculate_cluster_size(num_chunks)
        max_levels = self.raptor_config.calculate_max_levels(num_chunks, cluster_size)

        level_0_nodes = []
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("source_metadata", {}).get("chunk_id")
            if not chunk_id:
                chunk_id = f"{document_id}_chunk_{idx}"

            content = _extract_content_from_element(chunk, self.rag_config)
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
            doc_initial_chain=doc_initial_chain,
            doc_iterative_chain=doc_iterative_chain,
        )

        return summary_nodes

    async def _build_tree_recursive(
        self,
        nodes: List[RAPTORNode],
        document_id: str,
        current_level: int,
        cluster_size: int,
        max_levels: int,
        doc_initial_chain,
        doc_iterative_chain,
    ) -> List[RAPTORNode]:
        """Recursively cluster and summarize nodes."""
        if current_level > max_levels:
            return []

        if len(nodes) <= cluster_size:
            summary_node = await self._generate_single_cluster_summary(
                cluster=nodes,
                cluster_idx=0,
                document_id=document_id,
                level=current_level,
                doc_initial_chain=doc_initial_chain,
                doc_iterative_chain=doc_iterative_chain,
            )
            return [summary_node]

        clusters = self._cluster_nodes(nodes, cluster_size)
        if not clusters:
            return []

        start_time = time.time()
        summary_nodes = await self._generate_cluster_summaries(
            clusters, document_id, current_level, doc_initial_chain, doc_iterative_chain
        )
        self.stats["time_summarization"] += time.time() - start_time
        self.stats["summaries_created"] += len(summary_nodes)

        higher_level_summaries = await self._build_tree_recursive(
            nodes=summary_nodes,
            document_id=document_id,
            current_level=current_level + 1,
            cluster_size=cluster_size,
            max_levels=max_levels,
            doc_initial_chain=doc_initial_chain,
            doc_iterative_chain=doc_iterative_chain,
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
        self, clusters: List[List[RAPTORNode]], document_id: str, level: int,
        doc_initial_chain, doc_iterative_chain
    ) -> List[RAPTORNode]:
        """Generate summaries for all clusters with controlled concurrency."""
        MAX_CONCURRENT_CLUSTERS = 3
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLUSTERS)
        
        async def _summarize_with_limit(cluster, cluster_idx):
            async with semaphore:
                return await self._generate_single_cluster_summary(
                    cluster, cluster_idx, document_id, level, doc_initial_chain, doc_iterative_chain
                )
        
        tasks = [
            _summarize_with_limit(cluster, cluster_idx)
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
        level: int,
        doc_initial_chain,
        doc_iterative_chain,
    ) -> RAPTORNode:
        """Generate summary for a single cluster using iterative enrichment if needed."""
        tokenizer = _get_tokenizer(self.rag_config)
        texts = [node.content for node in cluster]
        combined_text = "\n\n".join(texts)

        current_tokens = _token_length(combined_text, self.rag_config)
        
        if current_tokens <= self.rag_config.summarizer.max_chunk_length:
            summary = await self._generate_initial_summary(combined_text, doc_initial_chain)
        else:
            text_chunks = _split_text_into_chunks(
                combined_text, tokenizer, self.rag_config.summarizer.max_chunk_length,
                self.rag_config.summarizer.chunk_overlap
            )
            
            summary = await self._generate_initial_summary(text_chunks[0], doc_initial_chain)
            for chunk in text_chunks[1:]:
                summary = await self._generate_enriched_summary(summary, chunk, doc_iterative_chain)
        
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
    
    async def _generate_initial_summary(self, text: str, chain) -> str:
        """Generate initial summary using document_summary_prompt."""
        try:
            summary = await chain.ainvoke(
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
    
    async def _generate_enriched_summary(self, previous_summary: str, new_chunk: str, chain) -> str:
        """Generate enriched summary using iterative_summary_prompt."""
        try:
            summary = await chain.ainvoke(
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
        """Format summary nodes as VDB records.
        
        Stores chunk indices instead of full chunk_id strings to reduce JSON size.
        Example: ["doc_chunk_0", ..., "doc_chunk_130"] → [0, ..., 130]
        """
        records = []
        for node in summary_nodes:
            # Extract indices from chunk_ids to save space
            chunk_indices = self._extract_chunk_indices(node.chunk_ids, node.document_id)
            
            content_metadata = {
                "type": "summary",
                "level": node.level,
                "document_id": node.document_id,
                "cluster_id": node.cluster_id,
                "node_id": node.node_id,
                "num_children": len(node.chunk_ids),
                **node.metadata,
            }
            
            # Store indices (much smaller) or fallback to full chunk_ids
            if chunk_indices is not None:
                content_metadata["chunk_indices"] = chunk_indices
            else:
                content_metadata["chunk_ids"] = node.chunk_ids
            
            # Store summary_ids
            if node.summary_ids:
                content_metadata["summary_ids"] = node.summary_ids
            
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
                "content_metadata": content_metadata,
            }
            records.append(record)
        return records
    
    def _extract_chunk_indices(
        self, chunk_ids: List[str], document_id: str
    ) -> Optional[List[int]]:
        """Extract indices from chunk_ids following pattern: {document_id}_chunk_{index}"""
        if not chunk_ids:
            return []
        
        indices = []
        expected_prefix = f"{document_id}_chunk_"
        
        for chunk_id in chunk_ids:
            if not chunk_id.startswith(expected_prefix):
                return None  # Non-standard format
            
            suffix = chunk_id[len(expected_prefix):]
            if not suffix.isdigit():
                return None  # Non-numeric suffix
            
            indices.append(int(suffix))
        
        return indices


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
    
    async def two_stage_retrieval(
        self,
        query: str,
        vdb_op: Any,
        reranker_model: Any,
        vdb_top_k: int = 100,
        reranker_top_k: int = 25,
        collection_names: List[str] = None,
        filter_expr: str = "",
        exclude_summaries_from_context: bool = False,
    ) -> List[Any]:
        """
        Two-stage RAPTOR retrieval for improved accuracy.
        
        Stage 1: Summary-level retrieval
        - Retrieve top summaries (all levels) from VDB
        - Rerank summaries to get top N (configurable via RAPTOR_SUMMARY_TOP_K)
        - Identify relevant documents from top summaries
        
        Stage 2: Document-focused retrieval
        - Retrieve content from identified documents
        - Option 1: Summaries + Chunks (hierarchical context)
        - Option 2: Chunks only (exclude summaries via VDB filter)
        - Rerank to get final top K items
        
        Args:
            query: User query
            vdb_op: VDB operator for search
            reranker_model: Reranker model for scoring
            vdb_top_k: Initial VDB retrieval size (default: 100)
            reranker_top_k: Final number of items to return (default: 25)
            collection_names: Collections to search
            filter_expr: Additional filter expression
            exclude_summaries_from_context: If True, add VDB filter to exclude summaries 
                                           from Stage 2, returning only original chunks (default: False)
        
        Returns:
            List of top reranked documents from relevant documents
        """
        logger.info("=" * 80)
        logger.info("RAPTOR Two-Stage Retrieval Started")
        logger.info("=" * 80)
        
        # ============================================================
        # STAGE 1: Summary-level retrieval to identify documents
        # ============================================================
        logger.info("STAGE 1: Summary-Level Document Identification")
        logger.info(f"  - Retrieving top {vdb_top_k} summaries from VDB")
        logger.info(f"  - Will select top {self.raptor_config.summary_top_k} after reranking")
        
        # Build filter for summaries only
        summary_filter = 'content_metadata["type"] == "summary"'
        if filter_expr:
            summary_filter = f"({summary_filter}) and ({filter_expr})"
        
        # Retrieve summaries from VDB (use same pattern as normal flow)
        collection_name = collection_names[0] if collection_names else ""
        vectorstore = vdb_op.get_langchain_vectorstore(collection_name)
        
        summary_results = vdb_op.retrieval_langchain(
            query=query,
            collection_name=collection_name,
            vectorstore=vectorstore,
            top_k=vdb_top_k,
            filter_expr=summary_filter,
        )
        
        logger.info(f"  ✓ Retrieved {len(summary_results)} summaries from VDB")
        
        if not summary_results:
            logger.warning("  ⚠ No summaries found, returning empty results")
            return []
        
        # Rerank summaries using same pattern as normal flow
        logger.info(f"  - Reranking summaries...")
        
        # Set top_n for Stage 1 reranking
        original_top_n = reranker_model.top_n
        reranker_model.top_n = self.raptor_config.summary_top_k
        
        context_reranker = RunnableAssign(
            {
                "context": lambda input: reranker_model.compress_documents(
                    query=input["question"], documents=input["context"]
                )
            }
        )
        
        reranked_result = await context_reranker.ainvoke(
            {"context": summary_results, "question": query},
            config={"run_name": "raptor_stage1_reranker"},
        )
        reranked_summaries = reranked_result.get("context", [])
        
        logger.info(f"  ✓ Reranked to top {len(reranked_summaries)} summaries")
        
        # Extract unique document IDs from top summaries
        document_ids = self._extract_document_ids_from_summaries(reranked_summaries)
        
        logger.info(f"  ✓ Identified {len(document_ids)} relevant documents:")
        for doc_id in document_ids:
            logger.info(f"    - {doc_id}")
        
        if not document_ids:
            logger.warning("  ⚠ No documents identified from summaries")
            # Restore original top_n
            reranker_model.top_n = original_top_n
            return []
        
        # ============================================================
        # STAGE 2: Document-focused retrieval with all content
        # ============================================================
        logger.info("")
        logger.info("STAGE 2: Document-Focused Content Retrieval")
        logger.info(f"  - Retrieving top {vdb_top_k} items from identified documents")
        logger.info(f"  - Will select top {reranker_top_k} after reranking")
        
        # Build filter for identified documents (includes all levels + chunks by default)
        doc_filter = self._build_document_filter(document_ids)
        
        # Optionally exclude summaries from Stage 2 retrieval (chunks only)
        if exclude_summaries_from_context:
            logger.info(f"  - Mode: Chunks only (excluding summaries)")
            summary_exclusion = 'content_metadata["type"] != "summary"'
            doc_filter = f"({doc_filter}) and ({summary_exclusion})"
        else:
            logger.info(f"  - Mode: Summaries + Chunks (hierarchical context)")
        
        if filter_expr:
            doc_filter = f"({doc_filter}) and ({filter_expr})"
        
        logger.info(f"  - Filter: {doc_filter}")
        
        # Retrieve all content from identified documents
        document_results = vdb_op.retrieval_langchain(
            query=query,
            collection_name=collection_name,
            vectorstore=vectorstore,
            top_k=vdb_top_k,
            filter_expr=doc_filter,
        )
        
        logger.info(f"  ✓ Retrieved {len(document_results)} items from documents")
        
        if not document_results:
            logger.warning("  ⚠ No content found from identified documents")
            # Restore original top_n
            reranker_model.top_n = original_top_n
            return []
        
        # Count summaries vs chunks
        num_summaries = sum(1 for d in document_results if self._is_summary_document(d))
        num_chunks = len(document_results) - num_summaries
        logger.info(f"    - Summaries: {num_summaries}, Chunks: {num_chunks}")
        
        # Rerank all content together using same pattern as normal flow
        logger.info(f"  - Reranking all content...")
        
        # Set top_n for Stage 2 reranking
        reranker_model.top_n = reranker_top_k
        
        final_reranker = RunnableAssign(
            {
                "context": lambda input: reranker_model.compress_documents(
                    query=input["question"], documents=input["context"]
                )
            }
        )
        
        final_result = await final_reranker.ainvoke(
            {"context": document_results, "question": query},
            config={"run_name": "raptor_stage2_reranker"},
        )
        final_results = final_result.get("context", [])
        
        # Restore original top_n
        reranker_model.top_n = original_top_n
        
        logger.info(f"  ✓ Final context: {len(final_results)} items")
        
        # Count final distribution
        final_summaries = sum(1 for d in final_results if self._is_summary_document(d))
        final_chunks = len(final_results) - final_summaries
        logger.info(f"    - Final Summaries: {final_summaries}, Final Chunks: {final_chunks}")
        
        logger.info("=" * 80)
        logger.info(f"RAPTOR Two-Stage Retrieval Complete: {len(final_results)} items selected")
        logger.info("=" * 80)
        
        return final_results
    
    def _extract_document_ids_from_summaries(self, summaries: List[Any]) -> List[str]:
        """Extract unique document IDs (source_id) from summary nodes."""
        document_ids = set()
        
        for summary in summaries:
            if hasattr(summary, "metadata"):
                # Extract source_id from metadata (standard location)
                source = summary.metadata.get("source", {})
                if isinstance(source, dict):
                    source_id = source.get("source_id")
                    if source_id:
                        document_ids.add(source_id)
        
        return sorted(list(document_ids))
    
    def _build_document_filter(self, document_ids: List[str]) -> str:
        """Build filter expression for multiple documents (Milvus format)."""
        if not document_ids:
            return ""
        
        # Escape document IDs and format for Milvus 'in' operator
        escaped_ids = [f'"{doc_id}"' for doc_id in document_ids]
        ids_list = ", ".join(escaped_ids)
        
        # Filter matches source_id in the list
        return f'source["source_id"] in [{ids_list}]'

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
        """Extract chunk IDs from summary metadata.
        
        Reconstructs full chunk_ids from stored indices.
        """
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            
            # Try indices format first (new format)
            chunk_indices = content_metadata.get("chunk_indices")
            if chunk_indices is not None:
                document_id = content_metadata.get("document_id", "")
                # Reconstruct: [0, 5, 10] → ["doc_chunk_0", "doc_chunk_5", "doc_chunk_10"]
                return [f"{document_id}_chunk_{idx}" for idx in chunk_indices]
            
            # Fallback to direct chunk_ids (legacy or non-standard)
            return content_metadata.get("chunk_ids", [])
        return []

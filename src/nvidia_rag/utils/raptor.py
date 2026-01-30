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
)

logger = logging.getLogger(__name__)

# Global semaphore to limit concurrent LLM calls across ALL files
# This ensures max_parallelization is respected system-wide
# Initialized once and shared by all RAPTORTreeBuilder instances
_global_llm_semaphore = None
_global_llm_semaphore_lock = asyncio.Lock()


async def _get_global_llm_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Get or create the global LLM semaphore for RAPTOR.
    
    This ensures max_parallelization limit is enforced across all files.
    If max_parallelization = 20, only 20 LLM calls run concurrently system-wide.
    """
    global _global_llm_semaphore
    
    async with _global_llm_semaphore_lock:
        if _global_llm_semaphore is None:
            _global_llm_semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(
                f"RAPTOR: Created global LLM semaphore with limit={max_concurrent}"
            )
        return _global_llm_semaphore


@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR tree building and retrieval"""

    # Hardcoded parameters (optimal defaults, no need to configure)
    min_cluster_size: int = 3  # Minimum chunks to form a cluster (safety threshold)
    batch_size: int = 50  # Embedding batch size (good balance)

    # Note: These are ALWAYS enabled during ingestion (core RAPTOR features)
    # - clustering_method: Always "sequential" (preserves document order)
    # - summary_max_tokens: Uses config.summarizer.max_chunk_length (via LLM parameter)
    # - tree_traversal: Always enabled during retrieval (fast, safe, beneficial)

    # Dynamic parameters (calculated per document during ingestion)
    # - cluster_size: sqrt(num_chunks) bounded to [5, 15]
    # - max_levels: log_cluster_size(num_chunks) bounded to [2, 5]

    @classmethod
    def from_config(cls, config: Any) -> "RAPTORConfig":
        """
        Create RAPTOR config from NvidiaRAGConfig.

        Reuses existing summarizer configuration:
        - LLM model: config.summarizer.model_name
        - Server URL: config.summarizer.server_url
        - Max tokens: config.summarizer.max_chunk_length
        - Temperature: config.summarizer.temperature
        - API key: config.summarizer.api_key

        No environment variables needed:
        - Ingestion: Use summarization_strategy="raptor" API parameter
        - Retrieval: Tree traversal always runs (no-op if no summaries)
        """
        return cls()

    @staticmethod
    def calculate_cluster_size(num_chunks: int) -> int:
        """
        Calculate optimal cluster size dynamically based on document size.

        Formula: sqrt(num_chunks) bounded to [5, 15]

        Reasoning:
        - Square root provides natural scaling
        - Small docs (25 chunks): sqrt(25) = 5
        - Medium docs (100 chunks): sqrt(100) = 10
        - Large docs (400 chunks): sqrt(400) = 20 → capped at 15

        Bounds [5, 15] based on RAPTOR research:
        - Min 5: Avoid too many tiny clusters
        - Max 15: Avoid overly generic summaries

        Args:
            num_chunks: Number of chunks in document

        Returns:
            Optimal cluster size (5-15)
        """
        cluster_size = int(math.ceil(math.sqrt(num_chunks)))
        return max(5, min(15, cluster_size))

    @staticmethod
    def calculate_max_levels(num_chunks: int, cluster_size: int) -> int:
        """
        Calculate optimal tree depth to reduce chunks to ~1 top node.

        Formula: log_cluster_size(num_chunks) bounded to [2, 5]

        Reasoning:
        - Each level reduces nodes by factor of cluster_size
        - Goal: Reach 1-3 nodes at top level
        - levels = log(num_chunks) / log(cluster_size)

        Examples:
        - 100 chunks, cluster=10: log₁₀(100) = 2 levels
        - 1000 chunks, cluster=10: log₁₀(1000) = 3 levels
        - 10000 chunks, cluster=15: log₁₅(10000) ≈ 4 levels

        Bounds [2, 5]:
        - Min 2: Need hierarchy (at least 2 levels of abstraction)
        - Max 5: Diminishing returns, processing overhead

        Args:
            num_chunks: Number of chunks in document
            cluster_size: Cluster size (from calculate_cluster_size)

        Returns:
            Optimal tree depth (2-5)
        """
        if num_chunks <= cluster_size:
            return 1  # No tree needed for very small documents

        # Calculate levels needed: cluster_size^levels ≈ num_chunks
        levels = int(math.ceil(math.log(num_chunks) / math.log(cluster_size)))
        return max(2, min(5, levels))


@dataclass
class RAPTORNode:
    """Represents a node in the RAPTOR tree"""

    content: str
    level: int  # 0 = chunk, 1+ = summary
    document_id: str
    node_id: str
    embedding: Optional[List[float]] = None
    chunk_ids: List[str] = field(default_factory=list)
    summary_ids: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAPTORTreeBuilder:
    """Builds RAPTOR trees from ingested chunks"""

    def __init__(self, rag_config: Any, raptor_config: RAPTORConfig):
        self.rag_config = rag_config
        self.raptor_config = raptor_config
        self.llm = None
        self.embedding_model = None
        self.llm_semaphore = None  # Will be initialized lazily with global semaphore

        # Statistics
        self.stats = {
            "documents_processed": 0,
            "summaries_created": 0,
            "time_summarization": 0.0,
            "time_embedding": 0.0,
        }

    async def initialize(self):
        """Lazy initialization of LLM and embedding model"""
        if self.llm is None:
            # Use summarizer configuration for RAPTOR summaries
            llm_params = {
                "model": self.rag_config.summarizer.model_name,
                "temperature": self.rag_config.summarizer.temperature,
                "top_p": self.rag_config.summarizer.top_p,
                "max_tokens": self.rag_config.summarizer.max_chunk_length,
                "api_key": self.rag_config.summarizer.get_api_key(),
                "timeout": 300,  # 5 min timeout for large summarization
            }

            if self.rag_config.summarizer.server_url:
                llm_params["llm_endpoint"] = self.rag_config.summarizer.server_url

            self.llm = get_llm(config=self.rag_config, **llm_params)
            self.prompts = get_prompts()

        if self.embedding_model is None:
            self.embedding_model = get_embedding_model(
                model=self.rag_config.embeddings.model_name,
                url=self.rag_config.embeddings.server_url,
                config=self.rag_config,
            )
        
        # Initialize global LLM semaphore (shared across all files)
        if self.llm_semaphore is None:
            self.llm_semaphore = await _get_global_llm_semaphore(
                self.rag_config.summarizer.max_parallelization
            )

    async def build_trees_from_results(
        self, results: List[List[Dict[str, Any]]], collection_name: str, vdb_op: Any
    ) -> None:
        """
        Build RAPTOR trees from NV-Ingest results.

        This is called when summarization_strategy="raptor".
        Uses the EXACT same pattern as regular summarization:
        - Get local semaphore for coordination
        - Acquire global slot via Redis for rate limiting
        - Process document
        - Release slot

        Args:
            results: NV-Ingest results (grouped by document)
            collection_name: Target collection
            vdb_op: VDB operator for uploading summaries
        """
        if not results:
            return

        # Initialize models
        await self.initialize()

        # Get semaphore (SAME as regular summarization line 417)
        semaphore = get_summarization_semaphore()

        # Group chunks by document
        documents_by_id = self._group_chunks_by_document(results)
        
        logger.info(f"RAPTOR: Processing {len(documents_by_id)} documents in collection '{collection_name}'")

        # Process each document WITH EXACT SAME PATTERN as regular summarization
        all_summary_nodes = []
        for doc_id, chunks in documents_by_id.items():
            slot_acquired = False
            try:
                if len(chunks) < self.raptor_config.min_cluster_size:
                    logger.info(
                        f"RAPTOR: Skipping {doc_id} - only {len(chunks)} chunks (min: {self.raptor_config.min_cluster_size})"
                    )
                    continue

                # EXACT SAME PATTERN as line 549 in summarization.py
                async with semaphore:
                    while not await acquire_global_summary_slot(self.rag_config):
                        await asyncio.sleep(0.5)
                    
                    slot_acquired = True

                    summary_nodes = await self._build_tree_for_document(doc_id, chunks)

                    if summary_nodes:
                        # Embed summaries
                        summary_nodes = await self._embed_summaries(summary_nodes)
                        all_summary_nodes.extend(summary_nodes)
                        
                        logger.info(f"RAPTOR: ✓ {doc_id} → {len(summary_nodes)} summary nodes created")

                    self.stats["documents_processed"] += 1

            except Exception as e:
                logger.error(f"RAPTOR: ✗ {doc_id} failed: {e}")
                # Re-raise to propagate error to caller for proper status update
                raise
            finally:
                # Release slot when done (or on error)
                if slot_acquired:
                    await release_global_summary_slot()

        # Upload all summaries to VDB
        if all_summary_nodes:
            try:
                records = self._format_as_vdb_records(
                    all_summary_nodes, collection_name
                )
                vdb_op.run(records)

                logger.info(f"RAPTOR: Uploaded {len(records)} summary nodes to VDB")
            except Exception as e:
                logger.error(f"RAPTOR: VDB upload failed: {e}")
                # Re-raise to mark file as failed
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
        """Build RAPTOR tree for a single document"""

        num_chunks = len(chunks)

        # Calculate optimal parameters for this document
        cluster_size = self.raptor_config.calculate_cluster_size(num_chunks)
        max_levels = self.raptor_config.calculate_max_levels(num_chunks, cluster_size)

        # Convert to RAPTORNode objects
        level_0_nodes = []
        for idx, chunk in enumerate(chunks):
            # Extract chunk ID
            chunk_id = (
                chunk.get("metadata", {}).get("source_metadata", {}).get("chunk_id")
            )
            if not chunk_id:
                chunk_id = f"{document_id}_chunk_{idx}"

            # Extract content from nv-ingest format: metadata.content
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

        # Recursively build tree with calculated parameters
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
        """Recursively cluster and summarize nodes"""

        if current_level > max_levels:
            return []

        if len(nodes) <= cluster_size:
            return []

        # Cluster nodes
        clusters = self._cluster_nodes(nodes, cluster_size)

        if not clusters:
            return []

        # Generate summaries for each cluster
        start_time = time.time()
        summary_nodes = await self._generate_cluster_summaries(
            clusters, document_id, current_level
        )
        self.stats["time_summarization"] += time.time() - start_time
        self.stats["summaries_created"] += len(summary_nodes)

        # Recursively process next level
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
        """
        Cluster nodes using sequential grouping.

        Sequential clustering preserves document order, which is important for:
        - Chronological content (reports, narratives)
        - Spatial locality (nearby chunks are related)
        - Simplicity and speed

        Args:
            nodes: Nodes to cluster
            cluster_size: Target cluster size

        Returns:
            List of clusters
        """
        clusters = []
        for i in range(0, len(nodes), cluster_size):
            cluster = nodes[i : i + cluster_size]
            if len(cluster) >= self.raptor_config.min_cluster_size:
                clusters.append(cluster)
            elif clusters:  # Merge small tail with previous cluster
                clusters[-1].extend(cluster)

        return clusters

    async def _generate_cluster_summaries(
        self, clusters: List[List[RAPTORNode]], document_id: str, level: int
    ) -> List[RAPTORNode]:
        """Generate summaries for clusters with semaphore-controlled concurrency.
        
        With global slot acquisition at document level (same as regular summarization),
        we can safely process all clusters in parallel. The semaphore at _generate_summary()
        ensures LLM calls stay under max_parallelization across the entire system.
        
        Flow:
        1. Document acquires global slot (blocks other documents)
        2. All clusters process in parallel
        3. Semaphore throttles actual LLM calls to max_parallelization
        4. When done, release slot for next document
        """
        
        # Create all tasks - document-level slot + LLM-level semaphore control concurrency
        tasks = [
            self._generate_single_cluster_summary(cluster, cluster_idx, document_id, level)
            for cluster_idx, cluster in enumerate(clusters)
        ]
        
        # Process all clusters - semaphore ensures only max_parallelization LLM calls run at once
        summary_nodes = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        results = []
        for idx, result in enumerate(summary_nodes):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to summarize cluster {idx} at level {level}: {result}"
                )
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
        """Generate summary for a single cluster with 2-pass approach if needed."""
        
        # Get tokenizer for proper token-based operations
        tokenizer = _get_tokenizer(self.rag_config)
        
        # Combine texts from cluster
        texts = [node.content for node in cluster]
        combined_text = "\n\n".join(texts)
        
        # Calculate safe token limit (reserve for prompt overhead and completion)
        # Prompt overhead: ~200 tokens, Completion: max_chunk_length tokens
        PROMPT_OVERHEAD = 200
        max_safe_input_tokens = (
            self.rag_config.summarizer.max_chunk_length - PROMPT_OVERHEAD
        )
        
        current_tokens = _token_length(combined_text, self.rag_config)
        
        # If combined text fits in context, use direct summarization
        if current_tokens <= max_safe_input_tokens:
            summary = await self._generate_summary(combined_text, level)
        else:
            # 2-Pass approach: Split -> Summarize in parallel -> Merge
            # Step 1: Split combined text into chunks
            text_chunks = _split_text_into_chunks(
                combined_text,
                tokenizer,
                max_safe_input_tokens,
                self.rag_config.summarizer.chunk_overlap
            )
            
            # Step 2: Summarize each chunk in parallel
            chunk_summaries = await asyncio.gather(
                *[self._generate_summary(chunk, level) for chunk in text_chunks]
            )
            
            # Step 3: Merge chunk summaries
            merged_text = "\n\n".join(chunk_summaries)
            merged_tokens = _token_length(merged_text, self.rag_config)
            
            # If merged summaries still too large (rare), split again recursively
            if merged_tokens > max_safe_input_tokens:
                # Split merged summaries again
                final_chunks = _split_text_into_chunks(
                    merged_text,
                    tokenizer,
                    max_safe_input_tokens,
                    self.rag_config.summarizer.chunk_overlap
                )
                
                # Summarize recursively in parallel
                recursive_summaries = await asyncio.gather(
                    *[self._generate_summary(chunk, level) for chunk in final_chunks]
                )
                
                # Final merge - if this is still too large, truncate as last resort
                final_merged = "\n\n".join(recursive_summaries)
                final_merged_tokens = _token_length(final_merged, self.rag_config)
                
                if final_merged_tokens > max_safe_input_tokens:
                    # Last resort: truncate to prevent infinite recursion
                    logger.warning(
                        f"RAPTOR: Cluster {cluster_idx} truncated from {final_merged_tokens} to {max_safe_input_tokens} tokens"
                    )
                    tokens = tokenizer.encode(final_merged, add_special_tokens=False)
                    truncated_tokens = tokens[:max_safe_input_tokens]
                    final_merged = tokenizer.decode(truncated_tokens)
                
                summary = await self._generate_summary(final_merged, level)
            else:
                # Final summary from merged chunk summaries
                summary = await self._generate_summary(merged_text, level)
        
        # Track children
        if level == 1:
            chunk_ids = [node.node_id for node in cluster]
            summary_ids = []
        else:
            summary_ids = [node.node_id for node in cluster]
            chunk_ids = []
            for node in cluster:
                chunk_ids.extend(node.chunk_ids)
        
        # Create summary node
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
    
    async def _generate_summary(self, text: str, level: int) -> str:
        """Generate summary using LLM with max_tokens enforced via parameter.
        
        Uses global semaphore to limit concurrent LLM calls across ALL files.
        With document-level slot acquisition (same as regular summarization),
        the system load is naturally limited, so semaphore wait times are minimal.
        """

        # Validate input
        if not text or not text.strip():
            raise ValueError("Cannot generate summary for empty text")
        
        # Check token count before sending to LLM
        current_tokens = _token_length(text, self.rag_config)
        max_context = self.rag_config.summarizer.max_chunk_length
        
        if current_tokens > max_context:
            # This shouldn't happen if 2-pass logic is working correctly
            raise ValueError(
                f"Input text has {current_tokens} tokens but max context is {max_context}. "
                f"Text should have been split earlier!"
            )

        # Construct prompt based on level
        if level == 1:
            prompt = """Summarize the following text chunks into a coherent, concise summary.
Focus on the main themes and key information.

Text chunks:
{text}

Summary:""".format(text=text)
        else:
            prompt = """Create a higher-level summary by synthesizing the following summaries.
Focus on overarching themes and connections.

Summaries to synthesize:
{text}

Higher-level summary:""".format(text=text)

        # Use global semaphore to limit concurrent LLM calls system-wide
        async with self.llm_semaphore:
            try:
                response = await self.llm.ainvoke(prompt)
                summary = (
                    response.content if hasattr(response, "content") else str(response)
                )
                
                if not summary or not summary.strip():
                    raise ValueError("LLM returned empty summary")
                
                return summary.strip()
            except asyncio.TimeoutError:
                logger.error(
                    f"LLM timeout at level {level} (input: {current_tokens} tokens)"
                )
                raise
            except Exception as e:
                logger.error(
                    f"LLM summary generation failed at level {level}: {type(e).__name__}: {e}"
                )
                raise

    async def _embed_summaries(
        self, summary_nodes: List[RAPTORNode]
    ) -> List[RAPTORNode]:
        """Embed all summary nodes"""

        start_time = time.time()
        texts = [node.content for node in summary_nodes]

        # Embed in batches
        all_embeddings = []
        batch_size = self.raptor_config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                batch_embeddings = await self.embedding_model.aembed_documents(
                    batch_texts
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"RAPTOR: Embedding batch failed: {e}")
                # Fallback: zero vectors
                all_embeddings.extend(
                    [[0.0] * self.rag_config.embeddings.dimensions] * len(batch_texts)
                )

        # Assign embeddings
        for node, embedding in zip(summary_nodes, all_embeddings):
            node.embedding = embedding

        self.stats["time_embedding"] += time.time() - start_time

        return summary_nodes

    def _format_as_vdb_records(
        self, summary_nodes: List[RAPTORNode], collection_name: str
    ) -> List[Dict[str, Any]]:
        """Format summary nodes as VDB records (NV-Ingest compatible)"""

        records = []

        for node in summary_nodes:
            record = {
                # Core fields (NV-Ingest standard format)
                "text": node.content,
                "vector": node.embedding,  # NV-Ingest uses "vector" not "embedding"
                "document_type": "text",  # RAPTOR summaries are text content
                # Source metadata (direct dict, not nested)
                "source": {
                    "source_id": node.document_id,
                    "source_name": node.document_id,
                    "collection_name": collection_name,
                    "source_type": "raptor_summary",
                },
                # Content metadata (direct dict, not nested)
                "content_metadata": {
                    "type": "summary",
                    "level": node.level,
                    "document_id": node.document_id,
                    "cluster_id": node.cluster_id,
                    "node_id": node.node_id,  # CRITICAL: Store node_id for unique identification
                    # Tree traversal metadata (enables tree-aware retrieval)
                    "chunk_ids": node.chunk_ids,
                    "summary_ids": node.summary_ids,
                    "num_children": len(node.chunk_ids),
                    **node.metadata,
                },
            }

            records.append(record)

        return records


class RAPTORRetriever:
    """Enhanced retrieval with RAPTOR tree traversal (filtering approach)"""

    def __init__(self, raptor_config: RAPTORConfig):
        self.raptor_config = raptor_config

    async def expand_with_tree_traversal(
        self, documents: List[Any], vdb_op: Any
    ) -> List[Any]:
        """
        Organize documents using RAPTOR tree structure.

        This DOES NOT fetch new documents. Instead, it:
        1. Identifies summaries in the top-k results
        2. Finds their children that are ALSO in top-k
        3. Groups them hierarchically for better LLM context

        Rationale: If a chunk is relevant, VDB already ranked it in top-k.
        We don't need to fetch children outside top-k - just organize what we have.

        Args:
            documents: Retrieved documents from VDB (top-k)
            vdb_op: VDB operator (not used in filter approach)

        Returns:
            Reorganized list of documents (same documents, better structure)
        """
        # Build index of document IDs (support multiple docs with same ID)
        doc_by_id = {}  # chunk_id -> list of documents
        for doc in documents:
            doc_id = self._get_document_id(doc)
            if doc_id not in doc_by_id:
                doc_by_id[doc_id] = []
            doc_by_id[doc_id].append(doc)

        # Reorganize: summaries followed by their children (if present in top-k)
        final_docs = []
        seen_docs = set()  # Track actual doc objects, not just IDs
        
        # Count summaries and chunks in input
        num_summaries = sum(1 for d in documents if self._is_summary_document(d))
        num_chunks = len(documents) - num_summaries
        
        total_children_added = 0

        for doc in documents:
            doc_id = self._get_document_id(doc)
            doc_obj_id = id(doc)

            if doc_obj_id in seen_docs:
                continue

            # Add the document
            final_docs.append(doc)
            seen_docs.add(doc_obj_id)

            # If it's a summary, add its children (if they're in top-k)
            if self._is_summary_document(doc):
                chunk_ids = self._get_chunk_ids(doc)

                # Filter: Only children that are ALSO in top-k
                for chunk_id in chunk_ids:
                    if chunk_id in doc_by_id:
                        # Add ALL documents with this chunk_id (handles multi-chunk-per-page)
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
        """Extract unique document/chunk ID from LangChain Document

        This constructs IDs that match the format used in RAPTOR's chunk_ids.
        """
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            source = doc.metadata.get("source", {})

            # For RAPTOR summaries: use node_id if available, else cluster_id
            if content_metadata.get("type") == "summary":
                node_id = content_metadata.get("node_id")
                if node_id:
                    return str(node_id)
                # Fallback to cluster_id (unique per summary)
                cluster_id = content_metadata.get("cluster_id")
                if cluster_id:
                    return str(cluster_id)

            # For regular chunks: Construct ID matching RAPTOR's chunk_id format
            # Format: {source_id}_chunk_{page_number - 1}
            if isinstance(source, dict) and isinstance(content_metadata, dict):
                source_id = source.get("source_id")
                page_number = content_metadata.get("page_number")

                if source_id and page_number is not None:
                    # RAPTOR uses 0-indexed chunks, pages are 1-indexed
                    chunk_index = page_number - 1
                    return f"{source_id}_chunk_{chunk_index}"

        # Fallback: use object ID
        return str(id(doc))

    def _is_summary_document(self, doc: Any) -> bool:
        """Check if document is a RAPTOR summary"""
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            return content_metadata.get("type") == "summary"
        return False

    def _get_chunk_ids(self, doc: Any) -> List[str]:
        """Extract chunk IDs from summary metadata"""
        if hasattr(doc, "metadata"):
            content_metadata = doc.metadata.get("content_metadata", {})
            return content_metadata.get("chunk_ids", [])
        return []

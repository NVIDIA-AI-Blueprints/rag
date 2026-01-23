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

"""
Document Summary Context Enhancement Module.

This module provides functionality to enhance RAG responses by including
document-level summaries alongside chunk-level context. This gives the LLM
broader document context when generating responses.

Functions:
1. extract_unique_documents(): Extract unique documents from retrieved chunks
2. fetch_document_summaries_async(): Fetch summaries for documents
3. format_summaries_for_prompt(): Format summaries for inclusion in prompt
4. enhance_context_with_summaries_async(): Main entry point
"""

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any

from langchain_core.documents import Document

from nvidia_rag.rag_server.response_generator import retrieve_summary

logger = logging.getLogger(__name__)


def extract_unique_documents(
    retrieved_chunks: list[Document],
) -> OrderedDict[str, dict[str, Any]]:
    """
    Extract unique document names from retrieved chunks.
    
    Args:
        retrieved_chunks: List of retrieved Document objects
    
    Returns:
        OrderedDict mapping filename to document info (preserves order of first occurrence)
    """
    unique_docs = OrderedDict()
    
    for chunk in retrieved_chunks:
        metadata = chunk.metadata
        
        # Try to extract filename from various metadata locations
        filename = None
        
        # Method 1: Direct filename in metadata
        if "filename" in metadata:
            filename = metadata["filename"]
        
        # Method 2: From content_metadata
        elif "content_metadata" in metadata and isinstance(metadata["content_metadata"], dict):
            content_meta = metadata["content_metadata"]
            if "filename" in content_meta:
                filename = content_meta["filename"]
        
        # Method 3: From source metadata
        elif "source_name" in metadata:
            filename = os.path.basename(metadata["source_name"])
        elif "document_name" in metadata:
            filename = metadata["document_name"]
        
        if filename and filename not in unique_docs:
            unique_docs[filename] = {
                "filename": filename,
                "first_seen_in_chunk": len(unique_docs),  # Order of appearance
            }
    
    logger.info(f"Extracted {len(unique_docs)} unique documents from {len(retrieved_chunks)} chunks")
    logger.debug(f"Unique documents: {list(unique_docs.keys())}")
    
    return unique_docs


async def fetch_document_summaries_async(
    unique_documents: OrderedDict[str, dict[str, Any]],
    collection_name: str,
    timeout: int = 10,
) -> dict[str, str]:
    """
    Fetch summaries for unique documents asynchronously.
    
    Args:
        unique_documents: OrderedDict of unique documents
        collection_name: Collection name for summary lookup
        timeout: Timeout for each summary fetch (seconds)
    
    Returns:
        dict: Mapping of filename to summary text
    """
    logger.info(f"Fetching summaries for {len(unique_documents)} documents from collection '{collection_name}'")
    
    # Create async tasks to fetch summaries in parallel
    async def fetch_single_summary(filename: str) -> tuple[str, str | None]:
        """Fetch summary for a single document."""
        try:
            logger.debug(f"Fetching summary for: {filename}")
            response = await retrieve_summary(
                collection_name=collection_name,
                file_name=filename,
                wait=False,  # Non-blocking
                timeout=timeout,
            )
            
            status = response.get("status")
            summary = response.get("summary", "")
            
            if status == "SUCCESS" and summary:
                logger.debug(f"✓ Retrieved summary for {filename} ({len(summary)} chars)")
                return filename, summary
            else:
                logger.debug(f"⚠ No summary available for {filename} (status: {status})")
                return filename, None
        
        except Exception as e:
            logger.warning(f"Failed to fetch summary for {filename}: {e}")
            return filename, None
    
    # Fetch all summaries in parallel
    tasks = [fetch_single_summary(filename) for filename in unique_documents.keys()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    summaries = {}
    success_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Exception fetching summary: {result}")
            continue
        
        filename, summary = result
        if summary:
            summaries[filename] = summary
            success_count += 1
    
    logger.info(f"Successfully fetched {success_count}/{len(unique_documents)} summaries")
    
    return summaries


def format_summaries_for_prompt(summaries: dict[str, str]) -> str:
    """
    Format document summaries for inclusion in the LLM prompt.
    
    Args:
        summaries: Dict mapping filename to summary text
    
    Returns:
        str: Formatted summary text for prompt
    """
    if not summaries:
        return ""
    
    formatted_parts = ["## Document Summaries\n"]
    formatted_parts.append("The following are summaries of the documents that the retrieved context comes from:\n")
    
    for idx, (filename, summary) in enumerate(summaries.items(), 1):
        formatted_parts.append(f"\n### Document {idx}: {filename}")
        formatted_parts.append(f"{summary}\n")
    
    formatted_text = "\n".join(formatted_parts)
    logger.debug(f"Formatted summaries: {len(formatted_text)} total characters")
    
    return formatted_text


async def enhance_context_with_summaries_async(
    retrieved_chunks: list[Document],
    collection_name: str,
    timeout: int = 10,
) -> str:
    """
    Main entry point: Extract documents, fetch summaries, and format for prompt.
    
    This function:
    1. Extracts unique document names from retrieved chunks
    2. Fetches summaries for those documents (async, parallel)
    3. Formats summaries for inclusion in LLM prompt
    
    Args:
        retrieved_chunks: List of retrieved Document objects
        collection_name: Collection name for summary lookup
        timeout: Timeout for each summary fetch
    
    Returns:
        str: Formatted summary text ready to add to prompt (empty if no summaries found)
    """
    if not retrieved_chunks:
        logger.debug("No chunks provided, skipping summary enhancement")
        return ""
    
    logger.info("=" * 80)
    logger.info("STAGE: Document Summary Context Enhancement")
    logger.info("=" * 80)
    logger.info(f"Processing {len(retrieved_chunks)} retrieved chunks for summary enhancement")
    
    # Step 1: Extract unique documents
    unique_docs = extract_unique_documents(retrieved_chunks)
    
    if not unique_docs:
        logger.info("No unique documents identified, skipping summary enhancement")
        logger.info("-" * 80)
        return ""
    
    logger.info(f"Identified {len(unique_docs)} unique documents: {list(unique_docs.keys())}")
    
    # Step 2: Fetch summaries (async, parallel)
    summaries = await fetch_document_summaries_async(
        unique_docs, collection_name, timeout
    )
    
    if not summaries:
        logger.info("No summaries retrieved, skipping summary enhancement")
        logger.info("-" * 80)
        return ""
    
    logger.info(f"Retrieved {len(summaries)} summaries to enhance context")
    
    # Step 3: Format for prompt
    formatted_summaries = format_summaries_for_prompt(summaries)
    
    logger.info(f"✓ Summary context prepared: {len(formatted_summaries)} characters")
    logger.info(f"Summary preview: {formatted_summaries[:200]}...")
    logger.info("-" * 80)
    
    return formatted_summaries

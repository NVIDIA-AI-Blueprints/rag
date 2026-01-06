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

"""Module for shallow summary-based filtering to improve RAG accuracy.

This module implements a pre-processing stage that uses LLM to filter documents
based on their shallow summaries before performing vector retrieval.
"""

import asyncio
import logging
import re
from typing import Any

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field

from nvidia_rag.rag_server.response_generator import retrieve_summary
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


class RelevantFiles(BaseModel):
    """Structured output for relevant file names."""

    file_names: list[str] = Field(
        default_factory=list,
        description="List of relevant file names. Empty list if no documents are relevant.",
    )


async def get_summaries_for_collection(
    collection_name: str, vdb_op: VDBRag
) -> list[dict[str, str]]:
    """
    Get all summaries for documents in a collection.

    Args:
        collection_name: Name of the collection
        vdb_op: Vector database operator

    Returns:
        List of dicts with 'file_name' and 'summary' keys
    """
    try:
        documents_list = vdb_op.get_documents(collection_name)
        if not documents_list:
            logger.debug(f"No documents found in collection {collection_name}")
            return []

        summaries = []
        for doc in documents_list:
            file_name = doc.get("metadata", {}).get("filename")
            if not file_name:
                file_name = doc.get("document_name")
            if not file_name:
                continue

            summary_response = await retrieve_summary(
                collection_name=collection_name,
                file_name=file_name,
                wait=False,
                timeout=0,
            )

            if summary_response.get("status") == "SUCCESS" and summary_response.get(
                "summary"
            ):
                summaries.append(
                    {
                        "file_name": file_name,
                        "summary": summary_response.get("summary", ""),
                    }
                )

        return summaries
    except Exception as e:
        logger.error(
            f"Error retrieving summaries for collection {collection_name}: {e}",
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return []


def batch_summaries(
    summaries: list[dict[str, str]], batch_size: int
) -> list[list[dict[str, str]]]:
    """
    Batch summaries into groups for parallel LLM processing.

    Args:
        summaries: List of summary dicts with 'file_name' and 'summary'
        batch_size: Number of summaries per batch

    Returns:
        List of batches, each containing up to batch_size summaries
    """
    batches = []
    for i in range(0, len(summaries), batch_size):
        batches.append(summaries[i : i + batch_size])
    return batches


async def filter_summaries_with_llm(
    summaries_batch: list[dict[str, str]],
    query: str,
    llm: ChatNVIDIA,
    prompts: dict[str, Any],
) -> list[str]:
    """
    Use LLM with structured output to filter summaries and return relevant file names.

    Args:
        summaries_batch: Batch of summaries with 'file_name' and 'summary'
        query: User query
        llm: LLM instance
        prompts: Prompts dictionary

    Returns:
        List of relevant file names
    """
    try:
        prompt_config = prompts.get("summary_filter_prompt", {})
        system_prompt = prompt_config.get("system", "")
        human_template = prompt_config.get("human", "")

        summaries_text = ""
        for summary_item in summaries_batch:
            file_name = summary_item.get("file_name", "")
            summary = summary_item.get("summary", "")
            summaries_text += f"File: {file_name}\nSummary: {summary}\n\n"

        if human_template:
            prompt_text = human_template.format(query=query, summaries=summaries_text)
        else:
            prompt_text = f"""Given the following query and document summaries, return only the file names of documents that are relevant to answering the query.

Query: {query}

Document Summaries:
{summaries_text}

Return only the relevant file names, one per line. If no documents are relevant, return an empty response."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        # Use structured output with Pydantic model
        llm_with_structure = llm.with_structured_output(RelevantFiles)
        response = await llm_with_structure.ainvoke(messages)

        # Extract file names from structured response
        file_names = response.file_names if hasattr(response, "file_names") else []
        
        # Filter to only include valid filenames (with file extensions)
        valid_file_names = []
        for file_name in file_names:
            file_name = file_name.strip()
            if file_name and re.search(r'\.\w{2,5}$', file_name):
                valid_file_names.append(file_name)
            elif file_name:
                logger.debug(
                    f"Filtered out invalid filename from structured output: '{file_name}'"
                )

        return valid_file_names
    except Exception as e:
        logger.error(
            f"Error filtering summaries with LLM: {e}",
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        return []


async def get_relevant_file_names_from_summaries(
    collection_name: str,
    query: str,
    vdb_op: VDBRag,
    config: NvidiaRAGConfig | None = None,
    prompts: dict[str, Any] | None = None,
) -> list[str]:
    """
    Get relevant file names by filtering summaries with LLM.

    This function:
    1. Retrieves all summaries for the collection
    2. Batches summaries (configurable batch size)
    3. Makes parallel LLM calls (configurable parallelism)
    4. Extracts and returns relevant file names

    Args:
        collection_name: Name of the collection
        query: User query
        vdb_op: Vector database operator
        config: NvidiaRAGConfig instance. If None, creates a new one.
        prompts: Prompts dictionary. If None, loads from default.

    Returns:
        List of relevant file names
    """
    if config is None:
        config = NvidiaRAGConfig()

    if prompts is None:
        prompts = get_prompts()

    summaries_per_call = config.summary_filter.summaries_per_llm_call
    max_parallel_calls = config.summary_filter.max_parallel_llm_calls

    summaries = await get_summaries_for_collection(collection_name, vdb_op)
    total_summaries = len(summaries)

    if total_summaries == 0:
        logger.warning(f"No summaries found for collection {collection_name}")
        return []

    # Log all filenames available for summary filtering
    all_filenames = [s.get("file_name", "") for s in summaries]
    logger.info(
        f"Found {total_summaries} summaries in collection {collection_name}. "
        f"Filenames: {all_filenames}"
    )

    batches = batch_summaries(summaries, summaries_per_call)
    total_batches = len(batches)

    llm_settings = {
        "temperature": config.summary_filter.temperature,
        "top_p": config.summary_filter.top_p,
        "max_tokens": config.summary_filter.max_tokens,
        "api_key": config.summary_filter.get_api_key(),
    }

    llm = get_llm(
        model=config.summary_filter.model_name,
        llm_endpoint=config.summary_filter.server_url,
        **llm_settings,
    )

    semaphore = asyncio.Semaphore(max_parallel_calls)
    all_file_names = []

    async def process_batch(batch: list[dict[str, str]]) -> list[str]:
        async with semaphore:
            batch_filenames = [b.get("file_name", "") for b in batch]
            logger.info(
                f"Processing batch with {len(batch)} summaries. Files: {batch_filenames}"
            )
            result = await filter_summaries_with_llm(batch, query, llm, prompts)
            logger.info(
                f"LLM filtered batch result: selected {len(result)} files from {len(batch)}: {result}"
            )
            return result

    tasks = [process_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(
                f"Error processing batch: {result}",
                exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
            )
        elif isinstance(result, list):
            all_file_names.extend(result)

    unique_file_names = []
    seen = set()
    for file_name in all_file_names:
        if file_name not in seen:
            seen.add(file_name)
            unique_file_names.append(file_name)

    logger.info(
        f"Summary filter final result: Extracted {len(unique_file_names)} relevant file names "
        f"from {total_batches} LLM calls: {unique_file_names}"
    )

    return unique_file_names


def create_milvus_filter_from_file_names(file_names: list[str]) -> str:
    """
    Create a Milvus filter expression from a list of file names.

    Args:
        file_names: List of file names to filter by

    Returns:
        Milvus filter expression string
    """
    if not file_names:
        logger.warning("No file names provided to create Milvus filter")
        return ""

    filter_parts = []
    for file_name in file_names:
        escaped_name = file_name.replace('"', '\\"')
        filter_parts.append(f'content_metadata["filename"] == "{escaped_name}"')

    if len(filter_parts) == 1:
        filter_expr = filter_parts[0]
    else:
        filter_expr = "(" + " or ".join(filter_parts) + ")"
    
    logger.info(
        f"Created Milvus filter expression for {len(file_names)} files: {filter_expr}"
    )
    return filter_expr

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
Table/chart summary generation and VDB insertion.

After nv_ingest returns results, we filter structured elements (table/chart),
summarize each with an LLM, embed the summary, and insert one chunk per
summary into the same collection so retrieval can hit either the original or
the summary. At retrieval, if the hit is a summary chunk, the full table/chart
content is taken from metadata and included in the context.
"""

import asyncio
import copy
import logging
from typing import Any

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.llm import get_llm, get_prompts

logger = logging.getLogger(__name__)

# Milvus text field max length (nv_ingest schema)
MILVUS_TEXT_MAX_LENGTH = 65535
# Cap table/chart content sent to LLM to avoid context overflows
TABLE_CHART_LLM_CONTENT_MAX_CHARS = 12000
# Max records per VDB insert batch (align with RAPTOR: 1000; nv_ingest stream threshold is also 1000)
TABLE_CHART_INSERT_BATCH_SIZE = 1000
# Subtypes we summarize
TABLE_CHART_SUBTYPES = ("table", "chart")
CONTENT_METADATA_TYPE_SUMMARY = "table_chart_summary"

# Global semaphore per event loop so table/chart summary never exceeds max_concurrency across batches
_table_chart_semaphores: dict[int, asyncio.Semaphore] = {}


def _collect_table_chart_elements(results: list[list[dict]]) -> list[dict]:
    """Collect all structured elements with subtype table or chart from results."""
    elements: list[dict] = []
    for result_list in results:
        if not result_list:
            continue
        for element in result_list:
            if element.get("document_type") != "structured":
                continue
            metadata = element.get("metadata") or {}
            content_metadata = metadata.get("content_metadata") or {}
            subtype = content_metadata.get("subtype")
            if subtype not in TABLE_CHART_SUBTYPES:
                continue
            table_metadata = metadata.get("table_metadata") or {}
            table_content = table_metadata.get("table_content")
            if not table_content or not str(table_content).strip():
                continue
            elements.append(element)
    return elements


def _truncate_for_llm(text: str, max_chars: int = TABLE_CHART_LLM_CONTENT_MAX_CHARS) -> str:
    """Truncate content for LLM input to avoid context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for length ...]"


def _truncate_for_milvus_text(text: str, max_len: int = MILVUS_TEXT_MAX_LENGTH) -> str:
    """Ensure summary fits in Milvus VARCHAR text field."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + " [...]"


async def _summarize_one(
    table_chart_content: str,
    chain: Any,
) -> str | None:
    """Call LLM chain to summarize one table/chart content. Returns summary or None on failure."""
    content_for_llm = _truncate_for_llm(table_chart_content)
    try:
        summary = await chain.ainvoke(
            {"table_chart_content": content_for_llm},
            config={"run_name": "table_chart_summary"},
        )
        if summary and isinstance(summary, str) and summary.strip():
            return summary.strip()
    except Exception as e:
        logger.warning("Table/chart summary LLM call failed: %s", e, exc_info=logger.isEnabledFor(logging.DEBUG))
    return None


def _embed_summary_sync(summary_text: str, embedder: Any) -> list[float] | None:
    """Embed one summary string using the given embedder. Sync. Returns list of floats for Milvus."""
    try:
        vectors = embedder.embed_documents([summary_text])
        if not vectors or len(vectors) != 1:
            return None
        vec = vectors[0]
        if vec is None:
            return None
        if isinstance(vec, list):
            return vec
        if hasattr(vec, "__iter__") and not isinstance(vec, (str, bytes)):
            return list(vec)
        return None
    except Exception as e:
        logger.warning("Table/chart summary embedding failed: %s", e, exc_info=logger.isEnabledFor(logging.DEBUG))
    return None


def _build_summary_element(
    original_element: dict,
    summary_text: str,
    embedding: list[float],
) -> dict:
    """Build one nv_ingest-style element for a table/chart summary (document_type=text, etc.)."""
    metadata = original_element.get("metadata") or {}
    source_metadata = copy.deepcopy(metadata.get("source_metadata") or {})
    content_metadata_orig = copy.deepcopy(metadata.get("content_metadata") or {})
    table_metadata = metadata.get("table_metadata") or {}
    raw_content = table_metadata.get("table_content")
    original_table_content = str(raw_content) if raw_content is not None else ""

    content_metadata = {
        **content_metadata_orig,
        "type": CONTENT_METADATA_TYPE_SUMMARY,
        "original_table_content": original_table_content,
        "subtype": content_metadata_orig.get("subtype", "table"),
    }
    summary_for_storage = _truncate_for_milvus_text(summary_text)

    return {
        "document_type": "text",
        "metadata": {
            "content": summary_for_storage,
            "embedding": embedding,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
        },
    }


def _element_to_milvus_record(element: dict, collection_name: str) -> dict:
    """Convert a summary element to flat record for MilvusClient.insert (same pattern as RAPTOR)."""
    metadata = element.get("metadata") or {}
    source_metadata = metadata.get("source_metadata") or {}
    return {
        "text": metadata.get("content", ""),
        "vector": metadata.get("embedding", []),
        "document_type": element.get("document_type", "text"),
        "source": {
            "source_id": source_metadata.get("source_id", ""),
            "source_name": source_metadata.get("source_name", ""),
            "collection_name": collection_name,
            "source_type": "table_chart_summary",
        },
        "content_metadata": metadata.get("content_metadata", {}),
    }


async def generate_and_ingest_table_chart_summaries(
    results: list[list[dict]],
    collection_name: str,
    vdb_op: Any,
    config: NvidiaRAGConfig,
    prompts: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """
    Generate LLM summaries for table/chart elements and insert them into the same VDB collection.
    For Milvus, inserts via MilvusClient directly (same pattern as RAPTOR) so no CSV is needed.
    """
    if prompts is None:
        prompts = get_prompts()
    prompt_config = prompts.get("table_chart_summary_prompt")
    if not prompt_config:
        logger.warning("table_chart_summary_prompt not found in prompts; skipping table/chart summaries")
        return 0, 0
    elements = _collect_table_chart_elements(results)
    if not elements:
        logger.debug("No table/chart elements found for summary generation")
        return 0, 0
    total = len(elements)
    max_concurrent = max(1, getattr(config.nv_ingest, "table_chart_summary_max_concurrency", 20))
    logger.info(
        "Table/chart summary: %d to process (collection=%s, max_concurrent=%d)",
        total,
        collection_name,
        max_concurrent,
    )
    llm = get_llm(
        config=config,
        model=config.summarizer.model_name,
        temperature=config.summarizer.temperature,
        top_p=config.summarizer.top_p,
        api_key=config.summarizer.get_api_key(),
        llm_endpoint=config.summarizer.server_url or "",
    )
    system_msg = prompt_config.get("system", "")
    human_msg = prompt_config.get("human", "")
    if not human_msg:
        logger.warning("table_chart_summary_prompt has no 'human' template; skipping")
        return 0, 0
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", human_msg)]
    )
    chain = prompt | llm | StrOutputParser()
    embedder = get_embedding_model(
        model=config.embeddings.model_name,
        url=config.embeddings.server_url,
        config=config,
    )
    semaphore = _get_table_chart_semaphore(max_concurrent)

    async def process_one(element: dict) -> dict | None:
        async with semaphore:
            metadata = element.get("metadata") or {}
            table_metadata = metadata.get("table_metadata") or {}
            raw_content = table_metadata.get("table_content")
            table_content = str(raw_content).strip() if raw_content is not None else ""
            if not table_content:
                return None
            summary = await _summarize_one(table_content, chain)
            if not summary:
                return None
            embedding = await asyncio.to_thread(_embed_summary_sync, summary, embedder)
            if not embedding:
                return None
            return _build_summary_element(element, summary, embedding)

    summary_elements = []
    chunk_size = max_concurrent
    for start in range(0, total, chunk_size):
        batch = elements[start : start + chunk_size]
        batch_results = await asyncio.gather(*[process_one(el) for el in batch], return_exceptions=True)
        for r in batch_results:
            if isinstance(r, BaseException):
                logger.debug("Table/chart summary task failed: %s", r)
            elif r is not None:
                summary_elements.append(r)
        done = min(start + len(batch), total)
        failed_so_far = done - len(summary_elements)
        logger.info("Table/chart summary: %d/%d done (%d failed)", done, total, failed_so_far)
    failed = total - len(summary_elements)
    if not summary_elements:
        logger.warning("Table/chart summary: 0 produced, %d failed (collection=%s)", failed, collection_name)
        return 0, failed
    try:
        # Insert via MilvusClient directly (same as RAPTOR branch) – no nv_ingest run/CSV needed
        if getattr(vdb_op, "vdb_endpoint", None) and hasattr(vdb_op, "_get_milvus_token"):
            from pymilvus import MilvusClient

            client = MilvusClient(uri=vdb_op.vdb_endpoint, token=vdb_op._get_milvus_token())
            for i in range(0, len(summary_elements), TABLE_CHART_INSERT_BATCH_SIZE):
                batch = summary_elements[i : i + TABLE_CHART_INSERT_BATCH_SIZE]
                records = [_element_to_milvus_record(el, collection_name) for el in batch]
                client.insert(collection_name=collection_name, data=records)
        else:
            # Fallback for non-Milvus VDB (e.g. Elasticsearch)
            for i in range(0, len(summary_elements), TABLE_CHART_INSERT_BATCH_SIZE):
                vdb_op.run(summary_elements[i : i + TABLE_CHART_INSERT_BATCH_SIZE])
        logger.info(
            "Table/chart summary: inserted %d, failed %d (collection=%s)",
            len(summary_elements),
            failed,
            collection_name,
        )
    except Exception as e:
        logger.error(
            "Failed to insert table/chart summary chunks into VDB: %s",
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return 0, failed + len(summary_elements)
    return len(summary_elements), failed


def _get_table_chart_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Get or create a global semaphore for table/chart summary (per event loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError("No running event loop - cannot create table/chart summary semaphore") from e
    loop_id = id(loop)
    if loop_id not in _table_chart_semaphores:
        _table_chart_semaphores[loop_id] = asyncio.Semaphore(max_concurrent)
        logger.debug("Table/chart summary semaphore initialized (max_concurrent=%d)", max_concurrent)
    return _table_chart_semaphores[loop_id]

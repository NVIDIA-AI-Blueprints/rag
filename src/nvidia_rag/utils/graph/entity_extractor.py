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

"""LLM-based entity and relationship extraction for knowledge graph construction.

Extracts Subject-Predicate-Object triples from text chunks using a structured
prompt loaded from prompt.yaml. Uses /no_think and temperature=0 for
deterministic, non-reasoning results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.graph.graph_store import (
    Entity,
    GraphStore,
    Relationship,
)
from nvidia_rag.utils.llm import get_llm

logger = logging.getLogger(__name__)


def _chunk_id(text: str) -> str:
    """Generate a stable short ID for a text chunk."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _parse_extraction_response(response: str) -> dict[str, Any]:
    """Parse the LLM response, handling common formatting issues."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse entity extraction response: %s", text[:200])
        return {"entities": [], "relationships": []}


def _parse_query_entities(response: str) -> list[str]:
    """Parse entity names from query extraction response."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(e).strip() for e in result if str(e).strip()]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
                if isinstance(result, list):
                    return [str(e).strip() for e in result if str(e).strip()]
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse query entity extraction response")
    return []


def _build_prompt(prompts: dict, prompt_key: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate from the prompts dict, same as other components."""
    prompt_config = prompts.get(prompt_key, {})
    system_prompt = prompt_config.get("system", "")
    human_prompt = prompt_config.get("human", prompt_config.get("user", ""))
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])


def _get_graph_llm(config: NvidiaRAGConfig) -> Any:
    """Create an LLM instance for graph extraction using GraphRAG config."""
    graph_cfg = config.graph_rag
    llm_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 4096,
    }
    if graph_cfg.entity_extraction_model:
        llm_kwargs["model"] = graph_cfg.entity_extraction_model
    if graph_cfg.entity_extraction_server_url:
        llm_kwargs["llm_endpoint"] = graph_cfg.entity_extraction_server_url
    api_key = graph_cfg.get_api_key()
    if api_key:
        llm_kwargs["api_key"] = api_key
    return get_llm(config=config, **llm_kwargs)


async def extract_entities_from_chunks(
    chunks: list[Document],
    graph_store: GraphStore,
    collection_name: str,
    config: NvidiaRAGConfig | None = None,
    prompts: dict | None = None,
) -> dict[str, int]:
    """Extract entities and relationships from document chunks and store them in the graph.

    Args:
        chunks: List of LangChain Document objects with page_content.
        graph_store: GraphStore instance to write to.
        collection_name: Name of the collection.
        config: NvidiaRAGConfig instance.
        prompts: Prompts dict loaded from prompt.yaml via get_prompts().

    Returns:
        Dict with counts: {"entities_added", "relationships_added", "chunks_processed"}.
    """
    if config is None:
        config = NvidiaRAGConfig()
    if prompts is None:
        from nvidia_rag.utils.llm import get_prompts
        prompts = get_prompts()

    graph_cfg = config.graph_rag
    llm = _get_graph_llm(config)

    prompt = _build_prompt(prompts, "graph_entity_extraction_prompt")
    chain = prompt | llm | StrOutputParser()

    semaphore = asyncio.Semaphore(graph_cfg.max_parallelization)

    total_entities = 0
    total_relationships = 0
    chunks_processed = 0

    max_retries = 3

    async def _process_chunk(chunk: Document) -> tuple[list[Entity], list[Relationship], bool]:
        """Extract entities from a single chunk with concurrency control and retries."""
        text = chunk.page_content
        if not text or len(text.strip()) < 50:
            return [], [], False

        chunk_hash = _chunk_id(text)
        response = None
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    response = await chain.ainvoke(
                        {
                            "text": text,
                            "max_entities": graph_cfg.max_entities_per_chunk,
                            "max_relationships": graph_cfg.max_relationships_per_chunk,
                        },
                        config={"run_name": "graph-entity-extraction"},
                    )
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            "Entity extraction attempt %d/%d failed for chunk %s, retrying in %ds",
                            attempt + 1, max_retries, chunk_hash, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(
                            "Entity extraction failed for chunk %s after %d attempts",
                            chunk_hash, max_retries, exc_info=True,
                        )
                        return [], [], False

        if response is None:
            return [], [], False

        parsed = _parse_extraction_response(response)

        entities = []
        for e in parsed.get("entities", []):
            if not e.get("name"):
                continue
            entities.append(Entity(
                name=e["name"],
                entity_type=e.get("type", "unknown"),
                description=e.get("description", ""),
                source_chunk_ids=[chunk_hash],
                metadata={
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page_number", ""),
                },
            ))

        entity_name_set = {e.key for e in entities}
        relationships = []
        for r in parsed.get("relationships", []):
            subj = (r.get("subject") or "").strip()
            obj = (r.get("object") or "").strip()
            pred = (r.get("predicate") or "related_to").strip()
            if not subj or not obj or not pred:
                continue
            if subj.lower() not in entity_name_set or obj.lower() not in entity_name_set:
                for missing_name in [subj, obj]:
                    if missing_name.lower() not in entity_name_set:
                        entities.append(Entity(
                            name=missing_name,
                            entity_type="unknown",
                            description="",
                            source_chunk_ids=[chunk_hash],
                        ))
                        entity_name_set.add(missing_name.lower())

            relationships.append(Relationship(
                source=subj,
                target=obj,
                relation_type=pred,
                description=r.get("description", ""),
                weight=1.0,
                source_chunk_ids=[chunk_hash],
            ))

        return entities, relationships, True

    results = await asyncio.gather(
        *[_process_chunk(chunk) for chunk in chunks],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            logger.warning("Chunk extraction task failed: %s", result)
            continue
        entities, relationships, success = result
        if entities:
            total_entities += graph_store.add_entities(entities, collection_name)
        if relationships:
            total_relationships += graph_store.add_relationships(relationships, collection_name)
        if success:
            chunks_processed += 1

    graph_store.persist()
    logger.info(
        "Entity extraction complete for '%s': %d entities, %d relationships from %d chunks",
        collection_name,
        total_entities,
        total_relationships,
        chunks_processed,
    )

    return {
        "entities_added": total_entities,
        "relationships_added": total_relationships,
        "chunks_processed": chunks_processed,
    }


async def extract_entities_from_query(
    query: str,
    config: NvidiaRAGConfig | None = None,
    prompts: dict | None = None,
) -> list[str]:
    """Extract entity names from a user query for graph lookup.

    Args:
        query: User's natural language query.
        config: NvidiaRAGConfig instance.
        prompts: Prompts dict loaded from prompt.yaml via get_prompts().

    Returns:
        List of entity name strings.
    """
    if config is None:
        config = NvidiaRAGConfig()
    if prompts is None:
        from nvidia_rag.utils.llm import get_prompts
        prompts = get_prompts()

    graph_cfg = config.graph_rag
    llm_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 512,
    }
    if graph_cfg.entity_extraction_model:
        llm_kwargs["model"] = graph_cfg.entity_extraction_model
    if graph_cfg.entity_extraction_server_url:
        llm_kwargs["llm_endpoint"] = graph_cfg.entity_extraction_server_url
    api_key = graph_cfg.get_api_key()
    if api_key:
        llm_kwargs["api_key"] = api_key

    llm = get_llm(config=config, **llm_kwargs)

    prompt = _build_prompt(prompts, "graph_query_entity_extraction_prompt")
    chain = prompt | llm | StrOutputParser()

    try:
        response = await chain.ainvoke(
            {"query": query},
            config={"run_name": "graph-query-entity-extraction"},
        )
        entities = _parse_query_entities(response)
        logger.info("Extracted %d entities from query: %s", len(entities), entities)
        return entities
    except Exception:
        logger.warning("Query entity extraction failed", exc_info=True)
        return []

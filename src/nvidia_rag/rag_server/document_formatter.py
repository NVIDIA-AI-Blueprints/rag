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

"""Document formatting utilities for RAG responses."""

import logging
import math
import os
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _print_conversation_history(
    conversation_history: list[str] = None, query: str | None = None
) -> None:
    """Print conversation history for debugging.

    Args:
        conversation_history: List of (role, content) tuples
        query: Optional query string
    """
    if conversation_history is not None:
        for role, content in conversation_history:
            logger.debug("Role: %s", role)
            logger.debug("Content: %s\n", content)


def _normalize_relevance_scores(documents: list["Document"]) -> list["Document"]:
    """
    Normalize relevance scores in a list of documents to be between 0 and 1 using sigmoid function.

    Args:
        documents: List of Document objects with relevance_score in metadata

    Returns:
        The same list of documents with normalized scores
    """
    if not documents:
        return documents

    # Apply sigmoid normalization (1 / (1 + e^-x))
    for doc in documents:
        if "relevance_score" in doc.metadata:
            original_score = doc.metadata["relevance_score"]
            scaled_score = original_score * 0.1
            normalized_score = 1 / (1 + math.exp(-scaled_score))
            doc.metadata["relevance_score"] = normalized_score

    return documents


def _format_document_with_source(doc: "Document") -> str:
    """Format document content with its source filename.

    Args:
        doc: Document object with metadata and page_content

    Returns:
        str: Formatted string with filename and content if ENABLE_SOURCE_METADATA is True,
            otherwise returns just the content
    """
    # Debug log before formatting
    logger.debug(f"Before format_document_with_source - Document: {doc}")

    # Check if source metadata is enabled via environment variable
    enable_metadata = os.getenv("ENABLE_SOURCE_METADATA", "True").lower() == "true"

    # Return just content if metadata is disabled or doc has no metadata
    if not enable_metadata or not hasattr(doc, "metadata"):
        result = doc.page_content
        logger.debug(
            f"After format_document_with_source (metadata disabled) - Result: {result}"
        )
        return result

    # Handle nested metadata structure
    source = doc.metadata.get("source", {})
    source_path = (
        source.get("source_name", "") if isinstance(source, dict) else source
    )

    # If no source path is found, return just the content
    if not source_path:
        result = doc.page_content
        logger.debug(
            f"After format_document_with_source (no source path) - Result: {result}"
        )
        return result

    filename = os.path.splitext(os.path.basename(source_path))[0]
    logger.debug(f"Before format_document_with_source - Filename: {filename}")
    result = f"File: {filename}\nContent: {doc.page_content}"

    # Debug log after formatting
    logger.debug(f"After format_document_with_source - Result: {result}")

    return result

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

"""Utility for generating filter expressions from natural language using LLM."""

import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def _format_metadata_schema_for_prompt(metadata_schema: list[dict[str, Any]]) -> str:
    """Format metadata schema for inclusion in the prompt.

    Args:
        metadata_schema: List of metadata field definitions

    Returns:
        Formatted string representation of the schema
    """
    if not metadata_schema:
        return "No metadata schema available for this collection."

    return json.dumps(metadata_schema, separators=(",", ":"))


def _extract_filter_expression_from_response(response: Any) -> str | None:
    """Extract the filter expression from LLM response.

    Args:
        response: Raw response from LLM (can be AIMessage object or string)

    Returns:
        Extracted filter expression or None if not found
    """
    # Handle AIMessage objects by extracting their content
    if hasattr(response, "content"):
        response = response.content

    # Ensure response is a string
    if not isinstance(response, str):
        logger.warning(f"Unexpected response type: {type(response)}")
        return None

    response = response.strip()

    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    if "<think>" in response:
        response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)

    response = re.sub(r"\n\s*\n", "\n", response)
    response = response.strip()

    if (
        response.upper().startswith("NO_FILTER")
        or response.upper().startswith("UNSUPPORTED")
        or not response
    ):
        logger.warning("No filter expression found after removing thinking tokens")
        return None

    return response


def generate_filter_from_natural_language(
    user_request: str,
    collection_name: str,
    metadata_schema: list[dict[str, Any]],
    prompt_template: dict,
    llm: Any | None = None,
    existing_filter_expr: str | None = None,
) -> str | None:
    """Generate a filter expression from natural language request.

    Args:
        user_request: Natural language description of filtering requirements
        collection_name: Name of the collection to filter
        metadata_schema: Metadata schema for the collection
        prompt_template: Prompt template for filter generation
        llm: LLM instance to use (if None, will create one with default settings)
        existing_filter_expr: Existing filter expression to validate/improve (optional)

    Returns:
        Generated filter expression or None if no filter needed/possible
    """
    try:
        schema_text = _format_metadata_schema_for_prompt(metadata_schema)
        # Add existing filter expression to the prompt if provided
        existing_filter_context = ""
        if (
            existing_filter_expr
            and isinstance(existing_filter_expr, str)
            and existing_filter_expr.strip()
        ):
            existing_filter_context = f"\n**EXISTING FILTER EXPRESSION:**\n{existing_filter_expr}\n\nPlease validate this existing filter expression and improve it if needed based on the user request."

        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template.get("system")),
                ("human", prompt_template.get("human")),
            ]
        )

        # Prepare the input for the chain
        chain_input = {
            "metadata_schema": schema_text,
            "user_request": user_request,
            "collection_name": collection_name,
            "existing_filter_context": existing_filter_context,
        }

        if llm is None:
            logger.debug(
                "No LLM provided for filter expression generation, returning empty string"
            )
            return ""

        response = llm.invoke(filter_prompt.format_messages(**chain_input))

        filter_expr = _extract_filter_expression_from_response(response)

        return filter_expr

    except Exception as e:
        logger.error(f"Error generating filter expression: {str(e)}")
        return None

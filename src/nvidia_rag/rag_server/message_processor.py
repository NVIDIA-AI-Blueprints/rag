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

"""Message and prompt processing utilities for RAG server.

This module provides functionality for processing chat messages, handling conversation
history, and customizing prompts based on model-specific requirements. It includes
special handling for Nemotron models and their thinking mode capabilities.

Key functions:
- handle_prompt_processing(): Process chat history and templates into structured messages
"""

import logging
import os
from typing import Any

from nvidia_rag.rag_server.content_utils import _extract_text_from_content

logger = logging.getLogger(__name__)


def handle_prompt_processing(
    chat_history: list[dict[str, Any]],
    model: str,
    prompts: dict[str, Any],
    template_key: str = "chat_template",
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Handle common prompt processing logic for both LLM and RAG chains.

    This function processes chat history and prompt templates to prepare structured
    messages for LLM/RAG chains. It handles:
    - Extracting system and user prompts from templates
    - Processing conversation history
    - Special handling for Nemotron v1 models (thinking mode)
    - Merging system messages from chat history

    Args:
        chat_history: List of conversation messages with 'role' and 'content' keys.
                     Each message can have content as string or structured format.
        model: Name of the model used for generation. Used to detect Nemotron v1
               models and apply special system prompt handling.
        prompts: Dictionary containing prompt templates with keys like 'chat_template',
                'rag_template', etc. Each template should have 'system' and
                'human'/'user' keys.
        template_key: Key to get the appropriate template from prompts dictionary.
                     Defaults to "chat_template".

    Returns:
        Tuple containing:
        - system_message: List of system message tuples in format [("system", prompt)]
        - conversation_history: List of conversation history tuples in format
                              [(role, content), ...]
        - user_message: List of user message tuples from prompt template in format
                       [("user", prompt)] (empty list if no user template found)

    Notes:
        - For Nemotron v1 models (ending with "llama-3.3-nemotron-super-49b-v1"),
          the system prompt is overridden with "detailed thinking on" or
          "detailed thinking off" based on ENABLE_NEMOTRON_THINKING environment variable
        - System messages from chat_history are appended to the base system prompt
        - Supports both "human" and "user" keys in templates for backwards compatibility
        - All content is extracted as text using _extract_text_from_content utility

    Example:
        >>> prompts = {
        ...     "chat_template": {
        ...         "system": "You are a helpful assistant.",
        ...         "user": "Answer based on context: {context}"
        ...     }
        ... }
        >>> chat_history = [
        ...     {"role": "user", "content": "What is AI?"},
        ...     {"role": "assistant", "content": "AI stands for..."}
        ... ]
        >>> system_msg, history, user_msg = handle_prompt_processing(
        ...     chat_history, "llama-model", prompts
        ... )
    """

    # Get the base template
    system_prompt = prompts.get(template_key, {}).get("system", "")
    # Support both "human" and "user" keys with fallback
    template_dict = prompts.get(template_key, {})
    user_prompt = template_dict.get("human", template_dict.get("user", ""))
    conversation_history = []
    user_message = []

    is_nemotron_v1 = str(model).endswith("llama-3.3-nemotron-super-49b-v1")

    # Nemotron controls thinking using system prompt, if nemotron v1 model is used update system prompt to enable/disable think
    if is_nemotron_v1:
        logger.info("Nemotron v1 model detected, updating system prompt")
        if os.environ.get("ENABLE_NEMOTRON_THINKING", "false").lower() == "true":
            logger.info("Setting system prompt as detailed thinking on")
            system_prompt = "detailed thinking on"
        else:
            logger.info("Setting system prompt as detailed thinking off")
            system_prompt = "detailed thinking off"

    # Process chat history
    for message in chat_history:
        # Overwrite system message if provided in conversation history
        if message.get("role") == "system":
            content_text = _extract_text_from_content(message.get("content"))
            system_prompt = system_prompt + " " + content_text
        else:
            content_text = _extract_text_from_content(message.get("content"))
            conversation_history.append((message.get("role"), content_text))

    system_message = [("system", system_prompt)]
    if user_prompt:
        user_message = [("user", user_prompt)]

    return (
        system_message,
        conversation_history,
        user_message,
    )

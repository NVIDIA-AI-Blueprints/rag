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

"""The wrapper for interacting with llm models and pre or postprocessing LLM response.
1. get_prompts: Get the prompts from the YAML file.
2. get_llm: Get the LLM model. Uses the NVIDIA AI Endpoints or OpenAI.
3. streaming_filter_think: Filter the think tokens from the LLM response.
4. get_streaming_filter_think_parser: Get the parser for filtering the think tokens from the LLM response.
"""

import os
import logging
import yaml
import requests
from functools import lru_cache
from typing import Dict, Iterable
from pathlib import Path
from langchain.llms.base import LLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)

try:
  from langchain_openai import ChatOpenAI
except ImportError:
  logger.info("Langchain OpenAI is not installed.")
  pass

from langchain_core.language_models.chat_models import SimpleChatModel

from nvidia_rag.utils.common import get_config, sanitize_nim_url, combine_dicts
from nvidia_rag.utils.common import utils_cache

@lru_cache
def get_prompts() -> Dict:
    """Retrieves prompt configurations from YAML file and return a dict.
    """

    # default config taking from prompt.yaml
    default_config_path = os.path.join(os.environ.get("EXAMPLE_PATH", os.path.dirname(__file__)), "..", "rag_server", "prompt.yaml")
    cur_dir_path = os.path.join(os.path.dirname(__file__), "..", "rag_server", "prompt.yaml")
    default_config = {}
    if Path(default_config_path).is_file():
        with open(default_config_path, 'r', encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", default_config_path)
            default_config = yaml.safe_load(file)
    elif Path(cur_dir_path).is_file():
        # if prompt.yaml is not found in the default path, check in the current directory(use default config)
        # this is for packaging
        with open(cur_dir_path, 'r', encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", cur_dir_path)
            default_config = yaml.safe_load(file)
    else:
        logger.info("No prompts config file found")

    config_file = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")

    config = {}
    if Path(config_file).is_file():
        with open(config_file, 'r', encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", config_file)
            config = yaml.safe_load(file)

    config = combine_dicts(default_config, config)
    return config


@utils_cache
@lru_cache()
def get_llm(**kwargs) -> LLM | SimpleChatModel:
    """Create the LLM connection."""

    settings = get_config()

    # Sanitize the URL
    url = sanitize_nim_url(kwargs.get('llm_endpoint', ""), kwargs.get('model'), "chat")

    # Check if guardrails are enabled
    enable_guardrails = settings.enable_guardrails and kwargs.get('enable_guardrails', False) == True

    logger.debug("Using %s as model engine for llm. Model name: %s", settings.llm.model_engine, kwargs.get('model'))
    if settings.llm.model_engine == "nvidia-ai-endpoints":

        # Use ChatOpenAI with guardrails if enabled
        # TODO Add the ChatNVIDIA implementation when available
        if enable_guardrails:
            logger.info("Guardrails enabled, using ChatOpenAI with guardrails URL")
            guardrails_url = os.getenv("NEMO_GUARDRAILS_URL", "")
            if not guardrails_url:
                logger.warning("NEMO_GUARDRAILS_URL not set, falling back to default implementation")
            else:
                try:
                    # Parse URL and add scheme if missing
                    if not guardrails_url.startswith(('http://', 'https://')):
                        guardrails_url = 'http://' + guardrails_url

                    # Try to connect with a timeout of 5 seconds
                    response = requests.get(guardrails_url + "/v1/health", timeout=5)
                    response.raise_for_status()

                    x_model_authorization = {"X-Model-Authorization": os.environ.get("NGC_API_KEY", "")}
                    return ChatOpenAI(
                        model_name=kwargs.get('model'),
                        openai_api_base=f"{guardrails_url}/v1/guardrail",
                        openai_api_key="dummy-value",
                        default_headers=x_model_authorization,
                        temperature=kwargs.get('temperature', None),
                        top_p=kwargs.get('top_p', None),
                        max_tokens=kwargs.get('max_tokens', None)
                    )
                except (requests.RequestException, requests.ConnectionError) as e:
                    error_msg = f"Failed to connect to guardrails service at {guardrails_url}: {str(e)} Make sure the guardrails service is running and accessible."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        if url:
            logger.debug(f"Length of llm endpoint url string {url}")
            logger.info("Using llm model %s hosted at %s", kwargs.get('model'), url)
            return ChatNVIDIA(base_url=url,
                              model=kwargs.get('model'),
                              temperature=kwargs.get('temperature', None),
                              top_p=kwargs.get('top_p', None),
                              max_tokens=kwargs.get('max_tokens', None))

        logger.info("Using llm model %s from api catalog", kwargs.get('model'))
        return ChatNVIDIA(model=kwargs.get('model'),
                          temperature=kwargs.get('temperature', None),
                          top_p=kwargs.get('top_p', None),
                          max_tokens=kwargs.get('max_tokens', None))

    raise RuntimeError(
        "Unable to find any supported Large Language Model server. Supported engine name is nvidia-ai-endpoints.")


def streaming_filter_think(chunks: Iterable[str]) -> Iterable[str]:
    """
    This generator filters content between think tags in streaming LLM responses.
    It handles both complete tags in a single chunk and tags split across multiple tokens.

    Args:
        chunks (Iterable[str]): Chunks from a streaming LLM response

    Yields:
        str: Filtered content with think blocks removed
    """
    # Complete tags
    FULL_START_TAG = "<think>"
    FULL_END_TAG = "</think>"

    # Multi-token tags - core parts without newlines for more robust matching
    START_TAG_PARTS = ["<th", "ink", ">"]
    END_TAG_PARTS = ["</", "think", ">"]

    # States
    NORMAL = 0
    IN_THINK = 1
    MATCHING_START = 2
    MATCHING_END = 3

    state = NORMAL
    match_position = 0
    buffer = ""
    output_buffer = ""
    chunk_count = 0

    for chunk in chunks:
        content = chunk.content
        chunk_count += 1

        # Let's first check for full tags - this is the most reliable approach
        buffer += content

        # Check for complete tags first - most efficient case
        while state == NORMAL and FULL_START_TAG in buffer:
            start_idx = buffer.find(FULL_START_TAG)
            # Extract content before tag
            before_tag = buffer[:start_idx]
            output_buffer += before_tag

            # Skip over the tag
            buffer = buffer[start_idx + len(FULL_START_TAG):]
            state = IN_THINK

        while state == IN_THINK and FULL_END_TAG in buffer:
            end_idx = buffer.find(FULL_END_TAG)
            # Discard everything up to and including end tag
            buffer = buffer[end_idx + len(FULL_END_TAG):]
            content = buffer
            state = NORMAL

        # For token-by-token matching, use the core content without worrying about exact whitespace
        # Strip whitespace for comparison to make matching more robust
        content_stripped = content.strip()

        if state == NORMAL:
            if content_stripped == START_TAG_PARTS[0].strip():
                # Save everything except this start token
                to_output = buffer[:-len(content)]
                output_buffer += to_output

                buffer = content  # Keep only the start token in buffer
                state = MATCHING_START
                match_position = 1
            else:
                output_buffer += content  # Regular content, save it
                buffer = ""  # Clear buffer, we've processed this chunk

        elif state == MATCHING_START:
            expected_part = START_TAG_PARTS[match_position].strip()
            if content_stripped == expected_part:
                match_position += 1
                if match_position >= len(START_TAG_PARTS):
                    # Complete start tag matched
                    state = IN_THINK
                    match_position = 0
                    buffer = ""  # Clear the buffer
            else:
                # False match, revert to normal and recover the partial match
                state = NORMAL
                output_buffer += buffer  # Recover saved tokens
                buffer = ""

                # Check if this content is a new start tag
                if content_stripped == START_TAG_PARTS[0].strip():
                    state = MATCHING_START
                    match_position = 1
                    buffer = content  # Keep this token in buffer
                else:
                    output_buffer += content  # Regular content

        elif state == IN_THINK:
            if content_stripped == END_TAG_PARTS[0].strip():
                state = MATCHING_END
                match_position = 1
                buffer = content  # Keep this token in buffer
            else:
                buffer = ""  # Discard content inside think block

        elif state == MATCHING_END:
            expected_part = END_TAG_PARTS[match_position].strip()
            if content_stripped == expected_part:
                match_position += 1
                if match_position >= len(END_TAG_PARTS):
                    # Complete end tag matched
                    state = NORMAL
                    match_position = 0
                    buffer = ""  # Clear buffer
            else:
                # False match, revert to IN_THINK
                state = IN_THINK
                buffer = ""  # Discard content

                # Check if this is a new end tag start
                if content_stripped == END_TAG_PARTS[0].strip():
                    state = MATCHING_END
                    match_position = 1
                    buffer = content  # Keep this token in buffer

        # Yield accumulated output before processing next chunk
        if output_buffer:
            yield output_buffer
            output_buffer = ""

    # Yield any remaining content if not in a think block
    if state == NORMAL:
        if buffer:
            yield buffer
        if output_buffer:
            yield output_buffer

    logger.info("Finished streaming_filter_think processing after %d chunks", chunk_count)

def get_streaming_filter_think_parser():
    """
    Creates and returns a RunnableGenerator for filtering think tokens based on configuration.

    If FILTER_THINK_TOKENS environment variable is set to "true" (case-insensitive),
    returns a parser that filters out content between <think> and </think> tags.
    Otherwise, returns a pass-through parser that doesn't modify the content.

    Returns:
        RunnableGenerator: A parser for filtering (or not filtering) think tokens
    """
    from langchain_core.runnables import RunnableGenerator, RunnablePassthrough

    # Check environment variable
    filter_enabled = os.getenv('FILTER_THINK_TOKENS', 'true').lower() == 'true'

    if filter_enabled:
        logger.info("Think token filtering is enabled")
        return RunnableGenerator(streaming_filter_think)
    else:
        logger.info("Think token filtering is disabled")
        # If filtering is disabled, use a passthrough that passes content as-is
        return RunnablePassthrough()
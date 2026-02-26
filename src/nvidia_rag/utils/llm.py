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
3. extract_reasoning_and_content: Extract reasoning and content from response chunks.
4. streaming_filter_think: Filter the think tokens from the LLM response (sync).
5. get_streaming_filter_think_parser: Get the parser for filtering the think tokens (sync).
6. streaming_filter_think_async: Filter the think tokens from the LLM response (async).
7. get_streaming_filter_think_parser_async: Get the parser for filtering the think tokens (async).
"""

import logging
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import requests
import yaml
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.common import (
    NVIDIA_API_DEFAULT_HEADERS,
    combine_dicts,
    sanitize_nim_url,
    utils_cache,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    logger.info("Langchain OpenAI is not installed.")
    pass


def get_prompts(source: str | dict | None = None) -> dict:
    """Retrieves prompt configurations from source or YAML file and return a dict.

    Args:
        source: Optional path to a YAML/JSON file or a dictionary of prompts.
               If None, attempts to load from default locations or PROMPT_CONFIG_FILE env var.
    """

    # default config taking from prompt.yaml
    default_config_path = os.path.join(
        os.environ.get("EXAMPLE_PATH", os.path.dirname(__file__)),
        "..",
        "rag_server",
        "prompt.yaml",
    )
    cur_dir_path = os.path.join(
        os.path.dirname(__file__), "..", "rag_server", "prompt.yaml"
    )
    default_config = {}
    if Path(default_config_path).is_file():
        with open(default_config_path, encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", default_config_path)
            default_config = yaml.safe_load(file)
    elif Path(cur_dir_path).is_file():
        # if prompt.yaml is not found in the default path, check in the current directory(use default config)
        # this is for packaging
        with open(cur_dir_path, encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", cur_dir_path)
            default_config = yaml.safe_load(file)
    else:
        logger.info("No prompts config file found")

    # If source is provided, it takes precedence over environment variable
    config = {}

    if source is not None:
        if isinstance(source, dict):
            config = source
        elif isinstance(source, str) and Path(source).is_file():
            with open(source, encoding="utf-8") as file:
                logger.info("Using prompts config file from: %s", source)
                config = yaml.safe_load(file)
        else:
            logger.warning(f"Invalid source for prompts: {source}. Using defaults.")
    else:
        # Fallback to environment variable if no source provided
        config_file = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")
        if Path(config_file).is_file():
            with open(config_file, encoding="utf-8") as file:
                logger.info("Using prompts config file from: %s", config_file)
                config = yaml.safe_load(file)

    config = combine_dicts(default_config, config)
    return config


def _is_nvidia_endpoint(url: str | None) -> bool:
    """Detect if endpoint is NVIDIA-based using URL patterns."""
    if not url:
        return True  # Empty URL = API catalog or local NIM (default to NVIDIA)

    url_lower = url.lower()
    # Non-NVIDIA endpoints
    if any(
        provider in url_lower
        for provider in ["azure", "openai", "anthropic", "claude"]
    ):
        return False
    # NVIDIA URLs
    if "nvidia" in url_lower or "api.nvidia.com" in url_lower:
        return True
    # Unknown URL pattern - default to NVIDIA (likely local NIM)
    return True


def _is_nemotron_3(model: str | None) -> bool:
    """Detect Nemotron 3 model variants by checking for 'nemotron-3' in the model name."""
    if not model:
        return False
    return "nemotron-3" in model.lower()


def _is_nemotron_3_nano(model: str | None) -> bool:
    """Detect Nemotron 3 Nano models (30b-a3b and locally hosted variants)."""
    if not model:
        return False
    m = model.lower()
    return "nemotron-3-nano" in m


def _is_nemotron_nano_9b_v2(model: str | None) -> bool:
    """Detect legacy Nemotron Nano 9B v2."""
    if not model:
        return False
    return "nvidia/nvidia-nemotron-nano-9b-v2" in model


def _resolve_enable_thinking(config: NvidiaRAGConfig | None = None, **kwargs) -> bool:
    """Resolve enable_thinking from config, kwargs, or deprecated env var fallback."""
    if config is not None:
        enable = config.llm.parameters.enable_thinking
        if enable:
            return True
    enable = kwargs.get("enable_thinking", False)
    if enable:
        return True
    deprecated = os.getenv("ENABLE_NEMOTRON_3_NANO_THINKING")
    if deprecated is not None:
        logger.warning(
            "ENABLE_NEMOTRON_3_NANO_THINKING is deprecated, use LLM_ENABLE_THINKING instead"
        )
        return deprecated.lower() == "true"
    return False


def _bind_reasoning_config(
    llm: LLM | SimpleChatModel, config: NvidiaRAGConfig | None = None, **kwargs
) -> LLM | SimpleChatModel:
    """
    Bind reasoning parameters to the LLM based on model type and configuration.

    Reads enable_thinking, reasoning_budget, and low_effort from the config
    object (LLM_ENABLE_THINKING, LLM_REASONING_BUDGET, LLM_LOW_EFFORT env vars).
    kwargs can still override these for backward compatibility.

    Supports:
    - Nemotron 3 variants: enable_thinking, reasoning_budget, low_effort via chat_template_kwargs
    - Nemotron 3 Nano: enable_thinking + reasoning_budget (or nvext for local NIM)
    - Nemotron Nano 9B v2: legacy min_thinking_tokens / max_thinking_tokens
    - Other models: no reasoning features bound
    """
    model = kwargs.get("model", "")
    enable_thinking = _resolve_enable_thinking(config=config, **kwargs)
    params = config.llm.parameters if config is not None else None
    reasoning_budget = kwargs.get("reasoning_budget") or (params.reasoning_budget if params else 0)
    low_effort = kwargs.get("low_effort") or (params.low_effort if params else False)
    min_think = kwargs.get("min_thinking_tokens") or (params.min_thinking_tokens if params else 0) or 0
    max_think = kwargs.get("max_thinking_tokens") or (params.max_thinking_tokens if params else 0) or 0

    # Check specific variants first, then fall through to the general nemotron-3 check

    if _is_nemotron_3_nano(model):
        llm = llm.bind(chat_template_kwargs={"enable_thinking": enable_thinking})
        if enable_thinking and (reasoning_budget > 0 or max_think > 0):
            budget = reasoning_budget if reasoning_budget > 0 else max_think
            llm_endpoint = kwargs.get("llm_endpoint", "")
            if llm_endpoint:
                llm = llm.bind(nvext={"max_thinking_tokens": budget})
                logger.info("nemotron-3-nano (local): enable_thinking=%s, nvext.max_thinking_tokens=%d", enable_thinking, budget)
            else:
                llm = llm.bind(reasoning_budget=budget)
                logger.info("nemotron-3-nano (API): enable_thinking=%s, reasoning_budget=%d", enable_thinking, budget)
        else:
            logger.info("nemotron-3-nano: enable_thinking=%s", enable_thinking)
        return llm

    if _is_nemotron_nano_9b_v2(model):
        if min_think > 0 and max_think > 0:
            llm = llm.bind(min_thinking_tokens=min_think, max_thinking_tokens=max_think)
            logger.info("nemotron-nano-9b-v2: min_thinking_tokens=%d, max_thinking_tokens=%d", min_think, max_think)
        elif min_think > 0 or max_think > 0:
            raise ValueError(
                "nemotron-nano-9b-v2 requires both min_thinking_tokens and max_thinking_tokens "
                f"to be positive, got min={min_think}, max={max_think}"
            )
        return llm

    if _is_nemotron_3(model):
        template_kwargs: dict = {"enable_thinking": enable_thinking}
        if enable_thinking and low_effort:
            template_kwargs["low_effort"] = True
        llm = llm.bind(chat_template_kwargs=template_kwargs)
        budget = reasoning_budget if reasoning_budget > 0 else max_think
        if enable_thinking and budget > 0:
            llm = llm.bind(reasoning_budget=budget)
        logger.info(
            "nemotron-3: enable_thinking=%s, reasoning_budget=%d, low_effort=%s",
            enable_thinking, budget, low_effort,
        )
        return llm

    return llm


def get_llm(config: NvidiaRAGConfig | None = None, **kwargs) -> LLM | SimpleChatModel:
    """Create the LLM connection.

    Args:
        config: NvidiaRAGConfig instance. If None, creates a new one.
        **kwargs: Additional LLM configuration parameters
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Sanitize the URL
    url = sanitize_nim_url(kwargs.get("llm_endpoint", ""), kwargs.get("model"), "chat")

    # Check if guardrails are enabled
    enable_guardrails = (
        config.enable_guardrails and kwargs.get("enable_guardrails", False) is True
    )

    logger.debug(
        "Using %s as model engine for llm. Model name: %s",
        config.llm.model_engine,
        kwargs.get("model"),
    )
    if config.llm.model_engine == "nvidia-ai-endpoints":
        # Use ChatOpenAI with guardrails if enabled
        # TODO Add the ChatNVIDIA implementation when available
        if enable_guardrails:
            logger.info("Guardrails enabled, using ChatOpenAI with guardrails URL")
            guardrails_url = os.getenv("NEMO_GUARDRAILS_URL", "")
            if not guardrails_url:
                logger.warning(
                    "NEMO_GUARDRAILS_URL not set, falling back to default implementation"
                )
            else:
                try:
                    # Parse URL and add scheme if missing
                    if not guardrails_url.startswith(("http://", "https://")):
                        guardrails_url = "http://" + guardrails_url

                    # Try to connect with a timeout of 5 seconds
                    response = requests.get(guardrails_url + "/v1/health", timeout=5)
                    response.raise_for_status()

                    api_key = kwargs.get("api_key") or config.llm.get_api_key()
                    default_headers = {**NVIDIA_API_DEFAULT_HEADERS}
                    if api_key:
                        default_headers["X-Model-Authorization"] = api_key
                    openai_kwargs = {
                        "model_name": kwargs.get("model"),
                        "openai_api_base": f"{guardrails_url}/v1/guardrail",
                        "openai_api_key": "dummy-value",
                        "default_headers": default_headers,
                        "temperature": kwargs.get("temperature", None),
                        "top_p": kwargs.get("top_p", None),
                        "max_tokens": kwargs.get("max_tokens", None),
                    }
                    if kwargs.get("stop"):
                        openai_kwargs["stop"] = kwargs["stop"]
                    return ChatOpenAI(**openai_kwargs)
                except (requests.RequestException, requests.ConnectionError) as e:
                    error_msg = f"Guardrails NIM unavailable at {guardrails_url}. Please verify the service is running and accessible."
                    logger.exception(
                        "Connection error to guardrails at %s: %s", guardrails_url, e
                    )
                    raise APIError(
                        error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE
                    ) from e

        if url:
            logger.debug(f"Length of llm endpoint url string {url}")
            logger.debug("Using llm model %s hosted at %s", kwargs.get("model"), url)

            api_key = kwargs.get("api_key") or config.llm.get_api_key()
            # Detect endpoint type using URL patterns only
            is_nvidia = _is_nvidia_endpoint(url)

            # Build kwargs dict, only including parameters that are set
            # For non-NVIDIA endpoints, exclude NVIDIA-specific parameters
            # Do not pass stop=[] - some Nemotron 3 APIs reject empty stop arrays
            chat_nvidia_kwargs = {
                "base_url": url,
                "model": kwargs.get("model"),
                "api_key": api_key,
                "default_headers": NVIDIA_API_DEFAULT_HEADERS,
            }
            if kwargs.get("stop"):
                chat_nvidia_kwargs["stop"] = kwargs["stop"]
            if kwargs.get("temperature") is not None:
                chat_nvidia_kwargs["temperature"] = kwargs["temperature"]
            if kwargs.get("top_p") is not None:
                chat_nvidia_kwargs["top_p"] = kwargs["top_p"]
            if kwargs.get("max_tokens") is not None:
                chat_nvidia_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            # Only include NVIDIA-specific parameters for NVIDIA endpoints
            if is_nvidia:
                model_kwargs = {}
                if kwargs.get("min_tokens") is not None:
                    model_kwargs["min_tokens"] = kwargs["min_tokens"]
                if kwargs.get("ignore_eos") is not None:
                    model_kwargs["ignore_eos"] = kwargs["ignore_eos"]
                if model_kwargs:
                    chat_nvidia_kwargs["model_kwargs"] = model_kwargs

            llm = ChatNVIDIA(**chat_nvidia_kwargs)
            if is_nvidia:
                llm = _bind_reasoning_config(llm, config=config, **kwargs)
            return llm

        logger.debug("Using llm model %s from api catalog", kwargs.get("model"))

        api_key = kwargs.get("api_key") or config.llm.get_api_key()

        model_kwargs = {}
        if kwargs.get("min_tokens") is not None:
            model_kwargs["min_tokens"] = kwargs["min_tokens"]
        if kwargs.get("ignore_eos") is not None:
            model_kwargs["ignore_eos"] = kwargs["ignore_eos"]

        # Do not pass stop=[] - some Nemotron 3 APIs reject empty stop arrays
        chat_nvidia_kwargs = {
            "model": kwargs.get("model"),
            "api_key": api_key,
            "temperature": kwargs.get("temperature", None),
            "top_p": kwargs.get("top_p", None),
            "max_completion_tokens": kwargs.get("max_tokens", None),
            "default_headers": NVIDIA_API_DEFAULT_HEADERS,
            **({"model_kwargs": model_kwargs} if model_kwargs else {}),
        }
        if kwargs.get("stop"):
            chat_nvidia_kwargs["stop"] = kwargs["stop"]
        llm = ChatNVIDIA(**chat_nvidia_kwargs)
        llm = _bind_reasoning_config(llm, config=config, **kwargs)
        return llm

    raise RuntimeError(
        "Unable to find any supported Large Language Model server. Supported engine name is nvidia-ai-endpoints."
    )


def extract_reasoning_and_content(chunk) -> tuple[str, str]:
    """
    Extract both reasoning and content from a response chunk.
    
    Different models handle reasoning differently:
    - nvidia/nvidia-nemotron-nano-9b-v2: Uses <think> tags in content stream
    - nemotron-3-nano variants: Uses separate reasoning_content field
    - llama-3.3-nemotron-super-49b: Uses <think> tags in content stream (controlled by prompt)
    
    This function is designed to be robust and compatible with future changes:
    - Checks both reasoning_content and content fields
    - Returns whichever field has tokens, regardless of model behavior
    - If both have content, returns both separately
    
    This ensures that if the model server fixes the issue where reasoning is disabled
    but content still goes to reasoning_content, the code will still work correctly.
    
    Args:
        chunk: A response chunk from ChatNVIDIA or similar LLM interface
    
    Returns:
        tuple: (reasoning_text, content_text) - either may be empty string
        
    Example:
        >>> for chunk in llm.stream([HumanMessage(content="question")]):
        >>>     reasoning, content = extract_reasoning_and_content(chunk)
        >>>     if reasoning:
        >>>         print(f"[REASONING: {reasoning}]", end="", flush=True)
        >>>     if content:
        >>>         print(content, end="", flush=True)
    """
    reasoning = ""
    content = ""
    
    # Check for reasoning_content in additional_kwargs (nemotron-3-nano variants)
    # This field is populated by nemotron-3-nano models for reasoning output
    if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
        reasoning = chunk.additional_kwargs.get('reasoning_content', '')
    
    # Check for regular content
    # This field is populated by most models for regular output
    # For nemotron-nano-9b-v2 and llama-49b, this may include <think> tags
    if hasattr(chunk, 'content') and chunk.content:
        content = chunk.content
    
    # Robust fallback: If reasoning field has content but content field is empty,
    # treat reasoning as content. This handles the case where enable_thinking=false
    # but the model still populates reasoning_content instead of content.
    # This makes the code compatible with future fixes to the model server.
    if reasoning and not content:
        # If only reasoning has content, it might actually be the final response
        # (occurs when enable_thinking=false but model hasn't been updated)
        # Keep it in reasoning field but also check if it looks like a final answer
        pass  # Keep as-is, let the caller decide how to handle
    
    return reasoning, content


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
            buffer = buffer[start_idx + len(FULL_START_TAG) :]
            state = IN_THINK

        while state == IN_THINK and FULL_END_TAG in buffer:
            end_idx = buffer.find(FULL_END_TAG)
            # Discard everything up to and including end tag
            buffer = buffer[end_idx + len(FULL_END_TAG) :]
            content = buffer
            state = NORMAL

        # For token-by-token matching, use the core content without worrying about exact whitespace
        # Strip whitespace for comparison to make matching more robust
        content_stripped = content.strip()

        if state == NORMAL:
            if content_stripped == START_TAG_PARTS[0].strip():
                # Save everything except this start token
                to_output = buffer[: -len(content)]
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

    logger.info(
        "Finished streaming_filter_think processing after %d chunks", chunk_count
    )


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
    filter_enabled = os.getenv("FILTER_THINK_TOKENS", "true").lower() == "true"

    if filter_enabled:
        logger.info("Think token filtering is enabled")
        return RunnableGenerator(streaming_filter_think)
    else:
        logger.info("Think token filtering is disabled")
        # If filtering is disabled, use a passthrough that passes content as-is
        return RunnablePassthrough()


async def streaming_filter_think_async(chunks):
    """
    Async version of streaming_filter_think.
    This async generator filters content between think tags in streaming LLM responses.
    It handles both complete tags in a single chunk and tags split across multiple tokens.

    Args:
        chunks: Async iterable of chunks from a streaming LLM response

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

    async for chunk in chunks:
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
            buffer = buffer[start_idx + len(FULL_START_TAG) :]
            state = IN_THINK

        while state == IN_THINK and FULL_END_TAG in buffer:
            end_idx = buffer.find(FULL_END_TAG)
            # Discard everything up to and including end tag
            buffer = buffer[end_idx + len(FULL_END_TAG) :]
            content = buffer
            state = NORMAL

        # For token-by-token matching, use the core content without worrying about exact whitespace
        # Strip whitespace for comparison to make matching more robust
        content_stripped = content.strip()

        if state == NORMAL:
            if content_stripped == START_TAG_PARTS[0].strip():
                # Save everything except this start token
                to_output = buffer[: -len(content)]
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

    logger.info(
        "Finished streaming_filter_think_async processing after %d chunks", chunk_count
    )


def get_streaming_filter_think_parser_async():
    """
    Creates and returns an async RunnableGenerator for filtering think tokens.

    If FILTER_THINK_TOKENS environment variable is set to "true" (case-insensitive),
    returns a parser that filters out content between <think> and </think> tags.
    Otherwise, returns a pass-through parser that doesn't modify the content.

    Returns:
        RunnableGenerator: An async parser for filtering (or not filtering) think tokens
    """
    from langchain_core.runnables import RunnableGenerator, RunnablePassthrough

    # Check environment variable
    filter_enabled = os.getenv("FILTER_THINK_TOKENS", "true").lower() == "true"

    if filter_enabled:
        logger.info("Think token filtering is enabled (async)")
        return RunnableGenerator(streaming_filter_think_async)
    else:
        logger.info("Think token filtering is disabled (async)")
        # If filtering is disabled, use a passthrough that passes content as-is
        return RunnablePassthrough()
        
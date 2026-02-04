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

"""LLM Chain module for direct LLM/VLM generation without knowledge base.

This module provides functionality for executing simple LLM or VLM chains
when the `/generate` API is invoked with `use_knowledge_base` set to `False`.

The main function handles:
- Text-only queries using LLM
- Multimodal queries using VLM (when enabled)
- Conversation history management
- Prompt template processing
- Error handling and retries
- Streaming response generation

Public functions:
1. llm_chain(): Execute a simple LLM/VLM chain without knowledge base context (async).
2. vlm_direct_chain(): Execute a VLM chain without knowledge base context (async).
"""

import logging
import os
from traceback import print_exc
from typing import Any

import requests
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from requests import ConnectTimeout

from nvidia_rag.rag_server.response_generator import (
    APIError,
    ErrorCodeMapping,
    RAGResponse,
    generate_answer_async,
)
from nvidia_rag.rag_server.message_processor import handle_prompt_processing
from nvidia_rag.rag_server.content_utils import (
    _extract_text_from_content,
    _contains_images,
)
from nvidia_rag.rag_server.document_formatter import _print_conversation_history
from nvidia_rag.rag_server.stream_utils import (
    _async_iter,
    _eager_prefetch_astream,
)
from nvidia_rag.rag_server.vlm import VLM
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import (
    get_llm,
    get_streaming_filter_think_parser_async,
)
from nvidia_rag.utils.observability.otel_metrics import OtelMetrics

logger = logging.getLogger(__name__)


async def vlm_direct_chain(
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    query: str | list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    model: str = "",
    enable_citations: bool = True,
    metrics: OtelMetrics | None = None,
    vlm_settings: dict[str, Any] | None = None,
) -> RAGResponse:
    """Execute a VLM chain without knowledge base context.

    Used when enable_vlm_inference=True and use_knowledge_base=False.

    Args:
        config: Configuration object containing model and service settings
        prompts: Dictionary containing prompt templates
        query: The user's query (can be text or multimodal with images)
        chat_history: List of conversation messages
        model: Name of the model used for generation (for response metadata)
        enable_citations: Whether to enable citations in the response
        metrics: OpenTelemetry metrics client
        vlm_settings: Dictionary containing VLM settings

    Returns:
        RAGResponse: Streaming response from VLM
    """
    try:
        # Initialize vlm_settings if not provided
        vlm_settings = vlm_settings or {}

        # Limit conversation history to prevent overwhelming the model
        conversation_history_count = int(os.environ.get("CONVERSATION_HISTORY", 0))
        if conversation_history_count == 0:
            chat_history = []
        else:
            history_count = conversation_history_count * 2 * -1
            chat_history = chat_history[history_count:]

        # Resolve VLM settings from dict or config defaults
        vlm_model_cfg = vlm_settings.get("vlm_model") or config.vlm.model_name
        vlm_endpoint_cfg = vlm_settings.get("vlm_endpoint") or config.vlm.server_url
        vlm_temperature_cfg = vlm_settings.get("vlm_temperature") or config.vlm.temperature
        vlm_top_p_cfg = vlm_settings.get("vlm_top_p") or config.vlm.top_p
        vlm_max_tokens_cfg = vlm_settings.get("vlm_max_tokens") or config.vlm.max_tokens
        vlm_max_total_images_cfg = (
            vlm_settings.get("vlm_max_total_images") or config.vlm.max_total_images
        )

        # Extract text from query for logging
        query_text = _extract_text_from_content(query)
        has_images = _contains_images(query)

        logger.info("=" * 80)
        logger.info("STAGE: VLM Generation (Direct - no knowledge base)")
        logger.info("=" * 80)
        logger.info("VLM Configuration:")
        logger.info("  - Model: %s", vlm_model_cfg)
        logger.info("  - Endpoint: %s", vlm_endpoint_cfg)
        logger.info("  - Temperature: %s, Top-P: %s, Max Tokens: %s",
                   vlm_temperature_cfg, vlm_top_p_cfg, vlm_max_tokens_cfg)
        logger.info("  - Max Total Images: %s", vlm_max_total_images_cfg)
        logger.info("Input:")
        logger.info("  - Query: '%s'", query_text[:200] if query_text else "")
        logger.info("  - Has Images in Query: %s", has_images)
        logger.info("  - Chat History Messages: %d", len(chat_history) if chat_history else 0)
        logger.info("Starting VLM stream generation...")
        logger.info("-" * 80)

        vlm = VLM(
            vlm_model=vlm_model_cfg,
            vlm_endpoint=vlm_endpoint_cfg,
            config=config,
            prompts=prompts,
        )

        # Build full messages: prior history + current query as a final user turn
        vlm_messages = [
            *(chat_history or []),
            {"role": "user", "content": query},
        ]

        # Stream VLM response (no context documents in direct mode)
        vlm_generator = vlm.stream_with_messages(
            docs=[],  # No context documents
            messages=vlm_messages,
            context_text="",  # No retrieved context
            question_text=query_text,
            temperature=vlm_temperature_cfg,
            top_p=vlm_top_p_cfg,
            max_tokens=vlm_max_tokens_cfg,
            max_total_images=vlm_max_total_images_cfg,
        )

        # Eagerly prefetch first chunk to catch errors early
        prefetched_vlm_stream = await _eager_prefetch_astream(vlm_generator)

        logger.info("VLM stream initiated successfully (first chunk received)")
        logger.info("-" * 80)

        return RAGResponse(
            generate_answer_async(
                prefetched_vlm_stream,
                [],  # No context docs
                model=vlm_model_cfg,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.SUCCESS,
        )

    except APIError as e:
        # Catch APIError from VLM (raised during eager prefetch)
        logger.warning("APIError from VLM in vlm_direct_chain: %s", e.message)
        return RAGResponse(
            generate_answer_async(
                _async_iter([e.message]),
                [],
                model=model,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=e.status_code,
        )

    except (OSError, ValueError, ConnectionError) as e:
        vlm_url = vlm_settings.get("vlm_endpoint") if vlm_settings else None
        vlm_url = vlm_url or config.vlm.server_url
        error_msg = f"VLM NIM unavailable at {vlm_url}. Please verify the service is running and accessible."
        logger.exception("Connection error in VLM direct chain: %s", e)
        return RAGResponse(
            generate_answer_async(
                _async_iter([error_msg]),
                [],
                model=model,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
        )

    except Exception as e:
        error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
        logger.warning("Failed to generate VLM response: %s", error_msg)

        if "[403] Forbidden" in str(e):
            return RAGResponse(
                generate_answer_async(
                    _async_iter(
                        ["Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."]
                    ),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.FORBIDDEN,
            )
        elif "[404] Not Found" in str(e):
            vlm_model = vlm_settings.get("vlm_model") if vlm_settings else None
            vlm_model = vlm_model or config.vlm.model_name
            error_msg = f"VLM model '{vlm_model}' not found. Please verify the VLM model name and ensure it's available."
            logger.warning("VLM model not found: %s", error_msg)
            return RAGResponse(
                generate_answer_async(
                    _async_iter([error_msg]),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.NOT_FOUND,
            )
        else:
            return RAGResponse(
                generate_answer_async(
                    _async_iter([str(e)]),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.BAD_REQUEST,
            )


async def llm_chain(
    config: NvidiaRAGConfig,
    prompts: dict[str, Any],
    llm_settings: dict[str, Any],
    query: str | list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    model: str = "",
    enable_citations: bool = True,
    metrics: OtelMetrics | None = None,
    enable_vlm_inference: bool = False,
    vlm_settings: dict[str, Any] | None = None,
) -> RAGResponse:
    """Execute a simple LLM/VLM chain without knowledge base context.

    This function is called when the `/generate` API is invoked with
    `use_knowledge_base` set to `False`.

    Args:
        config: Configuration object containing model and service settings
        prompts: Dictionary containing prompt templates
        llm_settings: Dictionary containing LLM settings (endpoint, temperature, etc.)
        query: The user's query (can be text or multimodal with images)
        chat_history: List of conversation messages
        model: Name of the model used for generation
        enable_citations: Whether to enable citations in the response
        metrics: OpenTelemetry metrics client
        enable_vlm_inference: Whether to use VLM instead of LLM for generation
        vlm_settings: Dictionary containing VLM settings (model, endpoint, temperature, etc.)

    Returns:
        RAGResponse: Streaming response with status code

    Raises:
        APIError: If images are present in the query but VLM inference is not enabled.
    """
    # Check for images in query
    has_images_in_query = _contains_images(query)

    # Decision logic: VLM vs LLM vs Error
    # 1. If enable_vlm_inference=True -> Use VLM (with or without images)
    # 2. If has_images but VLM not enabled -> Error
    # 3. Otherwise -> Use LLM
    if enable_vlm_inference:
        # Use VLM for generation (works with or without images)
        return await vlm_direct_chain(
            config=config,
            prompts=prompts,
            query=query,
            chat_history=chat_history,
            model=model,
            enable_citations=enable_citations,
            metrics=metrics,
            vlm_settings=vlm_settings,
        )
    elif has_images_in_query:
        error_message = (
            "Visual Q&A is not supported without VLM inference enabled. "
            "Image-based queries require 'enable_vlm_inference' to be True. "
            "Please enable VLM inference to use visual Q&A features."
        )
        logger.warning(
            "Image detected in query with enable_vlm_inference=False. "
            "Returning error: %s",
            error_message,
        )
        raise APIError(error_message, ErrorCodeMapping.BAD_REQUEST)

    # LLM path (text-only, no VLM)
    try:
        # Limit conversation history to prevent overwhelming the model
        # conversation is tuple so it should be multiple of two
        # -1 is to keep last k conversation
        conversation_history_count = int(os.environ.get("CONVERSATION_HISTORY", 0))
        if conversation_history_count == 0:
            chat_history = []
        else:
            history_count = conversation_history_count * 2 * -1
            chat_history = chat_history[history_count:]

        # Use the new prompt processing method
        (
            system_message,
            conversation_history,
            user_message,
        ) = handle_prompt_processing(chat_history, model, prompts, "chat_template")

        logger.debug("System message: %s", system_message)
        logger.debug("User message: %s", user_message)
        logger.debug("Conversation history: %s", conversation_history)
        # Prompt template with system message, user message from prompt template
        message = system_message + user_message

        # If conversation history exists, add it as formatted message
        if conversation_history:
            # Format conversation history
            formatted_history = "\n".join(
                [
                    f"{role.title()}: {content}"
                    for role, content in conversation_history
                ]
            )
            message += [("user", f"Conversation history:\n{formatted_history}")]

        # Add user query to prompt
        user_query = []
        # Extract text from query for processing
        query_text = _extract_text_from_content(query)
        logger.info("Query is: %s", query_text)
        if query_text is not None and query_text != "":
            user_query += [("user", "Query: {question}")]

        # Add user query
        message += user_query

        _print_conversation_history(message, query_text)

        prompt_template = ChatPromptTemplate.from_messages(message)
        llm = get_llm(config=config, **llm_settings)

        logger.info("=" * 80)
        logger.info("STAGE: LLM Generation (Direct)")
        logger.info("=" * 80)
        logger.info("LLM Configuration:")
        logger.info("  - Model: %s", model)
        llm_endpoint_display = llm_settings.get("llm_endpoint") or "api catalog"
        logger.info("  - Endpoint: %s", llm_endpoint_display)
        logger.info("  - Temperature: %s, Top-P: %s, Max Tokens: %s",
                   llm_settings.get("temperature"),
                   llm_settings.get("top_p"),
                   llm_settings.get("max_tokens"))
        logger.info("Input:")
        logger.info("  - Query: '%s'", query_text[:200] if query_text else "")
        logger.info("Starting LLM stream generation...")
        logger.info("-" * 80)

        # Get streaming filter parser
        streaming_filter_think_parser = get_streaming_filter_think_parser_async()

        chain = (
            prompt_template
            | llm
            | streaming_filter_think_parser
            | StrOutputParser()
        )
        # Create async stream generator
        stream_gen = chain.astream(
            {"question": query_text}, config={"run_name": "llm-stream"}
        )
        # Eagerly fetch first chunk to trigger any errors before returning response
        prefetched_stream = await _eager_prefetch_astream(stream_gen)

        logger.info("LLM stream initiated successfully (first chunk received)")
        logger.info("-" * 80)

        return RAGResponse(
            generate_answer_async(
                prefetched_stream,
                [],
                model=model,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.SUCCESS,
        )
    except ConnectTimeout as e:
        logger.warning(
            "Connection timed out while making a request to the LLM endpoint: %s", e
        )
        return RAGResponse(
            generate_answer_async(
                _async_iter(
                    [
                        "Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."
                    ]
                ),
                [],
                model=model,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.REQUEST_TIMEOUT,
        )

    except (requests.exceptions.ConnectionError, ConnectionError, OSError):
        # Fallback for uncaught LLM connection errors
        llm_url = llm_settings.get("llm_endpoint") or config.llm.server_url
        error_msg = f"LLM NIM unavailable at {llm_url}. Please verify the service is running and accessible."
        logger.exception("Connection error (LLM)")
        return RAGResponse(
            generate_answer_async(
                _async_iter([error_msg]),
                [],
                model=model,
                collection_name="",
                enable_citations=enable_citations,
                otel_metrics_client=metrics,
            ),
            status_code=ErrorCodeMapping.SERVICE_UNAVAILABLE,
        )

    except Exception as e:
        # Extract just the error type and message for cleaner logs
        error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
        logger.warning(
            "Failed to generate response due to exception: %s", error_msg
        )

        # Only show full traceback at DEBUG level
        if logger.getEffectiveLevel() <= logging.DEBUG:
            print_exc()

        if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
            logger.warning(
                "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
            )
            return RAGResponse(
                generate_answer_async(
                    _async_iter(
                        [
                            "Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."
                        ]
                    ),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.FORBIDDEN,
            )
        elif "[404] Not Found" in str(e):
            # Check if this is a VLM-related error
            error_msg = "Model or endpoint not found. Please verify the API endpoint and your payload. Ensure that the model name is valid."
            logger.warning(f"Model not found: {error_msg}")

            return RAGResponse(
                generate_answer_async(
                    _async_iter([error_msg]),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.NOT_FOUND,
            )
        else:
            return RAGResponse(
                generate_answer_async(
                    _async_iter([str(e)]),
                    [],
                    model=model,
                    collection_name="",
                    enable_citations=enable_citations,
                    otel_metrics_client=metrics,
                ),
                status_code=ErrorCodeMapping.BAD_REQUEST,
            )

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
This module defines the VLM (Vision-Language Model) utilities for NVIDIA RAG pipelines.

Main functionalities:
- Analyze images using a VLM given user messages and full chat/context.
- Use an LLM to reason about the VLM's response and decide if it should be used.

Intended for use in NVIDIA's Retrieval-Augmented Generation (RAG) systems, compatible with LangChain and OpenAI-compatible VLM APIs.

Class:
    VLM: Provides methods for image analysis via messages and VLM/LLM reasoning.
"""

import base64
import io
import os
import re
from collections.abc import AsyncGenerator
from logging import getLogger
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image as PILImage

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import get_prompts
from nvidia_rag.utils.minio_operator import (
    get_minio_operator,
    get_unique_thumbnail_id,
)

logger = getLogger(__name__)


class VLM:
    """
    Handles image analysis using a Vision-Language Model (VLM).

    Image handling and limits
    -------------------------
    Images can come from:
    - User messages (multimodal ``content`` items with ``type == "image_url"``)
    - Retrieved context documents (thumbnails loaded from object storage)

    The effective image budget is controlled by ``max_total_images``:

    - ``None``: no explicit upper bound is enforced by this helper.
    - Integer ``N > 0``: hard cap on the **total** images (user + context)
      that will be included in the final VLM prompt.
    - ``0``: prevents **additional** images from being added from retrieved
      documents; user-supplied images already present in the messages are
      passed through unchanged.

    The limit is applied when assembling the LangChain messages, after
    normalizing incoming messages and skipping non-image document types.

    Methods
    -------
    analyze_with_messages(docs, messages, context_text, question_text):
        Build a VLM prompt similar to RAG chain prompts and analyze images (async).
    stream_with_messages(docs, messages, context_text, question_text):
        Stream VLM tokens for a multimodal conversation plus context (async generator).
    """

    def __init__(
        self,
        vlm_model: str,
        vlm_endpoint: str,
        config: NvidiaRAGConfig | None = None,
        prompts: dict | None = None,
    ):
        """
        Initialize the VLM with configuration and prompt templates.

        Args:
            vlm_model:
                VLM model name.
            vlm_endpoint:
                VLM server endpoint URL.
            config:
                NvidiaRAGConfig instance. If None, creates a new one.
            prompts:
                Optional prompts dictionary.

        Image budget semantics
        ----------------------
        The image budget is read from ``config.vlm.max_total_images`` and
        interpreted as:

        - ``None``: no explicit limit applied by this helper.
        - Integer ``N > 0``: at most ``N`` images (combined from user messages
          and retrieved context) are included in the VLM prompt.
        - ``0``: no **additional** images are taken from retrieved documents;
          any images already present in user/chat messages are left intact.

        This limit is enforced while building the prompt messages, after
        normalizing message content and skipping non-image document types.

        Raises
        ------
        EnvironmentError
            If VLM server URL or model name is not set in the environment.
        """
        if config is None:
            config = NvidiaRAGConfig()

        self.config = config
        self.invoke_url = vlm_endpoint
        self.model_name = vlm_model
        # Default VLM generation settings from configuration; can be overridden per call
        self.temperature: float = self.config.vlm.temperature
        self.top_p: float = self.config.vlm.top_p
        self.max_tokens: int = self.config.vlm.max_tokens
        self.max_total_images: int | None = self.config.vlm.max_total_images
        if not self.invoke_url or not self.model_name:
            raise OSError(
                "VLM server URL and model name must be set in the environment."
            )
        prompts = prompts or get_prompts()
        self.vlm_template = prompts["vlm_template"]
        logger.info(f"VLM Model Name: {self.model_name}")
        logger.info(f"VLM Server URL: {self.invoke_url}")

    @staticmethod
    def init_model(
        model: str, endpoint: str, api_key: str | None = None, **kwargs: Any
    ) -> ChatOpenAI:
        """
        Initialize and return the VLM ChatOpenAI model instance.

        Note
        ----
        This helper does **not** apply any image-count limits itself; those
        limits are enforced earlier when assembling the messages that will be
        sent to the model.
        """
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=endpoint,
            **kwargs,
        )

    @staticmethod
    def _normalize_messages(
        raw_messages: list[dict[str, Any]],
    ) -> tuple[list[HumanMessage | AIMessage | SystemMessage], int, str]:
        """Normalize raw messages; return (messages_without_system, last_human_idx, incoming_system_text)."""
        lc_messages: list[HumanMessage | AIMessage | SystemMessage] = []
        last_human_idx: int | None = None
        system_accum_text: str = ""

        def ensure_list_content(raw_content: Any) -> list[dict[str, Any]]:
            if isinstance(raw_content, str):
                return [{"type": "text", "text": raw_content}]
            if isinstance(raw_content, list):
                normalized: list[dict[str, Any]] = []
                for item in raw_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            normalized.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image_url":
                            url = (item.get("image_url") or {}).get("url", "")
                            if url:
                                # ensure images are PNG base64 data URLs
                                png_b64 = VLM._convert_image_url_to_png_b64(url)
                                normalized.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{png_b64}"
                                        },
                                    }
                                )
                    else:
                        # Fallback: treat non-dict items as plain text
                        normalized.append({"type": "text", "text": str(item)})
                return normalized
            return [
                {
                    "type": "text",
                    "text": str(raw_content) if raw_content is not None else "",
                }
            ]

        for m in raw_messages or []:
            role = (m or {}).get("role", "").strip()
            content = ensure_list_content((m or {}).get("content"))
            if role == "system":
                # Accumulate any incoming system text; do not add as a separate message
                system_text = "".join(
                    [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                        and part.get("text")
                    ]
                )
                if system_text:
                    system_accum_text = (system_accum_text + " " + system_text).strip()
            elif role == "assistant":
                # Assistant content should be a plain string
                assistant_text = "".join(
                    [
                        (
                            part.get("text", "")
                            if isinstance(part, dict) and part.get("type") == "text"
                            else str(part)
                        )
                        for part in content
                    ]
                )
                lc_messages.append(AIMessage(content=assistant_text))
            else:
                # User content can be multimodal list
                lc_messages.append(HumanMessage(content=content))
                last_human_idx = len(lc_messages) - 1

        if last_human_idx is None:
            lc_messages.append(HumanMessage(content=[{"type": "text", "text": ""}]))
            last_human_idx = len(lc_messages) - 1

        return lc_messages, last_human_idx, system_accum_text

    def extract_and_process_messages(
        self,
        vlm_template: dict[str, Any],
        docs: list[Any],
        incoming_messages: list[dict[str, Any]] | None,
        context_text: str | None,
        question_text: str | None,
        max_total_images: int | None = None,
    ) -> tuple[
        SystemMessage, HumanMessage, list[HumanMessage | AIMessage | SystemMessage]
    ]:
        """
        Build system and user messages from template, normalize chat history, and
        extract any query/context images to be attached to the last human message.
        Returns: (system_message, chat_history_messages, user_message, last_human_idx, query_images, context_images)
        """
        textual_context = (
            context_text if context_text is not None else self._format_docs_text(docs)
        )

        # Normalize chat history; keep images inline as image_url parts and collect incoming system text
        chat_history_messages, _, incoming_system_text = self._normalize_messages(
            incoming_messages or []
        )

        # Build system + citations instruction/user prompt
        system_text = (vlm_template.get("system") or "").strip()
        if incoming_system_text:
            system_text = (system_text + " " + incoming_system_text).strip()
        human_template = vlm_template.get("human") or "{context}\n\n{question}"
        formatted_human = human_template.format(
            context=textual_context or "",
            question=(question_text or "").strip(),
        )
        # System must be plain string, not list content
        system_message = SystemMessage(content=system_text)

        # Build citations_instruct_user_message as multimodal: text + citation images from MinIO (if any)
        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": formatted_human}
        ]

        # Count images already present in chat history to respect overall image budget
        existing_image_count = 0
        try:
            for msg in chat_history_messages:
                parts = msg.content if isinstance(msg.content, list) else []
                for p in parts:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        existing_image_count += 1
        except Exception:
            existing_image_count = 0

        remaining_image_budget = None
        if isinstance(max_total_images, int) and max_total_images >= 0:
            remaining_image_budget = max(0, max_total_images - existing_image_count)

        try:
            for doc in docs or []:
                metadata = getattr(doc, "metadata", {}) or {}
                content_md = metadata.get("content_metadata", {}) or {}
                doc_type = content_md.get("type")
                if doc_type not in ["image", "structured"]:
                    continue

                if remaining_image_budget is not None and remaining_image_budget <= 0:
                    break

                # Inputs for unique thumbnail id
                collection_name = metadata.get("collection_name") or ""
                source_meta = metadata.get("source", {}) or {}
                source_id = (
                    source_meta.get("source_id", "")
                    if isinstance(source_meta, dict)
                    else ""
                )
                file_name = os.path.basename(source_id) if source_id else ""
                page_number = content_md.get("page_number")
                location = content_md.get("location")

                if not (
                    collection_name
                    and file_name
                    and page_number is not None
                    and location is not None
                ):
                    continue

                try:
                    unique_thumbnail_id = get_unique_thumbnail_id(
                        collection_name=collection_name,
                        file_name=file_name,
                        page_number=page_number,
                        location=location,
                    )
                    payload = get_minio_operator().get_payload(
                        object_name=unique_thumbnail_id
                    )
                    content_b64 = (payload or {}).get("content", "")
                    if not content_b64:
                        continue
                    png_b64 = VLM._convert_image_url_to_png_b64(content_b64)
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        }
                    )
                    if remaining_image_budget is not None:
                        remaining_image_budget -= 1
                except Exception:
                    # best-effort: skip if anything fails per document
                    continue
        except Exception:
            # best-effort overall
            pass

        citations_instruct_user_message = HumanMessage(content=content_parts)

        # Return citations_instruct_user_message as second element by contract
        return (system_message, citations_instruct_user_message, chat_history_messages)

    @staticmethod
    def assemble_messages(
        system_message: SystemMessage,
        citations_instruct_user_message: HumanMessage,
        chat_history_messages: list[HumanMessage | AIMessage | SystemMessage],
    ) -> list[HumanMessage | AIMessage | SystemMessage]:
        """Assemble final message list as [system] + [citations user] + chat history."""
        return [system_message, citations_instruct_user_message, *chat_history_messages]

    @staticmethod
    async def invoke_model_async(
        model: ChatOpenAI,
        messages: list[HumanMessage | AIMessage | SystemMessage],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        """Invoke the VLM model asynchronously and return the complete response string."""
        logger.info(
            f"Invoking VLM async with temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}"
        )
        result = await model.ainvoke(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        return result.content.strip()

    @staticmethod
    def invoke_model(
        model: ChatOpenAI,
        messages: list[HumanMessage | AIMessage | SystemMessage],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        """Invoke the VLM model and return the complete response string."""
        logger.info(
            f"Invoking VLM with temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}"
        )
        return model.invoke(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        ).content.strip()

    @staticmethod
    def _convert_image_url_to_png_b64(image_url: str) -> str:
        """
        Convert an image URL (data URL or base64 string) to PNG format base64.

        Parameters
        ----------
        image_url : str
            Image URL in data URL format or base64 string

        Returns
        -------
        str
            Base64-encoded PNG image string
        """
        try:
            # Handle data URL format (e.g., "data:image/jpeg;base64,/9j/4AAQ...")
            if image_url.startswith("data:image/"):
                # Extract base64 data from data URL
                match = re.match(r"data:image/[^;]+;base64,(.+)", image_url)
                if match:
                    b64_data = match.group(1)
                else:
                    logger.warning(f"Invalid data URL format: {image_url[:100]}...")
                    return image_url
            else:
                # Assume it's already a base64 string
                b64_data = image_url

            # Decode base64 to bytes
            image_bytes = base64.b64decode(b64_data)

            # Open image with PIL and convert to RGB (in case it's RGBA or other format)
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

            # Convert to PNG format
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                png_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.debug("Successfully converted image to PNG format")
            return png_b64

        except Exception as e:
            logger.warning(f"Failed to convert image URL to PNG: {e}")
            # Return original if conversion fails
            return image_url

    def _redact_messages_for_logging(
        self, messages: list[HumanMessage | AIMessage | SystemMessage]
    ) -> list[dict[str, Any]]:
        """
        Create a redacted, log-safe representation of the messages where any
        Base64 image data in data URLs is removed.
        """
        safe: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                role = "user"

            raw_content = m.content
            parts = (
                raw_content
                if isinstance(raw_content, list)
                else [{"type": "text", "text": str(raw_content)}]
            )

            safe_parts: list[dict[str, Any]] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if (
                        isinstance(url, str)
                        and url.startswith("data:image/")
                        and ";base64," in url
                    ):
                        redacted_url = re.sub(
                            r"^data:image/[^;]+;base64,.*$",
                            "data:image/png;base64,[REDACTED]",
                            url,
                        )
                    else:
                        redacted_url = url
                    safe_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": redacted_url},
                        }
                    )
                elif isinstance(p, dict) and p.get("type") == "text":
                    safe_parts.append({"type": "text", "text": "[TEXT REDACTED]"})
                else:
                    safe_parts.append({"type": "text", "text": str(p)})

            safe.append({"role": role, "content": safe_parts})
        return safe

    def _format_docs_text(self, docs: list[Any]) -> str:
        """
        Build a textual context string from retrieved docs, skipping image/structured types
        because those are passed as images to the VLM.
        """
        parts: list[str] = []
        for doc in docs or []:
            try:
                metadata = getattr(doc, "metadata", {}) or {}
                content_md = metadata.get("content_metadata", {}) or {}
                doc_type = content_md.get("type")
                if doc_type in ["image", "structured"]:
                    # will be sent as image
                    continue
                # filename from nested source
                source = metadata.get("source", {})
                source_path = (
                    source.get("source_name", "")
                    if isinstance(source, dict)
                    else source
                )
                filename = (
                    os.path.splitext(os.path.basename(source_path))[0]
                    if source_path
                    else ""
                )
                header = f"File: {filename}\n" if filename else ""
                content = getattr(doc, "page_content", "")
                if content:
                    parts.append(f"{header}Content: {content}")
            except Exception:
                # best-effort
                content = getattr(doc, "page_content", "")
                if content:
                    parts.append(content)
        return "\n\n".join(parts)

    async def analyze_with_messages(
        self,
        docs: list[Any],
        messages: list[dict[str, Any]],
        context_text: str | None = None,
        question_text: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_total_images: int | None = None,
        **_: Any,
    ) -> str:
        """
        Send the full conversation messages to the VLM asynchronously, appending any relevant images
        from user messages and retrieved context. Ensures images are provided as
        base64 PNG data URLs.

        Parameters
        ----------
        docs : List[Any]
            Retrieved documents that may contain image thumbnails in storage.
            Each item is expected to behave like a LangChain ``Document``
            (i.e., exposing ``page_content`` and ``metadata`` attributes).
        messages : List[dict]
            Full conversation messages with roles and content. Content can be
            a string or multimodal list with items of shape {type: text|image_url}.

        Returns
        -------
        str
            The VLM's response as a string, or an empty string on error.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning("No messages provided for VLM analysis.")
            return ""

        # Resolve effective settings (function overrides > instance defaults)
        eff_temperature = temperature if temperature is not None else self.temperature
        eff_top_p = top_p if top_p is not None else self.top_p
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        eff_max_total_images = (
            max_total_images if max_total_images is not None else self.max_total_images
        )

        vlm = self.init_model(
            self.model_name, self.invoke_url, api_key=self.config.vlm.get_api_key()
        )

        (
            system_message,
            citations_instruct_user_message,
            chat_history_messages,
        ) = self.extract_and_process_messages(
            self.vlm_template,
            docs,
            messages,
            context_text,
            question_text,
            max_total_images=eff_max_total_images,
        )

        lc_messages = self.assemble_messages(
            system_message, citations_instruct_user_message, chat_history_messages
        )

        # Log final prompt with images redacted
        safe_prompt = self._redact_messages_for_logging(lc_messages)
        logger.info("VLM final prompt (images redacted): %s", safe_prompt)

        try:
            vlm_response = await self.invoke_model_async(
                vlm,
                lc_messages,
                temperature=eff_temperature,
                top_p=eff_top_p,
                max_tokens=eff_max_tokens,
            )
            logger.info(f"VLM Response: {vlm_response}")
            return str(vlm_response or "")
        except Exception as e:
            error_type = type(e).__name__
            if (
                "Connection" in error_type
                or "Connect" in error_type
                or isinstance(e, ConnectionError | OSError)
            ):
                vlm_url = self.invoke_url or "VLM service"
                error_msg = f"VLM NIM unavailable at {vlm_url}. Please verify the service is running and accessible."
                logger.error("Connection error in VLM analysis: %s", e)
                raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e
            logger.warning(
                f"Exception during VLM call with messages: {e}", exc_info=True
            )
            return ""

    async def stream_with_messages(
        self,
        docs: list[Any],
        messages: list[dict[str, Any]],
        context_text: str | None = None,
        question_text: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_total_images: int | None = None,
        **_: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the VLM asynchronously given full conversation and retrieved context.
        Yields incremental text chunks as they arrive.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning("No messages provided for VLM streaming.")
            return

        try:
            # Resolve effective settings (function overrides > instance defaults)
            eff_temperature = (
                temperature if temperature is not None else self.temperature
            )
            eff_top_p = top_p if top_p is not None else self.top_p
            eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            eff_max_total_images = (
                max_total_images
                if max_total_images is not None
                else self.max_total_images
            )

            vlm = self.init_model(
                self.model_name, self.invoke_url, api_key=self.config.vlm.get_api_key()
            )

            (
                system_message,
                citations_instruct_user_message,
                chat_history_messages,
            ) = self.extract_and_process_messages(
                self.vlm_template,
                docs,
                messages,
                context_text,
                question_text,
                max_total_images=eff_max_total_images,
            )

            lc_messages = self.assemble_messages(
                system_message, citations_instruct_user_message, chat_history_messages
            )

            # Log final prompt with images redacted
            safe_prompt = self._redact_messages_for_logging(lc_messages)
            logger.info("VLM final streaming prompt (images redacted): %s", safe_prompt)

            # Stream response chunks asynchronously
            idx = 0
            async for chunk in vlm.astream(
                lc_messages,
                temperature=eff_temperature,
                top_p=eff_top_p,
                max_tokens=eff_max_tokens,
            ):
                try:
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        yield content
                except Exception as e:
                    # Best-effort streaming; log and skip malformed chunks
                    logger.debug(
                        "Skipping malformed VLM stream chunk at index %s: %r; error: %s",
                        idx,
                        chunk,
                        e,
                        exc_info=True,
                    )
                idx += 1
        except Exception as e:
            error_type = type(e).__name__
            if (
                "Connection" in error_type
                or "Connect" in error_type
                or isinstance(e, ConnectionError | OSError)
            ):
                vlm_url = self.invoke_url or "VLM service"
                error_msg = f"VLM NIM unavailable at {vlm_url}. Please verify the service is running and accessible."
                logger.error("Connection error in VLM streaming: %s", e)
                raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e
            logger.warning(
                f"Exception during VLM streaming call with messages: {e}", exc_info=True
            )
            return

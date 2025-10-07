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
- Analyze up to 4 images using a VLM given a user question.
- Merge and resize images for VLM input.
- Extract and process images from document context (e.g., MinIO storage).
- Use an LLM to reason about the VLM's response and decide if it should be used.

Intended for use in NVIDIA's Retrieval-Augmented Generation (RAG) systems, compatible with LangChain and OpenAI-compatible VLM APIs.

Class:
    VLM: Provides methods for image analysis, merging, and VLM/LLM reasoning.
"""

import base64
import io
import os
import re
from logging import getLogger
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
import binascii

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.llm import get_llm, get_prompts
from nvidia_rag.utils.minio_operator import get_minio_operator, get_unique_thumbnail_id

logger = getLogger(__name__)


class VLM:
    """
    Handles image analysis and response reasoning using a Visual Language Model (VLM).

    Methods
    -------
    analyze_image(image_b64_list, question):
        Analyze up to 4 images with a VLM given a question.
    analyze_images_from_context(docs, question):
        Extracts images from document context and analyzes them with the VLM.
    reason_on_vlm_response(question, vlm_response, docs, llm_settings):
        Uses an LLM to reason about the VLM's response and decide if it should be used.
    """

    def __init__(self, vlm_model: str, vlm_endpoint: str):
        """
        Initialize the VLM with configuration and prompt templates.

        Raises
        ------
        EnvironmentError
            If VLM server URL or model name is not set in the environment.
        """

        self.invoke_url = vlm_endpoint
        self.model_name = vlm_model
        if not self.invoke_url or not self.model_name:
            raise OSError(
                "VLM server URL and model name must be set in the environment."
            )
        prompts = get_prompts()
        self.vlm_template = prompts["vlm_template"]
        self.vlm_response_reasoning_template = prompts[
            "vlm_response_reasoning_template"
        ]
        logger.info(f"VLM Model Name: {self.model_name}")
        logger.info(f"VLM Server URL: {self.invoke_url}")

    def analyze_image(
        self,
        image_b64_list: list[str],
        question: str,
        query_image_list: list[str] | None = None,
    ) -> str:
        """
        Analyze up to 4 images using the VLM for a given question.

        Parameters
        ----------
        image_b64_list : List[str]
            Base64 PNG images from context. Will be truncated to fit total limit.
        question : str
            The question to ask the VLM about the images.
        query_image_list : Optional[List[str]]
            Base64 PNG images directly associated with the user query (max 2).

        Returns
        -------
        str
            The VLM's response as a string, or an empty string on error.
        """
        if not image_b64_list and not query_image_list:
            logger.warning("No images provided for VLM analysis.")
            return ""

        vlm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base=self.invoke_url,
        )

        formatted_prompt = self.vlm_template.format(question=question)
        message = HumanMessage(content=[{"type": "text", "text": formatted_prompt}])

        config = get_config()
        max_total_images = max(0, int(config.vlm.max_total_images))
        max_query_images = max(0, int(config.vlm.max_query_images))
        max_context_images = max(0, int(config.vlm.max_context_images))

        logger.info(
            "VLM image limits - max_total_images=%d, max_query_images=%d, max_context_images=%d",
            max_total_images,
            max_query_images,
            max_context_images,
        )

        if max_query_images + max_context_images > max_total_images:
            logger.error(
                "Configured max_query_images (%d) + max_context_images (%d) exceed max_total_images (%d). Skipping VLM call.",
                max_query_images,
                max_context_images,
                max_total_images,
            )
            return ""

        # Determine query images to add (up to per-type cap)
        query_imgs = (query_image_list or [])[:max_query_images]
        if query_imgs:
            self._add_image_urls(query_imgs, max_query_images, message)

        # Add context images within remaining slots (up to per-type cap)
        if image_b64_list:
            remaining_slots = max(0, max_total_images - len(query_imgs))
            context_limit = min(max_context_images, remaining_slots)
            context_images = image_b64_list[:context_limit]
            if context_images:
                self._add_image_urls(context_images, max_context_images, message)

        vlm_response = vlm.invoke([message]).content.strip()
        logger.info(f"VLM Response: {vlm_response}")

        try:
            return vlm_response
        except Exception as e:
            logger.warning(f"Exception during VLM call: {e}", exc_info=True)
            return ""

    def _convert_image_url_to_png_b64(self, image_url: str) -> str:
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

    def _add_image_urls(
        self, image_list: list[str], max_images: int, message: HumanMessage
    ):
        """
        Add image URLs to message.content

        Parameters
        ----------
        image_list : List[str]
            List of base64-encoded PNG images.
        max_images : int
            Maximum number of images to add.
        message : Message
            The message to add the image URLs to.
        """
        for b64 in image_list[:max_images]:
            message.content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

    def analyze_images_from_context(
        self, docs: list[dict], question: str | list[dict[str, Any]]
    ) -> str:
        """
        Extract images from document context and analyze them with the VLM.

        Parameters
        ----------
        docs : List[dict]
            List of document objects with metadata containing image info.
        question : str | list[dict[str, Any]]
            The question to ask the VLM about the images.

        Returns
        -------
        str
            The VLM's response as a string, or an empty string if no images found.

        Raises
        ------
        ValueError
            If collection_name is not provided.
        """
        image_objects = []

        if not docs:
            logger.warning("No documents provided for image context analysis.")
            return ""

        logger.info("Number of documents: %s", len(docs))
        for doc in docs:
            try:
                content_metadata = doc.metadata.get("content_metadata", {})
                doc_type = content_metadata.get("type")
                if doc_type in ["image", "structured"]:
                    file_name = os.path.basename(
                        doc.metadata.get("source", {}).get("source_id", "")
                    )
                    page_number = content_metadata.get("page_number")
                    location = content_metadata.get("location")

                    unique_thumbnail_id = get_unique_thumbnail_id(
                        collection_name=doc.metadata.get("collection_name"),
                        file_name=file_name,
                        page_number=page_number,
                        location=location,
                    )

                    payload = get_minio_operator().get_payload(
                        object_name=unique_thumbnail_id
                    )
                    content = payload.get("content", "")
                    if not content:
                        logger.warning(
                            "Empty image content for %s; skipping", unique_thumbnail_id
                        )
                        continue
                    try:
                        image_bytes = base64.b64decode(content)
                    except (binascii.Error, ValueError) as decode_err:
                        logger.warning(
                            "Invalid base64 image content for %s; skipping (%s)",
                            unique_thumbnail_id,
                            decode_err,
                        )
                        continue

                    try:
                        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                    except UnidentifiedImageError:
                        logger.warning(
                            "Unidentified image content for %s; skipping", unique_thumbnail_id
                        )
                        continue
                    image_objects.append(img)
            except Exception as e:
                logger.warning(
                    f"Failed to process document for image extraction: {e}",
                    exc_info=True,
                )
                continue

        if not image_objects and isinstance(question, str):
            logger.warning(
                "Skipping VLM: no images extracted from context and no query images provided."
            )
            return ""

        image_b64_list = []
        for img in image_objects:
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_b64_list.append(image_b64)

        # Prepare the query for the VLM
        query_image_list = []
        vlm_query = question
        if isinstance(question, list):
            vlm_query = ""
            for item in question:
                if item.get("type") == "image_url":
                    # Convert image URL to PNG format for consistency
                    original_url = item.get("image_url").get("url")
                    png_b64 = self._convert_image_url_to_png_b64(original_url)
                    query_image_list.append(png_b64)
                elif item.get("type") == "text":
                    vlm_query = vlm_query + "\n" + item.get("text")
                vlm_query = vlm_query.strip()
            logger.debug("VLM query: %s", vlm_query)
            logger.debug(
                "Number of images found in the question: %s", len(query_image_list)
            )
        else:
            logger.debug("No image found in the question")
        return self.analyze_image(
            image_b64_list=image_b64_list,
            question=vlm_query,
            query_image_list=query_image_list,
        )

    def reason_on_vlm_response(
        self,
        question: str,
        vlm_response: str,
        docs: list[dict],
        llm_settings: dict[str, Any],
    ) -> bool:
        """
        Use an LLM to reason about the VLM's response and decide if it should be used.

        Parameters
        ----------
        question : str
            The original question posed to the VLM.
        vlm_response : str
            The response from the VLM.
        docs : List[dict]
            The document context used for reasoning.
        llm_settings : Dict[str, Any]
            Settings for initializing the LLM.

        Returns
        -------
        bool
            True if the LLM verdict is to USE the VLM response, False otherwise.
        """
        if not vlm_response.strip():
            logger.info("Empty VLM response provided for reasoning.")
            return False

        llm = get_llm(**llm_settings)

        template = get_prompts().get("vlm_response_reasoning_template", {})
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template.get("system", "")),
                ("human", template.get("human", "")),
            ]
        )

        parser = StrOutputParser()

        chain = prompt | llm | parser
        verdict = chain.invoke(
            {"question": question, "vlm_response": vlm_response, "text_context": docs}
        ).strip()
        logger.info("VLM response verdict: %s", verdict)
        return "USE" in verdict

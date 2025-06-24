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
import json
import os
from logging import getLogger
from typing import Any, Dict, List

import requests
import yaml
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image as PILImage

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
    _resize_and_merge_images(image_objects, target_height=400):
        Resize and merge images horizontally, returning a base64-encoded PNG.
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
            raise EnvironmentError(
                "VLM server URL and model name must be set in the environment."
            )
        prompts = get_prompts()
        self.vlm_template = prompts["vlm_template"]
        self.vlm_response_reasoning_template = prompts[
            "vlm_response_reasoning_template"
        ]
        logger.info(f"VLM Model Name: {self.model_name}")
        logger.info(f"VLM Server URL: {self.invoke_url}")

    def analyze_image(self, image_b64_list: List[str], question: str) -> str:
        """
        Analyze up to 4 images using the VLM for a given question.

        Parameters
        ----------
        image_b64_list : List[str]
            List of base64-encoded PNG images (max 4).
        question : str
            The question to ask the VLM about the images.

        Returns
        -------
        str
            The VLM's response as a string, or an empty string on error.
        """
        if not image_b64_list:
            logger.warning("No images provided for VLM analysis.")
            return ""

        vlm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base=self.invoke_url,
        )
        formatted_prompt = self.vlm_template.format(question=question)
        message = HumanMessage(content=[{"type": "text", "text": formatted_prompt}])

        if len(image_b64_list) > 4:
            image_b64_list = image_b64_list[:4]
            logger.warning(
                "VLM can only handle up to 4 images at a time. Only the first 4 images will be used."
            )
        for image_b64 in image_b64_list:
            message.content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
        try:
            return vlm.invoke([message]).content.strip()
        except Exception as e:
            logger.warning(f"Exception during VLM call: {e}", exc_info=True)
            return ""

    def _resize_and_merge_images(
        self, image_objects: List[PILImage.Image], target_height: int = 400
    ) -> str:
        """
        Resize images to a target height, merge them horizontally, and return as base64 PNG.

        Parameters
        ----------
        image_objects : List[PILImage.Image]
            List of PIL Image objects to merge.
        target_height : int, optional
            The height to resize all images to (default is 400).

        Returns
        -------
        str
            Base64-encoded PNG of the merged image.
        """
        if not image_objects:
            logger.warning("No image objects provided for merging.")
            return ""
        resized_images = []
        for img in image_objects:
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            resized_images.append(img.resize((new_width, target_height)))

        total_width = sum(img.width for img in resized_images)
        composite_image = PILImage.new(
            "RGB", (total_width, target_height), (255, 255, 255)
        )
        current_x = 0
        for img in resized_images:
            composite_image.paste(img, (current_x, 0))
            current_x += img.width

        with io.BytesIO() as buffer:
            composite_image.save(buffer, format="PNG")
            merged_image_bytes = buffer.getvalue()
            return base64.b64encode(merged_image_bytes).decode()

    def analyze_images_from_context(
        self, docs: List[dict], question: str
    ) -> str:
        """
        Extract images from document context and analyze them with the VLM.

        Parameters
        ----------
        docs : List[dict]
            List of document objects with metadata containing image info.
        question : str
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
                    image_bytes = base64.b64decode(content)
                    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_objects.append(img)
            except Exception as e:
                logger.warning(f"Failed to process document for image extraction: {e}", exc_info=True)
                continue

        if not image_objects:
            logger.warning("No valid images extracted from document context.")
            return ""

        image_b64_list = []
        for img in image_objects:
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_b64_list.append(image_b64)

        return self.analyze_image(image_b64_list=image_b64_list, question=question)

    def reason_on_vlm_response(
        self,
        question: str,
        vlm_response: str,
        docs: List[dict],
        llm_settings: Dict[str, Any],
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
        prompt = ChatPromptTemplate.from_template(self.vlm_response_reasoning_template)
        parser = StrOutputParser()

        chain = prompt | llm | parser
        verdict = chain.invoke(
            {"question": question, "vlm_response": vlm_response, "text_context": docs}
        ).strip()
        logger.info("VLM response verdict: %s", verdict)
        return "USE" in verdict

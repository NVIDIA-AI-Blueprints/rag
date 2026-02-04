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

"""Content utilities for parsing and processing multimodal content."""

from typing import Any


def _extract_text_from_content(content: Any) -> str:
    """Extract text content from either string or multimodal content.

    Args:
        content: Either a string or a list of content objects (multimodal)

    Returns:
        str: Extracted text content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from multimodal content
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            # Note: We ignore image_url content for text extraction
        return " ".join(text_parts)
    else:
        # Fallback for any other content type
        return str(content) if content is not None else ""


def _contains_images(content: Any) -> bool:
    """Check if content contains any images.

    Args:
        content: Either a string or a list of content objects (multimodal)

    Returns:
        bool: True if content contains images, False otherwise
    """
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return True
    return False


def _build_retriever_query_from_content(content: Any) -> tuple[str, bool]:
    """Build retriever query from either string or multimodal content.
    For multimodal content, includes both text and base64 images for VLM embedding support.

    Args:
        content: Either a string or a list of content objects (multimodal)

    Returns:
        tuple[str, bool]: Query string that may include base64 image data for VLM embeddings
        bool: True if image URL is provided, False otherwise
    """
    if isinstance(content, str):
        return content, False
    elif isinstance(content, list):
        # Build multimodal query with both text and base64 images
        query_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_content = item.get("text", "").strip()
                    if text_content:
                        query_parts.append(text_content)
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        # If image URL is provided, return it as is
                        return image_url, True
        # If no image URL is provided, return the text content
        return "\n\n".join(query_parts), False
    else:
        # Fallback for any other content type
        return (str(content) if content is not None else ""), False

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
Response handling utilities for integration tests
"""

import json
import logging
from typing import Any, Dict

import aiohttp

logger = logging.getLogger(__name__)


async def print_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
    """Helper to print API response and return JSON"""
    content_type = response.headers.get("content-type", "")

    # Handle streaming responses (Server-Sent Events)
    if "text/event-stream" in content_type:
        logger.debug(
            f"Detected streaming response with content-type: {content_type}"
        )
        streaming_result = await handle_streaming_response(response)
        return streaming_result

    # Handle regular JSON responses
    try:
        response_json = await response.json()
        logger.info(
            f"Response ({response.status}): {json.dumps(response_json, indent=2)}"
        )
        return response_json
    except Exception as e:
        text = await response.text()
        logger.error(f"Error parsing response: {e}")
        logger.debug(f"Raw response: {text}")
        return {"error": text}


async def handle_streaming_response(
    response: aiohttp.ClientResponse
) -> dict[str, Any]:
    """Handle streaming responses (Server-Sent Events) from RAG endpoints"""
    try:
        # Log raw response in debug mode
        raw_text = await response.text()
        logger.debug(f"Raw streaming response: {raw_text}")

        # Parse streaming response and extract text content
        response_text = ""
        first_chunk_with_citations = None
        last_usage: dict[str, Any] | None = None

        # Split by lines and process each chunk
        lines = raw_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            logger.debug(f"Processing line: {line}")
            if not line:
                continue

            # Handle Server-Sent Events format
            if line.startswith("data: "):
                line = line[len("data: ") :].strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON from line: {line}")
                continue

            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                last_usage = usage

            choices = data.get("choices", [])
            if not choices:
                continue

            # Save first chunk with citations for potential verification
            if first_chunk_with_citations is None and data.get("citations"):
                first_chunk_with_citations = data

            # Extract text content from streaming response
            delta = choices[0].get("delta", {})
            text = delta.get("content")
            if not text:
                message = choices[0].get("message", {})
                text = message.get("content", "")

            if text:
                response_text += text

        # Log the complete response text
        if response_text:
            logger.info(f"Streaming response text: {response_text}")
        else:
            logger.warning("No text content found in streaming response")

        # Prepare result with streaming response flag and citations if present
        result = {"streaming_response": True}

        # Store the extracted text for verification
        if response_text:
            result["extracted_text"] = response_text

        # Log citations if present and include them in the result
        if first_chunk_with_citations and first_chunk_with_citations.get(
            "citations"
        ):
            citations = first_chunk_with_citations["citations"]
            logger.info(
                f"Citations found: {len(citations.get('results', []))} results"
            )
            result["citations"] = citations

        if last_usage is not None:
            result["usage"] = last_usage

        return result

    except Exception as e:
        logger.error(f"Error handling streaming response: {e}")
        return {"streaming_response": False, "error": str(e)}


def extract_streaming_text(result: dict[str, Any]) -> str:
    """Extract text content from streaming response result"""
    try:
        # For streaming responses, the text is stored in the result by handle_streaming_response
        if "streaming_response" in result and "extracted_text" in result:
            return result["extracted_text"]
        return ""
    except Exception as e:
        logger.error(f"❌ Error extracting streaming text: {e}")
        return ""
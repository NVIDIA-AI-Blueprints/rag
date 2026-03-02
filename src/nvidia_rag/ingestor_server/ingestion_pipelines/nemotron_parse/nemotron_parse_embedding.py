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
This module contains the Nemotron Parse embedding model.
"""

from typing import Any

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.embedding import get_embedding_model

class NemotronParseEmbedding:

    def __init__(self, config: NvidiaRAGConfig):
        """
        Initialize the NemotronParseEmbedding class.
        Args:
            config: NvidiaRAGConfig - The configuration for the NemotronParseEmbedding class.
        Usage:
            nemotron_parse_embedding = NemotronParseEmbedding(config)
            result_element = nemotron_parse_embedding.embed_result_element(result_element)
        """
        self.config = config

        # Initialize embedding model
        self.document_embedder = get_embedding_model(
            model=self.config.embeddings.model_name,
            url=self.config.embeddings.server_url,
            config=self.config,
        )

    def __embed_text(self, text: str) -> list[float]:
        """
        Embed text.
        Args:
            text: str - The text to embed.
        Returns:
            list[float] - The embedding of the text.
        """
        return self.document_embedder.embed_documents([text])[0] if text else None

    def __pull_text(self, result_element: dict[str, Any]) -> str:
        """
        Pull text from a result element.
        Args:
            result_element: dict[str, Any] - The result element to pull text from.
        Returns:
            str - The text from the result element.
        """
        # Content to embed: we build a single string per element for the embedder.
        embed_content = None

        # ----- Text: use the raw text content as-is -----
        if result_element["document_type"] == "text":
            embed_content = result_element["metadata"]["content"]

        # ----- Structured (tables, charts, etc.) and image: may combine text + image -----
        elif result_element["document_type"] in ["structured", "image"]:
            # Optional base64 image payload (used when embed_images is True).
            image_content = content = result_element.get("metadata").get("content")

            # For structured elements (e.g. tables, charts), use table_content as text.
            if result_element["document_type"] == "structured":
                text = result_element["metadata"]["table_metadata"]["table_content"]

            # For images: use extracted text for page_image, otherwise use caption.
            if result_element["document_type"] == "image":
                if result_element["metadata"]["content_metadata"]["subtype"] == "page_image":
                    text = result_element["metadata"]["image_metadata"]["text"]
                else:
                    text = result_element["metadata"]["image_metadata"]["caption"]

            # If config says to embed images, combine text with base64 image so the model can use it.
            # Otherwise, embed only the text (caption/table content, etc.).
            if self.config.nemotron_parse.embed_images:
                embed_content = f"{text} data:image/png;base64,{image_content}"
            else:
                embed_content = text

        return embed_content

    def embed_result_element(self, result_element: dict[str, Any]) -> dict[str, Any]:
        """
        Embed a result element.
        Args:
            result_element: dict[str, Any] - The result element to embed.
        Returns:
            dict[str, Any] - The result element with the embedding.
        """
        text = self.__pull_text(result_element)
        embedding = self.__embed_text(text)
        result_element.get("metadata").update({"embedding": embedding})
        return result_element

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

"""Multimodal (VLM) reranker — requests-based compressor for text + image passages."""

import base64
import io
import logging
import os
import re
from urllib.parse import urlparse, urlunparse

import requests
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from PIL import Image as PILImage
from pydantic import Field, PrivateAttr

from nvidia_rag.utils.common import object_key_from_storage_uri, sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.object_store import get_object_store_operator

logger = logging.getLogger(__name__)


def _is_vlm_reranker_model(model: str) -> bool:
    """Return True when the configured reranker is the multimodal VL variant."""
    return "rerank-vl" in (model or "").lower()


def _build_vlm_rerank_invoke_url(url: str, model: str) -> str:
    """Build the correct hosted or self-hosted invoke URL for the VL reranker."""
    if not model:
        raise RuntimeError("VLM reranker requires an explicit model name.")

    if not url:
        model_path = model.split("/", 1)[-1]
        return (
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
            f"{model_path}/reranking"
        )

    base_url = sanitize_nim_url(url, model, "ranking")
    parsed = urlparse(base_url)
    host = (parsed.netloc or "").lower()
    path = parsed.path.rstrip("/")

    # Hosted NVIDIA endpoints use the full /retrieval/.../reranking path.
    if host in ("ai.api.nvidia.com", "api.nvcf.nvidia.com"):
        if path.endswith("/reranking"):
            return base_url.rstrip("/")
        model_path = model.split("/", 1)[-1]
        return (
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
            f"{model_path}/reranking"
        )

    # Self-hosted NIM serves reranking from the OpenAI-style /v1/ranking endpoint.
    if path.endswith("/ranking"):
        return base_url.rstrip("/")

    normalized_path = path
    if not normalized_path.endswith("/v1"):
        normalized_path = normalized_path + "/v1"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path + "/ranking",
            "",
            "",
            "",
        )
    )


def _image_to_png_b64(image_data: str) -> str:
    """Convert a base64 image string (or data URL) to PNG-format base64."""
    try:
        if image_data.startswith("data:image/"):
            match = re.match(r"data:image/[^;]+;base64,(.+)", image_data)
            if match:
                b64_data = match.group(1)
            else:
                logger.warning("Invalid data URL format: %s...", image_data[:100])
                return image_data
        else:
            b64_data = image_data

        image_bytes = base64.b64decode(b64_data)
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to convert image to PNG: %s", e)
        return image_data


class NVIDIAVLMRerank(BaseDocumentCompressor):
    """Requests-based reranker for multimodal passages (text + image)."""

    supports_image_passages: bool = True
    model: str = Field(description="The model to use for reranking.")
    url: str = Field(default="", description="URL endpoint for reranking service.")
    api_key: str | None = Field(default=None, description="Optional API key.")
    top_n: int = Field(default=5, ge=0, description="The number of documents to return.")
    default_headers: dict = Field(
        default_factory=dict,
        description="Default headers merged into all requests.",
    )
    timeout: int = Field(default=600, gt=0, description="Request timeout in seconds.")
    enable_image_input: bool = Field(
        default=False,
        description="When True, include images from retrieved citations in reranker passages.",
    )

    _session: requests.Session = PrivateAttr()
    _invoke_url: str = PrivateAttr()

    def __init__(
        self,
        *,
        model: str,
        url: str = "",
        api_key: str | None = None,
        top_n: int = 5,
        default_headers: dict | None = None,
        config: NvidiaRAGConfig | None = None,
        timeout: int = 600,
        enable_image_input: bool = False,
    ) -> None:
        if config is not None:
            enable_image_input = config.ranking.enable_vlm_image_input
        super().__init__(
            model=model,
            url=url,
            api_key=api_key,
            top_n=top_n,
            default_headers=default_headers or {},
            timeout=timeout,
            enable_image_input=enable_image_input,
        )
        self._invoke_url = _build_vlm_rerank_invoke_url(url, model)
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        """Build request headers for the VLM reranker API."""
        headers = {
            **self.default_headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_image_data_url(self, doc: Document) -> str | None:
        """Return a PNG data URL for a multimodal document when available."""
        metadata = getattr(doc, "metadata", {}) or {}
        content_md = metadata.get("content_metadata", {}) or {}
        doc_type = content_md.get("type")
        if doc_type not in ["image", "structured"]:
            return None

        collection_name = metadata.get("collection_name") or ""
        source_meta = metadata.get("source", {}) or {}
        source_id = (
            source_meta.get("source_id", "")
            or (source_meta.get("source_name", "") if isinstance(source_meta, dict) else "")
            if isinstance(source_meta, dict)
            else ""
        )
        file_name = os.path.basename(str(source_id)) if source_id else ""
        page_number = content_md.get("page_number")
        location = content_md.get("location")
        if not (
            collection_name
            and file_name
            and page_number is not None
            and location is not None
        ):
            return None

        try:
            source_location = doc.metadata.get("source").get("source_location")
            if source_location:
                object_name = object_key_from_storage_uri(source_location)
                raw_content = get_object_store_operator().get_object(object_name)
                content_b64 = base64.b64encode(raw_content).decode("ascii")
            else:
                content_b64 = ""
            if not content_b64:
                return None
            png_b64 = _image_to_png_b64(content_b64)
            return f"data:image/png;base64,{png_b64}"
        except Exception as e:
            logger.warning(
                "Unable to attach multimodal asset for reranking from %s: %s",
                (metadata.get("source") or {}).get("source_location"),
                e,
            )
            return None

    def _build_payload(
        self, query: str, documents: list[Document]
    ) -> dict[str, str | dict[str, str] | list[dict[str, str]]]:
        """Build the multimodal reranking payload expected by the API."""
        passages: list[dict[str, str]] = []
        for doc in documents:
            passage = {"text": doc.page_content}
            if self.enable_image_input:
                image_data_url = self._build_image_data_url(doc)
                if image_data_url:
                    passage["image"] = image_data_url
            passages.append(passage)

        return {
            "model": self.model,
            "query": {"text": query},
            "passages": passages,
        }

    def compress_documents(
        self,
        documents,
        query: str,
        callbacks: Callbacks | None = None,  # noqa: ARG002 - kept for BaseDocumentCompressor compatibility
    ) -> list[Document]:
        """Rerank documents and return them in API-provided order."""
        if not documents or self.top_n < 1:
            return []

        doc_list = list(documents)
        payload = self._build_payload(query=query, documents=doc_list)
        response = self._session.post(
            self._invoke_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        rankings = result.get("rankings", [])
        reranked_docs: list[Document] = []

        for ranking in rankings[: self.top_n]:
            index = ranking.get("index")
            if not isinstance(index, int) or not 0 <= index < len(doc_list):
                raise RuntimeError("invalid response from VLM reranker: index out of range")

            doc = doc_list[index]
            doc.metadata["relevance_score"] = ranking.get("logit")
            reranked_docs.append(doc)

        return reranked_docs

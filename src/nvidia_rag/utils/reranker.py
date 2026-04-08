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

"""The wrapper for interacting with reranking models.
1. _get_ranking_model: Creates the ranking model instance.
2. get_ranking_model: Returns the ranking model instance if it doesn't exist in cache.
"""

import base64
import logging
from functools import lru_cache

import requests
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_nvidia_ai_endpoints import NVIDIARerank
from pydantic import Field, PrivateAttr

from nvidia_rag.utils.common import NVIDIA_API_DEFAULT_HEADERS, sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig

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
    ) -> None:
        super().__init__(
            model=model,
            url=url,
            api_key=api_key,
            top_n=top_n,
            default_headers=default_headers or {},
            timeout=timeout,
        )
        self._invoke_url = _build_vlm_rerank_invoke_url(url, model)
        self._session = requests.Session()
        _ = config

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
                raw_content = get_minio_operator().get_object(object_name)
                content_b64 = base64.b64encode(raw_content).decode("ascii")
            else:
                content_b64 = ""
            if not content_b64:
                return None
            png_b64 = convert_image_url_to_png_b64(content_b64)
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


def _get_ranking_model(
    model="", url="", top_n=4, config: NvidiaRAGConfig | None = None
) -> BaseDocumentCompressor:
    """Create the ranking model.

    Args:
        model: Model name
        url: URL endpoint
        top_n: Number of top results
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        BaseDocumentCompressor: Base class for document compressors.

    Raises:
        RuntimeError: If the ranking model engine is not supported or initialization fails.
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "ranking")

    # Validate top_n
    if top_n is None:
        top_n = 4  # Use default for None
    elif not isinstance(top_n, int) or isinstance(top_n, bool):
        raise TypeError(
            f"reranker_top_k must be an integer, got {type(top_n).__name__}"
        )
    elif top_n <= 0:
        raise ValueError(f"reranker_top_k must be greater than 0, got {top_n}")

    if config.ranking.model_engine == "nvidia-ai-endpoints":
        api_key = config.ranking.get_api_key()

        if url:
            logger.info("Using ranking model hosted at %s", url)
            return NVIDIARerank(
                base_url=url,
                api_key=api_key,
                top_n=top_n,
                truncate="END",
                default_headers=NVIDIA_API_DEFAULT_HEADERS,
            )

        if model:
            logger.info("Using ranking model %s hosted at api catalog", model)
            return NVIDIARerank(
                model=model,
                api_key=api_key,
                top_n=top_n,
                truncate="END",
                default_headers=NVIDIA_API_DEFAULT_HEADERS,
            )

        # No model or URL provided
        raise RuntimeError(
            f"Ranking model configuration incomplete. "
            f"Either 'model' or 'url' must be provided. "
            f"Received: model='{model}', url='{url}'"
        )

    # Unsupported engine
    raise RuntimeError(
        f"Unsupported ranking model engine: '{config.ranking.model_engine}'. "
        f"Supported engines: 'nvidia-ai-endpoints'"
    )


def get_ranking_model(
    model="", url="", top_n=4, config: NvidiaRAGConfig | None = None
) -> BaseDocumentCompressor:
    """Create the ranking model.

    Args:
        model: Model name
        url: URL endpoint
        top_n: Number of top results
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        BaseDocumentCompressor: The ranking model instance.

    Raises:
        RuntimeError: If the ranking model cannot be created.
    """
    if config is None:
        config = NvidiaRAGConfig()
    return _get_ranking_model(model, url, top_n, config)

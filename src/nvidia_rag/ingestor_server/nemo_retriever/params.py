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

"""Adapter functions: NvidiaRAGConfig → NRL Pydantic param models.

All field-name mapping between RAG config and NRL params lives here.
When NRL renames a param field, only this file changes.

Param surface aligned with ``nemoretriever_playground_remote.py`` (extract / split /
store / embed). Extras are commented below for easy re-enable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    BatchTuningParams,
    CaptionParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    StoreParams,
    TextChunkParams,
)

from nvidia_rag.utils.object_store import DEFAULT_BUCKET_NAME

if TYPE_CHECKING:
    from nvidia_rag.utils.configuration import NvidiaRAGConfig
    from nvidia_rag.utils.vdb.vdb_base import VDBRag


def make_extract_params(
    config: NvidiaRAGConfig,
    override: dict[str, Any] | None = None,
) -> ExtractParams:
    """Build ``ExtractParams`` from ``NvidiaRAGConfig``.

    Maps (playground-aligned):
        ``config.nv_ingest.extract_*`` and ``extract_page_as_image``
        ``config.nv_ingest.page_elements_invoke_url`` … ``table_structure_invoke_url``
        ``batch_tuning`` → ``BatchTuningParams(pdf_extract_workers=9)`` as in playground.

    The *override* dict is applied last so callers can force individual fields
    (e.g. ``{"extract_images": False}`` for shallow extraction).
    """
    params: dict[str, Any] = {
        "extract_text": config.nv_ingest.extract_text,
        "extract_tables": config.nv_ingest.extract_tables,
        "extract_charts": config.nv_ingest.extract_charts,
        "extract_infographics": config.nv_ingest.extract_infographics,
        "extract_images": config.nv_ingest.extract_images,
        "extract_page_as_image": config.nv_ingest.extract_page_as_image,

        "table_output_format": "markdown",

        "batch_tuning": BatchTuningParams(pdf_extract_workers=9),
    }

    # if config.nv_ingest.pdf_extract_method is not None:
    #     params["method"] = config.nv_ingest.pdf_extract_method

    # api_key = config.nv_ingest.get_api_key()
    # if api_key:
    #     params["api_key"] = api_key

    if config.nv_ingest.page_elements_invoke_url:
        params["page_elements_invoke_url"] = config.nv_ingest.page_elements_invoke_url
    if config.nv_ingest.graphic_elements_invoke_url:
        params["graphic_elements_invoke_url"] = config.nv_ingest.graphic_elements_invoke_url
    if config.nv_ingest.ocr_invoke_url:
        params["ocr_invoke_url"] = config.nv_ingest.ocr_invoke_url
    if config.nv_ingest.table_structure_invoke_url:
        params["table_structure_invoke_url"] = config.nv_ingest.table_structure_invoke_url

    # TODO: Map config.nv_ingest.extract_tables_method to
    # ExtractParams.nemotron_parse_invoke_url when the Nemotron Parse endpoint
    # URL is added to NvIngestConfig and extract_tables_method == "nemotron_parse".

    if override:
        params.update(override)

    return ExtractParams(**params)


def make_split_params(
    config: NvidiaRAGConfig,
) -> TextChunkParams:
    """Build ``TextChunkParams`` from ``NvidiaRAGConfig``.

    Maps (playground-aligned):
        ``config.nv_ingest.chunk_size``    → ``max_tokens``
        ``config.nv_ingest.chunk_overlap`` → ``overlap_tokens``

    The *options* dict can override any ``TextChunkParams`` field.
    """
    params: dict[str, Any] = {
        "max_tokens": config.nv_ingest.chunk_size,
        "overlap_tokens": config.nv_ingest.chunk_overlap,
        # "tokenizer_model_id": config.nv_ingest.tokenizer,
    }
    return TextChunkParams(**params)


def make_embed_params(config: NvidiaRAGConfig) -> EmbedParams:
    """Build ``EmbedParams`` from ``NvidiaRAGConfig``.

    Maps (playground-aligned):
        ``config.embeddings.server_url``   → ``embed_invoke_url``
        ``config.embeddings.model_name``   → ``model_name``
        ``config.nv_ingest.structured_elements_modality`` → ``embed_modality`` (when set)
    """
    params: dict[str, Any] = {}

    if config.embeddings.model_name:
        params["model_name"] = config.embeddings.model_name

    if config.embeddings.server_url:
        params["embed_invoke_url"] = config.embeddings.server_url

    # api_key = config.embeddings.get_api_key()
    # if api_key:
    #     params["api_key"] = api_key

    if config.nv_ingest.structured_elements_modality:
        params["embed_modality"] = config.nv_ingest.structured_elements_modality

    # if config.nv_ingest.structured_elements_modality:
    #     params["structured_elements_modality"] = (
    #         config.nv_ingest.structured_elements_modality
    #     )

    return EmbedParams(**params)


def make_caption_params(config: NvidiaRAGConfig) -> CaptionParams:
    """Build ``CaptionParams`` from ``NvidiaRAGConfig``.

    Playground has ``caption()`` commented out; when enabled it used only
    ``endpoint_url`` and ``model_name``.
    """
    params: dict[str, Any] = {
        "model_name": config.nv_ingest.caption_model_name,
    }

    if config.nv_ingest.caption_endpoint_url:
        params["endpoint_url"] = config.nv_ingest.caption_endpoint_url

    # api_key = config.nv_ingest.get_api_key()
    # if api_key:
    #     params["api_key"] = api_key

    return CaptionParams(**params)


def make_html_chunk_params(config: NvidiaRAGConfig) -> HtmlChunkParams:
    """Build ``HtmlChunkParams`` from ``NvidiaRAGConfig``.

    ``HtmlChunkParams`` inherits ``TextChunkParams`` with no extra fields;
    the same chunk_size / chunk_overlap config values are reused.
    """
    return HtmlChunkParams(
        max_tokens=config.nv_ingest.chunk_size,
        overlap_tokens=config.nv_ingest.chunk_overlap,
    )


def make_audio_chunk_params(config: NvidiaRAGConfig) -> AudioChunkParams:  # noqa: ARG001
    """Build ``AudioChunkParams`` from ``NvidiaRAGConfig``.

    NRL defaults (split_type="size", split_interval=450) are used; no
    audio-specific fields exist in NvIngestConfig yet.
    """
    return AudioChunkParams()


def make_asr_params(config: NvidiaRAGConfig) -> ASRParams:
    """Build ``ASRParams`` from ``NvidiaRAGConfig``.

    Maps:
        ``config.nv_ingest.segment_audio`` → ``ASRParams.segment_audio``
    All other ASR fields (endpoints, protocol, auth) use NRL defaults.
    """
    return ASRParams(segment_audio=config.nv_ingest.segment_audio)


def make_store_params(config: NvidiaRAGConfig, vdb_op: VDBRag) -> StoreParams:
    """Build ``StoreParams`` from ``NvidiaRAGConfig`` and the active VDB operation.

    Maps (playground-aligned):
        ``config.object_store.*``  → ``storage_options`` (key, secret, client_kwargs)
        ``vdb_op.collection_name`` → path under default bucket, ``.../images`` suffix
        ``public_base_url``        → same as ``storage_uri`` (optional in NRL; required for metadata)
    """
    collection = vdb_op.collection_name
    if config.object_store.backend == "filesystem":
        storage_uri = (
            config.object_store.storage_root
            / DEFAULT_BUCKET_NAME
            / collection
            / "images"
        ).as_uri()
        storage_options: dict[str, Any] = {}
    else:
        endpoint_url = config.object_store.endpoint_url
        storage_options = {
            "key": config.object_store.access_key.get_secret_value(),
            "secret": config.object_store.secret_key.get_secret_value(),
            "client_kwargs": {"endpoint_url": endpoint_url},
        }
        storage_uri = f"s3://{DEFAULT_BUCKET_NAME}/{collection}/images"
    return StoreParams(
        storage_uri=storage_uri,
        public_base_url=storage_uri,
        storage_options=storage_options,
    )

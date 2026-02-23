# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter: run Nemotron Parse extraction and map results to ingestor result schema.

Produces list[list[dict]] (one inner list per file) with document_type and full metadata
for NemotronParseEmbedding and vdb_op.run(). image_metadata.image_location (bbox) is
set here for nv_ingest client location; content_url/source_location are set later in
main.__put_content_to_minio after MinIO upload.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nvidia_rag.utils.configuration import NvidiaRAGConfig

from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction import (
    ExtractionPipeline,
    build_extraction_config,
)
from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.helper import (
    empty_hierarchy,
    get_document_type_for_class,
    get_subtype_for_structured,
)

logger = logging.getLogger(__name__)


def _bbox_to_location(bbox: Optional[Dict[str, float]]) -> List[float]:
    """Convert bbox dict to [x1, y1, x2, y2] list for image_location / location."""
    if not bbox or not isinstance(bbox, dict):
        return [-1, -1, -1, -1]
    return [
        round(float(bbox.get("xmin", bbox.get("left", -1))), 4),
        round(float(bbox.get("ymin", bbox.get("top", -1))), 4),
        round(float(bbox.get("xmax", bbox.get("right", -1))), 4),
        round(float(bbox.get("ymax", bbox.get("bottom", -1))), 4),
    ]


def _source_metadata(
    source_name: str,
    source_id: str,
    source_location: str = "",
    source_type: str = "PDF",
    collection_id: str = "",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "source_name": source_name,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": source_type,
        "collection_id": collection_id,
        "date_created": now,
        "last_modified": now,
        "summary": "",
        "partition_id": -1,
        "access_level": -1,
        "custom_content": None,
    }


# Default bbox when none available; nv_ingest client requires a non-falsy location.
_DEFAULT_BBOX = [0.0, 0.0, 1.0, 1.0]
_SENTINEL_NO_BBOX = [-1, -1, -1, -1]


def _content_metadata(
    doc_type: str,
    page_number: int,
    description: str,
    subtype: str = "",
    location: Any = None,
    page_count: int = -1,
) -> Dict[str, Any]:
    # Ensure location is always a 4-list for nv_ingest (avoids "failed to find location").
    loc = location if isinstance(location, list) and len(location) == 4 and location != _SENTINEL_NO_BBOX else _DEFAULT_BBOX
    return {
        "type": doc_type,
        "description": description,
        "page_number": page_number,
        "hierarchy": empty_hierarchy(page_number, page_count),
        "subtype": subtype,
        "start_time": -1,
        "end_time": -1,
        "custom_content": None,
        "location": loc,
    }


def _text_metadata(
    text_type: str = "page",
    text_location: Optional[List[float]] = None,
) -> Dict[str, Any]:
    return {
        "text_type": text_type,
        "summary": "",
        "keywords": "",
        "language": "en",
        "text_location": text_location or [-1, -1, -1, -1],
        "text_location_max_dimensions": [-1, -1],
        "custom_content": None,
    }


def _image_metadata(
    caption: str = "",
    text: str = "",
    image_location: Optional[List[float]] = None,
    image_location_max_dimensions: Optional[List[int]] = None,
    image_type: str = "png",
) -> Dict[str, Any]:
    # Use default bbox when missing so nv_ingest client does not log "failed to find location".
    loc = image_location if (isinstance(image_location, list) and len(image_location) == 4 and image_location != _SENTINEL_NO_BBOX) else _DEFAULT_BBOX
    max_d = image_location_max_dimensions if (isinstance(image_location_max_dimensions, list) and len(image_location_max_dimensions) == 2) else [-1, -1]
    return {
        "image_type": image_type,
        "structured_image_type": "unknown",
        "caption": caption or "",
        "text": text or "",
        "image_location": loc,
        "image_location_max_dimensions": max_d,
        "uploaded_image_url": "",
        "width": -1,
        "height": -1,
        "custom_content": None,
    }


def _table_metadata(
    table_content: str = "",
    caption: str = "",
    table_format: str = "text",
    table_location: Optional[List[float]] = None,
    table_location_max_dimensions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    # nv_ingest expects table_location; use default bbox when missing.
    loc = table_location if (isinstance(table_location, list) and len(table_location) == 4 and table_location != _SENTINEL_NO_BBOX) else _DEFAULT_BBOX
    max_d = table_location_max_dimensions if (isinstance(table_location_max_dimensions, list) and len(table_location_max_dimensions) == 2) else [-1, -1]
    return {
        "caption": caption or "",
        "table_content": table_content or "",
        "table_format": table_format,
        "table_location": loc,
        "table_location_max_dimensions": max_d,
    }


def _one_result_element(
    pipeline_item: Dict[str, Any],
    source_name: str,
    source_id: str,
    collection_id: str,
    page_count: int = -1,
) -> Dict[str, Any]:
    """
    Map one pipeline result item (text, type, image, metadata) to ingestor result element.
    """
    elem_type = pipeline_item.get("type", "Unknown")
    text = (pipeline_item.get("text") or "").strip()
    image_b64 = pipeline_item.get("image") or ""
    meta = pipeline_item.get("metadata") or {}
    page_number = meta.get("page_number", 1)
    bbox = meta.get("bbox")

    doc_type = get_document_type_for_class(elem_type)
    location = _bbox_to_location(bbox) if bbox else _SENTINEL_NO_BBOX
    if elem_type == "page":
        location = _DEFAULT_BBOX

    if doc_type == "text":
        content = text
        description = "Unstructured text from PDF document."
        subtype = ""
        content_meta = _content_metadata(
            "text", page_number, description, subtype=subtype, location=location, page_count=page_count
        )
        return {
            "document_type": "text",
            "metadata": {
                "content": content,
                "content_url": "",
                "embedding": None,
                "source_metadata": _source_metadata(source_name, source_id, source_name, "PDF", collection_id),
                "content_metadata": content_meta,
                "audio_metadata": None,
                "text_metadata": _text_metadata("page", location if location != _SENTINEL_NO_BBOX else None),
                "image_metadata": None,
                "table_metadata": None,
                "chart_metadata": None,
                "error_metadata": None,
                "info_message_metadata": None,
                "debug_metadata": None,
                "raise_on_failure": False,
                "total_pages": None,
                "original_source_id": None,
                "original_source_name": None,
                "custom_content": None,
            },
        }

    if doc_type == "image":
        content = image_b64
        description = "Image extracted from PDF document."
        subtype = "page_image" if elem_type == "page" else ""
        content_meta = _content_metadata(
            "image", page_number, description, subtype=subtype, location=location, page_count=page_count
        )
        img_meta = _image_metadata(
            caption=text if elem_type != "page" else "",
            text=text if elem_type == "page" else "",
            image_location=location,
        )
        return {
            "document_type": "image",
            "metadata": {
                "content": content,
                "content_url": "",
                "embedding": None,
                "source_metadata": _source_metadata(source_name, source_id, source_name, "PDF", collection_id),
                "content_metadata": content_meta,
                "audio_metadata": None,
                "text_metadata": None,
                "image_metadata": img_meta,
                "table_metadata": None,
                "chart_metadata": None,
                "error_metadata": None,
                "info_message_metadata": None,
                "debug_metadata": None,
                "raise_on_failure": False,
                "total_pages": None,
                "original_source_id": None,
                "original_source_name": None,
                "custom_content": None,
            },
        }

    # structured (e.g. Table)
    content = text  # embedding uses table_content; content can be text or image
    description = "Structured content from PDF document."
    subtype = get_subtype_for_structured(elem_type)
    content_meta = _content_metadata(
        "structured", page_number, description, subtype=subtype, location=location, page_count=page_count
    )
    table_meta = _table_metadata(table_content=text, table_format="text", table_location=location)
    return {
        "document_type": "structured",
        "metadata": {
            "content": content,
            "content_url": "",
            "embedding": None,
            "source_metadata": _source_metadata(source_name, source_id, source_name, "PDF", collection_id),
            "content_metadata": content_meta,
            "audio_metadata": None,
            "text_metadata": None,
            "image_metadata": None,
            "table_metadata": table_meta,
            "chart_metadata": None,
            "error_metadata": None,
            "info_message_metadata": None,
            "debug_metadata": None,
            "raise_on_failure": False,
            "total_pages": None,
            "original_source_id": None,
            "original_source_name": None,
            "custom_content": None,
        },
    }


def run_extraction(
    filepaths: List[str],
    rag_config: NvidiaRAGConfig,
    collection_name: str,
    max_pages_per_pdf: Optional[int] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Run Nemotron Parse extraction for each PDF and return results in ingestor shape.

    Args:
        filepaths: List of absolute paths to PDF files.
        rag_config: Nvidia RAG config (used for Nemotron Parse URL, pipeline_mode, etc.).
        collection_name: Collection name for source_metadata.
        max_pages_per_pdf: Optional cap on pages per PDF.

    Returns:
        (results, failures): results is list[list[dict]] (one list per file);
        each dict has document_type and metadata. failures is list of error dicts.
    """
    config = build_extraction_config(rag_config, max_pages_per_pdf=max_pages_per_pdf)
    pipeline = ExtractionPipeline(config)
    results: List[List[Dict[str, Any]]] = []
    failures: List[Dict[str, Any]] = []

    for filepath in filepaths:
        if not filepath or not str(filepath).lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF path: %s", filepath)
            results.append([])
            continue
        filepath = str(filepath)
        file_elements: List[Dict[str, Any]] = []
        try:
            for page_num, result_list in pipeline.process_pdf_generator(filepath):
                for item in result_list:
                    rec = _one_result_element(
                        item,
                        source_name=filepath,
                        source_id=filepath,
                        collection_id=collection_name,
                        page_count=-1,
                    )
                    file_elements.append(rec)
            results.append(file_elements)
        except Exception as e:
            logger.exception("Extraction failed for %s: %s", filepath, e)
            failures.append({"filepath": filepath, "error": str(e)})
            results.append([])

    return results, failures

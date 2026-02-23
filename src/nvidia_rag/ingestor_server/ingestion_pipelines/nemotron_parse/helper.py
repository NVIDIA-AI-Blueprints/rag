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
Helper functions for Nemotron Parse pipeline.

Includes class label -> document_type mapping (text | image | structured) for
ingestor result schema compatibility.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Map Nemotron Parse element type to ingestor document_type (text, image, or structured).
# All text-like classes -> "text"; Picture/page -> "image"; Table -> "structured".
CLASS_TO_DOCUMENT_TYPE: Dict[str, str] = {
    "Text": "text",
    "Title": "text",
    "Section-header": "text",
    "List-item": "text",
    "TOC": "text",
    "Bibliography": "text",
    "Footnote": "text",
    "Page-header": "text",
    "Page-footer": "text",
    "Formula": "text",
    "Caption": "text",
    "Picture": "image",
    "Table": "structured",
    "page": "image",  # full-page element
}
DEFAULT_DOCUMENT_TYPE = "text"


def get_document_type_for_class(element_type: str) -> str:
    """Return ingestor document_type (text, image, structured) for a Nemotron Parse class label."""
    return CLASS_TO_DOCUMENT_TYPE.get(element_type, DEFAULT_DOCUMENT_TYPE)


def get_subtype_for_structured(element_type: str) -> str:
    """Return content_metadata.subtype for structured elements (e.g. table, chart)."""
    if element_type == "Table":
        return "table"
    if element_type == "Picture":
        return "image"
    return "table"


def empty_hierarchy(page_number: int, page_count: int = -1) -> Dict[str, Any]:
    """Build empty hierarchy dict expected by ingestor schema."""
    return {
        "page_count": page_count,
        "page": page_number,
        "block": -1,
        "line": -1,
        "span": -1,
        "nearby_objects": {
            "text": {"content": [], "bbox": [], "type": []},
            "images": {"content": [], "bbox": [], "type": []},
            "structured": {"content": [], "bbox": [], "type": []},
        },
    }

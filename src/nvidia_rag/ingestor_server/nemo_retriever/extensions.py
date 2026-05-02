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

"""NRL extension metadata with no dependency on ``handler`` or ``nemo_retriever``.

``filters.filter_unsupported`` imports from here so that ``main.py`` can load
only the filter module on the NRL path without first importing ``GraphIngestor``.
"""

from __future__ import annotations

_PDF_DOC_EXTS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"})
_TEXT_EXTS: frozenset[str] = frozenset({".txt"})
_HTML_EXTS: frozenset[str] = frozenset({".html", ".htm"})
_AUDIO_VIDEO_EXTS: frozenset[str] = frozenset({".mp3", ".wav", ".mp4"})

_EXT_TO_TYPE: dict[str, str] = {
    **dict.fromkeys(_PDF_DOC_EXTS, "pdf_doc"),
    **dict.fromkeys(_IMAGE_EXTS, "image"),
    **dict.fromkeys(_TEXT_EXTS, "text"),
    **dict.fromkeys(_HTML_EXTS, "html"),
    **dict.fromkeys(_AUDIO_VIDEO_EXTS, "audio_video"),
}

_TYPE_ORDER: tuple[str, ...] = ("pdf_doc", "image", "text", "html", "audio_video")

NRL_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXT_TO_TYPE)

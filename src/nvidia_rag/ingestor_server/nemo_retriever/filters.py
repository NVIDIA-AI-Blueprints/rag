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

"""File-type filtering utilities for the NRL ingestion pipeline.

``filter_unsupported`` is the NRL-backend equivalent of
``__remove_unsupported_files`` / ``__get_non_supported_files`` from the
NV-Ingest path in ``ingestor_server/main.py``.  Keeping it here rather than
inline in ``main.py`` gives it a single home and keeps ``main.py`` slim.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nvidia_rag.ingestor_server.nemo_retriever.extensions import NRL_SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Human-readable extension list built once at import time — used in error messages.
_SUPPORTED_EXTS_DISPLAY: str = ", ".join(
    sorted(ext.lstrip(".") for ext in NRL_SUPPORTED_EXTENSIONS)
)
_UNSUPPORTED_ERROR: str = (
    f"Unsupported file type, supported file types are: {_SUPPORTED_EXTS_DISPLAY}"
)


def filter_unsupported(
    filepaths: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split *filepaths* into NRL-supported and unsupported groups.

    Unsupported files are never sent to ``NemoRetrieverHandler.ingest()``;
    instead they are returned as pre-built failure tuples so the caller can
    surface a clear ``"Unsupported file type"`` error to the end user.

    Parameters
    ----------
    filepaths:
        Absolute paths of documents intended for NRL ingestion.

    Returns
    -------
    tuple[list[str], list[tuple[str, str]]]
        ``(supported_filepaths, unsupported_failures)``

        *supported_filepaths*
            Files whose extensions are in ``NRL_SUPPORTED_EXTENSIONS``;
            safe to pass directly to ``NemoRetrieverHandler.ingest()``.

        *unsupported_failures*
            ``(filepath, error_message)`` pairs for files whose extensions
            are not recognised by NRL.  The error message begins with
            ``"Unsupported file type"`` so existing test assertions and
            response validators continue to match.
    """
    supported: list[str] = []
    unsupported_failures: list[tuple[str, str]] = []

    for fp in filepaths:
        ext = Path(fp).suffix.lower()
        if ext in NRL_SUPPORTED_EXTENSIONS:
            supported.append(fp)
        else:
            logger.warning(
                "NRL: unsupported file extension %r for %r — "
                "will be reported as 'Unsupported file type'",
                ext,
                Path(fp).name,
            )
            unsupported_failures.append((fp, _UNSUPPORTED_ERROR))

    if unsupported_failures:
        logger.info(
            "NRL file-type filter: %d supported file(s), %d unsupported file(s)",
            len(supported),
            len(unsupported_failures),
        )

    return supported, unsupported_failures

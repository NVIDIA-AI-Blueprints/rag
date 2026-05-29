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

"""NeMo-Retriever Library integration package for the ingestor server.

Public surface (used when ``config.nv_ingest.backend == "nrl"`` — see
``ingestor_server/main.py`` for lazy imports):

* :class:`NemoRetrieverHandler` — async façade over ``GraphIngestor``.
* :class:`IngestSchemaManager` — stable accessor API over the NRL DataFrame.
* :func:`filter_unsupported` — split filepaths into supported / unsupported
  before invoking ``NemoRetrieverHandler.ingest()``.

Submodules are not imported at package load time; use :func:`__getattr__` or
import submodules directly (as ``main.py`` does) so optional ``nemo_retriever``
wheels are not required for the default NV-Ingest backend.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NemoRetrieverHandler",
    "IngestSchemaManager",
    "filter_unsupported",
]


def __getattr__(name: str) -> Any:
    if name == "NemoRetrieverHandler":
        from nvidia_rag.ingestor_server.nemo_retriever.handler import (
            NemoRetrieverHandler,
        )

        return NemoRetrieverHandler
    if name == "IngestSchemaManager":
        from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
            IngestSchemaManager,
        )

        return IngestSchemaManager
    if name == "filter_unsupported":
        from nvidia_rag.ingestor_server.nemo_retriever.filters import (
            filter_unsupported,
        )

        return filter_unsupported
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

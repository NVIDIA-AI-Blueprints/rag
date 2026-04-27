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

Public surface (imported by ``ingestor_server/main.py`` when
``config.nv_ingest.backend == "nrl"``):

* :class:`NemoRetrieverHandler` — async façade over ``GraphIngestor``.
* :class:`IngestSchemaManager` — stable accessor API over the NRL DataFrame.
* :func:`filter_unsupported` — split filepaths into supported / unsupported
  before invoking ``NemoRetrieverHandler.ingest()``.
"""

from nvidia_rag.ingestor_server.nemo_retriever.filters import filter_unsupported
from nvidia_rag.ingestor_server.nemo_retriever.handler import NemoRetrieverHandler
from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
    IngestSchemaManager,
)

__all__ = [
    "NemoRetrieverHandler",
    "IngestSchemaManager",
    "filter_unsupported",
]

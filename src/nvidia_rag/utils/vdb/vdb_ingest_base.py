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
This module provides VDBRagIngest, a VDBRag subclass with nv_ingest support.

VDBRagIngest combines VDBRag (pure abstract base) with VDB from nv_ingest_client,
providing full ingestion capabilities. This class should be used by ingestor_server
and any components that require nv_ingest functionality.

For components that only need retrieval operations (like rag_server), use VDBRag
from vdb_base.py instead to avoid the nv_ingest dependency.
"""

import logging

from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

# Flag to track if nv_ingest is available
NV_INGEST_AVAILABLE = False

try:
    from nv_ingest_client.util.vdb.adt_vdb import VDB

    NV_INGEST_AVAILABLE = True

    class VDBRagIngest(VDBRag, VDB):
        """
        VDBRag with nv_ingest VDB support for ingestion operations.

        This class combines the VDBRag abstract interface with the VDB class from
        nv_ingest_client, providing full support for both RAG retrieval and
        nv_ingest ingestion operations.

        Use this class when:
        - You need to perform document ingestion via nv_ingest
        - You're working in the ingestor_server context

        Use VDBRag instead when:
        - You only need retrieval operations
        - You want to avoid nv_ingest dependencies (e.g., rag_server)
        """

        pass

except ImportError:
    logger.warning(
        "Optional nv_ingest_client module not installed. "
        "VDBRagIngest will be an alias for VDBRag without VDB capabilities."
    )
    # Fallback: VDBRagIngest is just VDBRag without nv_ingest support
    VDBRagIngest = VDBRag


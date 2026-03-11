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
This module provides VDBRagIngest, a VDBRag subclass with nv_ingest support,
and SerializedVDBWrapper for serializing concurrent VDB write operations.

VDBRagIngest combines VDBRag (pure abstract base) with VDB from nv_ingest_client,
providing full ingestion capabilities. This class should be used by ingestor_server
and any components that require nv_ingest functionality.

For components that only need retrieval operations (like rag_server), use VDBRag
from vdb_base.py instead to avoid the nv_ingest dependency.
"""

import hashlib
import logging
import threading

from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

try:
    from nv_ingest_client.util.vdb.adt_vdb import VDB

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

    class _ResolvedFuture:
        """Thin wrapper so a pre-resolved value satisfies the .result() protocol."""

        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class SerializedVDBWrapper:
        """Wraps a VDB op to serialize write operations while keeping reads parallel.

        When multiple batches run extraction concurrently, their VDB writes can
        overlap and cause indexing timeouts (e.g., GPU_CAGRA JIT compilation takes
        longer than the client's patience window). This wrapper uses a threading
        lock to ensure only one batch writes to the VDB at a time.

        Additionally injects a deterministic ``chunk_hash`` into each record's
        ``content_metadata`` before the VDB write.  The hash uses the same
        algorithm as ``entity_extractor._chunk_id`` (SHA-256, first 16 hex chars)
        so that graph entity ``source_chunk_ids`` can be matched to Milvus
        documents at query time.
        """

        def __init__(self, vdb_op):
            self._vdb_op = vdb_op
            self._write_lock = threading.Lock()

        @staticmethod
        def _inject_chunk_hashes(records):
            """Add ``chunk_hash`` to ``content_metadata`` for every text/structured record.

            Uses the same hash as ``entity_extractor._chunk_id``:
            ``hashlib.sha256(text.encode()).hexdigest()[:16]``
            """
            for result in records:
                if result is None:
                    continue
                items = [result] if isinstance(result, dict) else result
                for element in items:
                    if not isinstance(element, dict):
                        continue
                    metadata = element.get("metadata", {})
                    doc_type = element.get("document_type", "")
                    content = None
                    if doc_type == "text":
                        content = metadata.get("content")
                    elif doc_type == "structured":
                        content = (metadata.get("table_metadata") or {}).get(
                            "table_content"
                        )
                    if content:
                        h = hashlib.sha256(str(content).encode()).hexdigest()[:16]
                        cm = metadata.setdefault("content_metadata", {})
                        cm["chunk_hash"] = h

        def run_async(self, records):
            with self._write_lock:
                if hasattr(records, "result"):
                    resolved = records.result()
                    self._inject_chunk_hashes(resolved)
                    return self._vdb_op.run_async(_ResolvedFuture(resolved))
                self._inject_chunk_hashes(records)
                return self._vdb_op.run_async(records)

        def run(self, records):
            with self._write_lock:
                self._inject_chunk_hashes(records)
                return self._vdb_op.run(records)

        def write_to_index(self, records, **kwargs):
            with self._write_lock:
                return self._vdb_op.write_to_index(records, **kwargs)

        def create_index(self, **kwargs):
            with self._write_lock:
                return self._vdb_op.create_index(**kwargs)

        def __getattr__(self, name):
            return getattr(self._vdb_op, name)

    VDB.register(SerializedVDBWrapper)

except ImportError:
    logger.warning(
        "Optional nv_ingest_client module not installed. "
        "VDBRagIngest will be an alias for VDBRag without VDB capabilities."
    )
    # Fallback: VDBRagIngest is just VDBRag without nv_ingest support
    VDBRagIngest = VDBRag
    SerializedVDBWrapper = None

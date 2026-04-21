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

"""NemoRetrieverHandler: async façade over the synchronous GraphIngestor.

The handler owns the ``GraphIngestor`` lifecycle and runs it on a
``ThreadPoolExecutor`` so callers remain async.  One executor slot is kept
(``max_workers=1``) because NRL / Ray manages its own thread pool internally
and submitting overlapping pipelines would race for GPU resources.

Threading model::

    FastAPI request
        └── INGESTION_TASK_HANDLER.submit_task(_task)
                └── asyncio background task
                        └── loop.run_in_executor(
                                NemoRetrieverHandler._executor,
                                handler._run_sync,
                                ingestor
                            )
                                └── GraphIngestor.ingest()  [blocking, in thread]
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pandas as pd
from nemo_retriever.graph_ingestor import GraphIngestor

from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
    IngestSchemaManager,
)
from nvidia_rag.ingestor_server.nemo_retriever.params import (
    make_caption_params,
    make_embed_params,
    make_extract_params,
    make_split_params,
    make_store_params,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig

if TYPE_CHECKING:
    from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


class NemoRetrieverHandler:
    """Single object the rest of ingestor-server talks to for NRL-backed ingestion.

    Parameters
    ----------
    config:
        Full ``NvidiaRAGConfig`` — used to build all NRL param objects.
    """

    def __init__(self, config: NvidiaRAGConfig) -> None:
        self._config = config
        # Read run_mode added to NvIngestConfig (see NRL_INTEGRATION_PLAN.md §9).
        # Fall back to "inprocess" until configuration.py is updated.
        self._run_mode: str = getattr(
            config.nv_ingest, "nrl_run_mode", "batch"
        )
        # One pipeline at a time: NRL / Ray owns its own worker threads.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        logger.info(
            "NemoRetrieverHandler initialised (run_mode=%s)", self._run_mode
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        filepaths: list[str],
        vdb_op: VDBRag | None,
        split_options: dict[str, Any] | None = None,
        extract_override: dict[str, Any] | None = None,
        store_images: bool = True,
    ) -> IngestSchemaManager:
        """Run the full extraction → split → (caption) → (store) → embed pipeline.

        Parameters
        ----------
        filepaths:
            Absolute paths (or glob patterns) of documents to ingest.
        vdb_op:
            Active ``VDBRag`` instance; controls whether embed / store stages
            are added.  ``None`` skips both.
        split_options:
            When not ``None``, a ``TextChunkParams``-compatible override dict is
            merged with config defaults and a split stage is added.  Pass ``{}``
            to split with config defaults only.
        extract_override:
            Field overrides forwarded directly to ``make_extract_params``.
        store_images:
            When ``True`` *and* ``vdb_op`` is not ``None``, a store stage for
            extracted images is added before embedding.

        Returns
        -------
        IngestSchemaManager
            Stable wrapper around the raw NRL result DataFrame.
        """
        ingestor = self._build_ingestor(
            filepaths, split_options, extract_override, vdb_op, store_images
        )
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(self._executor, self._run_sync, ingestor)
        return IngestSchemaManager(df)

    async def ingest_shallow(self, filepaths: list[str]) -> IngestSchemaManager:
        """Text-only extraction with no embed and no VDB write — for fast summarisation.

        Equivalent to ``extract(images=False, tables=False, charts=False,
        infographics=False)`` with no ``.embed()`` stage.
        """
        ingestor = self._build_shallow_ingestor(filepaths)
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(self._executor, self._run_sync, ingestor)
        return IngestSchemaManager(df)

    # ------------------------------------------------------------------
    # Internal pipeline builders
    # ------------------------------------------------------------------

    def _build_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,
        extract_override: dict[str, Any] | None,
        vdb_op: VDBRag | None,
        store_images: bool,
    ) -> GraphIngestor:
        """Construct and configure the full GraphIngestor pipeline.

        Does NOT call ``.vdb_upload()`` — VDB write is handled by the caller
        via ``VectorStore`` backends after the ingest completes.

        TODO(NRL-VDB): When NRL PR #1822 merges and they add a VDBRag-compatible
        upload stage, wire it as:
            gi = gi.vdb_upload(backend)  # backend implements VectorStore ABC
        and remove the post-ingest ``write_rows()`` call in main.py.
        """
        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract(make_extract_params(self._config, extract_override))

        gi = gi.split(make_split_params(self._config))

        if self._config.nv_ingest.extract_images:
            gi = gi.caption(make_caption_params(self._config))

        if store_images and vdb_op is not None:
            gi = gi.store(make_store_params(self._config, vdb_op))

        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))

        return gi

    def _build_shallow_ingestor(self, filepaths: list[str]) -> GraphIngestor:
        """Construct a text-only GraphIngestor for fast summarisation."""
        shallow_override: dict[str, Any] = {
            "extract_images": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_infographics": False,
        }
        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract(make_extract_params(self._config, shallow_override))
        return gi

    # ------------------------------------------------------------------
    # Synchronous execution (runs inside ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _run_sync(self, ingestor: GraphIngestor) -> pd.DataFrame:
        """Call ``ingestor.ingest()`` and materialise the result as a DataFrame.

        ``inprocess`` mode returns a ``pandas.DataFrame`` directly.
        ``batch`` mode returns a Ray Dataset that must be materialised with
        ``take_all()`` before this thread exits.

        TODO(NRL-ASYNC): When NRL adds progress callbacks to GraphIngestor,
        wire them into ``IngestionStateManager.update_document_status()`` here.
        Expected interface (speculative):
            ingestor.on_document_complete = lambda doc_id: state_mgr.mark_completed(doc_id)
            ingestor.on_document_failed   = lambda doc_id, err: state_mgr.mark_failed(doc_id, err)
        """
        logger.info(
            "NemoRetrieverHandler._run_sync starting (run_mode=%s)", self._run_mode
        )
        result = ingestor.ingest()

        if self._run_mode == "batch":
            # result is a ray.data.Dataset; materialise to a local list.
            # Ray is already initialised by GraphIngestor.ingest() in batch mode,
            # so we don't need to import it here — take_all() is a method on the
            # Dataset object that GraphIngestor returned.
            records = result.take_all()
            df = pd.DataFrame(records)
        else:
            df = result  # already a pandas.DataFrame in inprocess mode

        # ------------------------------------------------------------------------------------------
        # Ingestion summary for debugging.
        chunks = df.to_dict(orient="records")
        ct_counts = Counter(dict(r).get("_content_type") for r in chunks)
        logger.info("Ingestion using Nemo Retriever summary:")
        logger.info("  - Number of chunks: %d", len(chunks))
        logger.info("  - _content_type counts: %s", dict(ct_counts))
        if chunks:
            logger.info(
                "  - Contains embeddings: %s",
                chunks[0].get("_contains_embeddings"),
            )
        # ------------------------------------------------------------------------------------------

        logger.debug(
            "NemoRetrieverHandler._run_sync complete (%d rows)", len(df)
        )
        return df

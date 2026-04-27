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
                        └── for each type group (sequential):
                                loop.run_in_executor(
                                    NemoRetrieverHandler._executor,
                                    handler._run_sync,
                                    ingestor,
                                    type_label,
                                )
                                    └── GraphIngestor.ingest()  [blocking, in thread]

Supported file types and pipelines
-----------------------------------
Each extension is classified into one of five type groups.  A separate
``GraphIngestor`` is built per non-empty group and executed **sequentially**
(each group completes before the next starts).  The NRL reference example
(``graph_pipeline.py``) also runs one ingestor per file type — parallelising
across types is not safe because Ray manages its own internal concurrency.

    pdf_doc     — .pdf / .docx / .pptx
                  → extract → split → [caption] → [embed] → [store]

    image       — .jpg / .jpeg / .png / .tiff / .tif / .bmp
                  → extract_image_files → [caption] → [embed] → [store]

    text        — .txt
                  → extract_txt  (chunking built-in via TextChunkParams)
                  → [embed]

    html        — .html / .htm
                  → extract_html  (chunking built-in via HtmlChunkParams)
                  → [embed]

    audio_video — .mp3 / .wav / .mp4
                  → extract_audio  (chunking built-in via AudioChunkParams
                                    + ASR via ASRParams)
                  → [embed]

Files with unrecognised extensions are logged as warnings and routed to the
``pdf_doc`` pipeline (backward-compatible fallback).

Results from all type groups are ``pd.concat``-ed into a single DataFrame
before being wrapped in ``IngestSchemaManager``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from nemo_retriever.graph_ingestor import GraphIngestor

from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
    IngestSchemaManager,
)
from nvidia_rag.ingestor_server.nemo_retriever.params import (
    make_asr_params,
    make_audio_chunk_params,
    make_caption_params,
    make_embed_params,
    make_extract_params,
    make_html_chunk_params,
    make_split_params,
    make_store_params,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig

if TYPE_CHECKING:
    from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-type extension sets and lookup table
# ---------------------------------------------------------------------------

_PDF_DOC_EXTS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"})
_TEXT_EXTS: frozenset[str] = frozenset({".txt"})
_HTML_EXTS: frozenset[str] = frozenset({".html", ".htm"})
_AUDIO_VIDEO_EXTS: frozenset[str] = frozenset({".mp3", ".wav", ".mp4"})

# Flat extension → type-key lookup built once at import time.
_EXT_TO_TYPE: dict[str, str] = {
    **dict.fromkeys(_PDF_DOC_EXTS, "pdf_doc"),
    **dict.fromkeys(_IMAGE_EXTS, "image"),
    **dict.fromkeys(_TEXT_EXTS, "text"),
    **dict.fromkeys(_HTML_EXTS, "html"),
    **dict.fromkeys(_AUDIO_VIDEO_EXTS, "audio_video"),
}

# Processing order: deterministic for concat ordering and log output.
_TYPE_ORDER: tuple[str, ...] = ("pdf_doc", "image", "text", "html", "audio_video")

# Public constant: all extensions recognised by the NRL pipeline.
# Exposed so callers (e.g. main.py) can pre-filter unsupported files before
# calling ingest(), mirroring SUPPORTED_FILE_TYPES in the NV-Ingest path.
NRL_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXT_TO_TYPE)


class NemoRetrieverHandler:
    """Single object the rest of ingestor-server talks to for NRL-backed ingestion.

    Parameters
    ----------
    config:
        Full ``NvidiaRAGConfig`` — used to build all NRL param objects.
    """

    def __init__(self, config: NvidiaRAGConfig) -> None:
        self._config = config
        self._run_mode: str = getattr(config.nv_ingest, "nrl_run_mode", "batch")
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
        """Run the full extraction → split → (caption) → embed → (store) pipeline.

        Files are first classified by extension into type groups.  A dedicated
        ``GraphIngestor`` pipeline is built for each non-empty group using the
        appropriate NRL extraction method.  Type groups are processed
        **sequentially** — one group fully completes before the next begins.
        Results from all groups are concatenated into a single
        ``IngestSchemaManager``.

        Parameters
        ----------
        filepaths:
            Absolute paths of documents to ingest.  Mixed file types are
            supported; see module docstring for the full extension matrix.
        vdb_op:
            Active ``VDBRag`` instance; controls whether embed / store stages
            are added.  ``None`` skips both.
        split_options:
            Reserved for future per-call split overrides (not yet applied).
        extract_override:
            Field overrides forwarded directly to ``make_extract_params``
            (applies to pdf_doc and image groups only).
        store_images:
            When ``True`` *and* ``vdb_op`` is not ``None``, a store stage for
            extracted images is added to the pdf_doc and image pipelines.

        Returns
        -------
        IngestSchemaManager
            Stable wrapper around the concatenated NRL result DataFrame.
        """
        logger.info(
            "ingest() called with %d filepath(s), vdb_op=%s, store_images=%s",
            len(filepaths),
            vdb_op is not None,
            store_images,
        )

        type_ingestors = self._build_ingestors(
            filepaths, split_options, extract_override, vdb_op, store_images
        )

        if not type_ingestors:
            logger.warning(
                "No supported files found among %d filepath(s); returning empty result",
                len(filepaths),
            )
            return IngestSchemaManager(pd.DataFrame())

        logger.info(
            "Executing %d type group(s) sequentially: %s",
            len(type_ingestors),
            [tk for tk, _ in type_ingestors],
        )

        loop = asyncio.get_running_loop()
        frames: list[pd.DataFrame] = []

        # Sequential execution: each group is awaited before the next starts.
        # Parallelising across type groups is not safe — Ray manages its own
        # internal concurrency and running overlapping pipelines would race
        # for GPU resources (same reason max_workers=1 is used).
        for idx, (type_key, gi) in enumerate(type_ingestors, start=1):
            logger.info(
                "Starting type group %d/%d: %s",
                idx,
                len(type_ingestors),
                type_key,
            )
            t0 = time.perf_counter()
            df = await loop.run_in_executor(
                self._executor, self._run_sync, gi, type_key
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "Finished type group %d/%d: %s — %d rows in %.2fs",
                idx,
                len(type_ingestors),
                type_key,
                len(df),
                elapsed,
            )
            frames.append(df)

        combined = (
            pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        )
        logger.info(
            "All type groups complete: %d total rows from %d group(s)",
            len(combined),
            len(type_ingestors),
        )
        return IngestSchemaManager(combined)

    async def ingest_shallow(self, filepaths: list[str]) -> IngestSchemaManager:
        """Text-only extraction with no embed and no VDB write — for fast summarisation.

        Only PDF / DOCX / PPTX files are processed; other file types are skipped
        with a warning because the shallow pipeline runs text-only PDF extraction.
        Equivalent to ``extract(images=False, tables=False, charts=False,
        infographics=False)`` with no ``.embed()`` stage.
        """
        logger.info("ingest_shallow() called with %d filepath(s)", len(filepaths))

        supported = [
            fp for fp in filepaths if Path(fp).suffix.lower() in _PDF_DOC_EXTS
        ]
        skipped_count = len(filepaths) - len(supported)
        if skipped_count:
            skipped_paths = [
                fp for fp in filepaths
                if Path(fp).suffix.lower() not in _PDF_DOC_EXTS
            ]
            logger.warning(
                "ingest_shallow: skipping %d non-PDF/DOC/PPTX file(s) "
                "(shallow extraction is text-only): %s",
                skipped_count,
                skipped_paths,
            )

        if not supported:
            logger.warning(
                "ingest_shallow: no supported files remain after filtering; "
                "returning empty result"
            )
            return IngestSchemaManager(pd.DataFrame())

        logger.info(
            "ingest_shallow: processing %d PDF/DOC/PPTX file(s)", len(supported)
        )
        ingestor = self._build_shallow_ingestor(supported)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        df = await loop.run_in_executor(
            self._executor, self._run_sync, ingestor, "pdf_doc_shallow"
        )
        logger.info(
            "ingest_shallow complete: %d rows in %.2fs", len(df), time.perf_counter() - t0
        )
        return IngestSchemaManager(df)

    # ------------------------------------------------------------------
    # File classification
    # ------------------------------------------------------------------

    def _classify_filepaths(self, filepaths: list[str]) -> dict[str, list[str]]:
        """Group *filepaths* by file-type key using ``_EXT_TO_TYPE``.

        Unknown extensions are warned about and assigned to ``pdf_doc`` as a
        backward-compatible fallback so existing integrations are not broken.

        Returns
        -------
        dict[str, list[str]]
            Keys are ``_TYPE_ORDER`` entries; values are (possibly empty) lists
            of file paths belonging to that group.
        """
        classified: dict[str, list[str]] = {t: [] for t in _TYPE_ORDER}

        for fp in filepaths:
            ext = Path(fp).suffix.lower()
            type_key = _EXT_TO_TYPE.get(ext)
            if type_key is None:
                logger.warning(
                    "Unrecognised file extension %r for %r; "
                    "defaulting to pdf_doc pipeline",
                    ext,
                    fp,
                )
                type_key = "pdf_doc"
            classified[type_key].append(fp)

        non_empty = {k: len(v) for k, v in classified.items() if v}
        logger.info(
            "File classification result (%d file(s) total): %s", len(filepaths), non_empty
        )
        return classified

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def _build_ingestors(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,
        extract_override: dict[str, Any] | None,
        vdb_op: VDBRag | None,
        store_images: bool,
    ) -> list[tuple[str, GraphIngestor]]:
        """Classify *filepaths* and build one ``GraphIngestor`` per type group.

        Uses a dispatch table keyed on type strings so adding a new file type
        requires only a new entry in the table and a corresponding
        ``_build_*_ingestor`` method — no branching logic here.

        Returns
        -------
        list[tuple[str, GraphIngestor]]
            Ordered by ``_TYPE_ORDER``; only non-empty groups are included.
        """
        classified = self._classify_filepaths(filepaths)

        builders: dict[str, Any] = {
            "pdf_doc":     self._build_pdf_doc_ingestor,
            "image":       self._build_image_ingestor,
            "text":        self._build_text_ingestor,
            "html":        self._build_html_ingestor,
            "audio_video": self._build_audio_video_ingestor,
        }

        result: list[tuple[str, GraphIngestor]] = []
        for type_key in _TYPE_ORDER:
            paths = classified[type_key]
            if not paths:
                continue
            logger.debug(
                "Building %s ingestor for %d file(s): %s",
                type_key,
                len(paths),
                paths,
            )
            gi = builders[type_key](paths, split_options, extract_override, vdb_op, store_images)
            result.append((type_key, gi))

        logger.info(
            "_build_ingestors: %d ingestor(s) prepared for type group(s): %s",
            len(result),
            [tk for tk, _ in result],
        )
        return result

    def _build_pdf_doc_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,  # noqa: ARG002
        extract_override: dict[str, Any] | None,
        vdb_op: VDBRag | None,
        store_images: bool,
    ) -> GraphIngestor:
        """Pipeline: extract → split → [caption] → [embed] → [store].

        Does NOT call ``.vdb_upload()`` — VDB write is handled by the caller
        via ``VectorStore`` backends after the ingest completes.

        TODO(NRL-VDB): When NRL PR #1822 merges and they add a VDBRag-compatible
        upload stage, wire it as:
            gi = gi.vdb_upload(backend)  # backend implements VectorStore ABC
        and remove the post-ingest ``write_rows()`` call in main.py.
        """
        stages = ["extract"]
        if self._config.nv_ingest.enable_split:
            stages.append("split")
        if self._config.nv_ingest.extract_images:
            stages.append("caption")
        if vdb_op is not None:
            stages.append("embed")
        if store_images and vdb_op is not None:
            stages.append("store")
        logger.debug("pdf_doc pipeline stages: %s", " → ".join(stages))

        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract(make_extract_params(self._config, extract_override))
        if self._config.nv_ingest.enable_split:
            gi = gi.split(make_split_params(self._config))
        if self._config.nv_ingest.extract_images:
            gi = gi.caption(make_caption_params(self._config))
        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))
        if store_images and vdb_op is not None:
            gi = gi.store(make_store_params(self._config, vdb_op))
        return gi

    def _build_image_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,  # noqa: ARG002
        extract_override: dict[str, Any] | None,
        vdb_op: VDBRag | None,
        store_images: bool,
    ) -> GraphIngestor:
        """Pipeline: extract_image_files → [caption] → [embed] → [store].

        No split stage: each detected page element (text region, table, chart)
        produced by the image extraction pipeline is already its own row;
        splitting image-derived text is not meaningful.
        """
        stages = ["extract_image_files"]
        if self._config.nv_ingest.extract_images:
            stages.append("caption")
        if vdb_op is not None:
            stages.append("embed")
        if store_images and vdb_op is not None:
            stages.append("store")
        logger.debug("image pipeline stages: %s", " → ".join(stages))

        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract_image_files(make_extract_params(self._config, extract_override))
        if self._config.nv_ingest.extract_images:
            gi = gi.caption(make_caption_params(self._config))
        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))
        if store_images and vdb_op is not None:
            gi = gi.store(make_store_params(self._config, vdb_op))
        return gi

    def _build_text_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,  # noqa: ARG002
        extract_override: dict[str, Any] | None,  # noqa: ARG002
        vdb_op: VDBRag | None,
        store_images: bool,  # noqa: ARG002
    ) -> GraphIngestor:
        """Pipeline: extract_txt → [embed].

        Chunking is built into ``TextChunkParams`` passed to ``extract_txt``;
        no separate split stage is added.  Caption and store stages are not
        applicable to plain-text content.
        """
        stages = ["extract_txt"]
        if vdb_op is not None:
            stages.append("embed")
        logger.debug("text pipeline stages: %s", " → ".join(stages))

        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract_txt(make_split_params(self._config))
        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))
        return gi

    def _build_html_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,  # noqa: ARG002
        extract_override: dict[str, Any] | None,  # noqa: ARG002
        vdb_op: VDBRag | None,
        store_images: bool,  # noqa: ARG002
    ) -> GraphIngestor:
        """Pipeline: extract_html → [embed].

        Chunking is built into ``HtmlChunkParams`` passed to ``extract_html``;
        no separate split stage is added.  Caption and store stages are not
        applicable to HTML content.
        """
        stages = ["extract_html"]
        if vdb_op is not None:
            stages.append("embed")
        logger.debug("html pipeline stages: %s", " → ".join(stages))

        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract_html(make_html_chunk_params(self._config))
        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))
        return gi

    def _build_audio_video_ingestor(
        self,
        filepaths: list[str],
        split_options: dict[str, Any] | None,  # noqa: ARG002
        extract_override: dict[str, Any] | None,  # noqa: ARG002
        vdb_op: VDBRag | None,
        store_images: bool,  # noqa: ARG002
    ) -> GraphIngestor:
        """Pipeline: extract_audio → [embed].

        Audio/video chunking and ASR transcription are configured via
        ``AudioChunkParams`` and ``ASRParams``; no separate split stage is
        added.  Caption and store stages are not applicable to transcribed
        audio content.
        """
        stages = ["extract_audio"]
        if vdb_op is not None:
            stages.append("embed")
        logger.debug("audio_video pipeline stages: %s", " → ".join(stages))

        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract_audio(
            make_audio_chunk_params(self._config),
            asr_params=make_asr_params(self._config),
        )
        if vdb_op is not None:
            gi = gi.embed(make_embed_params(self._config))
        return gi

    def _build_shallow_ingestor(self, filepaths: list[str]) -> GraphIngestor:
        """Construct a text-only GraphIngestor for fast summarisation.

        Callers must pre-filter *filepaths* to PDF/DOC/PPTX before calling
        this method; ``ingest_shallow`` handles that filtering.
        """
        shallow_override: dict[str, Any] = {
            "extract_images": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_infographics": False,
        }
        logger.debug(
            "pdf_doc_shallow pipeline: extract (text-only) for %d file(s)",
            len(filepaths),
        )
        gi = GraphIngestor(run_mode=self._run_mode)
        gi = gi.files(filepaths)
        gi = gi.extract(make_extract_params(self._config, shallow_override))
        return gi

    # ------------------------------------------------------------------
    # Synchronous execution (runs inside ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _run_sync(
        self, ingestor: GraphIngestor, type_label: str = ""
    ) -> pd.DataFrame:
        """Call ``ingestor.ingest()`` and materialise the result as a DataFrame.

        ``inprocess`` mode returns a ``pandas.DataFrame`` directly.
        ``batch`` mode returns a Ray Dataset that must be materialised with
        ``take_all()`` before this thread exits.

        Parameters
        ----------
        ingestor:
            Fully configured ``GraphIngestor`` ready to execute.
        type_label:
            Short string identifying the file-type group (e.g. ``"pdf_doc"``,
            ``"audio_video"``).  Prepended to every log line so mixed-type
            runs remain easy to trace in the logs.

        TODO(NRL-ASYNC): When NRL adds progress callbacks to GraphIngestor,
        wire them into ``IngestionStateManager.update_document_status()`` here.
        Expected interface (speculative):
            ingestor.on_document_complete = lambda doc_id: state_mgr.mark_completed(doc_id)
            ingestor.on_document_failed   = lambda doc_id, err: state_mgr.mark_failed(doc_id, err)
        """
        label = f"[{type_label}] " if type_label else ""
        logger.info(
            "NemoRetrieverHandler._run_sync %sstarting (run_mode=%s)",
            label,
            self._run_mode,
        )

        result = ingestor.ingest()

        if self._run_mode == "batch":
            # result is a ray.data.Dataset; materialise to a local list.
            # Ray is already initialised by GraphIngestor.ingest() in batch mode,
            # so we don't need to import it here — take_all() is a method on the
            # Dataset object that GraphIngestor returned.
            logger.debug("%smaterialising Ray Dataset via take_all()", label)
            records = result.take_all()
            df = pd.DataFrame(records)
        else:
            df = result  # already a pandas.DataFrame in inprocess mode

        # Per-type ingestion summary for debugging.
        chunks = df.to_dict(orient="records")
        ct_counts = Counter(dict(r).get("_content_type") for r in chunks)
        logger.info("Ingestion %ssummary:", label)
        logger.info("  - Number of chunks: %d", len(chunks))
        logger.info("  - _content_type counts: %s", dict(ct_counts))
        if chunks:
            logger.info(
                "  - Contains embeddings: %s",
                chunks[0].get("_contains_embeddings"),
            )

        logger.debug(
            "NemoRetrieverHandler._run_sync %scomplete (%d rows)", label, len(df)
        )
        return df
